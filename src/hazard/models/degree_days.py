from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import PosixPath, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    Indicator,
    MultiYearAverageIndicatorBase,
    ThresholdBasedAverageIndicator,
)
from hazard.utilities.map_utilities import (
    check_map_bounds,
    transform_epsg4326_to_epsg3857,
)

from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon
from hazard.protocols import OpenDataset, ReadWriteDataArray, WriteDataArray

from dask.distributed import Client
import numpy as np
from rasterio.crs import CRS  # type: ignore
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int


class DegreeDays(IndicatorModel[BatchItem]):
    """Calculates degree days from temperature data sets."""

    def __init__(
        self,
        threshold: float = 32,
        window_years: int = 20,
        gcms: Iterable[str] = [
            "ACCESS-CM2",
            "CMCC-ESM2",
            "CNRM-CM6-1",
            "MIROC6",
            "MPI-ESM1-2-LR",
            "NorESM2-MM",
        ],
        scenarios: Iterable[str] = ["historical", "ssp126", "ssp245", "ssp585"],
        central_year_historical: int = 2005,
        central_years: Iterable[int] = [2030, 2040, 2050],
    ):
        """Construct model to calculate degree days from temperature data sets.

        Args:
            threshold (float, optional): Degree days above threshold are calculated. Defaults to 32.
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation. Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation. Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_years (Iterable[int], optional): Central years to include in calculation. Defaults to [2010, 2030, 2040, 2050].
        """

        self.threshold: float = 273.15 + threshold  # in Kelvin; degree days above 32C
        # 1995 to 2014 (2010), 2021 tp 2040 (2030), 2031 to 2050 (2040), 2041 to 2060 (2050)
        self.window_years = window_years
        self.gcms = gcms
        self.scenarios = scenarios
        self.central_years = central_years
        self.central_year_historical = central_year_historical

    def batch_items(self) -> Iterable[BatchItem]:
        """Items to process."""
        # just one for now
        resource = self._resource()
        for gcm in self.gcms:
            for scenario in self.scenarios:
                central_years = (
                    [self.central_year_historical]
                    if scenario == "historical"
                    else self.central_years
                )
                for central_year in central_years:
                    yield BatchItem(
                        resource=resource,
                        gcm=gcm,
                        scenario=scenario,
                        central_year=central_year,
                    )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._resource()]

    def _resource(self):
        with open(os.path.join(os.path.dirname(__file__), "degree_days.md"), "r") as f:
            description = f.read().replace('\u00c2\u00b0', 'u00b0')
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="mean_degree_days/above/32c",
            indicator_model_gcm="{gcm}",
            path="chronic_heat/osc/v2/mean_degree_days_v2_above_32c_{gcm}_{scenario}_{year}",
            display_name="Mean degree days above 32°C/{gcm}",
            description=description,
            params={"gcm": list(self.gcms)},
            group_id="",
            display_groups=["Mean degree days"],
            map=MapInfo(  # type:ignore
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_index=255,
                    max_value=4000.0,
                    units="degree days",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                path="mean_degree_days_v2_above_32c_{gcm}_{scenario}_{year}_map",
                source="map_array",
            ),
            units="degree days",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource

    def run_single(
        self,
        item: BatchItem,
        source: OpenDataset,
        target: ReadWriteDataArray,
        client: Client,
    ):
        average_deg_days = self._average_degree_days(client, source, target, item)
        average_deg_days.attrs["crs"] = CRS.from_epsg(4326)
        pp = self._item_path(item)
        logger.info(f"Writing array to {str(pp)}")
        target.write(str(pp), average_deg_days)
        pp = self._item_path(item)
        pp_map = pp.with_name(pp.name + "_map")
        self._generate_map(
            str(pp),
            str(pp_map),
            item.resource.map.bounds if item.resource.map is not None else None,
            target,
        )
        return

    def _generate_map(
        self,
        path: str,
        map_path: str,
        bounds: Optional[List[Tuple[float, float]]],
        target: ReadWriteDataArray,
    ):
        logger.info(f"Generating map projection for file {path}; reading file")
        da = target.read(path)
        logger.info(f"Reprojecting to EPSG:3857")
        # reprojected = transform_epsg4326_to_epsg3857(average_deg_days.sel(latitude=slice(85, -85)))
        reprojected = transform_epsg4326_to_epsg3857(da)
        # sanity check bounds:
        (top, right, bottom, left) = check_map_bounds(reprojected)
        if top > 85.05 or bottom < -85.05:
            raise ValueError("invalid range")
        logger.info(f"Writing map file {map_path}")
        target.write(map_path, reprojected)
        return

    def _average_degree_days(
        self,
        client: Client,
        source: OpenDataset,
        target: WriteDataArray,
        item: BatchItem,
    ):
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        years = range(
            item.central_year - self.window_years // 2,
            item.central_year + self.window_years // 2 + (self.window_years % 2),
        )
        logger.info(
            f"Calculating average degree days, gcm={item.gcm}, scenario={item.scenario}, years={list(years)}"
        )
        futures = []
        for year in years:
            future = client.submit(
                self._degree_days, source, item.gcm, item.scenario, year
            )
            futures.append(future)
        deg_days = client.gather(futures)
        average = sum(deg_days) / float(len(years))
        return enforce_conventions_lat_lon(average)

    def _degree_days(
        self, source: OpenDataset, gcm: str, scenario: str, year: int
    ) -> xr.DataArray:
        """Calculate degree days for Dataset provided."""
        # check DataArray
        logger.info(f"Starting calculation for year {year}")
        with source.open_dataset_year(gcm, scenario, "tasmax", year) as ds:
            result = self._degree_days_indicator(ds, year)
            logger.info(f"Calculation complete for year {year}")
            return result

    def _degree_days_indicator(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        if any(coord not in ds.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (ds.time.dt.year != year).any():
            raise ValueError("unexpected year found")
        # normalize to 365 days
        scale = 365.0 / len(ds.time)
        # will raise error if taxmax not present
        return scale * xr.where(
            ds.tasmax > self.threshold, ds.tasmax - self.threshold, 0
        ).sum(dim=["time"])

    def _item_path(self, item: BatchItem) -> PosixPath:
        return PosixPath(
            item.resource.path.format(
                gcm=item.gcm, scenario=item.scenario, year=item.central_year
            )
        )


class AboveBelow(Enum):
    BELOW = 0
    ABOVE = 1


class HeatingCoolingDegreeDays(ThresholdBasedAverageIndicator):
    def __init__(
        self,
        threshold_temps_c: List[float] = [16, 20, 24],
        window_years: int = MultiYearAverageIndicatorBase._default_window_years,
        gcms: Iterable[str] = MultiYearAverageIndicatorBase._default_gcms,
        scenarios: Iterable[str] = MultiYearAverageIndicatorBase._default_scenarios,
        central_year_historical: int = MultiYearAverageIndicatorBase._default_central_year_historical,
        central_years: Iterable[
            int
        ] = MultiYearAverageIndicatorBase._default_central_years,
    ):
        """Create indicators based on average number of days above different temperature thresholds.

        Args:
            threshold_temps_c (List[float], optional): Temperature thresholds in degrees C. Defaults to [14, 16, 20, 24, 28].
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation. Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation. Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int): Central year to include in calculation for historical scenario. Defaults to 2005.
            central_years (Iterable[int], optional): Central years to include in calculation. Defaults to [2010, 2030, 2040, 2050].
        """
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c
        self.resources = self._resource()

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return list(self._resource().values())

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        """For a single year and batch item calculate the indicators (i.e. one per threshold temperature)."""
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            above = self._degree_days_indicator(tas, year, AboveBelow.ABOVE)
            below = self._degree_days_indicator(tas, year, AboveBelow.BELOW)
        logger.info(f"Calculation complete for year {year}")
        return [
            Indicator(
                above,
                PurePosixPath(
                    self.resources["above"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
                self.resources["above"].map.bounds,
            ),
            Indicator(
                below,
                PurePosixPath(
                    self.resources["below"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
                self.resources["below"].map.bounds,
            ),
        ]

    def _degree_days_indicator(
        self, tas: xr.DataArray, year: int, above_below: AboveBelow
    ) -> xr.DataArray:
        if any(coord not in tas.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (tas["time"].dt.year != year).any():
            raise ValueError("unexpected year found")
        # normalize to 365 days
        scale = 365.0 / len(tas["time"])
        # will raise error if tax not present
        da = xr.DataArray(
            coords={
                "index": self.threshold_temps_c,
                "lat": tas.coords["lat"],
                "lon": tas.coords["lon"],
            },
            dims=["index", "lat", "lon"],
        )
        for i, threshold in enumerate(self.threshold_temps_c):
            threshold_k = 273.15 + threshold
            da[i, :, :] = (
                scale
                * xr.where(tas > threshold_k, tas - threshold_k, 0).sum(dim=["time"])
                if above_below == AboveBelow.ABOVE
                else scale
                * xr.where(tas < threshold_k, threshold_k - tas, 0).sum(dim=["time"])
            )
        return da

    def _resource(self):
        resources: Dict[str, HazardResource] = {}
        for above_below in ["above", "below"]:
            with open(
                os.path.join(os.path.dirname(__file__), "degree_days.md"), "r"
            ) as f:
                description = f.read()
            resource = HazardResource(
                hazard_type="ChronicHeat",
                indicator_id=f"mean_degree_days/{above_below}/index",
                indicator_model_gcm="{gcm}",
                path="chronic_heat/osc/v2/mean_degree_days_"
                + f"{above_below}"
                + "_index_{gcm}_{scenario}_{year}",
                display_name="Mean degree days "
                + f"{above_below}"
                + " index value/{gcm}",
                description=description,
                params={"gcm": list(self.gcms)},
                group_id="",
                display_groups=["Mean degree days"],
                map=MapInfo(  # type:ignore
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=4000.0,
                        units="degree days",
                    ),
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -60.0),
                        (-180.0, -60.0),
                    ],
                    index_values=self.threshold_temps_c,
                    path="mean_degree_days_"
                    + f"{above_below}"
                    + "_index_{gcm}_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="degree days",
                scenarios=[
                    Scenario(id="historical", years=[self.central_year_historical]),
                    Scenario(id="ssp126", years=list(self.central_years)),
                    Scenario(id="ssp245", years=list(self.central_years)),
                    Scenario(id="ssp585", years=list(self.central_years)),
                ],
            )
            resources[above_below] = resource
        return resources
