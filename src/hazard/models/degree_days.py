"""Climate Hazard Indicator: Degree Days Calculation."""

import logging
import os
from contextlib import ExitStack
from enum import Enum
from pathlib import PosixPath, PurePosixPath
from typing import Sequence
from typing_extensions import Dict, Iterable, List, Optional

import xarray as xr
from dask.distributed import Client
from rasterio.crs import CRS  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    BatchItem,
    Indicator,
    MultiYearAverageIndicatorBase,
    ThresholdBasedAverageIndicator,
)
from hazard.sources.osc_zarr import OscZarr
from hazard.protocols import OpenDataset, ReadWriteDataArray, WriteDataArray
from hazard.utilities.description_utilities import get_indicator_period_descriptions
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon

logger = logging.getLogger(__name__)


class DegreeDays(IndicatorModel[BatchItem]):
    """Calculates degree days from temperature data sets."""

    def __init__(
        self,
        threshold: float = 32,
        window_years: Optional[int] = 20,
        gcms: Sequence[str] = [
            "ACCESS-CM2",
            "CMCC-ESM2",
            "CNRM-CM6-1",
            "MIROC6",
            "MPI-ESM1-2-LR",
            "NorESM2-MM",
        ],
        scenarios: Sequence[str] = ["historical", "ssp126", "ssp245", "ssp585"],
        central_year_historical: int = 2005,
        central_years: Sequence[int] = [2030, 2040, 2050],
        source_dataset: str = "NEX-GDDP-CMIP6",
    ):
        """Construct model to calculate degree days from temperature data sets.

        Args:
            threshold (float, optional): Degree days above threshold are calculated. Defaults to 32.
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Sequence[str], optional): Global Circulation Models to include in calculation.
                Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Sequence[str], optional): Scenarios to include in calculation.
                Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int, optional): Central reference year for historical data.
                Defaults to 2005.
            central_years (Sequence[int], optional): Central years to include in calculation.
                Defaults to [2010, 2030, 2040, 2050].
            source_dataset (str, optional): Name of the dataset used as the data source.
                Defaults to "NEX-GDDP-CMIP6".

        """

        self.threshold: float = (
            273.15 + threshold
        )  # in Kelvin; degree days above {threshold}C
        self.threshold_c: float = threshold
        # 1995 to 2014 (2010), 2021 tp 2040 (2030), 2031 to 2050 (2040), 2041 to 2060 (2050)
        self.window_years = window_years
        self.gcms = gcms
        self.scenarios = scenarios
        self.central_years = central_years
        self.central_year_historical = central_year_historical
        self.source_dataset = source_dataset
        self.resource = self._resource()

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

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images for the resource."""
        create_tiles_for_resource(source, target, self.resource)

    def _resource(self):
        description = self._generate_description()

        scenarios = []

        if "historical" in self.scenarios:
            scenarios.append(
                Scenario(id="historical", years=[self.central_year_historical])
            )

        for s in self.scenarios:
            if s == "historical":
                continue
            scenarios.append(Scenario(id=s, years=list(self.central_years)))

        path = (
            f"chronic_heat/osc/v2/mean_degree_days_v2_above_{self.threshold_c}c"
            + "_{gcm}_{scenario}_{year}"
        )
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id=f"mean_degree_days/above/{self.threshold_c}c",
            indicator_model_gcm="{gcm}",
            path=path,
            display_name=f"Mean degree days above {self.threshold_c}°C/" + "{gcm}",
            description=description,
            params={"gcm": list(self.gcms)},
            group_id="",
            license="Creative Commons",
            source="",
            version="",
            resolution="1800 m",
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
                bbox=[-180.0, -60.0, 180.0, 85.0],
                path="maps/" + path,
                source="map_array_pyramid",
            ),
            units="degree days",
            scenarios=scenarios,
        )
        return resource

    def _generate_description(self):
        current_file_dir_name = os.path.dirname(__file__)
        with open(os.path.join(current_file_dir_name, "degree_days.md"), "r") as f:
            description = f.read().replace("\u00c2\u00b0", "\u00b0")

        sources_dir_name = os.path.join(
            os.path.dirname(current_file_dir_name), "sources"
        )
        source_dataset_filename = self.source_dataset.lower().replace("-", "_")
        with open(
            os.path.join(sources_dir_name, f"{source_dataset_filename}.md"), "r"
        ) as source_dataset_description:
            description += "\n" + source_dataset_description.read()

        description += "\n" + get_indicator_period_descriptions(
            self.scenarios,
            self.central_year_historical,
            self.central_years,
            self.window_years,
        )

        return description

    def run_single(
        self,
        item: BatchItem,
        source: OpenDataset,
        target: ReadWriteDataArray,
        client: Client,
    ):
        """Process a single batch item and write the data to the Zarr store."""
        average_deg_days = self._average_degree_days(client, source, target, item)
        average_deg_days.attrs["crs"] = CRS.from_epsg(4326)
        pp = self._item_path(item)
        logger.info(f"Writing array to {str(pp)}")
        target.write(str(pp), average_deg_days)

    def _average_degree_days(
        self,
        client: Client,
        source: OpenDataset,
        target: WriteDataArray,
        item: BatchItem,
    ):
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        window = self.window_years or 1  # O el valor que tenga sentido por defecto
        years = range(
            item.central_year - window // 2,
            item.central_year + window // 2 + (window % 2),
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
    """Enumeration for threshold comparison results.

    Attributes
        BELOW (int): Represents a value below the threshold.
        ABOVE (int): Represents a value above the threshold.

    """

    BELOW = 0
    ABOVE = 1


class HeatingCoolingDegreeDays(ThresholdBasedAverageIndicator):
    """Computes heating and cooling degree days based on temperature thresholds."""

    def __init__(
        self,
        threshold_temps_c: Sequence[float] = [16, 20, 24],
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
            threshold_temps_c (Sequence[float], optional): Temperature thresholds in degrees C.
            Defaults to [14, 16, 20, 24, 28].
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation.
            Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation.
            Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int): Central year to include in calculation for historical scenario.
            Defaults to 2005.
            central_years (Iterable[int], optional): Central years to include in calculation.
            Defaults to [2010, 2030, 2040, 2050].

        """
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c
        self.resource = self._resource()

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
        bounds_above = (
            self.resource["above"].map.bounds if self.resource["above"].map else []
        )
        bounds_below = (
            self.resource["below"].map.bounds if self.resource["below"].map else []
        )
        return [
            Indicator(
                above,
                PurePosixPath(
                    self.resource["above"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
                bounds_above,
            ),
            Indicator(
                below,
                PurePosixPath(
                    self.resource["below"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
                bounds_below,
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

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        for resource in self.inventory():
            create_tiles_for_resource(source, target, resource)

    def _resource(self) -> Dict[str, HazardResource]:  # type: ignore[override]
        # Unsure why this returns a dict vs a single resource
        resources: Dict[str, HazardResource] = {}
        for above_below in ["above", "below"]:
            with open(
                os.path.join(os.path.dirname(__file__), "degree_days.md"), "r"
            ) as f:
                description = f.read()
            path = (
                f"chronic_heat/osc/v2/mean_degree_days_{above_below}"
                + "_index_{gcm}_{scenario}_{year}"
            )
            resource = HazardResource(
                hazard_type="ChronicHeat",
                indicator_id=f"mean_degree_days/{above_below}/index",
                indicator_model_gcm="{gcm}",
                path=path,
                display_name=f"Mean degree days {above_below}" + " index value/{gcm}",
                attribution="",
                description=description,
                params={"gcm": ["ACCESS-CM2"]},
                group_id="",
                display_groups=["Mean degree days"],
                resolution="1800 m",
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
                    path="maps/" + path,
                    source="map_array_pyramid",
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
