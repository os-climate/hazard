"""Climate Hazard Indicator: Degree Days Calculation."""

import logging
import os
from pathlib import PurePosixPath
from typing import Sequence
from typing_extensions import Dict, Iterable, List, Optional

import xarray as xr
from dask.distributed import Client
from rasterio.crs import CRS  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    BatchItem,
    RunConfig,
    ThresholdBasedAverageIndicator,
)
from hazard.models.indicator_conventions import (
    indicator_description,
    threshold_indicator_resource,
)
from hazard.sources.osc_zarr import OscZarr
from hazard.protocols import OpenDataset, ReadWriteDataArray, WriteDataArray
from hazard.utilities.description_utilities import get_indicator_period_descriptions
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon

logger = logging.getLogger(__name__)


class HeatingCoolingDegreeDays(ThresholdBasedAverageIndicator):
    """Cooling and heating degree days: the time integral of daily-mean temperature above or below thresholds.

    Two indicators are produced, "above" (cooling degree days) and "below" (heating degree
    days), from the daily mean near-surface temperature ('tas'), each with its own reference
    temperatures (thresholds).
    """

    quantities = ["tas"]

    # reference temperatures in °C. Heating: from 10 °C, the balance point of well-insulated buildings
    # with high internal gains, through the UK base of 15.5 °C and 18 °C (about 65 °F, the US base),
    # to 22 °C for buildings heated to high internal temperatures or with low internal gains
    default_heating_thresholds = [10, 12, 14, 15.5, 18, 22]
    default_cooling_thresholds = [18, 22, 24]

    def __init__(
        self,
        heating_thresholds: Optional[Sequence[float]] = None,
        cooling_thresholds: Optional[Sequence[float]] = None,
        run: Optional[RunConfig] = None,
    ):
        """Construct the model.

        Args:
            heating_thresholds (Optional[Sequence[float]], optional): Reference temperatures of the heating
                degree days in °C. Defaults to default_heating_thresholds.
            cooling_thresholds (Optional[Sequence[float]], optional): Reference temperatures of the cooling
                degree days in °C. Defaults to default_cooling_thresholds.
            run (Optional[RunConfig], optional): GCMs, scenarios, periods and source dataset of the
                run. Defaults to RunConfig(), i.e. the v3 settings.

        """
        self.heating_thresholds = list(
            self.default_heating_thresholds
            if heating_thresholds is None
            else heating_thresholds
        )
        self.cooling_thresholds = list(
            self.default_cooling_thresholds
            if cooling_thresholds is None
            else cooling_thresholds
        )
        # the thresholds of the model: all of those of the resources
        super().__init__(
            thresholds=sorted(set(self.heating_thresholds + self.cooling_thresholds)),
            run=run,
        )

    def resource_thresholds(self, key: str) -> List[float]:
        """Thresholds of the cooling ("above") or heating ("below") degree days."""
        return self.cooling_thresholds if key == "above" else self.heating_thresholds

    def _single_year_arrays(
        self, inputs: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        tas_c = inputs["tas"] - 273.15  # convert from K to °C
        scale = 365.0 / len(tas_c["time"])  # normalize to 365 days
        return {
            "above": self._stack_thresholds(
                lambda threshold: scale
                * (tas_c - threshold).clip(min=0.0).sum(dim="time"),
                self.cooling_thresholds,
            ),
            "below": self._stack_thresholds(
                lambda threshold: scale
                * (threshold - tas_c).clip(min=0.0).sum(dim="time"),
                self.heating_thresholds,
            ),
        }

    def _resources(self) -> Dict[str, HazardResource]:
        description = indicator_description("heating_cooling_degree_days.md", self.run)
        return {
            key: threshold_indicator_resource(
                run=self.run,
                thresholds=self.resource_thresholds(key),
                indicator_id=f"{name}_degree_days",
                path_name=f"{name}_degree_days",
                display_name=f"{name.capitalize()} degree days ({key} threshold in °C) "
                + "({gcm})",
                display_group="Degree days",
                description=description,
                # parsable by pint as a temperature difference times time ("degree days" is not)
                units="°C day",
                max_value=max_value,
            )
            for key, name, max_value in [
                ("above", "cooling", 2000.0),
                ("below", "heating", 5000.0),
            ]
        }


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
            display_name=f"Mean degree days above {self.threshold_c}°C" + " ({gcm})",
            description=description,
            params={"gcm": list(self.gcms)},
            group_id="",
            license="Creative Commons",
            source="",
            version="",
            resolution="0.25°",
            display_groups=["Mean degree days"],
            map=MapInfo(  # type:ignore
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_index=255,
                    max_value=2000.0,
                    units="degree days",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 85.0],
                path=path + "_map",
                source="map_array",
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

    def _item_path(self, item: BatchItem) -> PurePosixPath:
        return PurePosixPath(
            item.resource.path.format(
                gcm=item.gcm, scenario=item.scenario, year=item.central_year
            )
        )
