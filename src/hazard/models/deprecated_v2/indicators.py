"""Deprecated (v2) indicator models; see hazard.models.deprecated_v2."""

import logging
import os
from abc import abstractmethod
from contextlib import ExitStack
from enum import Enum
from pathlib import PurePosixPath
from typing import Sequence

import numpy as np
import xarray as xr
from typing_extensions import Dict, Iterable, List, deprecated

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    BatchItem,
    Indicator,
    MultiYearAverageIndicatorBase,
)
from hazard.protocols import OpenDataset
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.description_utilities import get_indicator_period_descriptions
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class ThresholdBasedAverageIndicatorDeprecated(
    MultiYearAverageIndicatorBase[BatchItem]
):
    """Model for calculating threshold-based multi-year average indicators.

    Deprecated: superseded by ThresholdBasedAverageIndicator, which has thresholds as the index
    dimension of each array and the v3 conventions. Kept for the v2 indicators derived from it, so
    that the existing v2 arrays remain described in the inventory.
    """

    def batch_items(self) -> Iterable[BatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
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
        """Get the inventory item(s)."""
        return [self._resource()]  # .expand()

    def _get_indicators(
        self, item: BatchItem, data_arrays: List[xr.DataArray], param: str
    ) -> List[Indicator]:
        """Generate a list of indicators for given thresholds and data arrays.

        Args:
            item (BatchItem): _description_
            data_arrays (List[xr.DataArray]): _description_
            param (str): _description_

        Returns:
            List[Indicator]: _description_

        """
        resource = item.resource
        paths = [
            item.resource.path.format(
                temp_c=threshold,
                gcm=item.gcm,
                scenario=item.scenario,
                year=item.central_year,
            )
            for threshold in resource.params[param]
        ]
        return [
            Indicator(array=array, path=PurePosixPath(paths[i]))
            for i, array in enumerate(data_arrays)
        ]

    @abstractmethod
    def _resource(self) -> HazardResource: ...  # noqa:E704


@deprecated(
    "WetBulbGlobeTemperatureAboveIndicator (v2) is superseded by DaysWBGTAboveIndicator (v3); "
    "it is kept only to describe the existing v2 arrays."
)
class WetBulbGlobeTemperatureAboveIndicator(
    ThresholdBasedAverageIndicatorDeprecated,
):
    """Calculates days exceeding specified WBGT temperature thresholds based on climate data for various scenarios and years, storing results for global heat risk assessment.

    Deprecated: superseded by DaysWBGTAboveIndicator (v3), which also gives days with daily-maximum
    WBGT above thresholds, a multi-model mean and the v3 conventions. Kept so that the existing v2
    arrays (chronic_heat/osc/v2/days_wbgt_above_...) remain in the inventory.

    Attributes
        threshold_temps_c : List[float]
            Temperature thresholds in Celsius for calculating exceedance days.
        window_years : int
            Number of years over which to average data.
        gcms : Iterable[str]
            Climate models to use for data analysis.
        scenarios : Iterable[str]
            Emission scenarios to consider for projections.
        central_year_historical : int
            Central year for historical scenario calculations.
        central_years : Iterable[int]
            Central years for projected scenario calculations.

    """

    def __init__(
        self,
        threshold_temps_c: Sequence[float] = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
        ],
        window_years: int = 20,
        gcms: Sequence[str] = [
            "ACCESS-CM2",
            "CMCC-ESM2",
            "CNRM-CM6-1",
            "MPI-ESM1-2-LR",
            "MIROC6",
            "NorESM2-MM",
        ],
        scenarios: Sequence[str] = [
            "historical",
            "ssp126",
            "ssp245",
            "ssp370",
            "ssp585",
        ],
        central_year_historical: int = 2005,
        central_years: Sequence[int] = [2030, 2040, 2050, 2060, 2070, 2080, 2090],
    ):
        """Initialize the WBGT indicator with specified thresholds, climate models, and scenarios.

        Args:
            threshold_temps_c : Sequence[float]
                Temperature thresholds for exceedance calculations in Celsius.
            window_years : int
                Number of years for averaging data.
            gcms : Sequence[str]
                Global Climate Models to include.
            scenarios : Sequence[str]
                Emission scenarios to evaluate.
            central_year_historical : int
                Central year for historical averages.
            central_years : Sequence[int]
                Target years for future scenario calculations.

        """
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            hurs = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "hurs", year)
            ).hurs
            results = self._days_wbgt_above_indicators(tas, hurs)
        path = item.resource.path.format(
            gcm=item.gcm, scenario=item.scenario, year=item.central_year
        )
        result = [Indicator(results, PurePosixPath(path))]
        logger.info(f"Calculation complete for year {year}")
        return result

    def _days_wbgt_above_indicators(
        self, tas: xr.DataArray, hurs: xr.DataArray
    ) -> xr.DataArray:
        """Create DataArrays containing indicators the thresholds for a single year."""
        tas_c = tas - 273.15  # convert from K to C
        # vpp is water vapour partial pressure in hPa
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
        scale = 365.0 / len(wbgt.time)
        if any(coord not in wbgt.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        coords = {
            "index": self.threshold_temps_c,
            "lat": wbgt.coords["lat"].values,
            "lon": wbgt.coords["lon"].values,
        }
        output = xr.DataArray(coords=coords, dims=list(coords.keys()))
        for i, threshold_c in enumerate(self.threshold_temps_c):
            output[i, :, :] = xr.where(wbgt > threshold_c, scale, 0.0).sum(dim=["time"])
        return output

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        create_tiles_for_resource(source, target, self._resource())

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "wet_bulb_globe_temp.md"), "r"
        ) as f:
            description = f.read().replace("\u00c2\u00b0", "\u00b0")
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="days_wbgt_above",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": list(self.gcms)},
            path="chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}",
            display_name="Days with wet-bulb globe temperature above threshold in °C ({gcm})",
            description=description,
            resolution="0.25°",
            license="Creative Commons",
            source="",
            version="",
            display_groups=[
                "Days with wet-bulb globe temperature above threshold in °C"
            ],  # display names of groupings
            group_id="",
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=365,
                    max_index=255,
                    units="days/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                bbox=[-180.0, -85.0, 180.0, 85.0],
                path="maps/chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}_map",
                index_values=self.threshold_temps_c,
                source="map_array_pyramid",
            ),
            units="days/year",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp370", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource


@deprecated(
    "DaysTasAboveIndicatorDeprecated (v2) is superseded by DaysTasAboveIndicator (v3); "
    "it is kept only to describe the existing v2 arrays."
)
class DaysTasAboveIndicatorDeprecated(ThresholdBasedAverageIndicatorDeprecated):
    """Indicator for counting the number of days with temperatures above given thresholds.

    This class calculates the number of days where the daily average temperature exceeds
    specified threshold values over a given time window. The calculations use temperature
    data from different Global Circulation Models (GCMs) and climate scenarios.

    Deprecated: superseded by DaysTasAboveIndicator (v3), which also gives days with daily-maximum
    temperature above thresholds, a multi-model mean and the v3 conventions. Kept so that the
    existing v2 arrays (chronic_heat/osc/v2/days_tas_above_...) remain in the inventory.
    """

    def __init__(
        self,
        threshold_temps_c: Sequence[float] = [25, 30, 35, 40, 45, 50, 55],
        window_years: int = MultiYearAverageIndicatorBase._default_window_years,
        gcms: Iterable[str] = MultiYearAverageIndicatorBase._default_gcms,
        scenarios: Iterable[str] = MultiYearAverageIndicatorBase._default_scenarios,
        central_year_historical: int = MultiYearAverageIndicatorBase._default_central_year_historical,
        central_years: Iterable[
            int
        ] = MultiYearAverageIndicatorBase._default_central_years,
        source_dataset: str = MultiYearAverageIndicatorBase._default_source_dataset,
    ):
        """Create indicators based on average number of days above different temperature thresholds.

        Args:
            threshold_temps_c (Sequence[float], optional): Temperature thresholds in degrees C.
                Defaults to [25, 30, 35, 40, 45, 50, 55].
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation.
                Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation.
                Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int): Central year to include in calculation for historical scenario.
                Defaults to 2005.
            central_years (Iterable[int], optional): Central years to include in calculation.
                Defaults to [2010, 2030, 2040, 2050].
            source_dataset (str): Name of the dataset used for temperature data.

        """
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
            source_dataset=source_dataset,
        )
        self.threshold_temps_c = threshold_temps_c
        self.resource = self._resource()

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        """For a single year and batch item calculate the indicators (i.e. one per threshold temperature)."""
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            results = self._days_tas_above_indicators(tas, year, self.threshold_temps_c)
        logger.info(f"Calculation complete for year {year}")
        # Get Indicators for reach array, looking up path using "threshold"
        return self._get_indicators(item, results, "temp_c")

    def _days_tas_above_indicators(
        self, tas: xr.DataArray, year: int, threshold_temps_c: Sequence[float]
    ) -> List[xr.DataArray]:
        """Create DataArrays containing indicators the thresholds for a single year."""
        if any(coord not in tas.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (tas.time.dt.year != year).any():
            raise ValueError("unexpected year found")
        # normalize to 365 days
        scale = 365.0 / len(tas.time)
        # will raise error if tas not present
        return [
            scale * xr.where(tas > (273.15 + threshold_c), 1.0, 0.0).sum(dim=["time"])
            for threshold_c in threshold_temps_c
        ]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self.resource)

    def _resource(self) -> HazardResource:
        """Create resource."""
        scenarios = []

        if "historical" in self.scenarios:
            scenarios.append(
                Scenario(id="historical", years=[self.central_year_historical])
            )

        for s in self.scenarios:
            if s == "historical":
                continue
            scenarios.append(Scenario(id=s, years=list(self.central_years)))

        description = self._generate_description()
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="days_tas/above/{temp_c}c",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={
                "temp_c": [str(t) for t in self.threshold_temps_c],
                "gcm": list(self.gcms),
            },
            path="chronic_heat/osc/v2/days_tas_above_{temp_c}c_{gcm}_{scenario}_{year}",
            display_name="Days with average temperature above {temp_c}°C ({gcm})",
            description=description,
            display_groups=[
                "Days with average temperature above"
            ],  # display names of groupings
            group_id="",
            license="Creative Commons",
            source="",
            version="",
            resolution="0.25°",
            attribution="",
            map=MapInfo(
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=100,
                    max_index=255,
                    units="days/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 85.0],
                index_values=None,
                path="chronic_heat/osc/v2/days_tas_above_{temp_c}c_{gcm}_{scenario}_{year}_map",
                source="map_array",
            ),
            units="days/year",
            store_netcdf_coords=False,
            scenarios=scenarios,
        )
        return resource

    def _generate_description(self):
        current_file_dir_name = os.path.dirname(__file__)
        with open(os.path.join(current_file_dir_name, "days_tas_above.md"), "r") as f:
            description = f.read().replace("\u00c2\u00b0", "\u00b0")

        # hazard/sources, from hazard/models/deprecated_v2
        sources_dir_name = os.path.join(
            os.path.dirname(os.path.dirname(current_file_dir_name)), "sources"
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


class AboveBelow(Enum):
    """Enumeration for threshold comparison results.

    Attributes
        BELOW (int): Represents a value below the threshold.
        ABOVE (int): Represents a value above the threshold.

    """

    BELOW = 0
    ABOVE = 1


@deprecated(
    "HeatingCoolingDegreeDaysDeprecated (v2) is superseded by HeatingCoolingDegreeDays (v3); "
    "it is kept only to describe the existing v2 arrays."
)
class HeatingCoolingDegreeDaysDeprecated(ThresholdBasedAverageIndicatorDeprecated):
    """Computes heating and cooling degree days based on temperature thresholds.

    Deprecated: superseded by HeatingCoolingDegreeDays (v3), which gives both cooling (above) and
    heating (below) degree days for all GCMs, a multi-model mean and the v3 conventions. Kept so
    that the existing v2 arrays (chronic_heat/osc/v2/mean_degree_days_above_index_...) remain in
    the inventory.
    """

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
        return [
            Indicator(
                above,
                PurePosixPath(
                    self.resource["above"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
            ),
            Indicator(
                below,
                PurePosixPath(
                    self.resource["below"].path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
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
        # will raise error if tas not present
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
        # for above_below in ["above", "below"]:
        for above_below in ["above"]:
            with open(
                os.path.join(os.path.dirname(__file__), "degree_days_tas.md"), "r"
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
                display_name=f"Mean degree days {above_below}"
                + " threshold in °C ({gcm})",
                attribution="",
                description=description,
                params={"gcm": ["ACCESS-CM2"]},
                group_id="",
                display_groups=["Mean degree days"],
                resolution="0.25°",
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
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -60.0),
                        (-180.0, -60.0),
                    ],
                    index_values=self.threshold_temps_c,
                    path=path + "_map",
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
