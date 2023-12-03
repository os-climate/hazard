from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
import logging
import os
from typing import Iterable, List
from hazard.indicator_model import IndicatorModel

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    BatchItem,
    Indicator,
    MultiYearAverageIndicatorBase,
    ThresholdBasedAverageIndicator,
)
from hazard.protocols import OpenDataset

import xarray as xr

logger = logging.getLogger(__name__)


class DaysTasAboveIndicator(ThresholdBasedAverageIndicator):
    def __init__(
        self,
        threshold_temps_c: List[float] = [25, 30, 35, 40, 45, 50, 55],
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
            threshold_temps_c (List[float], optional): Temperature thresholds in degrees C. Defaults to [25, 30, 35, 40, 45, 50, 55].
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
        self, tas: xr.DataArray, year: int, threshold_temps_c: List[float]
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

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "days_tas_above.md"), "r"
        ) as f:
            description = f.read()
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
            display_name="Days with average temperature above {temp_c}°C/{gcm}",
            description=description,
            display_groups=[
                "Days with average temperature above"
            ],  # display names of groupings
            group_id="",
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
                path="days_tas_above_{temp_c}c_{gcm}_{scenario}_{year}_map",
                source="map_array",
            ),
            units="days/year",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource
