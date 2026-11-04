"""Climate Hazard Indicator: Days Above Threshold Temperature."""

import logging
from typing_extensions import Dict

import xarray as xr

from hazard.inventory import HazardResource

from hazard.models.multi_year_average import (
    ThresholdBasedAverageIndicator,
)
from hazard.models.indicator_conventions import (
    indicator_description,
    threshold_indicator_resource,
)

logger = logging.getLogger(__name__)


class DaysTasAboveIndicator(ThresholdBasedAverageIndicator):
    """Days per year with daily-mean and daily-maximum near-surface temperature above thresholds.

    Two indicators are produced, "mean" and "max", from the daily mean ('tas') and daily
    maximum ('tasmax') near-surface temperature respectively.
    """

    quantities = ["tas", "tasmax"]

    # temperature thresholds in °C
    default_thresholds = [25, 30, 35, 40, 45, 50, 55]

    def _single_year_arrays(
        self, inputs: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        # convert from K to °C
        return {
            "mean": self._days_above(inputs["tas"] - 273.15),
            "max": self._days_above(inputs["tasmax"] - 273.15),
        }

    def _resources(self) -> Dict[str, HazardResource]:
        description = indicator_description("days_tas_above.md", self.run)
        return {
            key: threshold_indicator_resource(
                run=self.run,
                thresholds=self.thresholds,
                indicator_id=f"days_{quantity}_above",
                path_name=f"days_{quantity}_above",
                display_name=f"Days with daily-{'mean' if key == 'mean' else 'maximum'} "
                + "temperature above threshold in °C ({gcm})",
                display_group="Days with temperature above threshold in °C",
                description=description,
                units="days/year",
                max_value=100.0,
            )
            for key, quantity in [("mean", "tas"), ("max", "tasmax")]
        }
