"""Deprecated (v2) indicator models.

These are superseded by the v3 models (DaysWBGTAboveIndicator, DaysTasAboveIndicator and
HeatingCoolingDegreeDays), which are derived from ThresholdBasedAverageIndicator. They are kept
only so that the existing v2 arrays remain described in the inventory, and can be removed with
this package (including the v2 descriptions alongside the code) when those arrays are
retired.
"""

from hazard.models.deprecated_v2.indicators import (
    AboveBelow,
    DaysTasAboveIndicatorDeprecated,
    HeatingCoolingDegreeDaysDeprecated,
    ThresholdBasedAverageIndicatorDeprecated,
    WetBulbGlobeTemperatureAboveIndicator,
)

__all__ = [
    "AboveBelow",
    "DaysTasAboveIndicatorDeprecated",
    "HeatingCoolingDegreeDaysDeprecated",
    "ThresholdBasedAverageIndicatorDeprecated",
    "WetBulbGlobeTemperatureAboveIndicator",
]
