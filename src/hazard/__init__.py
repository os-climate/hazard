"""__init__ for hazard.

Contains the get_hazards_onboarding functions.
"""

from typing import Dict, Type

from hazard.onboarder import Onboarder
from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.flopros_flood import FLOPROSFloodStandardOfProtection
from hazard.onboard.global_seismic_hazard_v2023_1 import GEMSeismicHazard
from hazard.onboard.indicator_model_onboarder import (
    DaysTasAboveIndicator,
    DaysTasAboveIndicatorDeprecated,
    DaysWBGTAboveIndicator,
    DegreeDays,
    DroughtIndicator,
    HeatingCoolingDegreeDays,
    HeatingCoolingDegreeDaysDeprecated,
    WaterTemperatureAboveIndicator,
    WetBulbGlobeTemperatureAboveIndicator,
    WorkLossIndicator,
)
from hazard.onboard.ipcc_drought import IPCCDrought
from hazard.onboard.iris_wind import IRISIndicator
from hazard.onboard.jrc_landslides import JRCLandslides
from hazard.onboard.jrc_river_flood import JRCRiverFlood
from hazard.onboard.jrc_subsidence import JRCSubsidence
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.probabilistic_european_wildfire import FireRiskIndicators
from hazard.onboard.rain_european_winter_storm import RAINEuropeanWinterStorm
from hazard.onboard.tudelft_flood import TUDelftCoastalFlood, TUDelftRiverFlood
from hazard.onboard.tudelft_wildfire import TUDelftFire
from hazard.onboard.tudelft_wind import TUDelftConvectiveWindstorm
from hazard.onboard.wisc_european_winter_storm import WISCEuropeanWinterStorm
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk

# Every hazard that can be onboarded via onboard_hazards/the CLI.
_onboarder_classes: list[Type[Onboarder]] = [
    DavydzenkaEtAlLandSubsidence,
    DaysTasAboveIndicator,
    DaysTasAboveIndicatorDeprecated,
    DaysWBGTAboveIndicator,
    DegreeDays,
    DroughtIndicator,
    ETHZurichLitPop,
    FireRiskIndicators,
    FLOPROSFloodStandardOfProtection,
    GEMSeismicHazard,
    HeatingCoolingDegreeDays,
    HeatingCoolingDegreeDaysDeprecated,
    IPCCDrought,
    IRISIndicator,
    JRCLandslides,
    JRCRiverFlood,
    JRCSubsidence,
    Jupiter,
    RAINEuropeanWinterStorm,
    TUDelftCoastalFlood,
    TUDelftConvectiveWindstorm,
    TUDelftFire,
    TUDelftRiverFlood,
    WaterTemperatureAboveIndicator,
    WetBulbGlobeTemperatureAboveIndicator,
    WISCEuropeanWinterStorm,
    WorkLossIndicator,
    WRIAqueductFlood,
    WRIAqueductWaterRisk,
]

# Subset of _onboarder_classes considered ready to publish in the inventory/catalog
# (see get_inventory_hazards) -- e.g. excludes hazards still being worked on. Order
# matches the original hand-written list in test_create_inventory, preserved here so
# regenerating inventory.json produces a diffable result rather than a full reorder.
_inventory_classes: list[Type[Onboarder]] = [
    WRIAqueductFlood,
    DegreeDays,
    Jupiter,
    WorkLossIndicator,
    DaysTasAboveIndicatorDeprecated,
    IRISIndicator,
    HeatingCoolingDegreeDaysDeprecated,
    WaterTemperatureAboveIndicator,
    WetBulbGlobeTemperatureAboveIndicator,
    WRIAqueductWaterRisk,
    DroughtIndicator,
    TUDelftRiverFlood,
    FLOPROSFloodStandardOfProtection,
    DavydzenkaEtAlLandSubsidence,
    WISCEuropeanWinterStorm,
    FireRiskIndicators,
    JRCRiverFlood,
]

# v3 replacements of the v2 chronic heat indicators: used in place of these in the inventory
# when get_inventory_hazards is called with heat_v3=True
_inventory_heat_v3_replacements: Dict[Type[Onboarder], Type[Onboarder]] = {
    DaysTasAboveIndicatorDeprecated: DaysTasAboveIndicator,
    HeatingCoolingDegreeDaysDeprecated: HeatingCoolingDegreeDays,
    WetBulbGlobeTemperatureAboveIndicator: DaysWBGTAboveIndicator,
}

hazard_map: Dict[str, Type[Onboarder]] = {
    cls.__name__: cls for cls in _onboarder_classes
}
inventory_hazard_map: Dict[str, Type[Onboarder]] = {
    cls.__name__: cls for cls in _inventory_classes
}


def get_hazards_onboarding() -> Dict[str, Type[Onboarder]]:
    """Return a dictionary with the hazards that can be onboarded."""
    return hazard_map


def get_inventory_hazards(heat_v3: bool = False) -> Dict[str, Type[Onboarder]]:
    """Return the subset of onboardable hazards considered ready to be published in
    the inventory/catalog -- see _inventory_classes.

    Args:
        heat_v3 (bool, optional): If True, the v3 chronic heat indicators replace the v2 ones
            (see _inventory_heat_v3_replacements), in the same positions. Defaults to False.
    """
    if not heat_v3:
        return inventory_hazard_map
    classes = [
        _inventory_heat_v3_replacements.get(cls, cls) for cls in _inventory_classes
    ]
    return {cls.__name__: cls for cls in classes}
