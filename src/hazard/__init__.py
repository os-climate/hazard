"""__init__ for hazard.

Contains the get_hazards_onboarding functions.
"""

from typing import Dict, Type

from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.flopros_flood import FLOPROSFloodStandardOfProtection
from hazard.onboard.global_seismic_hazard_v2023_1 import GEMSeismicHazard
from hazard.onboard.indicator_model_onboarder import (
    DaysTasAboveIndicator,
    DegreeDays,
    DroughtIndicator,
    HeatingCoolingDegreeDays,
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
from hazard.onboarder import Onboarder

# Every hazard that can be onboarded via onboard_hazards/the CLI.
_onboarder_classes: list[type[Onboarder]] = [
    DavydzenkaEtAlLandSubsidence,
    DaysTasAboveIndicator,
    DegreeDays,
    DroughtIndicator,
    ETHZurichLitPop,
    FireRiskIndicators,
    FLOPROSFloodStandardOfProtection,
    GEMSeismicHazard,
    HeatingCoolingDegreeDays,
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
_inventory_classes: list[type[Onboarder]] = [
    WRIAqueductFlood,
    DegreeDays,
    Jupiter,
    WorkLossIndicator,
    DaysTasAboveIndicator,
    IRISIndicator,
    HeatingCoolingDegreeDays,
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

hazard_map: dict[str, type[Onboarder]] = {
    cls.__name__: cls for cls in _onboarder_classes
}
inventory_hazard_map: dict[str, type[Onboarder]] = {
    cls.__name__: cls for cls in _inventory_classes
}


def get_hazards_onboarding() -> dict[str, type[Onboarder]]:
    """Return a dictionary with the hazards that can be onboarded."""
    return hazard_map


def get_inventory_hazards() -> dict[str, type[Onboarder]]:
    """Return the subset of onboardable hazards considered ready to be published in
    the inventory/catalog -- see _inventory_classes.
    """
    return inventory_hazard_map
