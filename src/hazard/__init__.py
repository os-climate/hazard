"""__init__ for hazard.

Contains the get_hazards_onboarding functions.
"""

from typing import Dict, Type

from hazard.onboarder import Onboarder
from hazard.onboard.probabilistic_european_wildfire import FireRiskIndicators
from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.ipcc_drought import IPCCDrought
from hazard.onboard.iris_wind import IRISIndicator
from hazard.onboard.jrc_landslides import JRCLandslides
from hazard.onboard.jrc_subsidence import JRCSubsidence
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.rain_european_winter_storm import RAINEuropeanWinterStorm
from hazard.onboard.tudelft_flood import TUDelftCoastalFlood, TUDelftRiverFlood
from hazard.onboard.tudelft_wildfire import TUDelftFire
from hazard.onboard.tudelft_wind import TUDelftConvectiveWindstorm
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk
from hazard.onboard.wisc_european_winter_storm import WISCEuropeanWinterStorm
from hazard.onboard.global_seismic_hazard_v2023_1 import GEMSeismicHazard
from hazard.onboard.jrc_riverflood import JRCRiverFlood

_onboarder_classes: list[Type[Onboarder]] = [
    DavydzenkaEtAlLandSubsidence,
    ETHZurichLitPop,
    IPCCDrought,
    IRISIndicator,
    JRCLandslides,
    JRCSubsidence,
    Jupiter,
    RAINEuropeanWinterStorm,
    TUDelftCoastalFlood,
    TUDelftConvectiveWindstorm,
    TUDelftFire,
    TUDelftRiverFlood,
    WRIAqueductFlood,
    WRIAqueductWaterRisk,
    WISCEuropeanWinterStorm,
    GEMSeismicHazard,
    JRCRiverFlood,
    FireRiskIndicators,
]

hazard_map: Dict[str, Type[Onboarder]] = {cls.__name__: cls for cls in _onboarder_classes}


def get_hazards_onboarding() -> Dict[str, Type[Onboarder]]:
    """Return a dictionary with the hazards that can be onboarded."""
    return hazard_map
