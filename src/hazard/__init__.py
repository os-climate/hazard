"""__init__ for hazard.

Contains the get_hazards_onboarding functions.
"""

from typing import Dict

from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.water_temp import WaterTemperatureAboveIndicator
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
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

hazard_map = {
    "DavydzenkaEtAlLandSubsidence": DavydzenkaEtAlLandSubsidence,
    "DaysTasAboveIndicator": DaysTasAboveIndicator,
    "ETHZurichLitPop": ETHZurichLitPop,
    "IPCCDrought": IPCCDrought,
    "IRISIndicator": IRISIndicator,
    "JRCLandslides": JRCLandslides,
    "JRCSubsidence": JRCSubsidence,
    "Jupiter": Jupiter,
    "RAINEuropeanWinterStorm": RAINEuropeanWinterStorm,
    "TUDelftCoastalFlood": TUDelftCoastalFlood,
    "TUDelftConvectiveWindstorm": TUDelftConvectiveWindstorm,
    "TUDelftFire": TUDelftFire,
    "TUDelftRiverFlood": TUDelftRiverFlood,
    "WaterTemperatureAboveIndicator": WaterTemperatureAboveIndicator,
    "WetBulbGlobeTemperatureAboveIndicator": WetBulbGlobeTemperatureAboveIndicator,
    "WRIAqueductFlood": WRIAqueductFlood,
    "WRIAqueductWaterRisk": WRIAqueductWaterRisk,
    "WISCEuropeanWinterStorm": WISCEuropeanWinterStorm,
    "GEMSeismicHazard": GEMSeismicHazard,
    "JRCRiverflood": JRCRiverFlood,
    "FireRiskIndicators": FireRiskIndicators,
}


def get_hazards_onboarding() -> Dict:
    """Return a dictionary with the hazards that can be onboarded."""
    return hazard_map
