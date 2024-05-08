from typing import List, Optional

import fire

from hazard import services as hazard_services


def days_tas_above_indicator(
    gcm_list: List[str] = ["NorESM2-MM"],
    scenario_list: List[str] = ["ssp585"],
    threshold_list: List[float] = [20],
    central_year_list: List[int] = [2090],
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    inventory_format: Optional[str] = "osc",
    extra_xarray_store: Optional[bool] = False,
):

    hazard_services.days_tas_above_indicator(
        gcm_list,
        scenario_list,
        threshold_list,
        central_year_list,
        window_years,
        bucket,
        prefix,
        store,
        extra_xarray_store,
        inventory_format,
    )


class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator


def cli():
    fire.Fire(Cli)
