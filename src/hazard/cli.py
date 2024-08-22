from typing import Any, Dict, List, Optional

import fire

from hazard import services as hazard_services
from hazard.sources import SourceDataset


def days_tas_above_indicator(
    source_dataset: SourceDataset = "NEX-GDDP-CMIP6",
    source_dataset_kwargs: Optional[Dict[str, Any]] = None,
    gcm_list: List[str] = ["NorESM2-MM"],
    scenario_list: List[str] = ["ssp585"],
    threshold_list: List[float] = [20],
    central_year_list: List[int] = [2090],
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    inventory_format: Optional[str] = "osc",
    write_xarray_compatible_zarr: Optional[bool] = False,
):
    hazard_services.days_tas_above_indicator(
        source_dataset,
        source_dataset_kwargs,
        gcm_list,
        scenario_list,
        threshold_list,
        central_year_list,
        central_year_historical,
        window_years,
        bucket,
        prefix,
        store,
        write_xarray_compatible_zarr,
        inventory_format,
    )


def degree_days_indicator(
    source_dataset: SourceDataset = "NEX-GDDP-CMIP6",
    source_dataset_kwargs: Optional[Dict[str, Any]] = None,
    gcm_list: List[str] = ["NorESM2-MM"],
    scenario_list: List[str] = ["ssp585"],
    threshold_temperature: float = 32,
    central_year_list: List[int] = [2090],
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    write_xarray_compatible_zarr: Optional[bool] = False,
    inventory_format: Optional[str] = "osc",
):
    hazard_services.degree_days_indicator(
        source_dataset,
        source_dataset_kwargs,
        gcm_list,
        scenario_list,
        threshold_temperature,
        central_year_list,
        central_year_historical,
        window_years,
        bucket,
        prefix,
        store,
        write_xarray_compatible_zarr,
        inventory_format,
    )


class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator
        self.degree_days_indicator = degree_days_indicator


def cli():
    fire.Fire(Cli)
