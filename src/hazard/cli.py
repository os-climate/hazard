"""Cli for hazard."""

from typing import Any, Dict, List, Optional, Sequence

from hazard import get_hazards_onboarding
import hazard.utilities.cli_utilities as cli_u
from hazard.onboard.general_onboarding import onboard_hazards
import typer
from enum import Enum
import ast

import fire


from hazard import services as hazard_services


app = typer.Typer(no_args_is_help=True)

credentials_option_default = typer.Option(
    False,
    help="If you want to make the onboard to your local machine",
)

hazards_option_default = typer.Argument(
    ...,
    help=f"Hazards to be onboarded: {cli_u.hazards_help_text()}",
    autocompletion=cli_u.autocomplete_hazards,
)


class SourceDataset(str, Enum):
    NEX_GDDP_CMIP6 = "NEX-GDDP-CMIP6"
    UKCP18 = "UKCP18"


def days_tas_above_indicator(
    source_dataset: SourceDataset = SourceDataset.NEX_GDDP_CMIP6,
    source_dataset_kwargs: Optional[Dict[str, Any]] = None,
    gcm_list: Sequence[str] = ["NorESM2-MM"],
    scenario_list: Sequence[str] = ["ssp585"],
    threshold_list: Sequence[float] = [20],
    central_year_list: Sequence[int] = [2090],
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    store_netcdf_coords: Optional[bool] = False,
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,  # To allow for extra parameters to the cli, due to how CWL will provide all input parameters
):
    hazard_services.days_tas_above_indicator(
        source_dataset,  # type: ignore
        source_dataset_kwargs,  # type: ignore
        gcm_list,
        scenario_list,
        threshold_list,
        central_year_list,
        central_year_historical,
        window_years,
        bucket,
        prefix,
        store,
        store_netcdf_coords,
        dask_cluster_kwargs,  # type: ignore
    )


def degree_days_indicator(
    source_dataset: SourceDataset = SourceDataset.NEX_GDDP_CMIP6,
    source_dataset_kwargs: Optional[Dict[str, Any]] = None,
    gcm_list: Sequence[str] = ["NorESM2-MM"],
    scenario_list: Sequence[str] = ["ssp585"],
    threshold_temperature: float = 32,
    central_year_list: Sequence[int] = [2090],
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    store_netcdf_coords: Optional[bool] = False,
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,  # To allow for extra parameters to the cli, due to how CWL will provide all input parameters
):
    hazard_services.degree_days_indicator(
        source_dataset,  # type: ignore
        source_dataset_kwargs,  # type: ignore
        gcm_list,
        scenario_list,
        threshold_temperature,
        central_year_list,
        central_year_historical,
        window_years,
        bucket,
        prefix,
        store,
        store_netcdf_coords,
        dask_cluster_kwargs,  # type: ignore
    )


class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator
        self.degree_days_indicator = degree_days_indicator


def cli():
    fire.Fire(Cli)


@app.command()
def list_hazards():
    """List all available hazards."""
    hazards = get_hazards_onboarding().keys()
    for haz in sorted(hazards):
        print(haz)


@app.command()
def onboard(
    local: Optional[bool] = credentials_option_default,
    credentials_path: Optional[str] = typer.Option(
        None, help="Path to the credentials for S3 access."
    ),
    source_dir_base: Optional[str] = typer.Option(None, help="Path to the data."),
    hazards: List[str] = hazards_option_default,
    download_dir: Optional[str] = typer.Option(None, help="Path to the data."),
):
    """Onboard desired hazards."""
    local = bool(local)

    onboard_hazards(local, credentials_path, source_dir_base, hazards, download_dir)


if __name__ == "__main__":
    app()
