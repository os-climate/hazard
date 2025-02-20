"""CLI for hazard."""

from typing import List, Optional

from hazard import get_hazards_onboarding
import hazard.utilities.cli_utilities as cli
from hazard.onboard.general_onboarding import onboard_hazards
import typer
from enum import Enum
import ast
from typing import Any, Dict, List, Optional

import fire


from hazard import services as hazard_services
from hazard.sources import SourceDataset


# class Cli(object):
#     def __init__(self) -> None:
#         self.days_tas_above_indicator = days_tas_above_indicator
#         self.degree_days_indicator = degree_days_indicator


# def run_cli():
#     fire.Fire(Cli)


app = typer.Typer(no_args_is_help=True)

credentials_option_default = typer.Option(
    False,
    help="If you want to make the onboard to your local machine",
)

hazards_option_default = typer.Argument(
    ...,
    help=f"Hazards to be onboarded: {cli.hazards_help_text()}",
    autocompletion=cli.autocomplete_hazards,
)


class SourceDataset(str, Enum):
    NEX_GDDP_CMIP6 = "NEX-GDDP-CMIP6"
    UKCP18 = "UKCP18"


@app.command()
def days_tas_above_indicator(
    source_dataset: SourceDataset = SourceDataset.NEX_GDDP_CMIP6,
    source_dataset_kwargs: Optional[
        str
    ] = None,  # Cambiado a str para luego convertirlo
    gcm_list: List[str] = None,
    scenario_list: List[str] = None,
    threshold_list: List[float] = None,
    central_year_list: List[int] = None,
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    write_xarray_compatible_zarr: Optional[bool] = False,
    dask_cluster_kwargs: Optional[str] = None,
):
    # Convierte los parámetros de cadena a listas o diccionarios cuando sea necesario
    if source_dataset_kwargs:
        source_dataset_kwargs = ast.literal_eval(
            source_dataset_kwargs
        )  # convierte la cadena en diccionario
    if central_year_list is None:
        central_year_list = [2090]
    if threshold_list is None:
        threshold_list = [20]
    if scenario_list is None:
        scenario_list = ["ssp585"]
    if gcm_list is None:
        gcm_list = ["NorESM2-MM"]
    if dask_cluster_kwargs:
        dask_cluster_kwargs = ast.literal_eval(dask_cluster_kwargs)
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
        dask_cluster_kwargs,
    )


@app.command()
def degree_days_indicator(
    source_dataset: SourceDataset = SourceDataset.NEX_GDDP_CMIP6,
    source_dataset_kwargs: Optional[str] = None,  # Cambiado a str
    gcm_list: List[str] = None,
    scenario_list: List[str] = None,
    threshold_temperature: float = 32,
    central_year_list: List[int] = None,
    central_year_historical: int = 2005,
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    write_xarray_compatible_zarr: Optional[bool] = False,
    dask_cluster_kwargs: Optional[str] = None,
):
    # Convierte los parámetros de cadena a listas o diccionarios cuando sea necesario
    if source_dataset_kwargs:
        source_dataset_kwargs = ast.literal_eval(source_dataset_kwargs)
    if central_year_list is None:
        central_year_list = [2090]
    if scenario_list is None:
        scenario_list = ["ssp585"]
    if gcm_list is None:
        gcm_list = ["NorESM2-MM"]
    if dask_cluster_kwargs:
        dask_cluster_kwargs = ast.literal_eval(dask_cluster_kwargs)
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
        dask_cluster_kwargs,
    )


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
