"""Legacy command line interface for hazard indicators (`os_climate_hazard`).

Kept for the CWL application package (hazard_workflow.cwl) and the Docker image,
which call e.g. `os_climate_hazard days_tas_above_indicator --store ...`. Each
command here is a hand-written wrapper around a specific (v2) model, so adding an
indicator means adding another command.

The generic way is to onboard hazards like any other: indicator models are wrapped
as onboarders (see hazard.onboard.indicator_model_onboarder) and run via
`osc_hazard onboard` (hazard.cli.main) or hazard.onboard.general_onboarding.
New indicators should not be added here.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import fire
from dask.distributed import Client, LocalCluster

from hazard.docs_store import DocStore  # type: ignore
from hazard.models.degree_days import DegreeDays
from hazard.models.deprecated_v2 import DaysTasAboveIndicatorDeprecated
from hazard.sources import get_source_dataset_instance
from hazard.sources.osc_zarr import OscZarr


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
    """Run the days_tas_above indicator generation for a list of models,scenarios, thresholds, central years and a given size of years window over which to compute the average.

    Store the result in a zarr store, locally if `store` is provided, else in an S3
    bucket if `bucket` and `prefix` are provided.
    An inventory filed is stored at the root of the zarr directory.
    """
    docs_store, target, client = setup(
        bucket, prefix, store, store_netcdf_coords, dask_cluster_kwargs
    )

    source = get_source_dataset_instance(source_dataset, source_dataset_kwargs or {})  # type: ignore

    model = DaysTasAboveIndicatorDeprecated(
        threshold_temps_c=threshold_list,
        window_years=window_years,
        gcms=gcm_list,
        scenarios=scenario_list,
        central_years=central_year_list,
        central_year_historical=central_year_historical,
        source_dataset=source_dataset,  # type: ignore
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)


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
    """Run the degree days indicator generation for a list of models,scenarios, a threshold temperature, central years and a given size of years window over which to compute the average.

    Store the result in a zarr store, locally if `store` is provided, else in an S3
    bucket if `bucket` and `prefix` are provided.
    An inventory filed is stored at the root of the zarr directory.
    """
    docs_store, target, client = setup(
        bucket, prefix, store, store_netcdf_coords, dask_cluster_kwargs
    )

    source = get_source_dataset_instance(source_dataset, source_dataset_kwargs or {})  # type: ignore

    model = DegreeDays(
        threshold=threshold_temperature,
        window_years=window_years,
        gcms=gcm_list,
        scenarios=scenario_list,
        central_years=central_year_list,
        central_year_historical=central_year_historical,
        source_dataset=source_dataset,  # type: ignore
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)


def setup(
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    store_netcdf_coords: Optional[bool] = False,
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DocStore, OscZarr, Client]:
    """Initialize output store, docs store and local dask client."""
    if store is not None:
        docs_store = DocStore(local_path=store)
        target = OscZarr(store=store, store_netcdf_coords=store_netcdf_coords)
    else:
        if bucket is None or prefix is None:
            raise ValueError(
                "either of `store`, or `bucket` and `prefix` together, must be provided"
            )
        else:
            docs_store = DocStore()
            target = OscZarr(
                bucket=bucket,
                store_netcdf_coords=store_netcdf_coords,
            )

    dask_cluster_kwargs = dask_cluster_kwargs or {}
    cluster = LocalCluster(processes=False, **dask_cluster_kwargs)

    client = Client(cluster)

    return docs_store, target, client


class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator
        self.degree_days_indicator = degree_days_indicator


def cli():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    )
    fire.Fire(Cli)
