import logging  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple

from dask.distributed import Client, LocalCluster  # noqa: E402
from fsspec.implementations.local import LocalFileSystem

from hazard.docs_store import DocStore  # type: ignore # noqa: E402
from hazard.models.days_tas_above import DaysTasAboveIndicator  # noqa: E402
from hazard.models.degree_days import DegreeDays  # noqa: E402
from hazard.sources import SourceDataset, get_source_dataset_instance
from hazard.sources.osc_zarr import OscZarr  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


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
    write_xarray_compatible_zarr: Optional[bool] = False,
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run the days_tas_above indicator generation for a list of models,scenarios, thresholds,
    central years and a given size of years window over which to compute the average.
    Store the result in a zarr store, locally if `store` is provided, else in an S3
    bucket if `bucket` and `prefix` are provided.
    An inventory filed is stored at the root of the zarr directory.
    """

    docs_store, target, client = setup(
        bucket, prefix, store, write_xarray_compatible_zarr, dask_cluster_kwargs
    )

    source_dataset_kwargs = (
        {} if source_dataset_kwargs is None else source_dataset_kwargs
    )
    source = get_source_dataset_instance(source_dataset, source_dataset_kwargs)

    model = DaysTasAboveIndicator(
        threshold_temps_c=threshold_list,
        window_years=window_years,
        gcms=gcm_list,
        scenarios=scenario_list,
        central_years=central_year_list,
        central_year_historical=central_year_historical,
        source_dataset=source_dataset,
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)


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
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run the degree days indicator generation for a list of models,scenarios, a threshold temperature,
    central years and a given size of years window over which to compute the average.
    Store the result in a zarr store, locally if `store` is provided, else in an S3
    bucket if `bucket` and `prefix` are provided.
    An inventory filed is stored at the root of the zarr directory.
    """

    docs_store, target, client = setup(
        bucket, prefix, store, write_xarray_compatible_zarr, dask_cluster_kwargs
    )

    source_dataset_kwargs = (
        {} if source_dataset_kwargs is None else source_dataset_kwargs
    )
    source = get_source_dataset_instance(source_dataset, source_dataset_kwargs)

    model = DegreeDays(
        threshold=threshold_temperature,
        window_years=window_years,
        gcms=gcm_list,
        scenarios=scenario_list,
        central_years=central_year_list,
        central_year_historical=central_year_historical,
        source_dataset=source_dataset,
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)


def setup(
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
    write_xarray_compatible_zarr: Optional[bool] = False,
    dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DocStore, OscZarr, Client]:
    """
    initialize output store, docs store and local dask client
    """

    if store is not None:
        docs_store = DocStore(fs=LocalFileSystem(), local_path=store)
        target = OscZarr(
            store=store, write_xarray_compatible_zarr=write_xarray_compatible_zarr
        )
    else:
        if bucket is None or prefix is None:
            raise ValueError(
                "either of `store`, or `bucket` and `prefix` together, must be provided"
            )
        else:
            docs_store = DocStore(bucket=bucket, prefix=prefix)
            target = OscZarr(
                bucket=bucket,
                prefix=prefix,
                write_xarray_compatible_zarr=write_xarray_compatible_zarr,
            )

    dask_cluster_kwargs = dask_cluster_kwargs or {}
    cluster = LocalCluster(processes=False, **dask_cluster_kwargs)

    client = Client(cluster)

    return docs_store, target, client
