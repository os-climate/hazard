import logging  # noqa: E402
from typing import List, Optional

import fire
from dask.distributed import Client, LocalCluster  # noqa: E402
from fsspec.implementations.local import LocalFileSystem

from hazard.docs_store import DocStore  # type: ignore # noqa: E402
from hazard.models.days_tas_above import DaysTasAboveIndicator  # noqa: E402
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6  # noqa: E402
from hazard.sources.osc_zarr import OscZarr  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


def days_tas_above_indicator(
    gcm_list: List[str] = ["NorESM2-MM"],
    scenario_list: List[str] = ["ssp585"],
    threshold_list: List[float] = [20],
    central_year_list: List[int] = [2090],
    window_years: int = 1,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    store: Optional[str] = None,
):
    """
    Run the days_tas_above indicator generation for a list of models,scenarios, thresholds,
    central years and a given size of years window over which to compute the average.
    Store the result in a zarr store, locally if `store` is provided, else in an S3
    bucket if `bucket` and `prefix` are provided.
    An inventory filed is stored at the root of the zarr directory.
    """

    if store is not None:
        docs_store = DocStore(fs=LocalFileSystem(), local_path=store)
        target = OscZarr(store=store)
    else:
        if bucket is None or prefix is None:
            raise ValueError("either of `store`, or `bucket` and `prefix` together, must be provided")
        else:
            docs_store = DocStore(prefix=prefix, bucket=bucket)
            target = OscZarr(bucket=bucket, prefix=prefix)

    cluster = LocalCluster(processes=False)

    client = Client(cluster)

    source = NexGddpCmip6()

    model = DaysTasAboveIndicator(
        threshold_temps_c=threshold_list,
        window_years=window_years,
        gcms=gcm_list,
        scenarios=scenario_list,
        central_years=central_year_list,
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)


class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator


def cli():
    fire.Fire(Cli)
