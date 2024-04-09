import logging  # noqa: E402

import fire
from dask.distributed import Client, LocalCluster  # noqa: E402

from hazard.docs_store import DocStore  # type: ignore # noqa: E402
from hazard.models.days_tas_above import DaysTasAboveIndicator  # noqa: E402
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6  # noqa: E402
from hazard.sources.osc_zarr import OscZarr  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)

def days_tas_above_indicator(bucket: str, prefix: str, gcm: str, scenario: str, year: int, threshold: int):
    """
    Run the days_tas_above indicator generation for a given gcm, scenario, year and temperature threshold.
    Store the result in a zarr store located in a given bucket and prefix.
    """

    docs_store = DocStore(prefix="hazard_test")

    cluster = LocalCluster(processes=False)

    client = Client(cluster)

    source = NexGddpCmip6()

    target = OscZarr(bucket=bucket, prefix=prefix)

    model = DaysTasAboveIndicator(
        threshold_temps_c=[threshold],
        window_years=1,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[year],
    )

    docs_store.update_inventory(model.inventory())

    model.run_all(source, target, client=client)

class Cli(object):
    def __init__(self) -> None:
        self.days_tas_above_indicator = days_tas_above_indicator
        
def cli():
    fire.Fire(Cli)
