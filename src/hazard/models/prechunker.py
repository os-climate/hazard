from concurrent import futures
from datetime import datetime
import time
import logging
import os
from pathlib import Path, PurePosixPath
from typing import MutableMapping, Sequence

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import xarray as xr

from hazard.protocols import OpenDataset
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

logger = logging.getLogger(__name__)


class PrechunkedSource(OpenDataset):
    def __init__(self, working_dir: Path, prechunked_zarr_store: MutableMapping):
        self.prechunked_zarr_store = prechunked_zarr_store

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ):
        try:
            ds = xr.open_zarr(
                store=self.prechunked_zarr_store,
                group=quantity + "_" + gcm + "_" + scenario,
                consolidated=False,
            )
            yield ds.sel(time=slice(datetime(year, 1, 1), datetime(year + 1, 1, 1)))
        finally:
            if ds is not None:
                ds.close()


class Prechunker:
    def __init__(
        self,
        working_dir: Path,
        prechunk_zarr_store: MutableMapping,
        gcms: Sequence[str],
        year_min: int,
        year_max: int,
        scenarios: Sequence[str],
        quantities: Sequence[str],
        lat_chunk_size: int = 40,
        lon_chunk_size: int = 40,
    ):
        """For certain sources which are not chunked, it can be faster to download data locally
        (in parallel) first before processing. This helper class handles that prechunking step.
        Currently implements NexGddpCmip6 case, but intended to be generalised.

        Args:
            working_dir (Path): local directory to store temporary downloaded files.
            prechunk_zarr_store (MutableMapping): destination Zarr store for XArrays.
            gcms (Sequence[str]): GCMs.
            year_min (int): Minimum year.
            year_max (int): Maximum year.
            scenarios (Sequence[str]): Scenarios.
            quantities (Sequence[str]): Quantities, e.g. "tas" etc.
            lat_chunk_size (int, optional): Latitude chunk size. Defaults to 40.
            lon_chunk_size (int, optional): Longitude chunhk size. Defaults to 40.
        """
        self.gcms = gcms
        self.lat_chunk_size, self.lon_chunk_size = lat_chunk_size, lon_chunk_size
        self.year_min = year_min
        self.year_max = year_max
        self.scenarios = scenarios
        self.quantities = quantities
        self.prechunk_zarr_store = prechunk_zarr_store
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.path_for_year = NexGddpCmip6().path
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=50)
        )

    def prechunk(self):
        years = list(range(self.year_min, self.year_max + 1))
        for gcm in self.gcms:
            for scenario in self.scenarios:
                for quantity in self.quantities:
                    logger.info(
                        f"Prechunking gcm={gcm} scenario={scenario} quantity={quantity}"
                    )
                    self._prechunk(gcm, scenario, quantity, years)

    def _prechunk(self, gcm: str, scenario: str, quantity: str, years: Sequence[int]):
        group = quantity + "_" + gcm + "_" + scenario
        try:
            existing_ds = xr.open_zarr(
                store=self.prechunk_zarr_store, group=group, consolidated=False
            )
            latest_year = pd.Timestamp(existing_ds["time"].values[-1]).year
            remaining_years = [y for y in years if y > latest_year]
        except Exception:
            remaining_years = years

        result = {}
        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_key = {
                executor.submit(self._download, gcm, scenario, quantity, y): y
                for y in remaining_years
            }
            for future in futures.as_completed(future_to_key):
                key = future_to_key[future]
                exception = future.exception()
                if exception:
                    result[key] = exception
                else:
                    result[key] = future.result()

        if any(isinstance(v, Exception) for v in result.values()):
            for k, v in result.items():
                if isinstance(v, Exception):
                    logger.error(f"Error downloading year {k}: {v}")
            raise RuntimeError("Error(s) occurred during download.")

        logger.info(f"Processing group {group} from downloads.")

        for year in remaining_years:
            logger.info(f"Adding year {year} to rechunked set.")
            ds = xr.open_dataset(result[year])
            ds = ds.load()
            ds = ds.chunk(
                {"time": 365, "lat": self.lat_chunk_size, "lon": self.lon_chunk_size}
            )
            if year == years[0]:
                ds.to_zarr(
                    store=self.prechunk_zarr_store,
                    group=group,
                    mode="w",
                    consolidated=False,
                )
            else:
                ds.to_zarr(
                    store=self.prechunk_zarr_store,
                    group=group,
                    append_dim="time",
                    consolidated=False,
                )
        logger.info(f"Prechunks created for group {group}.")

    def _download(self, gcm: str, scenario: str, quantity: str, year: int):
        if year < 2015:
            path = self.path_for_year(
                gcm=gcm, scenario="historical", quantity=quantity, year=year
            )
        else:
            path = self.path_for_year(
                gcm=gcm, scenario=scenario, quantity=quantity, year=year
            )
        return self._fetch(path[0])

    def _fetch(self, path: str):
        p = Path(path)
        local_file = self.working_dir / p.parts[-1]
        if local_file.exists():
            logger.info(f"File already exists, skipping download {path}")
            return local_file
        with open(str(local_file) + ".download", "wb") as data:
            for attempt in range(5):
                try:
                    logger.info(f"Starting download {path}, attempt {attempt}")
                    self.s3.download_fileobj(
                        p.parts[0], str(PurePosixPath(*p.parts[1:])), data
                    )
                    break
                except Exception as e:
                    if attempt == 4:
                        raise e
                    else:
                        logger.info(f"Retrying download, attempt {attempt}: {e}")
                        time.sleep(5)
                        pass
        os.rename(str(local_file) + ".download", str(local_file))
        logger.info(f"Completed download {path}")
        return local_file
