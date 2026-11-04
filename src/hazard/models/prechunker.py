from concurrent import futures
from contextlib import contextmanager
from datetime import datetime
import time
import logging
import os
import threading
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    MutableMapping,
    Protocol,
    Sequence,
    Tuple,
)

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import xarray as xr

from hazard.protocols import OpenDataset
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

logger = logging.getLogger(__name__)


class CachingSource(OpenDataset):
    def download_to_cache(
        self,
        gcms: Sequence[str],
        scenarios: Sequence[str],
        quantities: Sequence[str],
        years: Sequence[int],
    ): ...


def download_single(path: str, cache_dir: Path, client: Any):
    p = Path(path)
    local_file = cache_dir / p.parts[-1]
    if local_file.exists():
        logger.info(f"File already exists, skipping download {path}")
        return local_file
    with open(str(local_file) + ".download", "wb") as data:
        for attempt in range(5):
            try:
                logger.info(f"Starting download {path}, attempt {attempt}")
                client.download_fileobj(
                    p.parts[0], str(PurePosixPath(*p.parts[1:])), data
                )
                break
            except Exception as e:
                if attempt == 4:
                    raise e
                else:
                    logger.info(f"Retrying download, attempt {attempt}: {e}")
                    time.sleep(5 * 2**attempt)
                    pass
    os.rename(str(local_file) + ".download", str(local_file))
    logger.info(f"Completed download {path}")
    return local_file


def download_paths(paths: Sequence[str], cache_dir: Path, client: Any):
    result = {}
    executor = futures.ThreadPoolExecutor(max_workers=8)
    try:
        future_to_key = {
            executor.submit(download_single, p, cache_dir, client): p for p in paths
        }
        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if exception:
                result[key] = exception
            else:
                result[key] = future.result()
    except BaseException:
        # e.g. KeyboardInterrupt: do not start the queued downloads (those in progress still complete)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    if any(isinstance(v, Exception) for v in result.values()):
        for k, v in result.items():
            if isinstance(v, Exception):
                logger.error(f"Error downloading year {k}: {v}")
        raise RuntimeError("Error(s) occurred during download.")
    return result


# One lock per cached file, so that a file is downloaded only once if opened by several threads.
# These are module-level so that they are shared by all copies of a NexGddpCmip6CachedSource in
# the process (dask, for example, gives each task its own copy of the arguments).
_download_locks: Dict[str, threading.Lock] = {}
_download_locks_lock = threading.Lock()


def _download_lock(local_file: Path) -> threading.Lock:
    with _download_locks_lock:
        return _download_locks.setdefault(str(local_file), threading.Lock())


class NexGddpCmip6CachedSource(CachingSource):
    """NEX-GDDP-CMIP6 source that caches the files it reads in a local directory.

    open_dataset_year opens the file for the year from the cache, first downloading it (anonymously,
    from the public bucket) if it is not there. download_to_cache can be used to download many
    files in parallel up front.

    The source can be pickled (as dask does with the arguments of tasks); the S3 client is not
    pickled, a copy creating the default (anonymous) client when it needs one.
    """

    def __init__(self, cache_dir: Path, client=None):
        self.cache_dir = cache_dir
        self.source = NexGddpCmip6()
        self.path_for_year = self.source.path
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=50)
            )
        return self._client

    def __getstate__(self):
        # the S3 client cannot be pickled
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    @property
    def resolution(self) -> str:
        return self.source.resolution

    def gcms(self) -> Iterable[str]:
        return self.source.gcms()

    @contextmanager
    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> Generator[xr.Dataset, None, None]:
        local_file = self._cached_file(self._path(gcm, scenario, quantity, year))
        with xr.open_dataset(local_file, chunks=chunks) as ds:
            yield ds

    def _cached_file(self, path: str) -> Path:
        """Local path of the file in the cache, downloading it first if it is not there."""
        local_file = self.cache_dir / Path(path).parts[-1]
        if local_file.exists():
            return local_file
        with _download_lock(local_file):
            # download_single does nothing if another thread has downloaded the file meanwhile
            return download_single(path, self.cache_dir, self.client)

    def download_to_cache(
        self,
        gcms: Sequence[str],
        scenarios: Sequence[str],
        quantities: Sequence[str],
        years: Sequence[int],
    ):
        paths = [
            self._path(g, s, q, y)
            for y in years
            for q in quantities
            for s in scenarios
            for g in gcms
        ]
        download_paths(paths, self.cache_dir, self.client)

    def download_inputs(
        self, inputs: Iterable[Tuple[str, str, str, int]], dry_run: bool = False
    ) -> List[str]:
        """Download, in parallel, the files of inputs (GCM, scenario, quantity, year) not already in the cache.

        Returns:
            List[str]: Paths of the files that were (with dry_run, would be) downloaded.
        """
        if not dry_run:
            self.remove_stale_downloads()
        paths = sorted({self._path(*i) for i in inputs})
        missing = [p for p in paths if not (self.cache_dir / Path(p).name).exists()]
        logger.info(
            f"{len(paths)} files needed: {len(paths) - len(missing)} in the cache {self.cache_dir}, "
            f"{len(missing)} to download"
        )
        if not dry_run and missing:
            download_paths(missing, self.cache_dir, self.client)
        return missing

    def remove_stale_downloads(self, max_age_s: float = 3600) -> List[Path]:
        """Remove the partial (.download) files of interrupted downloads not modified for max_age_s.

        Partial files are never used (a download starts again from the beginning), so this only
        tidies the cache. Recently modified ones are kept, as another process may be downloading them.

        Returns:
            List[Path]: The files removed.
        """
        now = time.time()
        stale = [
            f
            for f in self.cache_dir.glob("*.download")
            if now - f.stat().st_mtime > max_age_s
        ]
        for f in stale:
            f.unlink(missing_ok=True)
        if stale:
            logger.info(
                f"removed {len(stale)} partial file(s) of interrupted downloads from {self.cache_dir}"
            )
        return stale

    def _path(self, gcm: str, scenario: str, quantity: str, year: int):
        if year < 2015:
            return self.path_for_year(
                gcm=gcm, scenario="historical", quantity=quantity, year=year
            )[0]
        else:
            return self.path_for_year(
                gcm=gcm, scenario=scenario, quantity=quantity, year=year
            )[0]


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
        cache: bool = True,
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
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=50)
        )
        self.source = (
            NexGddpCmip6CachedSource(self.working_dir, self.s3)
            if cache
            else NexGddpCmip6()
        )
        self.cache = cache

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
            try:
                latest_year = pd.Timestamp(existing_ds["time"].values[-1]).year
            except Exception:
                # in case of cftime.DatetimeNoLeap
                latest_year = existing_ds.indexes["time"].to_datetimeindex()[-1].year
            remaining_years = [y for y in years if y > latest_year]
        except Exception:
            remaining_years = list(years)

        if self.cache:
            assert isinstance(self.source, CachingSource)
            self.source.download_to_cache(
                [gcm], [scenario], [quantity], remaining_years
            )

        logger.info(f"Processing group {group}.")

        for year in remaining_years:
            logger.info(f"Adding year {year} to rechunked set.")
            with self.source.open_dataset_year(gcm, scenario, quantity, year) as ds:
                ds = ds.load()
                if year == years[0]:
                    ds = ds.chunk(
                        {
                            "time": 365,
                            "lat": self.lat_chunk_size,
                            "lon": self.lon_chunk_size,
                        }
                    )
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
