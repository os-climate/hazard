from concurrent import futures
from contextlib import contextmanager
from datetime import datetime
import time
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Iterator, MutableMapping, Protocol, Sequence

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import xarray as xr

from hazard.protocols import OpenDataset
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

logger = logging.getLogger(__name__)



class CachingSource(OpenDataset):
    def download_to_cache(gcms: Sequence[str], scenarios: Sequence[str], quantities: Sequence[str], years: Sequence[int]):
        ...

    
def download_single(path: str, cache_dir: Path, client: any):
    p = Path(path)
    local_file = cache_dir / p.parts[-1]
    if local_file.exists():
        logger.info(f"File already exists, skipping download {path}")  
        return local_file
    with open(str(local_file) + ".download", 'wb') as data:
        for attempt in range(5):
            try:
                logger.info(f"Starting download {path}, attempt {attempt}")
                client.download_fileobj(p.parts[0], str(PurePosixPath(*p.parts[1:])), data)
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


def download_paths(paths: Sequence[str], cache_dir: Path, client: any):
    result = {}
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_key = {executor.submit(download_single, p, cache_dir, client): p for p in paths}
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
    return result


class NexGddpCmip6CachedSource(CachingSource):
    def __init__(self, cache_dir: Path, client=None):
        self.cache_dir = cache_dir
        self.source = NexGddpCmip6()
        self.path_for_year = self.source.path
        self.client = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50)) if client is None else client

    @contextmanager
    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> Iterator[xr.Dataset]: 
        try: 
            ds = xr.open_dataset(self.cache_dir / Path(self._path(gcm, scenario, quantity, year)).parts[-1])
            yield ds
        finally:
            if ds is not None:
                ds.close()

    def download_to_cache(self, gcms: Sequence[str], scenarios: Sequence[str], quantities: Sequence[str], years: Sequence[int]):
        paths = [self._path(g, s, q, y) for y in years for q in quantities for s in scenarios for g in gcms]
        download_paths(paths, self.cache_dir, self.client)

    def _path(self, gcm: str, scenario: str, quantity: str, year: int):
        if year < 2015:
            return self.path_for_year(gcm=gcm, scenario="historical", quantity=quantity, year=year)[0]
        else:
            return self.path_for_year(gcm=gcm, scenario=scenario, quantity=quantity, year=year)[0]

#     ds = xr.open_zarr(
#         store=self.prechunked_zarr_store,
#         group=quantity + "_" + gcm + "_" + scenario,
#         consolidated=False
#     )
#     yield ds.sel(time=slice(datetime(year, 1, 1), datetime(year + 1, 1, 1)))

class Prechunker:
    def __init__(self, working_dir: Path, 
                prechunk_zarr_store: MutableMapping,
                gcms: Sequence[str],
                year_min: int,
                year_max: int,
                scenarios: Sequence[str],
                quantities: Sequence[str],
                lat_chunk_size: int=40,
                lon_chunk_size: int=40,
                cache: bool=True):
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
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))
        self.source = NexGddpCmip6CachedSource(self.working_dir, self.s3) if cache else NexGddpCmip6()
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
            existing_ds = xr.open_zarr(store=self.prechunk_zarr_store, group=group, consolidated=False)
            try:
                latest_year = pd.Timestamp(existing_ds["time"].values[-1]).year
            except:
                # in case of cftime.DatetimeNoLeap
                latest_year = existing_ds.indexes['time'].to_datetimeindex()[-1].year
            remaining_years = [y for y in years if y > latest_year]
        except Exception:
            remaining_years = years
        
        if self.cache:
            assert isinstance(self.source, CachingSource)
            self.source.download_to_cache([gcm], [scenario], [quantity], remaining_years)
 
        logger.info(f"Processing group {group}.")

        for year in remaining_years:
            logger.info(f"Adding year {year} to rechunked set.")
            with self.source.open_dataset_year(gcm, scenario, quantity, year) as ds:
                ds = ds.load()
                if year == years[0]:
                    ds = ds.chunk({"time": 365, "lat": self.lat_chunk_size, "lon": self.lon_chunk_size})
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



