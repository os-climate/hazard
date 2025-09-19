from concurrent import futures
import logging
import os
from pathlib import Path, PurePosixPath
from typing import MutableMapping, Sequence

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import xarray as xr

from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

logger = logging.getLogger(__name__)    

class Prechunker:
    def __init__(self, working_dir: Path, 
                 prechunk_zarr_store: MutableMapping,
                 gcms: Sequence[str],
                 year_min: int,
                 year_max: int,
                 scenarios: Sequence[str],
                 quantities: Sequence[str]):
        """For certain sources which are not chunked, it can be faster to download data locally
        first before processing. This class handles that prechunking step.

        Args:
            working_dir (Path): _description_
        """
        self.gcms = gcms
        self.lat_chunk_size, self.lon_chunk_size = 40, 40
        self.year_min = year_min
        self.year_max = year_max
        self.scenarios = scenarios
        self.quantities = quantities
        self.prechunk_zarr_store = prechunk_zarr_store
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.path_for_year = NexGddpCmip6().path_stac
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def prechunk(self):
        years = list(range(self.year_min, self.year_max + 1))
        for gcm in self.gcms:  
            for scenario in self.scenarios:
                for quantity in self.quantities:
                    logger.info(f"Prechunking gcm={gcm} scneario={scenario} quantity={quantity}")
                    self._prechunk(gcm, scenario, quantity, years)
        
    def _prechunk(self, gcm: str, scenario: str, quantity: str, years: Sequence[int]):
        result = {}
        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_key = {executor.submit(self._download, gcm, scenario, quantity, y): y for y in years}
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
 
        group = quantity + "_" + gcm + "_" + scenario
        logger.info(f"processing group {group}.")

        for year in years:
            ds = xr.open_dataset(result[year])
            #.chunk(
            #    {"time": 365, "lat": self.lat_chunk_size, "lon": self.lon_chunk_size}
            #)
            ds = ds.load()
            ds = ds.chunk({"time": 365, "lat": self.lat_chunk_size, "lon": self.lon_chunk_size})
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
            path = self.path_for_year(gcm=gcm, scenario="historical", quantity=quantity, year=year)
        else:
            path = self.path_for_year(gcm=gcm, scenario=scenario, quantity=quantity, year=year)
        return self._fetch(path)
        
    def _fetch(self, path: str):
        logger.info(f"Starting download {path}")  
        p = Path(path)
        local_file = self.working_dir / p.parts[-1]
        if local_file.exists():
            logger.info(f"File already exists, skipping download {path}")  
            return local_file
        with open(str(local_file) + ".download", 'wb') as data:
            self.s3.download_fileobj(p.parts[1], str(PurePosixPath(*p.parts[2:])), data)
        os.rename(str(local_file) + ".download", str(local_file))
        logger.info(f"Completed download {path}")  
        return local_file


