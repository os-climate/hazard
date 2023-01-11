from dataclasses import dataclass
import os, logging
from typing import Iterable

from hazard.utilities.xarray_utilities import enforce_conventions
from hazard.protocols import OpenDataset, WriteDataset

from dask.distributed import Client, LocalCluster
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

@dataclass
class BatchItem():
    gcm: str
    scenario: str
    central_year: int


class DegreeDays:  
    """Calculates degree days from temperature data sets."""

    def __init__(self,
            theshold: float=32,
            window_years: int=20,
            gcms: Iterable[str]=None,
            scenarios: Iterable[str]=None,
            central_years: Iterable[int]=None):
        """Construct model to calculate degree days from temperature data sets.

        Args:
            theshold (float, optional): degree days above threshold are calculated. Defaults to 32.
            window_years (int, optional): number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation. Defaults to None.
            scenarios (Iterable[str], optional): scenarios to include in calculation. Defaults to None.
            central_years (Iterable[int], optional): central years to include in calculation. Defaults to None.
        """


        self.threshold: float = 273.15 + theshold # in Kelvin; degree days above 32C
        self.window_years = window_years
        self.gcms = ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1",
            "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"] if gcms is None else gcms
        self.scenarios = ["historical", "ssp126", "ssp245", "ssp585"] if scenarios is None else scenarios
        # 1995, 2014 (2010)
        # 2021, 2040 (2030)
        # 2031, 2050 (2040)
        # 2041, 2060 (2050)
        self.central_years = [2010, 2030, 2040, 2050] if central_years is None else central_years


    def run(self, source: OpenDataset, target: WriteDataset, client: Client=None):
        if (client is None):
            cluster = LocalCluster(processes=False)
            client = Client(cluster)
        for item in self._batch_items():
            average_deg_days = self._average_degree_days(client, source, target, item)
            target.write(self._item_path(item), average_deg_days)


    def _average_degree_days(self, client: Client, source: OpenDataset, target: WriteDataset, item: BatchItem):
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        years = range(item.central_year - self.window_years // 2, item.central_year + self.window_years // 2 + (self.window_years % 2))
        logging.info(f"Calculating average degree days, gcm={item.gcm}, scenario={item.scenario}, years={list(years)}")
        futures = []
        for year in years:
            future = client.submit(self._degree_days, source, item.gcm, item.scenario, year)
            futures.append(future)
        deg_days = client.gather(futures)
        average = sum(deg_days) / float(len(years))
        return enforce_conventions(average)


    def _batch_items(self) -> Iterable[BatchItem]:
        """Items to process."""
        for gcm in self.gcms:
            for scenario in self.scenarios:
                for central_year in self.central_years:
                    yield BatchItem(gcm=gcm, scenario=scenario, central_year=central_year)    


    def _degree_days(self, source: OpenDataset, gcm: str, scenario: str, year: int ) -> xr.DataArray:
        """Calculate degree days for Dataset provided."""
        # check DataArray 
        with source.open_dataset_year(gcm, scenario, "tasmax", year) as ds:
            if any(coord not in ds.coords.keys() for coord in ['lat', 'lon', 'time']):
                raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
            if (ds.time.dt.year != year).any():
                raise ValueError("unexpected year found") 
            # normalize to 365 days
            scale = 365.0 / len(ds.time)
            # will raise error if taxmax not present    
            return scale * xr.where(ds.tasmax > self.threshold, ds.tasmax - self.threshold, 0).sum(dim=["time"])

    
    def _item_path(self, item: BatchItem) -> str:
        path = "chronic_heat/osc/v2" # v2 uses downscaled data
        typ = "mean_degree_days_v2" # need v2 in filename to make unique
        levels = ['above', '32c']
        return os.path.join(path, f"{typ}_{levels[0]}_{levels[1]}_{item.gcm}_{item.scenario}_{item.central_year}")



