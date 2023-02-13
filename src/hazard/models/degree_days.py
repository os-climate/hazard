from dataclasses import dataclass
import os, logging
from typing import Iterable, Optional
from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardModel

from hazard.utilities.xarray_utilities import enforce_conventions
from hazard.protocols import OpenDataset, WriteDataArray, WriteDataset

from dask.distributed import Client, LocalCluster
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

@dataclass
class BatchItem():
    gcm: str
    scenario: str
    central_year: int

class DegreeDays(IndicatorModel[BatchItem]):  
    """Calculates degree days from temperature data sets."""

    def __init__(self,
            threshold: float=32,
            window_years: int=20,
            gcms: Iterable[str]=["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1",
            "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"],
            scenarios: Iterable[str]=["historical", "ssp126", "ssp245", "ssp585"],
            central_years: Iterable[int]=[2010, 2030, 2040, 2050]):
        """Construct model to calculate degree days from temperature data sets.

        Args:
            threshold (float, optional): Degree days above threshold are calculated. Defaults to 32.
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation. Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation. Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_years (Iterable[int], optional): Central years to include in calculation. Defaults to [2010, 2030, 2040, 2050].
        """

        self.threshold: float = 273.15 + threshold # in Kelvin; degree days above 32C
        # 1995 to 2014 (2010), 2021 tp 2040 (2030), 2031 to 2050 (2040), 2041 to 2060 (2050)
        self.window_years = window_years
        self.gcms = gcms
        self.scenarios = scenarios
        self.central_years = central_years
        
    def batch_items(self) -> Iterable[BatchItem]:
        """Items to process."""
        for gcm in self.gcms:
            for scenario in self.scenarios:
                for central_year in self.central_years:
                    yield BatchItem(gcm=gcm, scenario=scenario, central_year=central_year)    

    def inventory(self) -> Iterable[HazardModel]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        yield HazardModel(
            type="ChronicHeat",
            path="")
    
    def run_single(self, item: BatchItem, source: OpenDataset, target: WriteDataArray, client: Client):
        average_deg_days = self._average_degree_days(client, source, target, item)
        target.write(self._item_path(item), average_deg_days)

    def _average_degree_days(self, client: Client, source: OpenDataset, target: WriteDataArray, item: BatchItem):
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



