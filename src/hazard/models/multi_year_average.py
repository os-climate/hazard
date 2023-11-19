from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
import logging
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Tuple, TypeVar

from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource, MapInfo
from hazard.utilities.map_utilities import generate_map

from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon
from hazard.protocols import Averageable, OpenDataset, ReadWriteDataArray, WriteDataArray

from dask.distributed import Client
from rasterio.crs import CRS # type: ignore
import xarray as xr

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Averageable)

@dataclass
class Indicator:
    array: xr.DataArray
    path: PurePosixPath
    bounds: List[Tuple[float, float]]


class MultiYearAverageIndicatorBase(IndicatorModel[T]):  
    """Indicator which is the average of indicators produced for a number of individual years.
    Such calculations can be split by year and run in parallel.
    """

    _default_window_years: int=20
    _default_gcms: Iterable[str]=["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"]
    _default_scenarios: Iterable[str]=["historical", "ssp126", "ssp245", "ssp585"]
    _default_central_year_historical: int = 2005
    _default_central_years: Iterable[int]=[2030, 2040, 2050] 

    def __init__(self,
            window_years: int=20,
            gcms: Iterable[str]=_default_gcms,
            scenarios: Iterable[str]=_default_scenarios,
            central_year_historical: int =_default_central_year_historical,
            central_years: Iterable[int]=_default_central_years): 
        """Construct model to calculate degree days from temperature data sets.

        Args:
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation. Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1", "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation. Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int): Central year to include in calculation for historical scenario. Defaults to 2005.
            central_years (Iterable[int], optional): Central years to include in calculation. Defaults to [2010, 2030, 2040, 2050].
        """

        # 1995 to 2014 (2010), 2021 tp 2040 (2030), 2031 to 2050 (2040), 2041 to 2060 (2050)
        self.window_years = window_years
        self.gcms = gcms
        self.scenarios = scenarios
        self.central_years = central_years
        self.central_year_historical = central_year_historical
        
    def run_single(self, item: T, source: OpenDataset, target: ReadWriteDataArray, client: Client):
        averaged_indicators = self._averaged_indicators(client, source, target, item)
        for indicator in averaged_indicators:
            indicator.array.attrs["crs"] = CRS.from_epsg(4326)
            logger.info(f"Writing array to {str(indicator.path)}")
            target.write(str(indicator.path), indicator.array)
            path_map = indicator.path.with_name(indicator.path.name + "_map")
            self._generate_map(str(indicator.path), str(path_map), indicator.bounds, target)
        return
    
    def _generate_map(self, path: str, map_path: str, bounds: Optional[List[Tuple[float, float]]], target: ReadWriteDataArray):
        generate_map(path, map_path, bounds, target)
        return
    
    def _averaged_indicators(self, client: Client, source: OpenDataset, target: WriteDataArray, item: Averageable) -> List[Indicator]:
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        years = range(item.central_year - self.window_years // 2, item.central_year + self.window_years // 2 + (self.window_years % 2))
        logger.info(f"Calculating average indicator for batch item {str(item)}, years={list(years)}")
        futures = []
        for year in years:
            future = client.submit(self._calculate_single_year_indicators, source, item, year)
            futures.append(future)
        single_year_sets: List[List[Indicator]] = list(client.gather(futures)) # indicators for each year
        indics_per_year = len(single_year_sets[0]) # number of indicators for each year
        res: List[Indicator] = []
        for i in range(indics_per_year):
            average = sum(set[i].array for set in single_year_sets) / float(len(years))
            assert isinstance(average, xr.DataArray) # must be non-zero
            res.append(Indicator(array=enforce_conventions_lat_lon(average), 
                                 path=single_year_sets[0][i].path,
                                 bounds=single_year_sets[0][i].bounds))
        return res

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get batch items (batch items can be calculated independently from one another)."""
        ...  
    
    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory."""
        ...

    @abstractmethod
    def _calculate_single_year_indicators(self, source: OpenDataset, item: T, year: int) -> List[Indicator]:
        """Calculate indicators for a single year for a single batch item. If just a single indicator per batch, a list
        of length one is expected."""
        ...


@dataclass
class BatchItem():
    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int

    def __str__(self):
        return f"gcm={self.gcm}, scenario={self.scenario}, central_year={self.central_year}"


class ThresholdBasedAverageIndicator(MultiYearAverageIndicatorBase[BatchItem]):
    
    def batch_items(self) -> Iterable[BatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
        resource = self._resource()
        for gcm in self.gcms:
            for scenario in self.scenarios:
                central_years = [self.central_year_historical] if scenario == "historical" else self.central_years
                for central_year in central_years:
                    yield BatchItem(resource=resource, gcm=gcm, scenario=scenario, central_year=central_year)    
    
    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return [self._resource()] #.expand()

    def _get_indicators(self, item: BatchItem, data_arrays: List[xr.DataArray], param: str) -> List[Indicator]:
        """Find the 

        Args:
            item (BatchItem): _description_
            data_arrays (List[xr.DataArray]): _description_
            param (str): _description_

        Returns:
            List[Indicator]: _description_
        """
        resource = item.resource
        paths = [item.resource.path.format(temp_c=threshold, gcm=item.gcm, scenario=item.scenario, year=item.central_year)
                 for threshold in resource.params[param]]
        assert isinstance(resource.map, MapInfo)      
        return [Indicator(array=array, path=PurePosixPath(paths[i]), bounds=resource.map.bounds) for i, array in enumerate(data_arrays)]

    @abstractmethod
    def _resource(self) -> HazardResource:
       ...