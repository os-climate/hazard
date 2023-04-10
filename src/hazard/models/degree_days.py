from dataclasses import dataclass
import os, logging
from pathlib import PosixPath
from typing import Iterable, Optional
from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardModel, MapInfo, Scenario

from hazard.utilities.xarray_utilities import enforce_conventions
from hazard.protocols import OpenDataset, WriteDataArray, WriteDataset

from dask.distributed import Client, LocalCluster
import numpy as np
import rasterio # type: ignore
from rasterio.crs import CRS # type: ignore
import rasterio.enums # type: ignore
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
            central_year_historical: int = 2010,
            central_years: Iterable[int]=[2030, 2040, 2050]):
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
        self.central_year_historical = central_year_historical
        
    def batch_items(self) -> Iterable[BatchItem]:
        """Items to process."""
        for gcm in self.gcms:
            for scenario in self.scenarios:
                central_years = [self.central_year_historical] if scenario == "historical" else self.central_years
                for central_year in central_years:
                    yield BatchItem(gcm=gcm, scenario=scenario, central_year=central_year)    

    def inventory(self) -> Iterable[HazardModel]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardModel(
                type="ChronicHeat",
                id="mean_degree_days_v2/above/32c",
                path="chronic_heat/osc/v2",
                display_name="Mean degree days above 32°C (NorESM2-MM)",
                array_name="mean_degree_days_v2_above_32c_NorESM2-MM_{scenario}_{year}",
                description="""
Degree days indicators are calculated by integrating over time the absolute difference in temperature
of the medium over a reference temperature.
                """,
                group_id = "",
                map = MapInfo( # type:ignore
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_value=12.0,
                        max_index=255,
                        units="degree days"),
                    array_name="mean_degree_days_v2_above_32c_NorESM2-MM_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="degree days",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[self.central_year_historical]),
                    Scenario(
                        id="ssp126",
                        years=list(self.central_years)),
                    Scenario(
                        id="ssp245",
                        years=list(self.central_years)),
                    Scenario(
                        id="ssp585",
                        years=list(self.central_years)),
                    ]
            )
        ]
    
    def run_single(self, item: BatchItem, source: OpenDataset, target: WriteDataArray, client: Client):
        average_deg_days = self._average_degree_days(client, source, target, item)
        pp = self._item_path(item)
        target.write(str(pp), average_deg_days)
        reprojected = average_deg_days.sel(lat=slice(85, -85)).rio.reproject("EPSG:3857", resampling=rasterio.enums.Resampling.max) #, shape=da.data.shape, nodata=0) # from EPSG:4326 to EPSG:3857 (Web Mercator)
        pp_map = pp.with_name(pp.name + "_map")
        target.write(str(pp_map), reprojected)

    def _average_degree_days(self, client: Client, source: OpenDataset, target: WriteDataArray, item: BatchItem):
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        years = range(item.central_year - self.window_years // 2, item.central_year + self.window_years // 2 + (self.window_years % 2))
        logger.info(f"Calculating average degree days, gcm={item.gcm}, scenario={item.scenario}, years={list(years)}")
        futures = []
        for year in years:
            future = client.submit(self._degree_days, source, item.gcm, item.scenario, year)
            futures.append(future)
        deg_days = client.gather(futures)
        average = sum(deg_days) / float(len(years))
        return enforce_conventions(average)

    def _degree_days(self, source: OpenDataset, gcm: str, scenario: str, year: int) -> xr.DataArray:
        """Calculate degree days for Dataset provided."""
        # check DataArray 
        with source.open_dataset_year(gcm, scenario, "tasmax", year) as ds:
            return self._degree_days_indicator(ds, year)
            
    def _degree_days_indicator(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        if any(coord not in ds.coords.keys() for coord in ['lat', 'lon', 'time']):
                raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (ds.time.dt.year != year).any():
            raise ValueError("unexpected year found") 
        # normalize to 365 days
        scale = 365.0 / len(ds.time)
        # will raise error if taxmax not present    
        return scale * xr.where(ds.tasmax > self.threshold, ds.tasmax - self.threshold, 0).sum(dim=["time"])

    def _item_path(self, item: BatchItem) -> PosixPath:
        path = "chronic_heat/osc/v2" # v2 uses downscaled data
        typ = "mean_degree_days_v2" # need v2 in filename to make unique
        levels = ['above', '32c']
        return PosixPath(path, f"{typ}_{levels[0]}_{levels[1]}_{item.gcm}_{item.scenario}_{item.central_year}")



