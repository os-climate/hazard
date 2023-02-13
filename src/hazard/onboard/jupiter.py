from dataclasses import dataclass
import os
from pathlib import PosixPath
from dask.distributed import Client
from fsspec.spec import AbstractFileSystem # type: ignore
import pandas as pd # type: ignore
from rasterio.crs import CRS # type: ignore
import rioxarray
import xarray as xr
from typing import Dict, Iterable
from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardModel, MapInfo, Scenario
from hazard.protocols import OpenDataset, WriteDataArray, WriteDataset
from hazard.utilities import xarray_utilities

@dataclass
class BatchItem:
    model: HazardModel # type of hazard
    csv_filename: str
    jupiter_array_name: str

class JupiterOscFileSource():
    def __init__(self, dir: str, fs: AbstractFileSystem):   
        """Source to load data set provided by Jupiter Intelligence for use by OS-Climate
        to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
        The data is provided as a set of csv files.

        Args:
            dir (str): Directory containing OSC_Distribution; path to files are e.g. {dir}/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv.
            fs (AbstractFileSystem): File system.
        """
        self._dir = dir
        self._fs = fs

    def read(self, csv_filename: str) -> Dict[str, xr.DataArray]:
        """Read Jupiter csv data and convert into a set of DataArrays. 

        Args:
            csv_filename (str): Name of csv file, e.g. etlfire.csv.

        Returns:
            Dict[str, xr.DataArray]: Data arrays, keyed by Jupiter name.
        """
        df = pd.read_csv(os.path.join(self._dir, "OSC_Distribution", "OS-C-DATA", "OS-C Tables", csv_filename))
        ids = [c for c in df.columns if c not in ["key", "latitude", "longitude"]]
        df_pv = df.pivot(index="latitude", columns="longitude", values=ids)
        arrays: Dict[str, xr.DataArray] = {}
        for id in ids:
            da = xr.DataArray(data=df_pv[id], attrs={"crs" : CRS.from_epsg(4326)})
            da = da.where(da.data > -9999) # Jupiter set no-data
            arrays[id] = da
        return arrays

class Jupiter(IndicatorModel):
    """On-board data set provided by Jupiter Intelligence for use by OS-Climate
    to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
    """
    
    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        for model in self.inventory():
            if model.id == "wildfire_probability":
                jupiter_array_name = "fire{scenario}{year}metric_mean"
            else:
                jupiter_array_name = ""
            yield BatchItem(model=model, csv_filename="etlfire.csv", jupiter_array_name=jupiter_array_name)

    def inventory(self) -> Iterable[HazardModel]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardModel(
                type="Wildfire",
                id="wildfire_probability",
                path="wildfire/jupiter/v1",
                display_name="Wildfire probability",
                array_name="wildfire_probability_{scenario}_{year}",
                description="wildfire_probability.md",
                map = MapInfo(
                    colormap=Colormap(
                        name="heating",
                        min_value=0.0,
                        max_value=1.0,
                        units="none"),
                    array_name="wildfire_probability_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ])]
    
    def run_single(self, item: BatchItem, source: JupiterOscFileSource, target: WriteDataArray, client: Client):
        """Run a single item of the batch."""
        arrays = source.read(item.csv_filename)
        model = [m for m in self.inventory() if m.id == "wildfire_probability"][0]
        for scenario in model.scenarios:
            for year in scenario.years:
                da = arrays[item.jupiter_array_name.format(scenario=scenario.id, year=year)]
                da = da.reindex(latitude=da.latitude[::-1]) # by convention latitude reversed
                pp = PosixPath(item.model.path, item.model.array_name.format(scenario=scenario.id, year=year))
                target.write(str(pp), da)
                reprojected = da.sel(latitude=slice(85, -85)).rio.reproject("EPSG:3857") #, shape=da.data.shape, nodata=0) # from EPSG:4326 to EPSG:3857 (Web Mercator)
                pp_map = pp.with_name(pp.name + "_map")
                target.write(str(pp_map), reprojected)
