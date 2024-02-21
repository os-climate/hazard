import os
import s3fs
import zarr
import numpy as np
import xarray as xr
import geopandas as gpd
from geocube.api.core import make_geocube

import logging
from pyproj.crs import CRS

from hazard.sources.osc_zarr import OscZarr

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set











from dataclasses import dataclass
import os
import numpy as np
import xarray as xr
from pathlib import PurePosixPath
from dask.distributed import Client
from fsspec.spec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
import xarray as xr
import logging

from hazard.indicator_model import IndicatorModel
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr

from typing import Any, Iterable, Optional
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    scenario: str
    central_year: int
    input_dataset_filename: str


class TUDelftFire(IndicatorModel[BatchItem]):

    def __init__(self, 
                 source_dir: str,
                 fs: Optional[AbstractFileSystem] = None):
        """
        Pan-European data sets of forest fire probability of occurrence under 
        present and future climate.

        METADATA:
        Link: https://data.4tu.nl/datasets/f9a134ad-fff9-44d5-ad4e-a0e9112b551e
        Data type: historical and scenario probability 
        Hazard indicator: Probability of Fire Weather Index (FWI) > 20 and > 45.
        Region: Pan-Europe
        Resolution: 49km
        Time range: 1971-2000,2021-2050,2071-2100
        Scenarios: RCP4.5, 8.5
        File type: Map (.nc)

        DATA DESCRIPTION:
        NetCDF files containing daily probabilities of high forest fire danger in 
        Europe under present and projected future climates. Includes gridded (NetCDF) 
        datasets of high forest fire danger probabilities for the present climate 
        (1981-2010) based on the ERA-Interim reanalysis and for the projected 
        climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
        and 2071-2100). 

        IMPORTANT NOTES:
        Tile maps do not work because very low resolution data.
        Coordinate system is rotated longitud and latitude. Needs to be transformed.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected. 
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """
        
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir

        # Download source data
        self._url = 'https://opendap.4tu.nl/thredds/fileServer/data2/uuid/a9f42f0a-1db4-4728-ad8c-d03a1d3f3c4d/'

        self._resource_fwi20 = self.inventory()[0]
        self._resource_fwi45 = self.inventory()[1]

    def batch_items(self) -> Iterable[BatchItem]:
        return [ BatchItem(scenario="historical", central_year=1980, input_dataset_filename="multi-model_fwi20_historical_1971-2000.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2035, input_dataset_filename="multi-model_fwi20_rcp45_2021-2050.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2035, input_dataset_filename="multi-model_fwi20_rcp85_2021-2050.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2085, input_dataset_filename="multi-model_fwi20_rcp45_2071-2100.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2085, input_dataset_filename="multi-model_fwi20_rcp85_2071-2100.nc"),
                 BatchItem(scenario="historical", central_year=1980, input_dataset_filename="multi-model_fwi45_historical_1971-2000.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2035, input_dataset_filename="multi-model_fwi45_rcp45_2021-2050.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2035, input_dataset_filename="multi-model_fwi45_rcp85_2021-2050.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2085, input_dataset_filename="multi-model_fwi45_rcp45_2071-2100.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2085, input_dataset_filename="multi-model_fwi45_rcp85_2071-2100.nc")
                 ]

    def prepare(self, item: BatchItem, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            nc_url = self._url + item.input_dataset_filename
            download_file(nc_url, working_dir, "tudelft_fire")
            for file in os.listdir(working_dir):
                with open(file, 'rb') as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read()) 
        else:
            # download and unzip directly in location
            source = PurePosixPath(self.source_dir)
            nc_url = self._url + item.input_dataset_filename
            download_file(nc_url, str(source), item.input_dataset_filename)

    def run_single(self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client):
        input = PurePosixPath(self.source_dir, item.input_dataset_filename)
        assert target == None or isinstance(target, OscZarr)
        filename = str(input)

        if 'fwi20' in item.input_dataset_filename:
            path_ = self._resource_fwi20.path.format(scenario=item.scenario, year=item.central_year)
        else:
            path_ = self._resource_fwi45.path.format(scenario=item.scenario, year=item.central_year)

        ds = xr.open_dataset(filename)
        da = ds.risk
        da = da.rename({'time':'index'})
        da['index'] = [0]
        da.data = np.nan_to_num(da.data)
        target.write(path_, da)

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        ...
        create_tiles_for_resource(source, target, self._resource_fwi20)
        create_tiles_for_resource(source, target, self._resource_fwi45)


    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="FWI",
                indicator_model_gcm = 'CLMcom-CCLM4-8-17-CLM3-5',
                path="fire/fire_tudelft/v1/fwi_greater_than_20_{scenario}_{year}",
                params={},
                display_name="FWI under 20 (TUDelft)",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in 
                Europe under present and projected future climates. Includes gridded (NetCDF) 
                datasets of high forest fire danger probabilities for the present climate 
                (1981-2010) based on the ERA-Interim reanalysis and for the projected 
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
                and 2071-2100). 
                """,
                group_id = "",
                display_groups=[],
                map = MapInfo(
                    bounds= [],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=100.0,
                        units="prob"),
                    path="maps/fire/fire_tudelft/v1/fwi_greater_than_20_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="probability",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1980]),
                    Scenario(
                        id="rcp4p5",
                        years=[2035, 2085]),
                    Scenario(
                        id="rcp8p5",
                        years=[2035, 2085]),

                    ]),
            HazardResource(
                hazard_type="Fire",
                indicator_id="FWI",
                indicator_model_gcm = 'CLMcom-CCLM4-8-17-CLM3-5',
                path="fire/fire_tudelft/v1/fwi_greater_than_45_{scenario}_{year}",
                params={},
                display_name="FWI under 45 (TUDelft)",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in 
                Europe under present and projected future climates. Includes gridded (NetCDF) 
                datasets of high forest fire danger probabilities for the present climate 
                (1981-2010) based on the ERA-Interim reanalysis and for the projected 
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
                and 2071-2100). 
                """,
                group_id = "",
                display_groups=[],
                map = MapInfo(
                    bounds= [],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=100.0,
                        units="prob"),
                    path="maps/fire/fire_tudelft/v1/fwi_greater_than_45_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="probability",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1980]),
                    Scenario(
                        id="rcp4p5",
                        years=[2035, 2085]),
                    Scenario(
                        id="rcp8p5",
                        years=[2035, 2085]),

                    ])
                    ]