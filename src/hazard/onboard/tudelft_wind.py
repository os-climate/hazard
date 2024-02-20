
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


class TUDelftConvectiveWindstorm(IndicatorModel[BatchItem]):

    def __init__(self, 
                 source_dir: str,
                 fs: Optional[AbstractFileSystem] = None):
        """
        Pan-European data sets of severe and extreme wind probability of occurrence
        under present and future climate.

        METADATA:
        Link: https://data.4tu.nl/datasets/93463344-a63d-4ab2-a16a-f69b989b0e13
        Data type: historical and scenario probability
        Hazard indicator: Probability
        Region: Pan-Europe
        Resolution: 49km
        Time range: 1971-2000,2021-2050,2071-2100
        Scenarios: RCP4.5, 8.5
        File type: Map (.nc)

        DATA DESCRIPTION:
        NetCDF files containing gridded annual probability of severe convective
        windstorms (wind gusts > 25 m/s) and of extremely severe convective
        windstorms (wind gusts > 32 m/s) for present day and the future climate.
        The fields are multi model means of 15 regional climate model simulations
        (CORDEX).

        IMPORTANT NOTES:
        Tile maps do not work because very high resolution data
        Coordinate system is rotated longitud and latitude. Needs to be transformed.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected. 
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """

        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir

        # Download source data
        self._url = 'https://opendap.4tu.nl/thredds/fileServer/data2/uuid/9be95957-ca8e-4eb9-aa00-074c55df9032/'

        self._resource_severe = self.inventory()[0]
        self._resource_extreme_severe = self.inventory()[1]

    def batch_items(self) -> Iterable[BatchItem]:
        return [ BatchItem(scenario="historical", central_year=1971, input_dataset_filename="wind_present.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2035, input_dataset_filename="wind_rcp45_2021_2050.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2035, input_dataset_filename="wind_rcp85_2021_2050.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2085, input_dataset_filename="wind_rcp45_2071_2100.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2085, input_dataset_filename="wind_rcp85_2071_2100.nc"),
                 BatchItem(scenario="historical", central_year=1971, input_dataset_filename="exwind_present.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2035, input_dataset_filename="exwind_rcp45_2021_2050.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2035, input_dataset_filename="exwind_rcp85_2021_2050.nc"),
                 BatchItem(scenario="rcp4p5", central_year=2085, input_dataset_filename="exwind_rcp45_2071_2100.nc"),
                 BatchItem(scenario="rcp8p5", central_year=2085, input_dataset_filename="exwind_rcp85_2071_2100.nc")
                 ]

    def prepare(self, item: BatchItem, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            nc_url = self._url + item.input_dataset_filename
            download_file(nc_url, working_dir, "tudelft_conv_wind")
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

        if 'present' in item.input_dataset_filename:
            da_var = item.input_dataset_filename.split('_')[0] + '_his_mean'
        else:
            da_var = item.input_dataset_filename.split('_')[0] + '_fut_mean'

        if 'exwind' in item.input_dataset_filename:
            path_ = self._resource_extreme_severe.path.format(scenario=item.scenario, year=item.central_year)
        else:
            path_ = self._resource_severe.path.format(scenario=item.scenario, year=item.central_year)

        ds = xr.open_dataset(filename)
        da = ds[da_var]

        da.data = np.nan_to_num(da.data)
        da = da.rename({'rlon': 'lon','rlat': 'lat'})
        target.write(path_, da)

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        ...
        create_tiles_for_resource(source, target, self._resource_severe)
        create_tiles_for_resource(source, target, self._resource_extreme_severe)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Wind",
                indicator_id="wind_gust_speed",
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="convective_windstorm/conv_wind_tudelft/v2/severe_gust_speed_{scenario}_{year}",
                params={},
                display_name="Convective Windstorm (TUDelft)",
                description="""
                NetCDF files containing gridded annual probability of severe convective
                windstorms (wind gusts > 25 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
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
                        max_value=50.0,
                        units="metres"),
                    path="maps/convective_windstorm/conv_wind_tudelft/v2/severe_gust_speed_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="metres",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1971]),
                    Scenario(
                        id="rcp4p5",
                        years=[2035, 2085]),
                    Scenario(
                        id="rcp8p5",
                        years=[2035, 2085]),

                    ]),
            HazardResource(
                hazard_type="Wind",
                indicator_id="wind_gust_speed",
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="convective_windstorm/conv_wind_tudelft/v2/extremely_severe_gust_speed_{scenario}_{year}",
                params={},
                display_name="Convective Windstorm (TUDelft)",
                description="""
                NetCDF files containing gridded annual probability of extremely severe convective
                windstorms (wind gusts > 32 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
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
                        max_value=50.0,
                        units="metres"),
                    path="maps/convective_windstorm/conv_wind_tudelft/v2/extremely_severe_gust_speed_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="metres",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1971]),
                    Scenario(
                        id="rcp4p5",
                        years=[2035, 2085]),
                    Scenario(
                        id="rcp8p5",
                        years=[2035, 2085]),

                    ])]