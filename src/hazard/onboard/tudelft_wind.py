"""Module for handling the onboarding and processing of TUDelft fire data."""

import logging
import os
from pathlib import PurePath
from typing_extensions import Iterable, Optional, override

import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem


from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class TUDelftConvectiveWindstorm(Onboarder):
    """On-board returns data set from TUDelft for wind hazard."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Pan-European data sets of severe and extreme wind probability of occurrence under present and future climate.

        METADATA:
        Link: https://data.4tu.nl/datasets/93463344-a63d-4ab2-a16a-f69b989b0e13
        Data type: historical and scenario probability
        Hazard indicator: Probability of wind gusts > 25 m/s and 32 m/s
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
        Tile maps do not work because very low resolution data.
        Coordinate system is rotated longitud and latitude. Needs to be transformed.

        Args:
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = PurePath(source_dir_base, "tudelft_wind").as_posix() + "/"

        # Download source data
        self._url = "https://opendap.4tu.nl/thredds/fileServer/data2/uuid/9be95957-ca8e-4eb9-aa00-074c55df9032/"

        self._resource_severe = list(self.inventory())[0]
        self._resource_extreme_severe = list(self.inventory())[1]

    @override
    def prepare(
        self,
        download_dir=None,
    ):
        self.fs.makedirs(self.source_dir, exist_ok=True)

        items = self._get_items_to_process()
        for item in items:
            nc_url = self._url + item["input_dataset_filename"]
            download_file(
                url=nc_url,
                directory=self.source_dir,
                filename=os.path.basename(item["input_dataset_filename"]),
            )

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if not os.path.exists(self.source_dir) or force or force_download:
            return False
        # Listar todos los archivos en el directorio
        try:
            files = os.listdir(self.source_dir)
        except FileNotFoundError:
            return False

        nc_files = [file for file in files if file.endswith(".nc")]

        # verificar que están los 10 .nc files
        return len(nc_files) == 10

    @override
    def onboard(self, target):
        items = self._get_items_to_process()
        for item in items:
            input = os.path.join(self.source_dir, item["input_dataset_filename"])
            assert target is None or isinstance(target, OscZarr)
            filename = str(input)

            if "present" in item["input_dataset_filename"]:
                da_var = item["input_dataset_filename"].split("_")[0] + "_his_mean"
            else:
                da_var = item["input_dataset_filename"].split("_")[0] + "_fut_mean"

            if "exwind" in item["input_dataset_filename"]:
                path_ = self._resource_extreme_severe.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                )
            else:
                path_ = self._resource_severe.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                )

            ds = xr.open_dataset(filename)
            da = ds[da_var]

            da.data = np.nan_to_num(da.data) / 100
            da = da.rename({"rlon": "lon", "rlat": "lat"})
            target.write(path_, da)

    def _get_items_to_process(self):
        """Get a list of all items to process as dictionaries."""
        return [
            {
                "scenario": "historical",
                "central_year": 1971,
                "input_dataset_filename": "wind_present.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "input_dataset_filename": "wind_rcp45_2021_2050.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "input_dataset_filename": "wind_rcp85_2021_2050.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "input_dataset_filename": "wind_rcp45_2071_2100.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "input_dataset_filename": "wind_rcp85_2071_2100.nc",
            },
            {
                "scenario": "historical",
                "central_year": 1971,
                "input_dataset_filename": "exwind_present.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "input_dataset_filename": "exwind_rcp45_2021_2050.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "input_dataset_filename": "exwind_rcp85_2021_2050.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "input_dataset_filename": "exwind_rcp45_2071_2100.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "input_dataset_filename": "exwind_rcp85_2071_2100.nc",
            },
        ]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(
            source,
            target,
            self._resource_severe,
            max_zoom=10,
            nodata_as_zero_coarsening=True,
        )
        create_tiles_for_resource(
            source,
            target,
            self._resource_extreme_severe,
            max_zoom=10,
            nodata_as_zero_coarsening=True,
        )
        # create_tiles_for_resource(source, target, self._resource_severe)
        # create_tiles_for_resource(source, target, self._resource_extreme_severe)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="ChronicWind",
                indicator_id="severe_windstorm_probability",
                indicator_model_id=None,
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="convective_windstorm/conv_wind_tudelft/v2/severe_gust_speed_{scenario}_{year}",
                params={},
                display_name="Annual Probability of Severe Convective Windstorm (TUDelft)",
                resolution="47100 m",
                description="""
                NetCDF files containing gridded annual probability of severe convective
                windstorms (wind gusts > 25 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
                """,
                version="",
                license="General terms of use for 4TU.Centre for Research Data",
                source="4TU Research Data: https://data.4tu.nl/datasets/93463344-a63d-4ab2-a16a-f69b989b0e13",
                group_id="wind_tudelft",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=1.0,
                        units="prob",
                    ),
                    path="maps/convective_windstorm/conv_wind_tudelft/v2/severe_gust_speed_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="probability",
                scenarios=[
                    Scenario(id="historical", years=[1971]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            ),
            HazardResource(
                hazard_type="ChronicWind",
                indicator_id="extreme_windstorm_probability",
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="convective_windstorm/conv_wind_tudelft/v2/extremely_severe_gust_speed_{scenario}_{year}",
                params={},
                display_name="Annual Probability of Extremely Severe Convective Windstorm (TUDelft)",
                resolution="47100 m",
                description="""
                NetCDF files containing gridded annual probability of extremely severe convective
                windstorms (wind gusts > 32 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
                """,
                group_id="wind_tudelft",
                version="",
                license="General terms of use for 4TU.Centre for Research Data",
                source="4TU Research Data: https://data.4tu.nl/datasets/93463344-a63d-4ab2-a16a-f69b989b0e13",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=1.0,
                        units="prob",
                    ),
                    path="maps/convective_windstorm/conv_wind_tudelft/v2/extremely_severe_gust_speed_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="probability",
                scenarios=[
                    Scenario(id="historical", years=[1971]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            ),
        ]
