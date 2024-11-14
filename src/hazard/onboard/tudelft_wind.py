"""Module for handling the onboarding and processing of TUDelft fire data."""

import logging
import os
from dataclasses import dataclass
from pathlib import PurePath
from typing_extensions import Any, Iterable, Optional, override

import numpy as np
import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represent a batch item for hazard processing.

    It includes scenario, central_year and input_dataset_filename.

    """

    scenario: str
    central_year: int
    input_dataset_filename: str


class TUDelftConvectiveWindstorm(IndicatorModel[BatchItem]):
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

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        return [
            BatchItem(
                scenario="historical",
                central_year=1971,
                input_dataset_filename="wind_present.nc",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2035,
                input_dataset_filename="wind_rcp45_2021_2050.nc",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2035,
                input_dataset_filename="wind_rcp85_2021_2050.nc",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2085,
                input_dataset_filename="wind_rcp45_2071_2100.nc",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2085,
                input_dataset_filename="wind_rcp85_2071_2100.nc",
            ),
            BatchItem(
                scenario="historical",
                central_year=1971,
                input_dataset_filename="exwind_present.nc",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2035,
                input_dataset_filename="exwind_rcp45_2021_2050.nc",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2035,
                input_dataset_filename="exwind_rcp85_2021_2050.nc",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2085,
                input_dataset_filename="exwind_rcp45_2071_2100.nc",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2085,
                input_dataset_filename="exwind_rcp85_2071_2100.nc",
            ),
        ]

    def onboard_single(
        self, target, download_dir=None, force_prepare=False, force_download=False
    ):
        """Onboard a single batch of hazard data into the system.

        Args:
            target: Target system for writing the processed data.
            download_dir (str): Directory where downloaded files will be stored.
            force_prepare(bool): Flag to force data preparation. Default is False
            force_download(bool):Flag to force re-download of data. Default is False

        """
        self.prepare(
            force=force_prepare,
            download_dir=download_dir,
            force_download=force_download,
        )
        self.run_all(source=None, target=target, client=None, debug_mode=False)
        self.create_maps(target, target)

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)
        if (
            not self.fs.exists(self.source_dir)
            or len(self.fs.listdir(self.source_dir)) == 0
            or force_download
        ):
            for batch_item in self.batch_items():
                nc_url = self._url + batch_item.input_dataset_filename
                download_file(
                    url=nc_url,
                    directory=self.source_dir,
                    filename=os.path.basename(batch_item.input_dataset_filename),
                    force_download=force_download,
                )

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
        """Process a single batch item and writes the data to the Zarr store."""
        input = os.path.join(
            self.source_dir, "conv_wind_tudelft", item.input_dataset_filename
        )
        assert target is None or isinstance(target, OscZarr)
        filename = str(input)

        if "present" in item.input_dataset_filename:
            da_var = item.input_dataset_filename.split("_")[0] + "_his_mean"
        else:
            da_var = item.input_dataset_filename.split("_")[0] + "_fut_mean"

        if "exwind" in item.input_dataset_filename:
            path_ = self._resource_extreme_severe.path.format(
                scenario=item.scenario, year=item.central_year
            )
        else:
            path_ = self._resource_severe.path.format(
                scenario=item.scenario, year=item.central_year
            )

        ds = xr.open_dataset(filename)
        da = ds[da_var]

        da.data = np.nan_to_num(da.data) / 100
        da = da.rename({"rlon": "lon", "rlat": "lat"})
        target.write(path_, da)

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self._resource_severe)
        create_tiles_for_resource(source, target, self._resource_extreme_severe)

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
                description="""
                NetCDF files containing gridded annual probability of severe convective
                windstorms (wind gusts > 25 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
                """,
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
                description="""
                NetCDF files containing gridded annual probability of extremely severe convective
                windstorms (wind gusts > 32 m/s) for present day and the future climate.
                The fields are multi model means of 15 regional climate model simulations (CORDEX).
                """,
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
