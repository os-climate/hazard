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


class TUDelftFire(Onboarder):
    """On-board returns data set from TUDelft for fire hazard."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Pan-European data sets of forest fire probability of occurrence under present and future climate.

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
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = PurePath(source_dir_base, "tudelft_wildfire").as_posix() + "/"
        # Download source data
        self._url = "https://opendap.4tu.nl/thredds/fileServer/data2/uuid/a9f42f0a-1db4-4728-ad8c-d03a1d3f3c4d/"

        self._resource_fwi20 = list(self.inventory())[0]
        self._resource_fwi45 = list(self.inventory())[1]

    @override
    def prepare(self, download_dir=None):
        self.fs.makedirs(self.source_dir, exist_ok=True)
        items = self._get_items_to_process()
        for item in items:
            nc_url = self._url + item["input_dataset_filename"]
            download_file(
                url=nc_url,
                directory=self.source_dir,
                filename=os.path.basename(item["input_dataset_filename"]),
                force_download=False,
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
            """Process a single item and writes the data to the Zarr store."""
            input = os.path.join(
                self.source_dir, "fire_tudelft", item["input_dataset_filename"]
            )
            assert target is None or isinstance(target, OscZarr)
            filename = str(input)

            if "fwi20" in item["input_dataset_filename"]:
                path_ = self._resource_fwi20.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                )
            else:
                path_ = self._resource_fwi45.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                )

            ds = xr.open_dataset(filename)
            da = ds.risk
            da = da.rename({"time": "index"})
            da["index"] = [0]
            da.data = np.nan_to_num(da.data) / 100
            target.write(path_, da)

    def _get_items_to_process(self):
        """Get a list of all items to process as dictionaries."""
        return [
            {
                "scenario": "historical",
                "central_year": 1980,
                "input_dataset_filename": "multi-model_fwi20_historical_1971-2000.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "input_dataset_filename": "multi-model_fwi20_rcp45_2021-2050.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "input_dataset_filename": "multi-model_fwi20_rcp85_2021-2050.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "input_dataset_filename": "multi-model_fwi20_rcp45_2071-2100.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "input_dataset_filename": "multi-model_fwi20_rcp85_2071-2100.nc",
            },
            {
                "scenario": "historical",
                "central_year": 1980,
                "input_dataset_filename": "multi-model_fwi45_historical_1971-2000.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "input_dataset_filename": "multi-model_fwi45_rcp45_2021-2050.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "input_dataset_filename": "multi-model_fwi45_rcp85_2021-2050.nc",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "input_dataset_filename": "multi-model_fwi45_rcp45_2071-2100.nc",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "input_dataset_filename": "multi-model_fwi45_rcp85_2071-2100.nc",
            },
        ]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self._resource_fwi20)
        create_tiles_for_resource(source, target, self._resource_fwi45)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="daily_probability_fwi20",
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="fire/fire_tudelft/v1/fwi_greater_than_20_{scenario}_{year}",
                params={},
                display_name="Daily Probabilities of High Forest Fire (FWI exceeding 20) (TUDelft)",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in
                Europe under present and projected future climates. Includes gridded (NetCDF)
                datasets of high forest fire danger probabilities for the present climate
                (1981-2010) based on the ERA-Interim reanalysis and for the projected
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050
                and 2071-2100).
                """,
                version="",
                license="General terms of use for 4TU.Centre for Research Data",
                source="4TU Research Data:  https://data.4tu.nl/datasets/f9a134ad-fff9-44d5-ad4e-a0e9112b551e",
                group_id="fire_tudelft",
                display_groups=[],
                resolution="72200 m",
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
                    path="maps/fire/fire_tudelft/v1/fwi_greater_than_20_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="probability",
                scenarios=[
                    Scenario(id="historical", years=[1980]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            ),
            HazardResource(
                hazard_type="Fire",
                indicator_id="daily_probability_fwi45",
                indicator_model_gcm="CLMcom-CCLM4-8-17-CLM3-5",
                path="fire/fire_tudelft/v1/fwi_greater_than_45_{scenario}_{year}",
                params={},
                display_name="Daily Probabilities of High Forest Fire (FWI exceeding 45) (TUDelft)",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in
                Europe under present and projected future climates. Includes gridded (NetCDF)
                datasets of high forest fire danger probabilities for the present climate
                (1981-2010) based on the ERA-Interim reanalysis and for the projected
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050
                and 2071-2100).
                """,
                version="",
                license="General terms of use for 4TU.Centre for Research Data",
                source="4TU Research Data:  https://data.4tu.nl/datasets/f9a134ad-fff9-44d5-ad4e-a0e9112b551e",
                group_id="fire_tudelft",
                resolution="72200 m",
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
                    path="maps/fire/fire_tudelft/v1/fwi_greater_than_45_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="probability",
                scenarios=[
                    Scenario(id="historical", years=[1980]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            ),
        ]
