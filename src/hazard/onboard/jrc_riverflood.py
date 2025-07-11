"""Module for onboarding and processing Joint Research Center (JRC) subsidence data for OS-Climate."""

import logging
import os
from pathlib import PurePosixPath, PurePath
import shutil
import rasterio
from typing_extensions import Iterable, Optional, override

import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.download_utilities import download_and_unzip

logger = logging.getLogger(__name__)


class JRCRiverFlood(Onboarder):
    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the Joint Research Center (JRC)
        river flood data.

        METADATA:
        Link: https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81
        Data type: historical return period
        Hazard indicator: flood depth
        Region: Pan-Europe
        Resolution: 100m
        Return periods: 10,20,50,100,200,500
        Time range: 1990-2016
        File type: Map (.tif)

        DATA DESCRIPTION:
        The maps depict flood prone areas for river flood events for six different flood
        frequencies (from 1-in-10-years to 1-in-500-years). The extent comprises most
        of the geographical Europe and all the river basins entering the Mediterranean
        and Black Seas in the Caucasus, Middle East and Northern Africa countries.
        Cell values indicate water depth (in m).
        """
        self.fs = fs if fs else LocalFileSystem()

        self.source_dir = PurePath(source_dir_base, "jrc_riverflood").as_posix() + "/"

        # Download source data
        self.return_periods_str = ["010", "020", "050", "100", "200", "500"]
        self.return_periods = [int(rt) for rt in self.return_periods_str]
        self.zip_urls = [
            "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/EuropeanMaps/floodMap_RP{}.zip".format(
                rp
            )
            for rp in self.return_periods_str
        ]
        self.zip_filenames = [url.split("/")[-1] for url in self.zip_urls]

        # Create tif files names and paths
        self.source_files = [
            "floodmap_EFAS_RP{}_C.tif".format(rp) for rp in self.return_periods_str
        ]
        self._resource = list(self.inventory())[0]
        # self.tif_paths = [
        #     os.path.join(self.path_to_extact_zip[axi], self.tif_filenames[axi])
        #     for axi in range(len(self.return_periods))
        # ]

        # # zarr parameters
        # hazard_type = "inundation_river"
        # data_source_name = "jrc"
        # version = "v1"
        # dataset_name = "flood_depth_historical_1990"
        # self.group_path_array = os.path.join(
        #     hazard_type, data_source_name, version, dataset_name
        # )

        #

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        # if not os.path.exists(os.path.join(download_dir, self.source_files)):
        #     msg = f"{self.__class__.__name__} requires the file {self.source_files} to be in the download_dir.\nThe download_dir was {download_dir}."
        #     raise FileNotFoundError(msg)

        self.fs.makedirs(self.source_dir, exist_ok=True)

        extracted_folder = os.path.join(download_dir, "riverflood_temp")

        for url in self.zip_urls:
            archive_name = os.path.splitext(os.path.basename(url))[0]
            download_and_unzip(url, extracted_folder, archive_name)
            extracted_dir = os.path.join(extracted_folder, archive_name)

            for root, _, files in os.walk(extracted_dir):
                # Verify the files are not already in the destination folder
                for file in files:
                    if file.lower().endswith(".tif"):
                        src = os.path.join(root, file)
                        dst = os.path.join(self.source_dir, file)
                        shutil.move(src, dst)

        shutil.rmtree(extracted_folder)

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        missing_files = {
            a
            for a in self.source_files
            if not self.fs.exists(PurePosixPath(self.source_dir, a))
        }
        return len(missing_files) == 0 and not force

    @override
    def onboard(self, target):
        """Process multiple large .tif files with memory optimization"""
        items = self._get_items_to_process()
        chunk_size = 512  # Reduced chunk size

        for item in items:
            input_file = os.path.join(self.source_dir, item["input_dataset_filename"])

            # Initialize zarr array with first file
            first_path = str(input_file).format(return_period=self.return_periods[0])
            with rasterio.open(first_path) as src:
                transform = src.transform
                crs = src.crs
                width, height = src.width, src.height

            z_depth = target.create_empty(
                self._resource.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                ),
                width,
                height,
                transform,
                str(crs),
                index_values=self.return_periods,
            )

            # Process each return period
            for i, return_period in enumerate(self.return_periods):
                full_path = str(input_file).format(return_period=return_period)

                da_depth = xr.open_dataarray(
                    full_path,
                    engine="rasterio",
                    chunks={"x": chunk_size, "y": chunk_size},
                )

                # Si el raster tiene la dimensión 'band', seleccionamos la primera banda
                if "band" in da_depth.dims:
                    da_depth = da_depth.isel(band=0)

                nodata_value = da_depth.attrs.get("nodatavals", [None])[0]

                # Process in blocks
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        y_end = min(y + chunk_size, height)
                        x_end = min(x + chunk_size, width)

                        block = da_depth[y:y_end, x:x_end].values
                        if nodata_value is not None:
                            block[block == nodata_value] = np.nan

                        z_depth[i, y:y_end, x:x_end] = block
                        del block

                # Cerrar explícitamente el DataArray
                da_depth.close()

                logger.info(f"Processed {return_period}")

    def _get_items_to_process(self):
        """Get the list of items to process."""
        return [
            {
                "scenario": "historical",
                "central_year": 1990,
                "input_dataset_filename": "floodmap_EFAS_RP{return_period:03d}_C.tif",
            },
        ]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(
            source,
            target,
            self._resource,
            max_zoom=4,
        )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_gcm="historical",
                path="inundation/jrc/v1/flood_depth_{scenario}_{year}",
                params={},
                display_name="River Flood Depth",
                description="""
                The maps depict flood prone areas for river flood events for six different flood
                frequencies (from 1-in-10-years to 1-in-500-years). The extent comprises most
                of the geographical Europe and all the river basins entering the Mediterranean
                and Black Seas in the Caucasus, Middle East and Northern Africa countries.
                Cell values indicate water depth (in m).
                """,
                resolution=" 9800 m",
                group_id="river_jrc",
                display_groups=[],
                map=MapInfo(
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=5.0,
                        units="meters",
                    ),
                    path="maps/inundation/jrc/flood_depth_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="none",
                store_netcdf_coords=False,
                scenarios=[
                    Scenario(id="historical", years=[1990]),
                ],
            )
        ]
