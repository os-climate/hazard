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
        Resolution: 100 m
        Return periods: 10,20,50,100,200,500
        Time range: 1990-2016
        File type: Map (.tif)

        DATA DESCRIPTION:
        See jrc_river_flood.md for the full description, including citation and
        validation details.
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

    @override
    def prepare(self, download_dir: Optional[str] = None, force_download: bool = False):
        # if not os.path.exists(os.path.join(download_dir, self.source_files)):
        #     msg = f"{self.__class__.__name__} requires the file {self.source_files} to be in the download_dir.\nThe download_dir was {download_dir}."
        #     raise FileNotFoundError(msg)
        if download_dir is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a download_dir for the downloaded archives."
            )

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
    def is_prepared(self) -> bool:
        """Check if the data is prepared."""
        missing_files = {
            a
            for a in self.source_files
            if not self.fs.exists(PurePosixPath(self.source_dir, a))
        }
        return len(missing_files) == 0

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

                # if the raster has a 'band' dimension, select the first band
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
        create_tiles_for_resource(
            source, target, self._resource, nodata_as_zero_coarsening=True, max_zoom=10
        )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        with open(
            os.path.join(os.path.dirname(__file__), "jrc_river_flood.md"), "r"
        ) as f:
            description = f.read()
        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_gcm="historical",
                path="inundation/jrc/v1/flood_depth_{scenario}_{year}",
                params={},
                display_name="Flood depth (JRC)",
                description=description,
                resolution="100 m",
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
                        units="metres",
                    ),
                    path="maps/inundation/jrc/flood_depth_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="metres",
                store_netcdf_coords=False,
                scenarios=[
                    Scenario(id="historical", years=[1990]),
                ],
            )
        ]
