"""Module for onboarding and processing Joint Research Center (JRC) subsidence data for OS-Climate."""

import logging
import os
from pathlib import PurePosixPath, PurePath
import shutil
from typing_extensions import Any, Iterable, Optional, override
import zipfile

import numpy as np
import xarray as xr
from affine import Affine
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class JRCSubsidence(Onboarder):
    """Onboards and processes Joint Research Center (JRC) subsidence data.

    This class handles reading subsidence susceptibility data from the JRC dataset, processes the raw
    data, categorizes it based on clay and sand content, and writes the results to a Zarr store. It
    also creates map tiles for visualizing the subsidence data.
    """

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the Joint Research Center (JRC) subsidence data.

        The data must be requested submitting a form in the next link:
        https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data

        Then, an email with instructions to downloading the data will be recieved.
        The data will be provided in Idrisi Raster Format (.rst) file type.

        METADATA:
        Link: https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data
        Hazard subtype: from Drought
        Data type: historical susceptability score
        Hazard indicator: Susceptability category of subsidence
        Region: Europe
        Resolution: 1km
        Time range: 1980
        File type: Restructured Text (.rst)

        DATA DESCRIPTION:
        A number of layers for soil properties have been created based on data from the European
        Soil Database in combination with data from the Harmonized World Soil Database (HWSD)
        and Soil-Terrain Database (SOTER). The available layers include: Total available water
        content, Depth available to roots, Clay content, Silt content, Sand content, Organic
        carbon, Bulk Density, Coarse fragments.

        IMPORTANT NOTES:
        To build the hazard indicator in the form of susceptability categories the next reference was used:
        https://publications.jrc.ec.europa.eu/repository/handle/JRC114120 (page 32)

        The categories depend on the percentage of soil and sand. The next categories are used:
        high (35% < clay < 60% and clay > 60 %)
        medium (35% > clay and sand < 15%)
        low (18% < clay < 35% and sand >= 15% , or 18% < clay and 15% < sand < 65%)
        norisk (18% < clay and > 65% sand).

        After downloading the data, the files STU_EU_T_SAND.rst/.RDC, STU_EU_T_CLAY.rst/.RDC must
        be placed in a directory to read it from the onboarding script. Another option is
        to onboard the raw file to a folder in the S3 bucket to read it from there in the
        onboarding process.
        Map tile creation is not working for CRS 3035.
        Install osgeo using conda if pip fails building wheels.

        Args:
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()

        self.source_dir = PurePath(source_dir_base, "jrc_subsidence").as_posix() + "/"

        self.dataset_filename = "STU_EU_Layers.zip"

        self.dataset_filename_sand = "STU_EU_S_SAND.rst"
        self.dataset_filename_clay = "STU_EU_S_CLAY.rst"
        # The RDC files are needed to properly read the rst files.
        self.rdfiles = ["STU_EU_S_SAND.RDC", "STU_EU_S_CLAY.RDC"]
        self.source_files = {
            self.dataset_filename_sand,
            self.dataset_filename_clay,
            *self.rdfiles,
        }
        self._resource = list(self.inventory())[0]

        # etrs laea coordinate system. BOUNDS
        self.min_xs = 1500000
        self.max_xs = 7400000
        self.min_ys = 900000
        self.max_ys = 5500000

        # Map bounds and crs
        self.width = 5900
        self.height = 4600
        self.crs = "3035"

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        if not os.path.exists(os.path.join(download_dir, self.dataset_filename)):
            msg = f"{self.__class__.__name__} requires the file {self.dataset_filename} to be in the download_dir.\nThe download_dir was {download_dir}."
            raise FileNotFoundError(msg)

        self.fs.makedirs(self.source_dir, exist_ok=True)

        extracted_folder = os.path.join(download_dir, "subsidence__temp")

        with zipfile.ZipFile(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as z:
            z.extractall(extracted_folder)

        for _, _, files in os.walk(extracted_folder):
            # Verify the files are not already in the destination folder
            for file_name in files:
                if file_name in self.source_files:
                    self.fs.copy(
                        os.path.join(extracted_folder, file_name),
                        self.source_dir,
                    )

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

    def read_raw_data(self, filename: str) -> xr.DataArray:
        """Read raw raster data from the specified filename.

        Args:
            filename (str): Path to the raster file.

        Returns:
            xr.DataArray: DataArray containing the raster data.

        """
        data = xr.open_dataarray(filename, engine="rasterio")
        assert isinstance(data, xr.DataArray)

        return data

    @override
    def onboard(self, target):
        items_to_process = self._get_items_to_process()
        for item in items_to_process:
            input_sand = PurePosixPath(self.source_dir, self.dataset_filename_sand)
            input_clay = PurePosixPath(self.source_dir, self.dataset_filename_clay)
            assert target is None or isinstance(target, OscZarr)
            filename_sand = str(input_sand)
            filename_clay = str(input_clay)

            # Read raw data
            raw_data_sand = self.read_raw_data(filename_sand)
            raw_data_clay = self.read_raw_data(filename_clay)

            # Compute transform
            transform = self.create_affine_transform_from_mapbounds_3035()

            # Create categories
            data_cat = self.create_categories(
                data_clay=raw_data_clay, data_sand=raw_data_sand
            )

            z = target.create_empty(
                self._resource.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                ),
                self.width,
                self.height,
                transform,
                self.crs,
            )
            z[0, :, :] = data_cat[0]  # type: ignore[index]

    def create_affine_transform_from_mapbounds_3035(self) -> Affine:
        """Create an affine transformation from map point and shape of bounds.

        Maybe add to map utilities
        """
        # Create Affine transformation
        bounds = (self.min_xs, self.min_ys, self.max_xs, self.max_ys)

        # Compute the parameters of the georeference
        a = (bounds[2] - bounds[0]) / self.width
        b = 0
        c = 0
        d = -(bounds[3] - bounds[1]) / self.height
        e = bounds[0]
        f = bounds[3]

        transform = Affine(a, b, e, c, d, f)
        return transform

    def create_categories(
        self, data_clay: xr.DataArray, data_sand: xr.DataArray
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """https://publications.jrc.ec.europa.eu/repository/handle/JRC114120.

        assess the subsidence susceptibility for different classes:
        No risk: Coarse soil texture (clay < 18% and sand > 65%)
        Low risk: Medium (18% < clay < 35% and sand >= 15%, or clay > 18% and 15% < sand < 65%)
        Medium risk: Medium fine (clay > 35% and sand < 15%)
        High risk: Fine (35% < clay < 60%) and Very fine (clay > 60%).
        """
        # Initialize category array with zeros
        mask = (data_clay + data_sand) > 100
        data_cat = xr.zeros_like(~mask)

        # 1) No risk
        data_cat = xr.where((data_clay < 18) & (data_sand > 65) & ~mask, 1, data_cat)

        # 2) Low risk
        data_cat = xr.where(
            ((data_clay > 18) & (data_clay < 35) & (data_sand >= 15) & ~mask)
            | ((data_clay > 18) & (data_sand > 15) & (data_sand < 65) & ~mask),
            2,
            data_cat,
        )

        # 3) Medium risk
        data_cat = xr.where((data_clay < 35) & (data_sand < 15) & ~mask, 3, data_cat)

        # 4) High ris
        # Only mark "4" if clay > 35 *and* sand >= 15, so we don't overwrite category "3".
        data_cat = xr.where((data_clay > 35) & ~mask, 4, data_cat)
        return data_cat.values

    def _get_items_to_process(self):
        """Get the list of items to process."""
        return [
            {
                "scenario": "historical",
                "central_year": 1980,
                "input_dataset_filename": "susceptability_{scenario}_{year}",
            },
        ]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self._resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Subsidence",
                indicator_id="subsidence_susceptability",
                indicator_model_id=None,
                indicator_model_gcm="historical",
                path="subsidence/subsidence_jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Subsidence Susceptability (JRC)",
                resolution="1000 m",
                description="""
                A number of layers for soil properties have been created based on data from the European
                Soil Database in combination with data from the Harmonized World Soil Database (HWSD)
                and Soil-Terrain Database (SOTER). The available layers include: Total available water
                content, Depth available to roots, Clay content, Silt content, Sand content, Organic
                carbon, Bulk Density, Coarse fragments.
                """,
                version="-",
                attribution="European Commission, Joint Research Centre (2016): Ground deformation mapping and monitoring by satellite based multi-temporal DInSAR technique (2016-10-17). [Dataset]. PID: http://data.europa.eu/89h/f539651c-1c40-4362-8e9a-fe360923dbd3",
                source="The data must be requested submitting a form in the next link: https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data",
                license="“European Commission Re-use and Copyright Notice” (link on the page). The notice says “Reuse is authorised, provided the source is acknowledged” and points to Commission Decision 2011/833/EU, which makes re-use (commercial or not) the default.",
                group_id="subsidence_jrc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=4.0,
                        units="index",
                    ),
                    path="maps/subsidence/subsidence_jrc/v1/susceptability_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="index",
                scenarios=[
                    Scenario(id="historical", years=[1980]),
                ],
            )
        ]
