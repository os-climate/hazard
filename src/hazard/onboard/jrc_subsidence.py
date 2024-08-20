import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

import numpy as np
import rioxarray
import xarray as xr
from affine import Affine
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    scenario: str
    central_year: int
    input_dataset_filename: str


class JRCSubsidence(IndicatorModel[BatchItem]):
    def __init__(self, source_dir: str, fs: Optional[AbstractFileSystem] = None):
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        subsidence data.

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
        very high (clay > 60 %)
        high (35% < clay < 60%)
        medium (18% < clay < 35% and >= 15% sand, or 18% < clay and 15% < sand < 65%)
        low (18% < clay and > 65% sand)

        After downloading the data, the files STU_EU_T_SAND.rst/.RDC, STU_EU_T_CLAY.rst/.RDC must
        be placed in a directory to read it from the onboarding script. Another option is
        to onboard the raw file to a folder in the S3 bucket to read it from there in the
        onboarding process.
        Map tile creation is not working for CRS 3035.
        Install osgeo using conda if pip fails building wheels.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """

        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.dataset_filename_sand = "STU_EU_S_SAND.rst"  # Sub soil
        self.dataset_filename_clay = "STU_EU_S_CLAY.rst"  # Sub soil
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

    def batch_items(self) -> Iterable[BatchItem]:
        return [
            BatchItem(
                scenario="historical",
                central_year=1980,
                input_dataset_filename="susceptability_{scenario}_{year}",
            )
        ]

    def read_raw_data(self, filename: str) -> xr.DataArray:
        data = rioxarray.open_rasterio(filename)
        assert isinstance(data, xr.DataArray)

        return data

    def create_affine_transform_from_mapbounds_3035(self) -> Affine:
        """
        Create an affine transformation from map point and shape of bounds.

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
        """
        https://publications.jrc.ec.europa.eu/repository/handle/JRC114120

        assess the subsidence susceptibility for different classes:
        very high (clay > 60 %)
        high (35% < clay < 60%)
        medium (18% < clay < 35% and >= 15% sand, or 18% < clay and 15% < sand < 65%)
        low (18% < clay and > 65% sand).
        """

        # From matrix index to _etrs_laea using transform
        data_cat = np.zeros_like(data_clay)

        # Condition 1
        data_cat = xr.where((data_clay > 18) & (data_sand > 65), 1, data_cat)

        # Condition 2
        data_cat = xr.where(
            (data_clay >= 18) & (data_clay <= 35) & (data_sand <= 15), 2, data_cat
        )

        # Condition 3
        data_cat = xr.where((data_clay > 35) & (data_clay <= 60), 3, data_cat)

        # Condition 4
        data_cat = xr.where(data_clay > 60, 4, data_cat)

        return data_cat

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
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
        data_cat = self.create_categories(raw_data_sand, raw_data_clay)

        z = target.create_empty(
            self._resource.path.format(scenario=item.scenario, year=item.central_year),
            self.width,
            self.height,
            transform,
            self.crs,
        )
        z[0, :, :] = data_cat.data[0, :, :]  # type: ignore[index]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        ...
        create_tiles_for_resource(source, target, self._resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""

        return [
            HazardResource(
                hazard_type="Drought",
                indicator_id="subsidence_susceptability",
                indicator_model_id=None,
                indicator_model_gcm="historical",
                path="drought/subsidence_jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Subsidence Susceptability (JRC)",
                description="""
                A number of layers for soil properties have been created based on data from the European
                Soil Database in combination with data from the Harmonized World Soil Database (HWSD)
                and Soil-Terrain Database (SOTER). The available layers include: Total available water
                content, Depth available to roots, Clay content, Silt content, Sand content, Organic
                carbon, Bulk Density, Coarse fragments.
                """,
                group_id="",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=5.0,
                        units="index",
                    ),
                    path="maps/drought/subsidence_jrc/v1/susceptability_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="index",
                scenarios=[
                    Scenario(id="historical", years=[1980]),
                ],
            )
        ]
