import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

import numpy as np
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


class JRCLandslides(IndicatorModel[BatchItem]):
    def __init__(self, source_dir: str, fs: Optional[AbstractFileSystem] = None):
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        landslide data.

        The data must be requested submitting a form in the next link:
        https://esdac.jrc.ec.europa.eu/content/european-landslide-susceptibility-map-elsus-v2

        Then, an email with instructions to downloading the data will be recieved.
        The data will be provided in Esri ASCII Grid (.asc) file type.

        METADATA:
        Link: https://data.jrc.ec.europa.eu/dataset/jrc-esdac-78
        Data type: historical susceptability score
        Hazard indicator: Susceptability Score of landslide
        Region: Europe
        Resolution: 200m
        Time range: 1980
        File type: Esri ASCII Grid (.asc)

        DATA DESCRIPTION:
        The spatial dataset (GIS map) shows landslide susceptibility levels at European scale,
        derived from heuristic-statistical modelling of main landslide conditioning factors
        using also landslide location data. It covers all EU member states except Malta, in
        addition to Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR Macedonia, Iceland,
        Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.

        IMPORTANT NOTES:
        After downloading the data, the file elsus_v2.asc must be placed in a directory to read it from
        the onboarding script. Another option is to onboard the raw file to a folder in the S3 bucket
        to read it from there in the onboarding process.
        Map tile creation is not working for CRS 3035.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """

        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.dataset_filename = "elsus_v2.asc"
        self._resource = list(self.inventory())[0]

        # Affine matrix metadata
        self.cell_size = 200
        self.XLLCORNER = 2636073.6872550002
        self.YLLCORNER = 1385914.3968890002

        # Map bounds and crs
        self.width = 19453
        self.height = 20151
        self.crs = "3035"

    def batch_items(self) -> Iterable[BatchItem]:
        return [
            BatchItem(
                scenario="historical",
                central_year=1980,
                input_dataset_filename="susceptability_{scenario}_{year}",
            )
        ]

    def read_raw_data(self, filename: str) -> np.ndarray:
        data = np.loadtxt(filename, skiprows=6)
        return data

    def create_affine_transform_from_mapbounds_3035(self) -> Affine:
        """
        Create an affine transformation from map point and shape of bounds.
        MOVE THIS METHOD TO MAP UTILITIES
        """

        # Create Affine transformation
        min_xs = self.XLLCORNER
        max_xs = min_xs + self.cell_size * self.width
        min_ys = self.YLLCORNER
        max_ys = min_ys + self.cell_size * self.height

        bounds = (min_xs, min_ys, max_xs, max_ys)

        # Compute the parameters of the georeference
        a = (bounds[2] - bounds[0]) / self.width
        b = 0
        c = 0
        d = -(bounds[3] - bounds[1]) / self.height
        e = bounds[0]
        f = bounds[3]

        transform = Affine(a, b, e, c, d, f)
        return transform

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
        input = PurePosixPath(self.source_dir, self.dataset_filename)
        assert target is None or isinstance(target, OscZarr)
        filename = str(input)

        # Read raw data
        raw_data = self.read_raw_data(filename)

        transform = self.create_affine_transform_from_mapbounds_3035()

        z = target.create_empty(
            self._resource.path.format(scenario=item.scenario, year=item.central_year),
            self.width,
            self.height,
            transform,
            self.crs,
        )
        z[0, :, :] = raw_data[:, :]

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
                indicator_id="landslide_susceptability",
                indicator_model_id=None,
                indicator_model_gcm="historical",
                path="drought/landslide_jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Landslide Susceptability (JRC)",
                description="""
                The spatial dataset (GIS map) shows landslide susceptibility levels at European scale,
                derived from heuristic-statistical modelling of main landslide conditioning factors
                using also landslide location data. It covers all EU member states except Malta, in
                addition to Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR Macedonia, Iceland,
                Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.
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
                    path="maps/drought/landslide_jrc/v1/susceptability_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="index",
                scenarios=[
                    Scenario(id="historical", years=[1980]),
                ],
            )
        ]
