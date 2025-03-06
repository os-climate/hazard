import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_and_unzip
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import affine_to_coords, global_crs_transform

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    scenario: str
    year: int
    key: str


class ETHZurichLitPop(IndicatorModel[BatchItem]):
    def __init__(
        self,
        source_dir: str,
        fs: Optional[AbstractFileSystem] = None,
    ):
        """
        Define every attribute of the onboarding class for the ETH Zurich LitPop data.

        METADATA:
        Link: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/409595/essd-12-817-2020.pdf
        Data type: historical
        Hazard indicator: LitPop
        Region: Global
        Resolution: 30 arc-seconds (or 1km roughly)
        Time range: 2014
        File type: File type: CSV

        DATA DESCRIPTION:
        CSV files containing the estimated physical asset values of 2014 at grid point in current USD of 224
        countries. The dataset is generated from national total physical asset values downscaled proportionally
        to the normalised product of nightlight intensity (Lit, based on NASA Earth at Night) and population count
        (Pop, based on Gridded Population of the World, Version 4.1).

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none,
            a LocalFileSystem is used.
        """
        self.fs = fs if fs else LocalFileSystem()
        self.zip_url = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/331316/LitPop_v1_2.tar"
        self.source_dir = os.path.join(
            source_dir, self.zip_url.split("/")[-1].split(".")[0]
        )
        if not (os.path.exists(self.source_dir)):
            self.prepare(source_dir)
        self.resources = self._resources()

    def batch_items(self) -> Iterable[BatchItem]:
        return [
            BatchItem(scenario="historical", year=2014, key=key)
            for key in self.resources
        ]

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3; download to specified working directory,
            # but then copy to self.source_dir
            assert working_dir is not None
            download_and_unzip(
                self.zip_url, working_dir, self.zip_url.split("/")[-1].split(".")[0]
            )
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            source = PurePosixPath(self.source_dir)
            download_and_unzip(self.zip_url, str(source.parent), source.parts[-1])

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
        assert item.key in self.resources
        assert target is None or isinstance(target, OscZarr)

        width, height = 120 * 360, 120 * 180
        _, transform = global_crs_transform(width, height)
        coords = affine_to_coords(transform, width, height, x_dim="lon", y_dim="lat")
        data = xr.DataArray(coords=coords, dims=list(coords.keys()))

        column = "region_id" if item.key == "litpop" else "value"

        for file in os.listdir(self.source_dir):
            logger.info(f"Loading {file}")
            df = pd.read_csv(
                os.path.join(self.source_dir, file),
                usecols=[column, "latitude", "longitude"],
            )
            image_coords = OscZarr._get_coordinates(
                df["longitude"], df["latitude"], transform
            )
            image_coords = np.floor(image_coords).astype(int)
            for i_lat, i_lon, value in zip(
                image_coords[1, :], image_coords[0, :], df[column]
            ):
                data[i_lat, i_lon] = value
            logger.info("Loading complete for {file}")

        path = self.resources[item.key].path.format(
            scenario=item.scenario, year=item.year
        )
        logger.info(f"Writing array to {path}")
        target.write(path, data)
        logger.info(f"Writing complete for {path}")

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        for key in self.resources:
            create_tiles_for_resource(source, target, self.resources[key])

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return self.resources.values()

    def _resources(self) -> Dict[str, HazardResource]:
        """Create resource."""
        resources: Dict[str, HazardResource] = dict()
        resource_map = {
            "country_code": {
                "units": "",
                "max_value": 999,
                "display_name": "ISO 3166-1 numeric code",
                "description": """
ISO 3166-1 numeric codes are three-digit country codes defined in ISO 3166-1, part of the ISO 3166 standard
published by the International Organization for Standardization (ISO), to represent countries, dependent
territories, and special areas of geographical interest.
""",
            },
            "litpop": {
                "units": "USD",
                "max_value": 500000000,
                "display_name": "LitPop (ETH Zurich)",
                "description": """
Asset exposure data being either unavailable or restricted to single countries or regions, ETH Zurich produced
a global high-resolution asset exposure dataset using “lit population” (LitPop), a globally consistent methodology
to disaggregate asset value data proportional to a combination of nightlight intensity (NASA) and geographical
population data (SEDAC). National total physical asset values are downscaled proportionally to the normalised
product of nightlight intensity (Lit, based on NASA Earth at Night) and population count (Pop, based on Gridded
Population  of the World, Version 4.1) using the LitPop-module of the probabilistic natural catastrophe damage
model CLIMADA. National total physical asset values are produced capital of 2014 from The World Bank's "Wealth
Accounting" available for 140 countries. For 84 countries, non-financial wealth is estimated from the country's
GDP in 2014 and the GDP-to-wealth ratios as estimated in the Credit Suisse Research Institute's "Global Wealth
Report 2017".
""",
            },
        }
        for key in resource_map:
            path = (
                "spatial_distribution/ethz/v1/{indicator_id}".format(indicator_id=key)
                + "_{scenario}_{year}"
            )
            resources[key] = HazardResource(
                hazard_type="SpatialDistribution",
                indicator_id=key,
                indicator_model_gcm="",
                indicator_model_id=None,
                path=path,
                params={},
                display_name=str(resource_map[key]["display_name"]),
                description=str(resource_map[key]["description"]),
                group_id="",
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
                        name="heating",
                        min_value=0.0,
                        max_value=resource_map[key]["max_value"],  # type:ignore
                        units=resource_map[key]["units"],  # type:ignore
                    ),
                    index_values=None,
                    path="maps/" + path + "_map",
                    source="map_array_pyramid",
                ),
                units=str(resource_map[key]["units"]),
                scenarios=[
                    Scenario(id="historical", years=[2014]),
                ],
            )
        return resources
