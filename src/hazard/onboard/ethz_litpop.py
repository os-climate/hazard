"""Module for handling the onboarding and processing of ETHZLitpop data."""

import logging
import os
from pathlib import PurePath
import tarfile
from typing_extensions import Dict, Iterable, Optional, override

import numpy as np
import xarray as xr
import dask
import dask.array as da
import dask.dataframe as dd
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import affine_to_coords, global_crs_transform

logger = logging.getLogger(__name__)


class ETHZurichLitPop(Onboarder):
    """On-board returns data set from ETHZurichLitPop."""

    def __init__(
        self,
        source_dir_base: str,
        fs: Optional[AbstractFileSystem] = None,
    ):
        """Define every attribute of the onboarding class for the ETH Zurich LitPop data.

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
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none,
            a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.dataset_filename = "LitPop_v1_2.tar"

        self.urls = {
            "LitPop_v1_2.tar": "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/331316/LitPop_v1_2.tar"
        }
        self.source_dir = os.path.join(source_dir_base, "ethz_litpop")

        self.resources = self._resources()

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)

        download_file(
            url=self.urls[self.dataset_filename],
            directory=download_dir,
            filename=self.dataset_filename,
            force_download=force_download,
        )

        with tarfile.open(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as tar:
            tar2source = {
                member.name: PurePath(
                    self.source_dir, os.path.basename(member.name)
                ).as_posix()
                for member in tar.getmembers()
                if member.isfile()
            }

            for tname, target_file in tar2source.items():
                with self.fs.open(target_file, mode="wb") as mifi:
                    extracted_file = tar.extractfile(tname)
                    if extracted_file:
                        mifi.write(extracted_file.read())

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

        # Filtrar los archivos CSV
        csv_files = [file for file in files if file.endswith(".csv")]

        # Verificar si hay exactamente 229 archivos CSV
        return len(csv_files) == 229

    def _get_items_to_process(self):
        """Get the list of items to process."""
        return [
            {
                "scenario": "historical",
                "year": 2014,
                "key": key,
            }
            for key in self.resources
        ]

    @override
    def onboard(self, target):
        """Process a single item and writes the data to the Zarr store."""
        items = self._get_items_to_process()
        for item in items:
            assert item.key in self.resources
            assert target is None or isinstance(target, OscZarr)

            width, height = 120 * 360, 120 * 180
            chunks = (1000, 1000)
            _, transform = global_crs_transform(width, height)
            coords = affine_to_coords(
                transform, width, height, x_dim="lon", y_dim="lat"
            )
            data = xr.DataArray(
                data=da.zeros((height, width), chunks=chunks),
                coords=coords,
                dims=list(coords.keys()),
            )

            column = "region_id" if item.key == "litpop" else "value"

            @dask.delayed
            def process_single_file(file_path, column=column, transform=transform):
                try:
                    logger.info(f"Loading {file_path}")

                    df = dd.read_csv(
                        file_path,
                        usecols=[column, "latitude", "longitude"],
                    )
                    image_coords = OscZarr._get_coordinates(
                        df["longitude"].compute(), df["latitude"].compute(), transform
                    )
                    image_coords = np.floor(image_coords).astype(int)

                    return image_coords, df[column].compute(), file_path
                except UnicodeDecodeError:
                    logger.warning(f"Error reading {file_path}")
                    return None, None, None
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    return None, None, None

            files = [
                os.path.join(self.source_dir, f) for f in os.listdir(self.source_dir)
            ]

            results = [process_single_file(file) for file in files]

            for image_coords, values, file in dask.compute(*results):
                if image_coords is None:
                    continue
                for i_lat, i_lon, value in zip(
                    image_coords[1, :], image_coords[0, :], values, strict=False
                ):
                    data[i_lat, i_lon] = value

                logger.info(f"Finished writing data to DataArray for {file}")

            path = self.resources[item.key].path.format(
                scenario=item.scenario, year=item.year
            )
            logger.info(f"Writing array to {path}")
            target.write(path, data)
            logger.info(f"Writing complete for {path}")

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
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
                resolution="----",
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
