"""Module for onboarding and processing Joint Research Center (JRC) landslide data for OS-Climate."""

import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath, PurePath
from typing_extensions import Any, Iterable, Optional, override
import zipfile

import xarray as xr
import numpy as np
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
    """Represents a batch item for processing landslide data, including scenario, central year, and input dataset filename."""

    scenario: str
    central_year: int
    input_dataset_filename: str


class JRCLandslides(IndicatorModel[BatchItem]):
    """Onboards and processes Joint Research Center (JRC) landslide data.

    This class is responsible for reading landslide susceptibility data from the JRC dataset, processing the raw
    data, and saving the results to a Zarr store. Additionally, it generates map tiles for visualizing landslide
    susceptibility levels.

    To access the data, users must submit a request form at the following link:
    https://data.jrc.ec.europa.eu/dataset/jrc-esdac-78
    After submission, instructions for downloading the data will be sent via email.
    The data is provided in a .zip file containing GeoTIFF (.tif) files for various landslide return periods.

    ---

    **METADATA**:
    - **Link**: https://data.jrc.ec.europa.eu/dataset/jrc-esdac-78
    - **Data Type**: Historical susceptibility scores
    - **Hazard Indicator**: Landslide Susceptibility Score
    - **Region**: Europe
    - **Resolution**: 200m
    - **Time Range**: 1980
    - **File Format**: GeoTIFF (.tif)

    ---

    **DATA DESCRIPTION**:
    The spatial dataset (GIS map) represents landslide susceptibility levels at a European scale, derived using a
    heuristic-statistical model based on key landslide conditioning factors and landslide location data. The dataset
    covers all EU member states except Malta, as well as Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR
    Macedonia, Iceland, Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.

    ---

    **IMPORTANT NOTES**:
    1. After downloading, the data includes a separate .tif file for each available return period: [2, 5, 10, 20, 50, 100, 200, 500].
    2. Map tile generation is currently incompatible with CRS 3035.

    ---

    **FUTURE EXTENSIONS**:
    This class is built to accommodate multiple scenarios beyond historical data. Additional datasets can be integrated
    by defining new `Scenario` instances in the `inventory` method and providing the corresponding data files.
    """

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the Joint Research Center (JRC) landslide data.

        Args:
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        # Ensure it ends in a / so that we can copy to it as a folder.
        self.source_dir = PurePath(source_dir_base, "jrc_landslides").as_posix() + "/"
        # Zip file containing .tif files for different return periods, assume it is in the download folder
        self.dataset_filename = "LS_hazard.zip"

        self.return_periods = [2, 5, 10, 20, 50, 100, 200, 500]
        self.source_files = {f"LS_RP{a:03d}.tif" for a in self.return_periods}
        self._resource = list(self.inventory())[0]

    def batch_items(self) -> Iterable[BatchItem]:
        """Return a list of batch items for processing landslide data."""
        return [
            BatchItem(
                scenario="historical",
                central_year=2018,
                input_dataset_filename="LS_RP{return_period:03d}.tif",
            )
        ]

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        missing_files = {
            a
            for a in self.source_files
            if not self.fs.exists(PurePosixPath(self.source_dir, a))
        }
        if len(missing_files) == 0 and not force:
            return

        else:  # if len(missing_files) > 0 or force:
            if not os.path.exists(os.path.join(download_dir, self.dataset_filename)):
                msg = f"{self.__class__.__name__} requires the file {self.dataset_filename} to be in the download_dir.\nThe download_dir was {download_dir}."
                raise FileNotFoundError(msg)

        self.fs.makedirs(self.source_dir, exist_ok=True)

        with zipfile.ZipFile(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as z:
            zip2source = {
                name: PurePath(self.source_dir, os.path.basename(name)).as_posix()
                for name in z.namelist()
                if os.path.basename(name) in self.source_files
            }
            for zname, target_file in zip2source.items():
                with self.fs.open(target_file, mode="wb") as mifi:
                    mifi.write(z.read(name=zname))

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

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
        """Onboard a single batch item by processing and creating maps."""
        input_file = os.path.join(self.source_dir, item.input_dataset_filename)
        assert target is None or isinstance(target, OscZarr)
        coords_x, coords_y = None, None

        # Loop through each return period
        for i, return_period in enumerate(self.return_periods):
            # Format the file path using the return period
            full_path = str(input_file).format(return_period=return_period)

            logger.info(
                f"Reading .tif file {full_path} for return period {return_period}"
            )

            with self.fs.open(full_path, "rb") as fd:
                da_depth = xr.open_rasterio(fd).isel(  # type: ignore[attr-defined]
                    band=0
                )  # We assume we want the first band

                if i == 0:
                    # Store the coordinates for alignment checking
                    coords_x, coords_y = np.array(da_depth.x), np.array(da_depth.y)

                    # Create an empty target array in OscZarr for storing the data
                    z_depth = target.create_empty(
                        self._resource.path.format(
                            scenario=item.scenario, year=item.central_year
                        ),
                        da_depth.shape[1],  # width
                        da_depth.shape[0],  # height
                        da_depth.rio.transform(),  # Use the transform from the .tif
                        str(da_depth.rio.crs),  # Use CRS from the .tif metadata
                        index_values=self.return_periods,  # Create one slice per return period
                    )

                # Check that the coordinates match between all return periods
                lenx, leny = (
                    min(len(da_depth.x), len(coords_x)),  # type: ignore[arg-type]
                    min(len(da_depth.y), len(coords_y)),  # type: ignore[arg-type]
                )
                assert (
                    np.abs(np.array(da_depth.x[0:lenx]) - coords_x[0:lenx]).max() < 1e-4  # type: ignore[index]
                )
                assert (
                    np.abs(np.array(da_depth.y[0:leny]) - coords_y[0:leny]).max() < 1e-4  # type: ignore[index]
                )

                # Handle nodata values
                values_depth = np.array(da_depth.data, dtype="float32")
                nodata_value = da_depth.attrs.get("nodatavals", [None])[0]
                if nodata_value is not None:
                    values_depth[values_depth == nodata_value] = np.nan

                # Insert the values for this return period into the target array
                z_depth[i, 0 : len(da_depth.y), 0 : len(da_depth.x)] = values_depth[
                    :, :
                ]

                logger.info(
                    f"Successfully processed {full_path} for return period {return_period}"
                )

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(
            source,
            target,
            self._resource,
        )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Landslide",
                indicator_id="landslide_susceptability",
                indicator_model_id=None,
                indicator_model_gcm="historical",
                path="landslide/landslide_jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Landslide Susceptability (JRC)",
                description="""
                The spatial dataset (GIS map) shows landslide susceptibility levels at European scale,
                derived from heuristic-statistical modelling of main landslide conditioning factors
                using also landslide location data. It covers all EU member states except Malta, in
                addition to Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR Macedonia, Iceland,
                Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.
                """,
                group_id="landslide_jrc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="Oranges",
                        min_value=0.0,
                        max_value=5.0,
                        units="index",
                    ),
                    path="maps/landslide/landslide_jrc/v1/susceptability_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="index",
                scenarios=[
                    Scenario(id="historical", years=[2018]),
                ],
            )
        ]
