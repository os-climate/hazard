"""Module for onboarding and processing STORM wind data provided by Jupiter Intelligence."""

import os
from contextlib import ExitStack
from pathlib import PurePath
from typing_extensions import Dict, Iterable, Optional, override
import zipfile

from fsspec.implementations.local import LocalFileSystem
from fsspec import AbstractFileSystem
import numpy as np  # type: ignore
import xarray as xr
from affine import Affine  # type: ignore

from hazard.onboarder import Onboarder
from hazard.inventory import HazardResource
from hazard.utilities import xarray_utilities
from hazard.utilities.download_utilities import download_file


class STORMIndicator(Onboarder):
    """On-board data set provided by Jupiter Intelligence for use by OS-Climate to set up a OS-C ClimateScore API Service (“ClimateScore Service”)."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Source to load STORM wind data set.

        https://data.4tu.nl/articles/dataset/STORM_climate_change_tropical_cyclone_wind_speed_return_periods }.

        https://data.4tu.nl/authors/8a084c6a-3315-4ba7-9768-dd1ba1825dbc

        https://data.4tu.nl/articles/dataset/STORM_climate_change_tropical_cyclone_wind_speed_return_periods

        https://data.4tu.nl/articles/_/12706085/2

        Args:
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is  used.
            source_dir_base (str): Directory for storing downloaded files.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = PurePath(source_dir_base, "storm_wind").as_posix() + "/"
        self._basin_ids = {
            "EP": "Eastern Pacific",
            "NA": "North Atlantic",
            "NI": "North Indian",
            "SI": "South Indian",
            "SP": "South Pacific",
            "WP": "Western Pacific",
        }
        self._return_periods = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ]
        self.dataset_filename = (
            "STORM_FIXED_RETURN_PERIODS_HADGEM3-GC31-HM_TIF_FILES.zip"
        )

        self.urls = {
            "STORM_FIXED_RETURN_PERIODS_HADGEM3-GC31-HM_TIF_FILES.zip": "https://data.4tu.nl/file/504c838e-2bd8-4d61-85a1-d495bdc560c3/856f9530-56d7-489e-8005-18ae36db4804"
        }
        self._temp_dir = PurePath(source_dir_base, "downloads").as_posix() + "/"
        self.fs.makedirs(self._temp_dir, exist_ok=True)
        self._expected_size = (3600, 1800)  # width, height

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)

        if (
            not self.fs.exists(PurePath(download_dir, self.dataset_filename))
            or force_download
        ):
            download_file(
                url=self.urls[self.dataset_filename],
                directory=download_dir,
                filename=self.dataset_filename,
                force_download=force_download,
            )

        with zipfile.ZipFile(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as z:
            zip2source = {
                name: PurePath(self.source_dir, os.path.basename(name)).as_posix()
                for name in z.namelist()
            }
            for zname, target_file in zip2source.items():
                with self.fs.open(target_file, mode="wb") as mifi:
                    mifi.write(z.read(name=zname))

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if not os.path.exists(self.source_dir) or force or force_download:
            return False

        files = os.listdir(self.source_dir)

        tif_files = [file for file in files if file.endswith(".tif")]

        return len(tif_files) == 168

    @override
    def onboard(self, target):
        path = "storm_test"
        data_arrays: Dict[str, xr.DataArray] = {}
        for return_period in self._return_periods:
            with ExitStack() as stack:
                for basin_id in self._basin_ids:
                    file_name = f"STORM_FIXED_RETURN_PERIODS_HADGEM3-GC31-HM_{basin_id}_{return_period}_YR_RP.tif"
                    da: xr.DataArray = stack.enter_context(
                        xr.open_dataarray(
                            os.path.join(self.source_dir, file_name),
                            engine="rasterio",
                            chunks={"x": 512, "y": 512},  # o el chunk size que quieras
                        )  # type: ignore
                    )
                    # TODO: enforce that da follows standard conventions
                    data_arrays[basin_id] = da

            # check all children can be assembled via an integer col/row offset into parent
            xarray_utilities.assert_sources_combinable(list(data_arrays.values()))

            # infer parent size from first child array, assuming global coverage
            _, transform0, crs0 = xarray_utilities.get_array_components(
                list(data_arrays.values())[0]
            )
            size = np.array(~transform0 * (180, -90)) - np.array(
                ~transform0 * (-180, 90)
            )
            width, height = np.array(np.round(size), dtype=int)
            if not np.allclose([width, height], size):
                raise ValueError("size is not integer.")

            # create parent zarr array first time through
            if return_period == self._return_periods[0]:
                trans_parent = self._transform(width, height)
                zarr_parent = target.create_empty(
                    path,
                    width,
                    height,
                    trans_parent,
                    crs0.to_string(),
                    return_periods=self._return_periods,
                )
                # get data array as convenient way to access coordinates (although we will write to zarr directly)
                da_parent = target.read(path)
                da_parent = da_parent.rename({"longitude": "x", "latitude": "y"})

            index = self._return_periods.index(return_period)

            for da_child in data_arrays.values():
                if np.any(da_child.x > 180):
                    x = da_child.x
                    x = np.where(x > 180 + 1e-6, x - 360, x)
                    da_child["x"] = x
                da_child = da_child.squeeze(dim="band")
                xarray_utilities.add_children_to_parent(
                    da_parent, zarr_parent, index, da_child
                )

            for da in data_arrays.values():
                xi, yi = len(da["x"]) // 2, len(da["y"]) // 2
                value = da.data[0, yi, xi]
                xip, yip = np.round(
                    np.array(~trans_parent * (float(da["x"][xi]), float(da["y"][yi])))
                    - [0.5, 0.5]
                )
                check_value = zarr_parent[index, int(yip), int(xip)]
                if not np.allclose([value], [check_value]):
                    raise ValueError("check failed.")

    def _transform(self, width: int, height: int) -> Affine:
        """Affine transform of (col, row) into (x, y) or (lon, lat) for a EPSG:4326 CRS.

        Longitudes from -180 to 180 degrees and latitudes from 90 to -90 degrees.

        Args:
            width (int): Pixels in x or longitude direction.
            height (int): Pixels in y or latitude direction.

        Returns:
            Affine: Transform.

        """
        (sx, sy) = (360.0 / width, -180.0 / height)
        # or: affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
        return Affine.translation(-180 + 0.5 * sx, 90 + 0.5 * sy) * Affine.scale(sx, sy)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        raise NotImplementedError()
