import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import numpy as np
import s3fs  # type: ignore
import xarray as xr
import zarr  # type: ignore
import zarr.core
from affine import Affine  # type: ignore

import hazard.utilities.xarray_utilities as xarray_utilities
from hazard.protocols import WriteDataArray

default_dev_bucket = "physrisk-hazard-indicators-dev01"


class OscZarr(WriteDataArray):
    def __init__(
        self,
        bucket: str = default_dev_bucket,
        prefix: str = "hazard",
        s3: Optional[s3fs.S3File] = None,
        store: Optional[MutableMapping] = None,
    ):
        """For reading and writing to OSC Climate Zarr storage.
        If store is provided this is used, otherwise if S3File is provided, this is used.
        Otherwise, store is created using credentials in environment variables.

        Args:
            bucket: Name of S3 bucket.
            root: Path to Zarr Group, i.e. objects are located in S3://{bucket}/{prefix}/hazard.zarr/{rest of key}.
            store: If provided, Zarr will use this store.
            s3: S3File to use if present and if store not provided.
        """
        if store is None:
            if s3 is None:
                # zarr_utilities.load_dotenv() # to load environment variables
                s3 = s3fs.S3FileSystem(
                    anon=False,
                    key=os.environ["OSC_S3_ACCESS_KEY_DEV"],
                    secret=os.environ["OSC_S3_SECRET_KEY_DEV"],
                )
            group_path = str(PurePosixPath(bucket, prefix, "hazard.zarr"))
            store = s3fs.S3Map(root=group_path, s3=s3, check=False)

        self.root = zarr.group(store=store)

    def create_empty(
        self,
        path: str,
        width: int,
        height: int,
        transform: Affine,
        crs: str,
        overwrite=False,
        indexes=[0],
        chunks=None,
    ):
        return self._zarr_create(
            path,
            (len(indexes), height, width),
            transform,
            str(crs),
            overwrite,
            indexes=indexes,
            chunks=chunks,
        )

    def remove(self, path: str):
        try:
            self.root.pop(path)
        except Exception:
            pass  # if it already exists, remove it

    def read(self, path: str) -> xr.DataArray:
        """Read an OS-Climate array as an xarray DataArray. Coordinates are inferred from the
        coordinate reference system (CRS) and affine transform stored as zarr attributes.

        Args:
            path (str): relative path to zarr array.

        Returns:
            xr.DataArray: xarray DataArray.
        """
        z = self.root[path]
        return xarray_utilities.data_array_from_zarr(z)

    def read_dataset(self, path: str, index=0) -> xr.DataArray:
        da = xr.open_zarr(store=self.root.store, group=path)
        return da

    def read_floored(self, path, longitudes, latitudes):
        z = self.root[path]
        t = z.attrs["transform_mat3x3"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        index_values = z.attrs.get("index_values", [0])
        if index_values is None:
            index_values = [0]
        image_coords = OscZarr._get_coordinates(longitudes, latitudes, transform)
        image_coords = np.floor(image_coords).astype(int)
        iy = np.repeat(image_coords[1, :], len(index_values))
        ix = np.repeat(image_coords[0, :], len(index_values))
        if len(z.shape) == 3:
            iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1])  # type: ignore
            data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore
        else:
            data = z.get_coordinate_selection((iy, ix))
        if len(z.shape) == 3:
            iz = np.tile(np.arange(z.shape[0]), image_coords.shape[1])  # type: ignore
            data = z.get_coordinate_selection((iz, iy, ix))  # type: ignore
        else:
            data = z.get_coordinate_selection((iy, ix))
        return data

    def read_numpy(self, path: str, index=0) -> Tuple[np.ndarray, Affine, str]:
        """Read index as two dimensional numpy array and affine transform.
        This is intended for small datasets, otherwise recommended to
        use xarray.open_zarr."""
        z = self.root[path]
        t = z.attrs["transform_mat3x3"]  # type: ignore
        crs: str = z.attrs["crs"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        return z[index, :, :], transform, crs

    def read_zarr(self, path):
        return self.root[path]

    def if_exists_remove(self, path):
        if path in self.root:
            self.root.pop(path)

    def write(self, path: str, da: xr.DataArray, chunks: Optional[List[int]] = None):
        """Write DataArray according to the standard OS-Climate conventions.

        Args:
            path (str): Relative path.
            da (xr.DataArray): The DataArray.
        """
        da_norm = xarray_utilities.normalize_array(da)
        data, transform, crs = xarray_utilities.get_array_components(da_norm, assume_normalized=True)
        z = self._zarr_create(
            path,
            da_norm.shape,
            transform,
            crs.to_string(),
            indexes=da_norm.index.data,
            chunks=chunks,
        )
        z[:, :, :] = data[:, :, :]

    def write_slice(self, path, z_slice: slice, y_slice: slice, x_slice: slice, da: np.ndarray):
        z = self.root[path]
        z[z_slice, y_slice, x_slice] = np.expand_dims(da, 0)

    def new_empty_like(self, path: str, da: xr.DataArray) -> xr.DataArray:
        """Write a new empty Zarr array like the one supplied and return array.

        Args:
            path (str): _description_
            da (xr.DataArray): _description_

        Returns:
            xr.DataArray: New array.
        """
        _, transform, crs = xarray_utilities.get_array_components(da)
        z = self._zarr_create(
            path,
            da.shape,
            transform,
            crs.to_string(),
            overwrite=True,
            indexes=da.coords["index"].data,
        )
        return self._data_array_from_zarr(z)

    def write_data_array(self, path: str, da: xr.DataArray):
        """[You should probably rather use the 'write'method] Write DataArray to provided relative path.
        The array is saved as a dataset in the xarray native manner (xarray's to_zarr),
        with coordinates as separate arrays,
        but with extra attributes providing the coordinate reference system and affine transform.
        These extra attributes serve to make use of co-ordinate arrays optional.

        Args:
            path (str): Relative path.
            da (xr.DataArray): The DataArray.
        """
        # we expect the data to be called 'data'
        if "lon" not in da.dims and "longitude" not in da.dims:
            raise ValueError("longitude dimension not found.")
        if "lat" not in da.dims and "latitude" not in da.dims:
            raise ValueError("latitude dimension not found.")
        try:
            renamed = da.rename({"lat": "latitude", "lon": "longitude"})
        except Exception:
            renamed = da
        renamed.name = "data"
        if len(renamed.dims) == 2:
            renamed = renamed.expand_dims(dim={"unused": 1}, axis=0)
        _, transform, crs = xarray_utilities.get_array_components(renamed)
        self._add_attributes(renamed.attrs, transform, crs.to_string())
        renamed.to_dataset().to_zarr(self.root.store, compute=True, group=path, mode="w", consolidated=False)

    @staticmethod
    def _get_coordinates(longitudes, latitudes, transform: Affine):
        coords = np.vstack((longitudes, latitudes, np.ones(len(longitudes))))  # type: ignore
        inv_trans = ~transform
        mat = np.array(inv_trans).reshape(3, 3)
        frac_image_coords = mat @ coords
        return frac_image_coords

    def _data_array_from_zarr(self, z: zarr.core.Array) -> xr.DataArray:
        return xarray_utilities.data_array_from_zarr(z)

    def _zarr_create(
        self,
        path: str,
        shape: Union[np.ndarray, Tuple[int, ...]],
        transform: Affine,
        crs: str,
        overwrite=False,
        indexes=None,
        chunks=None,
    ):
        """
        Create Zarr array with given shape and affine transform.
        """
        try:
            self.root.pop(path)
        except Exception:
            pass  # if it already exists, remove it
        if len(shape) == 3:
            zarr_shape = shape
        elif len(shape) == 2:
            zarr_shape = (1 if indexes is None else len(indexes), shape[0], shape[1])
        else:
            raise ValueError("shape of DataArray must have length of 2 or 3.")
        if chunks is None:
            chunk_dim = 1000 if len(indexes) < 10 else 500
            chunks = (1 if indexes is None else len(indexes), chunk_dim, chunk_dim)
        z = self.root.create_dataset(
            path,
            shape=zarr_shape,
            chunks=chunks,
            dtype="f4",
            overwrite=overwrite,
            write_empty_chunks=False,
            fill_value=float("nan"),
        )  # array_path interpreted as path within group
        if isinstance(indexes, np.ndarray) and indexes.dtype in [
            "int16",
            "int32",
            "int64",
        ]:
            indexes = [int(i) for i in indexes]
        self._add_attributes(z.attrs, transform, crs, indexes)
        return z

    def _add_attributes(self, attrs: Dict[str, Any], transform: Affine, crs: str, indexes=None):
        trans_members = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]
        mat3x3 = [x * 1.0 for x in trans_members] + [0.0, 0.0, 1.0]
        attrs["crs"] = crs
        use_xy = crs.upper() == "EPSG:3857"
        attrs["transform_mat3x3"] = mat3x3
        attrs["dimensions"] = ["index", "y", "x"] if use_xy else ["index", "latitude", "longitude"]
        if indexes is not None:
            attrs["index_values"] = list(indexes)
            attrs["index_name"] = "return period (years)"

    @staticmethod
    def normalize_dims(da: xr.DataArray) -> xr.DataArray:
        return xarray_utilities.normalize_array(da)
