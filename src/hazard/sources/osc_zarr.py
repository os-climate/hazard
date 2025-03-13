import logging
import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import s3fs  # type: ignore
import xarray as xr
import zarr  # type: ignore
import zarr.core
from affine import Affine  # type: ignore

import hazard.utilities.xarray_utilities as xarray_utilities
from hazard.protocols import ReadWriteDataArray

default_dev_bucket = "physrisk-hazard-indicators-dev01"

logger = logging.getLogger(__name__)


class OscZarr(ReadWriteDataArray):
    __access_key = "OSC_S3_ACCESS_KEY_DEV"
    __secret_key = "OSC_S3_SECRET_KEY_DEV"
    __token = "OSC_S3_TOKEN_DEV"

    def __init__(
        self,
        bucket: str = default_dev_bucket,
        prefix: str = "hazard",
        s3: Optional[s3fs.S3File] = None,
        store: Optional[Any] = None,
        store_netcdf_coords: Optional[bool] = False,
    ):
        """For reading and writing to OSC Climate Zarr storage.
        If store is provided this is used, otherwise if S3File is provided, this is used.
        Otherwise, store is created using credentials in environment variables.

        Args:
            bucket: Name of S3 bucket.
            prefix: S3 bucket item prefix
            s3: S3File to use if present and if store not provided.
            store: If provided, Zarr will use this store.
            store_netcdf_coords: If true, an xarray compatible zarr
             will be created alongside the default zarr output
        """
        if store is None:
            if s3 is None:
                # zarr_utilities.load_dotenv() # to load environment variables
                access_key = os.environ.get(self.__access_key, None)
                secret_key = os.environ.get(self.__secret_key, None)
                token = os.environ.get(self.__token, None)
                if token:
                    s3 = s3fs.S3FileSystem(
                        key=access_key, secret=secret_key, token=token
                    )
                else:
                    s3 = s3fs.S3FileSystem(key=access_key, secret=secret_key)

            group_path = str(PurePosixPath(bucket, prefix, "hazard.zarr"))
            store = s3fs.S3Map(root=group_path, s3=s3, check=False)

        self.root = zarr.group(store=store)

        self.store_netcdf_coords = store_netcdf_coords

    def create_empty(
        self,
        path: str,
        width: int,
        height: int,
        transform: Affine,
        crs: str,
        overwrite=False,
        index_name: Optional[str] = "index",
        index_values: Any = [0],
        chunks=None,
    ):
        return self._zarr_create(
            path,
            (len(index_values), height, width),
            transform,
            str(crs),
            overwrite,
            index_name=index_name,
            index_values=index_values,
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

    def write(
        self,
        path: str,
        da: xr.DataArray,
        chunks: Optional[List[int]] = None,
        spatial_coords: Optional[bool] = True,
    ):
        if self.store_netcdf_coords and spatial_coords:
            # In this mode, the xarray is written to path including NetCDF-style co-ordinates.
            # The Zarr array containing the hazard indicator will be in path/indicator.
            self.write_data_array(path, da)
        else:
            self.write_zarr(path, da, chunks)

    def write_zarr(
        self, path: str, da: xr.DataArray, chunks: Optional[List[int]] = None
    ):
        """Write DataArray according to the standard OS-Climate conventions.

        Args:
            path (str): Relative path.
            da (xr.DataArray): The DataArray.
        """
        if "lon" not in da.dims and "longitude" not in da.dims and "x" not in da.dims:
            raise ValueError("longitude or x dimension not found.")
        if "lat" not in da.dims and "latitude" not in da.dims and "y" not in da.dims:
            raise ValueError("latitude dimension not found.")
        da_norm = xarray_utilities.normalize_array(da)
        data, transform, crs = xarray_utilities.get_array_components(
            da_norm, assume_normalized=True
        )
        z = self._zarr_create(
            path,
            da_norm.shape,
            transform,
            str(crs),
            index_name=str(da_norm.dims[0]),
            index_values=da_norm[
                da_norm.dims[0]
            ].data,  # the index dimension; allow to have name other than 'index'
            chunks=chunks,
        )
        z[:, :, :] = data[:, :, :]

    def write_slice(
        self, path, z_slice: slice, y_slice: slice, x_slice: slice, da: np.ndarray
    ):
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
            str(crs),
            overwrite=True,
            index_values=da[
                da.dims[0]
            ].data,  # the index dimension; alow to have name other than 'index'
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
        if "lon" not in da.dims and "longitude" not in da.dims and "x" not in da.dims:
            raise ValueError("longitude or x dimension not found.")
        if "lat" not in da.dims and "latitude" not in da.dims and "y" not in da.dims:
            raise ValueError("latitude or y dimension not found.")
        da_norm = xarray_utilities.normalize_array(da)
        _, transform, crs = xarray_utilities.get_array_components(
            da_norm, assume_normalized=True
        )
        self._add_attributes(
            da_norm.attrs,
            transform,
            str(crs),
            index_name=str(da_norm.dims[0]),
            index_values=da_norm[da_norm.dims[0]].data,
        )
        options = {"write_empty_chunks": False}
        if da.chunks is None:
            options["chunks"] = self._chunks(da_norm[da_norm.dims[0]].data)
        da_norm.to_dataset().to_zarr(
            self.root.store,
            compute=True,
            group=path,
            mode="w",
            consolidated=False,
            encoding={da_norm.name: options},
        )

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
        index_name: Optional[str] = None,
        index_values: Any = None,
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
            zarr_shape = (
                1 if index_values is None else len(index_values),
                shape[0],
                shape[1],
            )
        else:
            raise ValueError("shape of DataArray must have length of 2 or 3.")
        if chunks is None:
            chunks = self._chunks(index_values)
        z = self.root.create_dataset(
            path,
            shape=zarr_shape,
            chunks=chunks,
            dtype="f4",
            overwrite=overwrite,
            write_empty_chunks=False,
            fill_value=float("nan"),
        )  # array_path interpreted as path within group
        if isinstance(index_values, np.ndarray) and index_values.dtype in [
            "int16",
            "int32",
            "int64",
        ]:
            index_values = [int(i) for i in index_values]
        self._add_attributes(
            z.attrs, transform, crs, index_name=index_name, index_values=index_values
        )
        return z

    def _add_attributes(
        self,
        attrs: Dict[str, Any],
        transform: Affine,
        crs: str,
        index_name: Optional[str] = None,
        index_values: Any = None,
        index_units: str = "",
    ):
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
        use_xy = crs.upper() != "EPSG:4326"
        attrs["transform_mat3x3"] = mat3x3
        attrs["dimensions"] = (
            [index_name, "y", "x"] if use_xy else [index_name, "latitude", "longitude"]
        )
        if index_values is not None:
            attrs[f"{index_name}_values"] = list(index_values)
            attrs[f"{index_name}_units"] = index_units

    def _chunks(self, index_values: List[Any]):
        chunk_dim = 1000 if len(index_values) < 10 else 500
        chunks = (
            1 if index_values is None else len(index_values),
            chunk_dim,
            chunk_dim,
        )
        return chunks

    @staticmethod
    def normalize_dims(da: xr.DataArray) -> xr.DataArray:
        return xarray_utilities.normalize_array(da)
