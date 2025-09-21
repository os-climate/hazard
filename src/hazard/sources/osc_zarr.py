import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import s3fs
import xarray as xr
import zarr
import zarr.core
from affine import Affine

import hazard.utilities.xarray_utilities as xarray_utilities
from hazard.protocols import ReadWriteDataArray
from hazard.utilities.s3_utilities import get_store


logger = logging.getLogger(__name__)


class OscZarr(ReadWriteDataArray):
    """A class for reading and writing to OSC Climate Zarr storage.

    This class facilitates the interaction with Zarr storage in an S3 bucket, allowing
    for the creation, reading, and writing of multi-dimensional data arrays according to
    the OS-Climate conventions. It provides methods to handle both empty arrays and filled
    datasets, ensuring compatibility with xarray.

    """

    def __init__(
        self,
        s3: Optional[s3fs.S3FileSystem] = None,
        store: Optional[Any] = None,
        store_netcdf_coords: Optional[bool] = False,
        bucket: Optional[str] = None,
        group_path_suffix: str = "hazard/hazard.zarr",
        extra_s3fs_kwargs: Optional[dict] = None,
    ):
        """Class for reading and writing to OSC Climate Zarr storage.

        This class manages interactions with Zarr storage located in an S3 bucket.
        It can utilize an existing `store` directly if provided. If an `S3File` is
        given, it will be used unless a `store` is also provided. If neither is
        available, a new store will be created using credentials from environment
        variables.

        Args:
            s3: Optional[s3fs.S3FileSystem], optional
                An S3FileSystem instance to use. If provided and `store` is not
                specified, this instance will be used for storage operations.
            store: Optional[zarr.storage.Store], optional
                A Zarr store to be used for reading and writing. If provided, it
                overrides the creation of a new store.
            write_xarray_compatible_zarr: bool, optional
                If set to True, an xarray-compatible Zarr store will be created alongside
                the default Zarr output.
            bucket: Optional[str] = None
                bucket to use. If not provided, the value from the envvar
                (OSC_S3_BUCKET or OSC_S3_BUCKET_DEV) will be used.
            group_path_suffix: str = "hazard/hazard.zarr"
                The root zarr group is by convention `${bucket}/hazard/hazard.zarr`.
                This argument allows changing the `hazard/hazard.zarr` part.
            extra_s3fs_kwargs: Dict = {}
                A dictionary with the necessary environment variables to build the s3 file system

        Example:
            reader = OscZarr()

        """
        if extra_s3fs_kwargs is None:
            extra_s3fs_kwargs = {}

        if store is None:
            if s3 is None:
                store = get_store(
                    use_dev=True,
                    bucket=bucket,
                    group_path_suffix=group_path_suffix,
                    extra_s3fs_kwargs=extra_s3fs_kwargs,
                )
            else:
                store = get_store(
                    use_dev=True,
                    s3=s3,
                    bucket=bucket,
                    group_path_suffix=group_path_suffix,
                    extra_s3fs_kwargs=extra_s3fs_kwargs,
                )

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
        index_values: Any = None,
        chunks=None,
    ):
        """Create an empty Zarr array with specified dimensions and geospatial properties."""
        if index_values is None:
            index_values = [0]
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
        """Remove an item from the Zarr store at the specified path."""
        if path in self.root:
            try:
                self.root.pop(path)
            except KeyError:
                pass  # If the key doesn't exist, ignore it.

    def read(self, path: str) -> xr.DataArray:
        """Read an OS-Climate array as an xarray DataArray.

        Coordinates are inferred from the coordinate reference system (CRS) and the
        affine transform stored as Zarr attributes.

        Args:
            path (str):
                Relative path to the Zarr array.

        Returns:
            xr.DataArray:
                The data as an xarray DataArray.

        """
        z = self.root[path]
        return xarray_utilities.data_array_from_zarr(z)

    def read_dataset(self, path: str, index=0) -> xr.DataArray:
        """Read a dataset from the Zarr store as an xarray DataArray."""
        da = xr.open_zarr(store=self.root.store, group=path)
        return da

    def read_floored(self, path, longitudes, latitudes):
        """Read data from the Zarr store using floored coordinates.

        This method retrieves data from a specified path in the Zarr store,
        using provided longitude and latitude values. The coordinates are
        floored and transformed to indices to access the data array.

        Args:
            path (str):
                The relative path to the Zarr dataset.
            longitudes (array-like):
                Array of longitude values for data retrieval.
            latitudes (array-like):
                Array of latitude values for data retrieval.

        Returns:
            np.ndarray:
                The retrieved data corresponding to the floored coordinates.

        """
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
        """Read a Zarr array at a given index as a two-dimensional NumPy array along with its affine transform and CRS.

        This method is intended for small datasets. For larger datasets, it is recommended
        to use `xarray.open_zarr` for better performance and handling.

        Args:
            path (str):
                Relative path to the Zarr array.
            index (int, optional):
                Index of the array to read. Defaults to 0.

        Returns:
            Tuple[np.ndarray, Affine, str]:
                - NumPy array of the data at the specified index.
                - Affine transform of the data.
                - Coordinate reference system (CRS) as a string.

        """
        z = self.root[path]
        t = z.attrs["transform_mat3x3"]  # type: ignore
        crs: str = z.attrs["crs"]  # type: ignore
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        return z[index, :, :], transform, crs

    def read_zarr(self, path):
        """Read a Zarr array from the specified path."""
        return self.root[path]

    def if_exists_remove(self, path):
        """Remove a path from the Zarr store if it exists."""
        if path in self.root:
            self.root.pop(path)

    def write(
        self,
        path: str,
        da: xr.DataArray,
        chunks: Optional[Sequence[int]] = None,
        spatial_coords: Optional[bool] = True,
    ):
        """Write a DataArray to the Zarr store at the specified path.

        This method saves the given xarray DataArray to the Zarr store.
        If specified, it can also write a compatible xarray format.
        """
        if self.store_netcdf_coords and spatial_coords:
            # In this mode, the xarray is written to path including NetCDF-style co-ordinates.
            # The Zarr array containing the hazard indicator will be in path/indicator.
            self.write_data_array(path, da, chunks)
        else:
            self.write_zarr(path, da, chunks)

    def write_zarr(
        self, path: str, da: xr.DataArray, chunks: Optional[Sequence[int]] = None
    ):
        """Write DataArray according to the standard OS-Climate conventions.

        Args:
            path (str):
                The relative path where the DataArray will be stored in the
                Zarr store.
            da (xr.DataArray):
                The xarray DataArray to be written. It is normalized before
                being stored to ensure proper formatting and compatibility.
            chunks (Sequence[int]], optional):
                The desired chunk sizes for the DataArray. If provided,
                this allows for optimized storage and retrieval of data.

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
        """Write a 3D slice of data to a specified path in the Zarr store."""
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

    def write_data_array(
        self, path: str, da: xr.DataArray, chunks: Optional[Sequence[int]] = None
    ):
        """Write an xarray DataArray to the provided relative path.

        The DataArray is saved as a Zarr dataset in the native xarray format using
        `xarray.to_zarr`, with coordinates saved as separate arrays. Additional
        attributes are included to store the coordinate reference system (CRS)
        and affine transform, making coordinate arrays optional.

        Note:
            It's recommended to use the `write` method instead for general use.

        Args:
            path (str):
                Relative path to store the DataArray.
            da (xr.DataArray):
                The xarray DataArray to write.

        Raises:
            ValueError:
                If the DataArray lacks "latitude" or "longitude" dimensions.

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
        options: Dict[str, Any] = {"write_empty_chunks": False}
        if chunks is not None:
            options["chunks"] = chunks
        elif da.chunks is None:
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

    def _data_array_from_zarr(self, z: zarr.array) -> xr.DataArray:
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
        """Create Zarr array with given shape and affine transform."""
        if path in self.root:
            try:
                self.root.pop(path)
            except KeyError:
                pass  # If the key doesn't exist, ignore it.
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
        """Normalize the dimensions of an xarray DataArray."""
        return xarray_utilities.normalize_array(da)
