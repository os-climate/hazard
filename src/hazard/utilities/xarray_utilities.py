"""xarray_utilities."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import dask  # type: ignore
import dask.array
import numpy as np
import rasterio  # type: ignore
import rioxarray  # noqa: F401
import xarray as xr
import zarr  # type: ignore
import zarr.core
from affine import Affine  # type: ignore
from rasterio.crs import CRS  # type: ignore


def add_children_to_parent(
    da_parent: xr.DataArray,
    zarr_parent: zarr.array,
    parent_index: int,
    da_child: xr.DataArray,
):
    """Add a child data array to a parent data array.

    The parent data arrays underlying zarr array is also provided for efficient writing.

    Args:
        da_parent (xr.DataArray): Parent data array. Dimensions should be ["index", "y", "x"]
                                  (i.e. latitude renamed y; longitude renamed x).
        zarr_parent (zarr.array): Underlying zarr array of da_parent. Dimensions are ["index", "y", "x"].
        parent_index (int): Index of parent to write to.
        da_child (xr.DataArray): Child data array. Expects dimensions to be ["y", "x"].

    """
    # consider simplifying interface with: da_parent = data_array_from_zarr(zarr_parent)

    if da_child.dims != ("y", "x"):
        raise ValueError("da_child dims should be ('y', 'x').")

    if da_parent.dims != ("index", "y", "x"):
        raise ValueError("da_parent dims should be ('index', 'y', 'x').")

    _, trans_child, _ = get_array_components(da_child)
    _, trans_parent, _ = get_array_components(da_parent)
    width_child, height_child = da_child.sizes["x"], da_child.sizes["y"]
    width_parent, height_parent = da_parent.sizes["x"], da_parent.sizes["y"]

    # a couple of ways to find the indices of the parent corresponding to the child
    # 1) Use labels
    query_res_lon = da_parent.sel(x=da_child.x, method="nearest", tolerance=1e-6)

    indices_x = query_res_lon.dim_indexers["x"].data
    query_res_lat = da_parent.sel(y=da_child.y, method="nearest", tolerance=1e-6)

    indices_y = query_res_lat.dim_indexers["y"].data
    # 2) Use transform
    offset_x, offset_y = np.round(
        np.array(~trans_parent * trans_child * (0.5, 0.5)) - [0.5, 0.5]
    )
    indices_x_2 = np.arange(offset_x, offset_x + width_child, dtype=int) % width_parent
    indices_y_2 = (
        np.arange(offset_y, offset_y + height_child, dtype=int) % height_parent
    )
    # we do both and check!
    if not np.array_equal(indices_x, indices_x_2) or not np.array_equal(
        indices_y, indices_y_2
    ):
        raise ValueError("failed to find indices.")

    zarr_parent.set_orthogonal_selection(
        (parent_index, indices_y, indices_x), da_child.data[:, :]
    )


def affine_has_rotation(affine: Affine) -> bool:
    """Determine if an affine transformation includes a rotation component.

    Args:
        affine (Affine): Affine transform.

    """
    return affine.b == affine.d != 0


def affine_to_coords(
    affine: Affine, width: int, height: int, x_dim: str = "x", y_dim: str = "y"
) -> Dict[str, np.ndarray]:
    """Generate 1d pixel centered coordinates from affine.

    Args:
        affine (Affine): Affine transform.
        width (int): Width of array.
        height (int): Height of array.
        x_dim (str, optional): Name of the X dimension. Defaults to "x".
        y_dim (str, optional): Name of the Y dimension. Defaults to "y".

    Returns:
        Dict[str, np.ndarray]: co-ordinate arrays.

    """
    transform = affine * affine.translation(0.5, 0.5)
    if affine.is_rectilinear and not affine_has_rotation(affine):
        x_coords, _ = transform * (np.arange(width), np.zeros(width))
        _, y_coords = transform * (np.zeros(height), np.arange(height))
    else:
        x_coords, y_coords = transform * np.meshgrid(
            np.arange(width),
            np.arange(height),
        )
    return {y_dim: y_coords, x_dim: x_coords}


def assert_sources_combinable(sources: List[xr.DataArray]):
    """Check that children array all have same CRS (e.g. EPSG:4326) and represent fragments of a parent image that can be assembled without reprojection.

    Raises an exception if children do not meet criteria.
    """
    # handle case where all sources have CRS EPSG:4326 and represent
    # fragments of larger image
    _, transform0, crs0 = get_array_components(sources[0])
    # TODO check crs0
    for da in sources:
        _, transform, crs = get_array_components(da)
        if crs != crs0:
            raise ValueError("coordinate reference systems do not match.")
        _assert_transforms_consistent(~transform0, ~transform)


def coords_from_extent(width: int, height: int, x_dim: str = "x", y_dim: str = "y"):
    """Generate grid coordinates for a global extent based on the specified width and height.

    Args:
        width (int):
            Number of grid cells (pixels) along the x-axis (longitude).
        height (int):
            Number of grid cells (pixels) along the y-axis (latitude).
        x_dim (str, optional):
            Name to assign to the x-coordinate dimension. Defaults to "x".
        y_dim (str, optional):
            Name to assign to the y-coordinate dimension. Defaults to "y".

    Returns:
        dict:
            A dictionary containing coordinate arrays for the grid. The dictionary keys
            correspond to `x_dim` and `y_dim`, and the values are 1D arrays representing
            the center points of grid cells.

    """
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return affine_to_coords(affine, width, height, x_dim, y_dim)


def data_array_from_zarr(z: zarr.array) -> xr.DataArray:
    """Convert a Zarr array to an xarray DataArray with appropriate coordinates and CRS.

    Args:
        z: A Zarr array with metadata attributes such as "transform_mat3x3", "crs", and "index_values".

    """
    t = z.attrs["transform_mat3x3"]  # type: ignore
    crs: str = z.attrs.get("crs", "EPSG:4326")  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    index_name = z.attrs.get("dimensions", ["index"])[0]
    index_values = z.attrs.get(index_name + "_values", [0])
    if index_values is None:
        index_values = [0]
    coords = affine_to_coords(
        transform, z.shape[2], z.shape[1], x_dim="dim_2", y_dim="dim_1"
    )
    coords["dim_0"] = index_values
    da = xr.DataArray(
        data=dask.array.from_zarr(z), dims=["dim_0", "dim_1", "dim_2"], coords=coords
    )
    if "EPSG:4326" in crs.upper():
        da.rio.write_crs(4326, inplace=True)
        da = da.rename({"dim_0": index_name, "dim_1": "latitude", "dim_2": "longitude"})
    else:
        da.rio.write_crs(crs, inplace=True)
        da = da.rename({"dim_0": index_name, "dim_1": "y", "dim_2": "x"})
    return da


def enforce_conventions_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """By convention, underlying data should have decreasing latitude and should be centred on longitude 0."""
    if da.lat[-1] > da.lat[0]:
        da = da.reindex(lat=da.lat[::-1])
    if np.any(da.lon > 180):
        lon = da.lon
        lon = np.where(lon > 180, lon - 360, lon)
        da["lon"] = lon
        da = da.roll(lon=-len(da.lon) // 2, roll_coords=True)
    return da


def get_array_components(
    da: xr.DataArray, assume_normalized: bool = False
) -> Tuple[Any, Affine, Any]:
    """Extract the components of an xarray DataArray, including its data, affine transformation, and CRS.

    Args:
        da: An xarray DataArray, typically representing a spatially referenced dataset
            with dimensions such as (index, latitude, longitude) or (index, y, x).
        assume_normalized: If True, the input DataArray is assumed to be normalized (default is False).
            If False, the array will be normalized to the expected dimensionality.

    """
    renamed = da
    if not assume_normalized:
        renamed = normalize_array(renamed)
        # after this data will have dims (index, latitude, longitude) or (index, y, x)
        # index might have size 1

    if "latitude" in da.dims and "longitude" in da.dims:
        renamed = da.rename({"latitude": "y", "longitude": "x"})
    elif "x" not in da.dims or "y" not in da.dims:
        raise ValueError("dimensions not recognized.")

    data = renamed.data
    transform = renamed.rio.transform(recalc=True)
    crs = da.rio.crs
    if crs is None:
        # assumed default
        crs = da.attrs.get("crs", rasterio.CRS.from_epsg(4326))
    return (data, transform, crs)


def global_crs_transform(width: int = 3600, height: int = 1800):
    """Compute a global coordinate reference system (CRS) and affine transformation for a grid.

    Args:
        width: The number of grid cells (pixels) along the longitude axis (default is 3600).
        height: the number of grid cells (pixels) along the latitude axis (default is 1800).

    """
    crs = CRS.from_epsg(4326)
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return crs, affine


def normalize_array(da: xr.DataArray) -> xr.DataArray:  # noqa: C901
    """Ensure that DataArray follows the conventions expected by downstream algorithms.

    The conventions are:
    - dimensions must be (index, latitude, longitude) or (index, y, x) in that order; 'index' is most often
    used for return periods;
    - longitude and index are increasing; latitude is decreasing;
    - CRS is present.
    - longitude should be in range -180 and 180, not 0 to 360.
    """
    mappings = {"X": "x", "Y": "y", "lat": "latitude", "lon": "longitude"}
    to_rename = {}
    for dim in da.dims:
        assert isinstance(dim, str)
        if dim in mappings:
            to_rename[dim] = mappings[dim]
    da_norm = da.rename(to_rename)
    if "x" in da_norm.dims and "y" in da_norm.dims:
        index_dim = [d for d in da_norm.dims if d not in ["x", "y"]]
        dim_x, dim_y = "x", "y"
    elif "latitude" in da_norm.dims and "longitude" in da_norm.dims:
        index_dim = [d for d in da_norm.dims if d not in ["latitude", "longitude"]]
        dim_x, dim_y = "longitude", "latitude"
    else:
        raise ValueError(f"unexpected dims {da.dims}")

    if len(da.dims) == 3:
        if len(index_dim) != 1:
            raise Exception(f"unexpected dims {da.dims}")
        # keep the name the same name, but this has to be the first dimension
        da_norm = da_norm.transpose(index_dim[0], dim_y, dim_x)
    elif len(da.dims) == 2:
        da_norm = da_norm.expand_dims(dim={"index": 1}, axis=0)
        da_norm = da_norm.transpose("index", dim_y, dim_x)
    else:
        raise ValueError("2 or 3 dimensions expected.")

    if "latitude" in da_norm.dims and da_norm.latitude[-1] > da_norm.latitude[0]:
        da_norm = da_norm.reindex(latitude=da_norm.latitude[::-1])
    elif "y" in da_norm.dims and da_norm.y[-1] > da_norm.y[0]:
        da_norm = da_norm.reindex(y=da_norm.y[::-1])

    if "longitude" in da_norm.dims and np.any(da_norm.longitude > 180):
        longitude = da_norm.longitude
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        da_norm["longitude"] = longitude
        da_norm = da_norm.roll(longitude=-len(da_norm.longitude) // 2, roll_coords=True)
    da_norm.name = "indicator"
    return da_norm


def data_array(
    data: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
    name: str = "data",
    index_name: str = "index",
    index_units: str = "",
    index_values: Sequence[str] = ["0"],
):
    """Create an xarray DataArray from a 3D numpy array with spatial coordinates.

    Args:
        data: A 3D numpy array with shape (n_indices, height, width), where `n_indices`
            corresponds to the index dimension, and `height` and `width` correspond to
            the spatial dimensions (latitude/longitude or y/x).
        transform: The affine transformation that maps pixel coordinates to geographic coordinates.
        crs: The coordinate reference system to use (default is "EPSG:4326"). It affects
            the naming of spatial dimensions ("latitude"/"longitude" vs. "y"/"x").
        name: The name of the resulting DataArray (default is "data").
        index_name: The name of the index dimension (default is "index").
        index_units: The units of the index dimension, stored as an attribute in the DataArray (default is an empty string).
        index_values: A list of values for the index dimension, which should have the same length as
            the first dimension of the `data` array (default is [0]).

    """
    n_indices, height, width = data.shape

    assert len(index_values) == n_indices

    z_dim = index_name
    y_dim, x_dim = (
        ("latitude", "longitude") if crs.upper() == "EPSG:4326" else ("y", "x")
    )
    coords = affine_to_coords(transform, width, height, x_dim=x_dim, y_dim=y_dim)
    coords[z_dim] = np.array(index_values)
    da = xr.DataArray(data=data, coords=coords, dims=[z_dim, y_dim, x_dim], name=name)
    if index_units != "":
        da.attrs[z_dim + "_units"] = index_units
    return da


def empty_data_array(
    width: int,
    height: int,
    transform: Affine,
    crs: str = "EPSG:4326",
    index_name: str = "index",
    index_units: str = "",
    index_values: Sequence[str] = ["0"],
    chunks: Optional[Sequence[int]] = None,
):
    """Create an empty xarray DataArrau with the specified dimensions and spatial attributes.

    Args:
        width: The number of grid cells (pixels) along the width (longitude axis).
        height:  The number of grid cells (pixels) along the height (latitude axis).
        transform: The affine transformation that maps pixel coordinates to geographic coordinates.
        crs: The coordinate reference system to use (default is "EPSG:4326").
            It affects the naming of spatial dimensions ("latitude"/"longitude" vs. "y"/"x").
        index_name: The name of the index dimension (default is "index").
        index_units: The units of the index dimension, stored as an attribute in the DataArray (default is an empty string).
        index_values: A list of values for the index dimension, which should have the same length as
            the first dimension of the resulting array (default is [0])

    """

    shape = [len(index_values), height, width]

    if chunks is None:
        chunks = shape

    data = dask.array.empty(shape=shape, chunks=chunks)
    return data_array(
        data,
        transform,
        crs,
        index_name=index_name,
        index_units=index_units,
        index_values=index_values,
    )


def write_array(array: xr.DataArray, path: str):
    """Write an sarray DataArray to a raster file in Cloud Optimized GeoTIFF (COG) format.

    Args:
        array:
            The xarray DataArray to be written to the raster file. The DataArray must contain spatial data
            and be associated with a valid coordinate reference system (CRS).
        path: The file path where the raster will be saved.

    """
    array.rio.to_raster(raster_path=path, driver="COG")


def _assert_transforms_consistent(trans1: Affine, trans2: Affine):
    """Check transforms from (x, y)/(lon, lat) to (col, row) are consistent.

    If same point maps to (col, row) offset by integer amount then a parent array can be assembled from offset children
    with no reprojection.

    """
    if not np.allclose(
        (trans1.a, trans1.b, trans1.d, trans1.e),
        (trans2.a, trans2.b, trans2.d, trans2.e),
    ):
        raise ValueError("transforms have inconsistent scaling.")
    if not np.allclose(
        (trans1.c % 1 - trans2.c % 1, trans1.f % 1 - trans2.f % 1), (0.0, 0.0)
    ):
        raise ValueError("transforms have non-integer offset.")
