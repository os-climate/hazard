from typing import Any, Dict, Generator, Iterable, List, Tuple

import dask, dask.array  # type: ignore
import numpy as np
import rasterio  # type: ignore
import rioxarray
import xarray as xr
import zarr, zarr.core  # type: ignore
from affine import Affine  # type: ignore
from rasterio.crs import CRS  # type: ignore


def add_children_to_parent(
    da_parent: xr.DataArray,
    zarr_parent: zarr.array,
    parent_index: int,
    da_child: xr.DataArray,
):
    """Add a child data array to a parent data array. The parent data
       arrays underlying zarr array is also provided for efficient writing.

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
    query_res_lon = da_parent.xindexes["x"].sel(
        {"x": da_child.x}, method="nearest", tolerance=1e-6
    )
    indices_x = query_res_lon.dim_indexers["x"].data
    query_res_lat = da_parent.xindexes["y"].sel(
        {"y": da_child.y}, method="nearest", tolerance=1e-6
    )
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
    """Check that children array all have same CRS (e.g. EPSG:4326) and represent fragments
    of a parent image that can be assembled without reprojection. Raises an exception if
    children do not meet criteria.
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
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return affine_to_coords(affine, width, height, x_dim, y_dim)


def data_array_from_zarr(z: zarr.core.Array) -> xr.DataArray:
    t = z.attrs["transform_mat3x3"]  # type: ignore
    crs: str = z.attrs.get("crs", "EPSG:4326")  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    index_values = z.attrs.get("index_values", [0])
    # index_name = z.attrs.get("index_name", [0])
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
        da = da.rename({"dim_0": "index", "dim_1": "latitude", "dim_2": "longitude"})
    else:
        da = da.rename({"dim_0": "index", "dim_1": "y", "dim_2": "x"})
    return da


def enforce_conventions_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """By convention, underlying data should have decreasing latitude
    and should be centred on longitude 0."""
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
        crs = rasterio.CRS.from_epsg(4326)
    return (data, transform, crs)


def global_crs_transform(width: int = 3600, height: int = 1800):
    crs = CRS.from_epsg(4326)
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return crs, affine


def normalize_array(da: xr.DataArray) -> xr.DataArray:
    """Ensure that DataArray follows the conventions expected by downstream algorithms:
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
        da_norm.set_index({"index": index_dim[0]})
        da_norm = da_norm.transpose("index", dim_y, dim_x)
    elif len(da.dims) == 2:
        da_norm = da_norm.expand_dims(dim={"index": 1}, axis=0)
        da_norm = da_norm.transpose("index", dim_y, dim_x)
    else:
        raise ValueError("2 or 3 dimensions expected.")

    if "latitude" in da_norm.dims and da_norm.latitude[-1] > da_norm.latitude[0]:
        da_norm = da_norm.reindex(latitude=da_norm.latitude[::-1])
    elif "y" in da_norm.dims and da_norm.y[-1] > da_norm.y[0]:
        da_norm = da_norm.reindex(y=da_norm.y[::-1])

    if not da_norm.rio.crs:
        da_norm.rio.write_crs(4326, inplace=True)

    if "longitude" in da_norm.dims and np.any(da_norm.longitude > 180):
        longitude = da_norm.longitude
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        da_norm["longitude"] = longitude
        da_norm = da_norm.roll(longitude=-len(da_norm.longitude) // 2, roll_coords=True)

    return da_norm


def data_array(
    data: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
    index_values: List[float] = [0],
):
    n_indices, height, width = data.shape
    assert len(index_values) == n_indices

    z_dim = "index"
    y_dim, x_dim = (
        ("latitude", "longitude") if crs.upper() == "EPSG:4326" else ("y", "x")
    )
    coords = affine_to_coords(transform, width, height, x_dim=x_dim, y_dim=y_dim)
    coords[z_dim] = np.array(index_values, dtype=float)
    da = xr.DataArray(data=data, coords=coords, dims=[z_dim, y_dim, x_dim])
    return da


def empty_data_array(
    width: int,
    height: int,
    transform: Affine,
    crs: str = "EPSG:4326",
    index_values: List[float] = [0],
):
    data = dask.array.empty(shape=[len(index_values), height, width])
    return data_array(data, transform, crs, index_values)


def write_array(array: xr.DataArray, path: str):
    array.rio.to_raster(raster_path=path, driver="COG")


def _assert_transforms_consistent(trans1: Affine, trans2: Affine):
    """Check transforms from (x, y)/(lon, lat) to (col, row) are consistent. If same point maps
    to (col, row) offset by integer amount then a parent array can be assembled from offset childen
    with no reprojection
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
