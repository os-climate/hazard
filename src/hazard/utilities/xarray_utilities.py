from typing import Any, Dict, Generator, Iterable, List, Tuple

from affine import Affine # type: ignore
import dask # type: ignore
import numpy as np
import rasterio, rioxarray # type: ignore
import xarray as xr
import zarr # type: ignore


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


def add_children_to_parent(da_parent: xr.DataArray, zarr_parent: zarr.array, parent_index: int, da_child: xr.DataArray):
    """Add a child data array to a parent data array. The parent data arrays underlying zarr array is also provided for efficient writing.

    Args:
        da_parent (xr.DataArray): Parent data array. Dimensions should be ["index", "y", "x"] (i.e. latitude renamed y; longitude renamed x).
        zarr_parent (zarr.array): Underlying zarr array of da_parent. Dimensions are ["index", "y", "x"].  
        parent_index (int): Index of parent to write to.
        da_child (xr.DataArray): Child data array. Expects dimensions to be ["y", "x"].
    """
    if da_child.dims != ('y', 'x'):
        raise ValueError("da_child dims should be ('y', 'x').")
    
    if da_parent.dims != ('index', 'y', 'x'):
        raise ValueError("da_parent dims should be ('index', 'y', 'x').")

    _, trans_child, _ = get_array_components(da_child)
    _, trans_parent, _ = get_array_components(da_parent)
    width_child, height_child = da_child.sizes['x'], da_child.sizes['y']
    width_parent, height_parent = da_parent.sizes['x'], da_parent.sizes['y']

    # a couple of ways to find the indices of the parent corresponding to the child
    # 1) Use labels
    query_res_lon = da_parent.xindexes['x'].sel({'x': da_child.x}, method="nearest", tolerance=1e-6)
    indices_x = query_res_lon.dim_indexers['x'].data
    query_res_lat = da_parent.xindexes['y'].sel({'y': da_child.y}, method="nearest", tolerance=1e-6)
    indices_y = query_res_lat.dim_indexers['y'].data    
    # 2) Use transform
    offset_x, offset_y = np.round(np.array(~trans_parent * trans_child * (0.5, 0.5)) - [0.5, 0.5])
    indices_x_2 = np.arange(offset_x, offset_x + width_child, dtype=int) % width_parent
    indices_y_2 = np.arange(offset_y, offset_y + height_child, dtype=int) % height_parent
    # we do both and check!
    if not np.array_equal(indices_x, indices_x_2) or not np.array_equal(indices_y, indices_y_2):
        raise ValueError("failed to find indices.") 

    zarr_parent.set_orthogonal_selection((parent_index, indices_y, indices_x), da_child.data[:, :])


def coords_from_extent(width: int, height: int, x_dim: str = "x", y_dim: str = "y"):
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return affine_to_coords(affine, width, height, x_dim, y_dim)


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


def get_array_components(da: xr.DataArray) -> Tuple[Any, Affine, Any]:
    data = da.data
    if 'lat' in da.dims and 'lon' in da.dims:
        renamed = da.rename({ "lat": "y", "lon": "x" })
    elif 'latitude' in da.dims and 'longitude' in da.dims:
        renamed = da.rename({ "latitude": "y", "longitude": "x" })
    elif 'x' not in da.dims or 'y' not in da.dims:
        raise ValueError('dimensions not recognised,')
    else:
        renamed = da
    transform = renamed.rio.transform(recalc=True)
    crs = da.rio.crs   
    if crs is None:
        # assumed default
        crs = rasterio.CRS.from_epsg(4326)
    return (data, transform, crs)


def write_array(array: xr.DataArray, path: str):
    array.rio.to_raster(raster_path=path, driver="COG")


def return_period_data_array(width: int, height: int, transform: Affine, crs: str="EPSG:4326", return_periods: List[float]=[0]):
    z_dim = "return_period"
    y_dim, x_dim = ("latitude", "longitude") if crs.upper() == "EPSG:4326" else ("y", "x")
    coords = affine_to_coords(transform, width, height, x_dim=x_dim, y_dim=y_dim)
    coords[z_dim] = np.array(return_periods, dtype=float) 
    data = dask.array.empty(shape=[len(return_periods), height, width])
    da = xr.DataArray(data=data, coords=coords, dims=[z_dim, y_dim, x_dim])
    return da


def _assert_transforms_consistent(trans1: Affine, trans2: Affine):
    """Check transforms from (x, y)/(lon, lat) to (col, row) are consistent. If same point maps
    to (col, row) offset by integer amount then a parent array can be assembled from offset childen
    with no reprojection
    """
    if not np.allclose((trans1.a, trans1.b, trans1.d, trans1.e), (trans2.a, trans2.b, trans2.d, trans2.e)):
        raise ValueError("transforms have inconsistent scaling.")
    if not np.allclose((trans1.c % 1 - trans2.c % 1, trans1.f % 1 - trans2.f % 1), (0.0, 0.0)):
        raise ValueError("transforms have non-integer offset.")