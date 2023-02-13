from typing import Any, Dict, Generator, Tuple

from affine import Affine # type: ignore
import numpy as np
import rasterio, rioxarray # type: ignore
import xarray as xr


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


def coords_from_extent(width: int, height: int, x_dim: str = "x", y_dim: str = "y"):
    affine = Affine(2 * 180 / width, 0, -180.0, 0, -2 * 90 / height, 90)
    return affine_to_coords(affine, width, height, x_dim, y_dim)


def enforce_conventions(da: xr.DataArray) -> xr.DataArray:
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

