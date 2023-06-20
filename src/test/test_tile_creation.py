import math
import os
from typing import Optional
from affine import Affine
import dask
import mercantile
import ndpyramid
import numpy as np
import rasterio
from ndpyramid import pyramid_coarsen, pyramid_reproject
import zarr.core
import rasterio.transform
from rasterio.warp import transform, transform_geom, Resampling
from rasterio import CRS, profiles # type: ignore
import xarray as xr
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities
from hazard.utilities import xarray_utilities
from hazard.utilities.map_utilities import highest_zoom_slippy_maps
from .utilities import test_output_dir
from hazard.utilities.xarray_utilities import get_array_components


def test_xarray_writing(test_output_dir):
    lat = np.arange(90, -90, -0.01)
    lon = np.arange(-180, 180, 0.01)

    x = xr.Dataset(
        coords={
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
        },
        data_vars={
            "dsm": (
                ["lat", "lon"],
                dask.array.empty((lat.size, lon.size), chunks=(1024, 1024), dtype="uint8"),
            )
        },
    )
    store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard_test/test'))
    y = xr.open_zarr(store)
    y.dsm[0:256, 0:256] = np.random.randn(256, 256)
    y.to_zarr()
    #x.to_zarr(store)
    #x.dsm[0:256, 0:256] = np.random.randn(256, 256)
    #print(x)


def test_convert_tiles(test_output_dir):
    zarr_utilities.set_credential_env_variables() 
    source = OscZarr()
    id = "00000NorESM1-M"
    scenario = "rcp8p5"
    year = 2050
    path = f"inundation/wri/v2/inunriver_{scenario}_{id}_{year}"
    da = source.read(path) #.to_dataset()
    src_indices, src_height, src_width = da.sizes['index'], da.sizes['latitude'], da.sizes['longitude']

    #a = xr.DataArray(float('nan'), coords=[('x', np.arange(40000)), ('y', np.arange(20000))])
    #a = xr.DataArray(0.5, coords=[('x', np.arange(3)), ('y', ['a', 'b'])])

    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard_test', 'hazard.zarr')))
    
    da_local = target.new_empty_like(path + "_map", da)
    #da_local[src_indices - 1, :, :] = da[src_indices - 1, :, :]
    da_local[8, 10, 10] = 8.3
    #da_local[8, 10, 10] = da[8, 10, 10]
    dask.compute(da_local)
    #da_local.compute()

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
    #src_left, src_bottom, src_right, src_top = src.rio.bounds()
    # we have to truncate into range -85 to 85 
    #src_bottom, src_top = max(src_bottom, -85.0), min(src_top, 85.0), 

    level = int(round(math.log2(src_width / 256.0)))
    pixels_per_tile = 256
    dst_dim = 2**level * pixels_per_tile

    ulx, uly = mercantile.xy(*mercantile.ul(mercantile.Tile(x=0, y=0, z=level)))
    lrx, lry = mercantile.xy(*mercantile.ul(mercantile.Tile(x=0+1, y=1+1, z=level)))

    da_local.load()

    dst_transform = rasterio.transform.from_bounds(
            ulx, lry, lrx, uly, pixels_per_tile, pixels_per_tile
        )

    #dst_transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale(
    #    (20026376.39 * 2) / dst_dim, -(20048966.10 * 2) / dst_dim
    #)

    da_m = da_local.rio.reproject(
            'EPSG:3857',
            resampling=Resampling.bilinear,
            shape=(pixels_per_tile, pixels_per_tile),
            transform=dst_transform,
        )

    chunks = {'x': pixels_per_tile, 'y': pixels_per_tile}
    da_m = da_m.chunk(chunks)

    pyramid = pyramid_reproject(ds, levels=2, pixels_per_tile=256)
    print(pyramid)




def create_empty_zarr_dataset_like(z: zarr.core.Array):
    
    t = z.attrs["transform_mat3x3"]  # type: ignore
    crs: str = z.attrs.get("crs", "EPSG:4326")  # type: ignore
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    #coords = xarray_utilities.affine_to_coords(transform, z.shape[2], z.shape[1], x_dim="dim_2", y_dim="dim_1")
    #data = dask.array.from_zarr(self.root.store, path)
    #array = xr.DataArray(data=a)
    #coords = xarray_utilities.affine_to_coords(transform, z.shape[2], z.shape[1], x_dim="longitude", y_dim="latitude")
    #da = xr.DataArray(dask.array.from_zarr(z), coords=coords)
    #da = xr.DataArray(None, coords=coords)
    #        data=(["latitude", "longitude"], None)), coords=coords)
    ##ds = xr.Dataset(data_vars=dict(
    #ds = ds.squeeze("dim_0")
    #if crs.upper() == "EPSG:4326":
    #    ds = ds.rename({ "dim_1": "latitude", "dim_2": "longitude" })
    #else:
    #    ds = ds.rename({ "dim_1": "y", "dim_2": "x" })
    
    return ds




def create_tiles(da: xr.DataArray, zoom_levels: Optional[str]):
    

    # We are combing useful logic from a few sources.
    # First of all 
    # https://github.com/mapbox/rio-mbtiles
    # and also
    # https://github.com/carbonplan/ndpyramid/
    # (but we need slippy maps)
    # https://github.com/DHI-GRAS/terracotta
    # also interesting

    # we need to align to
    # https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    src_bounds = da.rio.bounds()
    src_crs = da.rio.crs   

    (west, east), (south, north) = transform(
        src_crs, "EPSG:4326", src_bounds[::2], src_bounds[1::2]
    )

    # Resolve the minimum and maximum zoom levels for export.

    

    if zoom_levels:
        minzoom, maxzoom = map(int, zoom_levels.split(".."))
    else:
        zw = int(round(math.log(360.0 / (east - west), 2.0)))
        zh = int(round(math.log(170.1022 / (north - south), 2.0)))
        minzoom = min(zw, zh)
        maxzoom = max(zw, zh)

    #log.debug("Zoom range: %d..%d", minzoom, maxzoom)
    # Constrain bounds.
    EPS = 1.0e-10
    west = max(-180 + EPS, west)
    south = max(-85.051129, south)
    east = min(180 - EPS, east)
    north = min(85.051129, north)

    tiles = mercantile.tiles(
        west, south, east, north, range(minzoom, maxzoom + 1)   
    )

    import xarray as xr
    import rioxarray
    from ndpyramid import pyramid_coarsen, pyramid_reproject

    # load a sampel xarray.Dataset
    ds = xr.tutorial.load_dataset('air_temperature')

    # make a coarsened pyramid
    pyramid = pyramid_coarsen(ds, factors=[16, 8, 4, 3, 2, 1], dims=['lat', 'lon'], boundary='trim')

    # make a reprojected (EPSG:3857) pyramid
    ds = ds.rio.write_crs('EPSG:4326')
    pyramid = pyramid_reproject(ds, levels=2)

    # write the pyramid to zarr
    pyramid.to_zarr('./path/to/write')

    # for zarr, reproject all for each level
    
    #dim = 512
    #rasterio.transform.Affine.translation(-20026376.39, 20048966.10) * rasterio.transform.Affine.scale((20026376.39 * 2) / dim, -(20048966.10 * 2) / dim)



