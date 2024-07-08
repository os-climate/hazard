import os
from sys import stdout

import numpy as np
import pytest
import rasterio.transform
import rasterio.warp
import xarray as xr
import zarr.core  # type: ignore
from rasterio.warp import Resampling

from hazard.onboard.tudelft_flood import TUDelftRiverFlood
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities
from hazard.utilities.tiles import create_tile_set, create_tiles_for_resource

from .conftest import test_output_dir  # noqa: F401


def test_convert_tiles_mocked(test_output_dir):  # noqa: F811
    """We are combining useful logic from a few sources.
    rio_tiler and titiler are very useful and also:
    https://github.com/mapbox/rio-mbtiles
    https://github.com/carbonplan/ndpyramid/
    (but we need slippy maps)
    https://github.com/DHI-GRAS/terracotta
    we need to align to:
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames.
    """

    # create rectangular are in EPSG:3035 and check the projection and
    # downscaling of this
    # bounds of the rectangular region in latitude / longitude
    feat_left, feat_bottom, feat_right, feat_top = 5, 51, 6, 52
    left, bottom, right, top = -10, 40, 30, 55
    step = 0.01
    lon = np.linspace(left + step / 2, right - step / 2, num=int((right - left) / 0.01))
    lat = np.linspace(bottom + step / 2, top - step / 2, num=int((top - bottom) / 0.01))
    da = xr.DataArray(
        data=np.zeros((3, len(lat), len(lon))),
        coords=dict(index=np.array([0, 1, 2]), y=lat[::-1], x=lon),
        attrs=dict(description="Test array"),
    )
    np.testing.assert_array_equal(da.rio.bounds(), [-10, 40, 30, 55])
    da.sel(index=2, x=slice(feat_left, feat_right), y=slice(feat_top, feat_bottom))[
        :, :
    ] = 1.0
    da.rio.write_crs(4326, inplace=True)
    t_left, t_bottom, t_right, t_top = rasterio.warp.transform_bounds(
        "EPSG:4326", "EPSG:3035", left, bottom, right, top
    )
    dst_transform = rasterio.transform.from_bounds(
        t_left, t_bottom, t_right, t_top, len(lon), len(lat)
    )
    da_3035 = da.rio.reproject(
        "EPSG:3035",
        shape=(len(lat), len(lon)),
        transform=dst_transform,
        resampling=Resampling.nearest,
        nodata=float("nan"),
    )
    da_3035.assign_coords(index=("index", [0, 1, 2]))
    assert da_3035[2, :, :].max().values == 1
    local_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard", "hazard.zarr")
    )
    source = OscZarr(store=local_store)
    source.write("test_set/test0", da_3035)
    create_tile_set(source, "test_set/test0", source, "test_set/test0_map", indices=[2])
    da_map = source.read("test_set/test0_map/2")
    # find the location that should contain the region
    m_left, m_bottom, m_right, m_top = rasterio.warp.transform_bounds(
        "EPSG:4326", "EPSG:3857", feat_left, feat_bottom, feat_right, feat_top
    )
    mapped_region = da_map.sel(
        index=2, x=slice(m_left, m_right), y=slice(m_top, m_bottom)
    )
    # this should contain our original feature
    assert mapped_region.max().values == 1


@pytest.mark.skip(reason="Example not test")
def test_map_tiles_from_model(test_output_dir):  # noqa: F811
    local_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard", "hazard.zarr")
    )
    source = OscZarr(store=local_store)
    target = source

    models = [
        TUDelftRiverFlood(None),
        # WRIAqueductFlood(),
        # DegreeDays(),
        # Jupiter(),
        # WorkLossIndicator(),
        # DaysTasAboveIndicator(),
    ]
    for model in models:
        resources = model.inventory()
        # resources[0].
        for resource in resources:
            if resource.map is not None and resource.map.source == "map_array_pyramid":
                for resource in resource.expand():
                    create_tiles_for_resource(source, target, resource)


@pytest.mark.skip(reason="Requires mocking")
def test_convert_tiles(test_output_dir):  # noqa: F811
    zarr_utilities.set_credential_env_variables()
    id = "00000NorESM1-M"
    scenario = "rcp8p5"
    year = 2050
    path = f"inundation/wri/v2/inunriver_{scenario}_{id}_{year}"
    map_path = f"inundation/wri/v2/inunriver_{scenario}_{id}_{year}_map"
    # copy_zarr_local(test_output_dir, path)
    local_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard", "hazard.zarr")
    )
    source = OscZarr(store=local_store)
    target = source

    path = "inundation/river_tudelft/v2/flood_depth_historical_1971"
    map_path = "inundation/river_tudelft/v2/flood_depth_historical_1971_map"
    create_tile_set(source, path, target, map_path, max_zoom=10)


def copy_zarr_local(test_output_dir, path):  # noqa: F811
    local_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard_test", "hazard.zarr")
    )
    dest = zarr.open_group(store=local_store, mode="r+")
    if path not in dest:
        source = OscZarr(bucket=os.environ["OSC_S3_BUCKET"])
        zarr.copy(source.read_zarr(path), dest, name=path, log=stdout)
