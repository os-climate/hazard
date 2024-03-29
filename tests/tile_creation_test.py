import os
from sys import stdout
from typing import List

import dask
import numpy as np
import pytest
import xarray as xr
import zarr.core  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays
from hazard.models.work_loss import WorkLossIndicator
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities
from hazard.utilities.tiles import create_tile_set, create_tiles_for_resource

from .utilities import test_output_dir


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
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard_test/test"))
    y = xr.open_zarr(store)
    y.dsm[0:256, 0:256] = np.random.randn(256, 256)
    y.to_zarr()


@pytest.mark.skip(reason="Example not test")
def test_map_tiles_from_model(test_output_dir):
    local_store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard_test", "hazard.zarr"))
    source = OscZarr(store=local_store)
    target = source

    models: List[IndicatorModel] = [
        WRIAqueductFlood(),
        DegreeDays(),
        Jupiter(),
        WorkLossIndicator(),
        DaysTasAboveIndicator(),
    ]
    for model in models:
        resources = model.inventory()
        # resources[0].
        for resource in resources:
            if resource.map is not None and resource.map.source == "map_array_pyramid":
                for resource in resource.expand():
                    create_tiles_for_resource(source, target, resource)


@pytest.mark.skip(reason="Requires mocking")
def test_convert_tiles(test_output_dir):
    """We are combining useful logic from a few sources.
    rio_tiler and titiler are very useful and also:
    https://github.com/mapbox/rio-mbtiles
    https://github.com/carbonplan/ndpyramid/
    (but we need slippy maps)
    https://github.com/DHI-GRAS/terracotta
    we need to align to:
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames.
    """

    zarr_utilities.set_credential_env_variables()
    id = "00000NorESM1-M"
    scenario = "rcp8p5"
    year = 2050
    path = f"inundation/wri/v2/inunriver_{scenario}_{id}_{year}"
    map_path = f"inundation/wri/v2/inunriver_{scenario}_{id}_{year}_map"

    copy_zarr_local(test_output_dir, path)

    local_store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard_test", "hazard.zarr"))
    source = OscZarr(store=local_store)
    target = source

    create_tile_set(source, path, target, map_path)


def copy_zarr_local(test_output_dir, path):
    local_store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard_test", "hazard.zarr"))
    dest = zarr.open_group(store=local_store, mode="r+")
    if path not in dest:
        source = OscZarr(bucket=os.environ["OSC_S3_BUCKET"])
        zarr.copy(source.read_zarr(path), dest, name=path, log=stdout)
