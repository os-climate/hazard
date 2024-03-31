import os

import numpy as np  # type: ignore
import zarr  # type: ignore
from affine import Affine

from hazard.onboard.storm_wind import BatchItem, STORMIndicator  # type: ignore
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.xarray_utilities import data_array, empty_data_array, global_crs_transform

from .conftest import test_output_dir


def test_xarray_write_small(test_output_dir):  # noqa: F811
    _, affine = global_crs_transform(3600, 1800)
    da = empty_data_array(3600, 1800, affine)
    x = np.linspace(0, 1, 3600)
    y = np.linspace(0, 1, 1800)
    xx, yy = np.meshgrid(x, y)
    data = np.expand_dims(np.sin(xx) * np.sin(yy), 0)
    da = data_array(data, affine)
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    # to write to the dev bucket, use simply
    # target = OscZarr()
    target = OscZarr(store=store)
    target.write("test/test_small", da)
