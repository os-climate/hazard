import os
from typing import List, Optional

import numpy as np  # type: ignore
import xarray as xr
import zarr  # type: ignore

from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.xarray_utilities import (
    data_array,
    empty_data_array,
    global_crs_transform,
)

from .conftest import test_output_dir  # noqa:F401 used, it's a fixture


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


def test_xarray_write_net_cdf_coords(test_output_dir):  # noqa: F811
    """Test writing of XArrays with NetCDF-type style co-ordinates.
    Hazard indicators are generally maps of values, i.e. have two (spatial) dimensions, but often it is convenient
    to group a set of indicators (for a given scenario and year). For example, flood depths with different return
    periods or chronic indicators with different temperature thresholds and we want to be able look up maps using
    an index. To this end, an 'index' coordinate to the array is added.
    Physrisk requires an array with dimensions
    [index, "y", "x"] or
    [index, "latitude", "longitude"];
    the index co-ordinate could be for example "temperature", "flood_depth".
    """
    _, affine = global_crs_transform(3600, 1800)
    x = np.linspace(0, 1, 3600)
    y = np.linspace(0, 1, 1800)
    xx, yy = np.meshgrid(x, y)
    data = np.empty((2, 1800, 3600))
    data[0, :, :] = np.sin(xx) * np.sin(yy)
    data[1, :, :] = np.sin(xx) * np.sin(yy)
    da = data_array(
        data, affine, index_name="min_max", index_values=["min", "max"], name="sop"
    )
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    # to write to the dev bucket, use simply
    # target = OscZarr()
    target = OscZarr(store=store, store_netcdf_coords=True)
    target.write("test/test_netcdf_coords/sop", da, spatial_coords=True)

    z = target.read_zarr("test/test_netcdf_coords/sop/indicator")
    assert z.attrs["dimensions"][0] == "min_max"
    assert z.attrs["min_max_values"] == ["min", "max"]
    assert z.attrs["min_max_units"] == ""


def test_xarray_write_net_cdf_coords_huge(test_output_dir):  # noqa: F811
    """Do we need
    See distributed writes of:
    https://docs.xarray.dev/en/stable/user-guide/io.html

    Writing a large empty array requires a certain approach, add to protocol?
    da.to_dataset(name="flood_sop").to_zarr(store=store,
                                        group="inundation/flopros_riverine//v1/flood_sop",
                                        compute=False,
                                        mode="w",
                                        encoding={"flood_sop" : {
                                                 "chunks" : (1, 1000, 1000),
                                                 "write_empty_chunks": False,
                                             }
                                        })

    """
    ...
