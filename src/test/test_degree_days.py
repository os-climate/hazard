from datetime import datetime
import logging, os, sys
import logging.handlers
from typing import Dict
import pytest
from pytest import approx

import fsspec.implementations.local as local
from hazard.map_builder import MapBuilder
from hazard.protocols import OpenDataset, WriteDataset
import hazard.utilities.zarr_utilities as zarr_utilities
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.models.degree_days import BatchItem, DegreeDays
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr


class TestSource(OpenDataset):
    """Mocked source for testing."""
    def __init__(self, datasets: Dict[int, xr.Dataset]):
        self.datasets = datasets
    
    def open_dataset_year(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        return self.datasets[year]

class TestTarget(WriteDataset):
    """Mocked target for testing."""
    def write(self, path: str, dataset: xr.Dataset):
        self.dataset = dataset


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir
    # if we want to clean-up (most likely not)
    #shutil.rmtree(output_dir)


def _create_test_datasets() -> Dict[int, xr.Dataset]:
    return { 2029: _create_test_dataset(2029), 2030: _create_test_dataset(2030, 0.5) }


def _create_test_dataset(year: int, offset: float=0) -> xr.Dataset:
    """Create test xarray Dataset.
    Convention is that data is arranged in image-like way:
    - dimensions are ('latitude', 'longitude')
    or, for time-series-type data: ('time', 'latitude', 'longitude')
    - latitude is decreasing

    Returns:
       xr.Dataset : test Dataset
    """
    temperature_t1 = np.array([
        [293., 298., 310.], 
        [304., 302., 300.],
        [308., 290., 294.]]) + offset 
    temperature_t2 = temperature_t1 + 1. # temp at t1 + 1 degree 
    temperature_t3 = temperature_t2 + 2.
    # stack to give 3 time points
    temperature = np.stack((temperature_t1, temperature_t2, temperature_t3), axis=0)
    #time = pd.date_range("2030-06-01", periods=3) 
    time = pd.date_range(datetime(year, 6, 1), periods=3)    
    lat = np.arange(3., 0., -1.)
    lon = np.arange(0., 3., 1.)
    ds = xr.Dataset(
        data_vars=dict(
            tasmax=(["time", "lat", "lon"], temperature),
        ),
        coords=dict(
            time=time,
            lat=lat,
            lon=lon
        ),
        attrs=dict(description="Test array"),
    )
    return ds


def test_degree_days_mocked():
    """Test degree days calculation based on mocked data."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = TestSource(_create_test_datasets())
    target = TestTarget()
    # cut down the transform
    model = DegreeDays(window_years=2, gcms=[gcm], scenarios=[scenario], central_years=[year])  
    model.run(source, target)
    with source.open_dataset_year(gcm, scenario, "tasmax", 2029) as y0:
        with source.open_dataset_year(gcm, scenario, "tasmax", 2030) as y1:
            scale = 365.0 / len(y0.time)
            deg0 = scale * xr.where(y0.tasmax > (32 + 273.15), y0.tasmax - (32 + 273.15), 0).sum(dim=["time"])
            deg1 = scale * xr.where(y1.tasmax > (32 + 273.15), y1.tasmax - (32 + 273.15), 0).sum(dim=["time"])
            expected = (deg0 + deg1) / 2 
    assert expected.values == approx(target.dataset.values)

    #with source.open_dataset_year(gcm, scenario, "tasmax", year) as ds:
    #    store = zarr.DirectoryStore(os.path.join(working_dir, 'hazard_test2', 'hazard.zarr'))
    #    ds.to_zarr(store, compute=True, group="test_name", mode="w")

    #result.to_zarr()
    #map_builder=MapBuilder(zarr_store, working_directory=working_dir)


#@pytest.mark.skip(reason="inputs large and downloading slow")
def test_degree_days(test_output_dir):
    """Cut-down but still *slow* test that performs downloading of real datasets."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    years = [2029, 2030]
    download_test_datasets(test_output_dir, gcm, scenario, years)
    # source: read downloaded datasets from local file system
    fs = local.LocalFileSystem()
    source = NexGddpCmip6(root=os.path.join(test_output_dir, NexGddpCmip6.bucket), fs=fs)
    # target: write zarr to load fine system
    store = zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr'))
    target = OscZarr(store=store)
    # cut down the model and run
    model = DegreeDays(window_years=2, gcms=[gcm], scenarios=[scenario], central_years=[years[1]])
    model.run(source, target)
    # check one point...
    path = model._item_path(BatchItem(gcm, scenario, years[1]))
    calculated = target.read_floored(path, [32.625], [15.625])
    # against expected:
    with source.open_dataset_year(gcm, scenario, "tasmax", years[0]) as y0:
        with source.open_dataset_year(gcm, scenario, "tasmax", years[1]) as y1:
            assert y0.lat[302].values == approx(15.625)
            assert y0.lon[130].values == approx(32.625)
            scale = 365.0 / len(y0.time)
            y0p, y1p = y0.tasmax[:, 302, 130].values, y1.tasmax[:, 302, 130].values
            deg0 = scale * xr.where(y0p > (32 + 273.15), y0p - (32 + 273.15), 0).sum()
            deg1 = scale * xr.where(y1p > (32 + 273.15), y1p - (32 + 273.15), 0).sum()
            expected = (deg0 + deg1) / 2 
    assert calculated == approx(expected)


def example_run_degree_days():
    zarr_utilities.set_credential_env_variables()  
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = NexGddpCmip6()
    target = OscZarr(prefix="hazard_test") # test prefix is "hazard_test"; main one "hazard"
    # cut down the transform
    model = DegreeDays(window_years=1, gcms=[gcm], scenarios=[scenario], central_years=[year])
    model.run(source, target)
    assert True


def download_test_datasets(test_output_dir, gcm, scenario, years):
    store = NexGddpCmip6()
    s3 = s3fs.S3FileSystem(anon=True)
    for year in years:
        path, _ = store.path(gcm, scenario, "tasmax", year)
        if not os.path.exists(os.path.join(test_output_dir, path)):
            s3.download(path, os.path.join(test_output_dir, path))
    assert True


@pytest.mark.skip(reason="just example")
def test_load_dataset(test_output_dir):    
    fs = local.LocalFileSystem()
    store = NexGddpCmip6(root=os.path.join(test_output_dir, "nex-gddp-cmip6"), fs=fs)
    with store.open_dataset_year("NorESM2-MM", "ssp585", "tasmax", 2030) as ds:
        print(ds)
    assert True
