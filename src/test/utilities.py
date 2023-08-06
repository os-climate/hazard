import os
import pytest
from datetime import datetime
from typing import Dict, Tuple
import pytest
from pytest import approx

from hazard.protocols import OpenDataset, WriteDataset
import numpy as np
import pandas as pd # type: ignore
import xarray as xr


def working_directory():
    pass


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir
    # if we want to clean-up (most likely not)
    #shutil.rmtree(output_dir)


class TestSource(OpenDataset):
    """Mocked source for testing."""
    def __init__(self, datasets: Dict[Tuple[str, int], xr.Dataset]):
        self.datasets = datasets
    
    def open_dataset_year(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        return self.datasets[(quantity, year)] # ignore scenario and gcm: we test just a single one


class TestTarget(WriteDataset):
    """Mocked target for testing."""
    def __init__(self):
        self.datasets = {}

    def write(self, path: str, dataset: xr.Dataset):
        self.datasets[path] = dataset

    def read(self, path: str):
        return self.datasets[path].rename({ "lat": "latitude", "lon": "longitude" })


def _create_test_datasets_hurs(quantity: str = "hurs") -> Dict[Tuple[str, int], xr.Dataset]:
    return { (quantity, 2029): _create_test_dataset_hurs(2029, 0, quantity), (quantity, 2030): _create_test_dataset_hurs(2030, 0.5, quantity) }


def _create_test_datasets_tas(quantity: str = "tasmax") -> Dict[Tuple[str, int], xr.Dataset]:
    return { (quantity, 2029): _create_test_dataset_tas(2029, 0, quantity), (quantity, 2030): _create_test_dataset_tas(2030, 0.5, quantity) }


def _create_test_dataset_averaged() -> xr.Dataset:
    """An example 3x3 array that might result from some operation averaging over time."""
    temperature = np.array([
        [293., 298., 310.], 
        [304., 302., 300.],
        [308., 290., 294.]])  
    lat = np.arange(3., 0., -1.)
    lon = np.arange(0., 3., 1.)
    ds = xr.Dataset(
        data_vars=dict(
            tasmax=(["lat", "lon"], temperature),
        ),
        coords=dict(
            lat=lat,
            lon=lon
        ),
        attrs=dict(description="Test array"),
    )
    return ds


def _create_test_dataset_tas(year: int, offset: float=0, quantity: str = "tasmax") -> xr.Dataset:
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
    time = pd.date_range(datetime(year, 1, 1), periods=3)    
    lat = np.arange(3., 0., -1.)
    lon = np.arange(0., 3., 1.)
    ds = xr.Dataset(
        data_vars={
            quantity: (["time", "lat", "lon"], temperature),
        },
        coords=dict(
            time=time,
            lat=lat,
            lon=lon
        ),
        attrs=dict(description="Test array"),
    )
    return ds

def _create_test_dataset_hurs(year: int, offset: float=0, quantity="hurs") -> xr.Dataset:
    """Create test xarray Dataset.
    Convention is that data is arranged in image-like way:
    - dimensions are ('latitude', 'longitude')
    or, for time-series-type data: ('time', 'latitude', 'longitude')
    - latitude is decreasing

    Returns:
       xr.Dataset : test Dataset
    """
    hurs_t1 = np.array([
        [70., 72., 69.], 
        [72., 71., 70.],
        [80., 74., 68.]]) + offset 
    hurs_t2 = hurs_t1 + 5. # hurs at t1 + 5% 
    hurs_t3 = hurs_t2 + 10.
    # stack to give 3 time points
    temperature = np.stack((hurs_t1, hurs_t2, hurs_t3), axis=0)
    #time = pd.date_range("2030-06-01", periods=3) 
    time = pd.date_range(datetime(year, 1, 1), periods=3)    
    lat = np.arange(3., 0., -1.)
    lon = np.arange(0., 3., 1.)
    ds = xr.Dataset(
        data_vars={
            quantity: (["time", "lat", "lon"], temperature),
        },
        coords=dict(
            time=time,
            lat=lat,
            lon=lon
        ),
        attrs=dict(description="Test array"),
    )
    return ds