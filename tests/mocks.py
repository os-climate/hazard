"""Mocked sources, targets and datasets for testing indicator models."""

from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd  # type: ignore
import xarray as xr

from hazard.protocols import OpenDataset, ReadWriteDataArray


class TestSource(OpenDataset):
    """Mocked source for testing."""

    __test__ = False

    def __init__(
        self, datasets: Dict[Tuple[str, int], xr.Dataset], gcms: Iterable[str]
    ):
        self.datasets = datasets
        self._gcms = gcms

    def gcms(self) -> Iterable[str]:
        return self._gcms

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> xr.Dataset:
        return self.datasets[
            (quantity, year)
        ]  # ignore scenario and gcm: we test just a single one


class TestTarget(ReadWriteDataArray):
    """Mocked target for testing."""

    __test__ = False

    def __init__(self):
        self.datasets = {}

    def write(  # noqa:E704
        self,
        path: str,
        data_array: xr.DataArray,
        chunks: Optional[Sequence[int]] = None,
        spatial_coords: Optional[bool] = True,
    ):
        self.datasets[path] = data_array

    def read(self, path: str):
        return self.datasets[path].rename({"lat": "latitude", "lon": "longitude"})


def create_test_source_tas_tasmax_hurs(gcms: Iterable[str]) -> TestSource:
    """Mocked source with 'tas', 'tasmax' (3 K above 'tas') and 'hurs' for 2029 and 2030."""
    test_sets = create_test_datasets_tas(quantity="tas")
    test_sets.update(
        {
            ("tasmax", year): ds + 3.0
            for (_, year), ds in create_test_datasets_tas(quantity="tasmax").items()
        }
    )
    test_sets.update(create_test_datasets_hurs())
    return TestSource(test_sets, gcms)


def create_test_datasets_hurs(
    quantity: str = "hurs",
) -> Dict[Tuple[str, int], xr.Dataset]:
    return {
        (quantity, 2029): _create_test_dataset_hurs(2029, 0, quantity),
        (quantity, 2030): _create_test_dataset_hurs(2030, 0.5, quantity),
    }


def create_test_datasets_tas(
    quantity: str = "tasmax",
) -> Dict[Tuple[str, int], xr.Dataset]:
    return {
        (quantity, 2029): _create_test_dataset_tas(2029, 0, quantity),
        (quantity, 2030): _create_test_dataset_tas(2030, 0.5, quantity),
    }


def _create_test_dataset_tas(
    year: int, offset: float = 0, quantity: str = "tasmax"
) -> xr.Dataset:
    """Create test xarray Dataset.
    Convention is that data is arranged in image-like way:
    - dimensions are ('latitude', 'longitude')
    or, for time-series-type data: ('time', 'latitude', 'longitude')
    - latitude is decreasing

    Returns:
       xr.Dataset : test Dataset
    """
    temperature_t1 = (
        np.array([[293.0, 298.0, 310.0], [304.0, 302.0, 300.0], [308.0, 290.0, 294.0]])
        + offset
    )
    temperature_t2 = temperature_t1 + 1.0  # temp at t1 + 1 degree
    temperature_t3 = temperature_t2 + 2.0
    # stack to give 3 time points
    temperature = np.stack((temperature_t1, temperature_t2, temperature_t3), axis=0)
    # time = pd.date_range("2030-06-01", periods=3)
    time = pd.date_range(datetime(year, 1, 1), periods=3)
    lat = np.arange(3.0, 0.0, -1.0)
    lon = np.arange(0.0, 3.0, 1.0)
    ds = xr.Dataset(
        data_vars={
            quantity: (["time", "lat", "lon"], temperature),
        },
        coords=dict(time=time, lat=lat, lon=lon),
        attrs=dict(description="Test array"),
    )
    return ds


def _create_test_dataset_hurs(
    year: int, offset: float = 0, quantity="hurs"
) -> xr.Dataset:
    """Create test xarray Dataset.
    Convention is that data is arranged in image-like way:
    - dimensions are ('latitude', 'longitude')
    or, for time-series-type data: ('time', 'latitude', 'longitude')
    - latitude is decreasing

    Returns:
       xr.Dataset : test Dataset
    """
    hurs_t1 = (
        np.array([[70.0, 72.0, 69.0], [72.0, 71.0, 70.0], [80.0, 74.0, 68.0]]) + offset
    )
    hurs_t2 = hurs_t1 + 5.0  # hurs at t1 + 5%
    hurs_t3 = hurs_t2 + 10.0
    # stack to give 3 time points
    temperature = np.stack((hurs_t1, hurs_t2, hurs_t3), axis=0)
    # time = pd.date_range("2030-06-01", periods=3)
    time = pd.date_range(datetime(year, 1, 1), periods=3)
    lat = np.arange(3.0, 0.0, -1.0)
    lon = np.arange(0.0, 3.0, 1.0)
    ds = xr.Dataset(
        data_vars={
            quantity: (["time", "lat", "lon"], temperature),
        },
        coords=dict(time=time, lat=lat, lon=lon),
        attrs=dict(description="Test array"),
    )
    return ds
