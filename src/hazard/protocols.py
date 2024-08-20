import typing
from typing import Iterable, List, Optional, Protocol

import xarray as xr


class Averageable(Protocol):
    """Write DataArray."""

    central_year: int


class OpenDataset(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified."""

    def gcms(self) -> Iterable[str]: ...  # noqa:E704

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ):  # -> Generator[xr.Dataset, None, None]:
        ...


class ReadDataArray(Protocol):
    """Read DataArray."""

    def read(self, path: str) -> xr.DataArray: ...  # noqa:E704


class WriteDataArray(Protocol):
    """Write DataArray."""

    def write(  # noqa:E704
        self,
        path: str,
        data_array: xr.DataArray,
        chunks: Optional[List[int]] = None,
        spatial_coords: Optional[bool] = True,
    ): ...


class ReadWriteDataArray(ReadDataArray, WriteDataArray): ...  # noqa: E701


class WriteDataset(Protocol):
    """Write DataArray."""

    def write(self, path: str, dataset: xr.Dataset): ...  # noqa:E704


T = typing.TypeVar("T")


class PTransform(Protocol):
    def batch_items(self) -> Iterable[T]: ...  # noqa:E704

    def process_item(self, item: T) -> xr.DataArray: ...  # noqa:E704

    def item_path(self, item: T) -> str: ...  # noqa:E704
