from typing import Generator, Iterable, Iterator, List, Optional, Protocol

import xarray as xr


class Averageable(Protocol):
    """Write DataArray."""

    central_year: int


class OpenDataset(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified."""

    def gcms(self) -> Iterable[str]: ...  # noqa:E704

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> Iterator[xr.Dataset]:
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
