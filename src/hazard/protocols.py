from typing import Iterable, List, Protocol
import typing
import xarray as xr

class OpenDataset(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified"""
    def gcms() -> Iterable[str]:
        ...
    def open_dataset_year(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        ...

class WriteDataset(Protocol):
    """Open XArray Dataset for Global Circulation Model (GCM), scenario and quantity for whole year specified"""
    def write(self, path: str, dataset: xr.Dataset):
        ...
        
T = typing.TypeVar('T')

class PTransform(Protocol):
    
    def batch_items(self) -> Iterable[T]:
        ...
    def process_item(self, item: T) -> xr.DataArray:
        ...
    def item_path(self, item: T) -> str:
        ...
