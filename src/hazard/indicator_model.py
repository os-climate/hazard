
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, List, Optional, TypeVar
from dask.distributed import Client, LocalCluster
from pydantic import BaseModel, Field
from hazard.inventory import HazardModel
from hazard.protocols import OpenDataset, WriteDataArray, WriteDataset

T = TypeVar('T')      # Declare type variable

class IndicatorModel(ABC, Generic[T]):
    """Generates a set of hazard indicators."""
        
    def run_all(self, source: OpenDataset, target: WriteDataArray, client: Optional[Client]=None):
        """Run all items in the batch."""
        if (client is None):
            cluster = LocalCluster(processes=False)
            client = Client(cluster)
        for item in self.batch_items():
            self.run_single(item, source, target, client)

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get a list of all batch items."""
        ...

    @abstractmethod
    def inventory(self) -> Iterable[HazardModel]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        ...
    
    @abstractmethod
    def run_single(self, item: T, source: Any, target: WriteDataArray, client: Client):
        """Run a single item of the batch."""
        ...
