
from abc import ABC, abstractmethod
import logging
from typing import Any, Generic, Iterable, List, Optional, TypeVar
from dask.distributed import Client, LocalCluster
from pydantic import BaseModel, Field
from hazard.inventory import HazardResource
from hazard.protocols import OpenDataset, ReadWriteDataArray, WriteDataArray, WriteDataset

logger = logging.getLogger(__name__)

T = TypeVar('T')      # Declare type variable

class IndicatorModel(ABC, Generic[T]):
    """Generates a set of hazard indicators."""
        
    def run_all(self, source: OpenDataset, target: ReadWriteDataArray, client: Optional[Client]=None):
        """Run all items in the batch."""
        if (client is None):
            cluster = LocalCluster(processes=False)
            client = Client(cluster)
        for item in self.batch_items():
            try:
                self.run_single(item, source, target, client)
            except:
                logger.error("Batch item failed", exc_info=True)

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get a list of all batch items."""
        ...

    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        ...
    
    @abstractmethod
    def run_single(self, item: T, source: Any, target: ReadWriteDataArray, client: Client):
        """Run a single item of the batch."""
        ...
