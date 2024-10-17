from abc import ABC, abstractmethod
import logging
from dask.distributed import LocalCluster, Client
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Any, Generic, Iterable, Optional, TypeVar

from hazard.inventory import HazardResource
from hazard.protocols import OpenDataset, ReadWriteDataArray, WriteDataArray

from abc import ABC, abstractmethod
import logging
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, Optional

from hazard.inventory import HazardResource
from hazard.protocols import WriteDataArray

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Onboarder(ABC, Generic[T]):
    """Onboard a hazard indicator data set into physrisk. Typically this is a pre-existing data set
    which requires a transformation which might be a simple conversion into physrisk conventions
    or something more complex.
    """

    def __init__(
        self, source_dir_base: str = "", fs: Optional[AbstractFileSystem] = None
    ):
        """Create Onboarder instance. The original file set is the starting point on the
        onboarding. This is stored in the (abstract) file system specified.

        Args:
            source_dir_base (str, optional): Base path of the source files. Defaults to "".
            fs (Optional[AbstractFileSystem], optional): File system for storing source files.
            Defaults to None.
        """
        self.source_dir_base = source_dir_base
        self.fs = fs if fs else LocalFileSystem()

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get a list of all batch items."""
        ...

    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Return resource(s) to add to Inventory."""
        ...

    @abstractmethod
    def onboard(self, target: ReadWriteDataArray):
        """Onboard the data, reading from the file source and writing to the target provided.

        Args:
            target (WriteDataArray): Hazard indicators are written to this target.
        """
        ...

    @abstractmethod
    def prepare(self, source_dir_base: str):
        """Create the source files in source_dir_base using abstract file system fs. Typically
        this might involve downloading, unzipping and rearranging the input files. The intent is
        to retain the data lineage.

        Args:
            working_dir (str): Path to local working directory for any temporary downloading and unzipping prior to
            copy to source directory.
        """
        ...

    @abstractmethod
    def run_single(
        self, item: T, source: Any, target: ReadWriteDataArray, client: Client
    ):
        """Run a single item of the batch."""
        ...

    def run_all(
        self,
        source: Any,
        target: ReadWriteDataArray,
        client: Optional[Client] = None,
        debug_mode=False,
    ):
        """Run all items in the batch."""
        if client is None:
            cluster = LocalCluster(processes=False)
            client = Client(cluster)
        if debug_mode:
            for item in self.batch_items():
                self.run_single(item, source, target, client)
        else:
            for item in self.batch_items():
                try:
                    self.run_single(
                        item=item, source=source, target=target, client=client
                    )
                except Exception:
                    logger.error("Batch item failed", exc_info=True)
