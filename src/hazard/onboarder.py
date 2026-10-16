# Standard library imports
import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional, TypeVar


from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem


# Local application imports
from hazard.inventory import HazardResource
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr


logger = logging.getLogger(__name__)


T = TypeVar("T")


class Onboarder(ABC):
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
    def prepare(
        self,
        download_dir: Optional[str] = None,
        force: bool = False,
        force_download: bool = False,
    ) -> None:
        """Create the source files in source_dir_base using abstract file system fs. Typically
        this might involve downloading, unzipping and rearranging the input files. The intent is
        to retain the data lineage.

        Args:
            download_dir: Path to local working directory for any temporary downloading and
                unzipping prior to copy to source directory.
            force: If True, forces preparation even if the source files already appear ready.
            force_download: If True, forces download of the source files even if they exist.

        """
        ...

    @abstractmethod
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the source files are prepared and available in the source directory.

        Returns:
            bool: True if the source files are prepared, False otherwise.
        """
        ...

    @abstractmethod
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create maps for the onboarded data."""
        pass
