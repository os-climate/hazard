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
        force_download: bool = False,
    ) -> None:
        """Create the source files in source_dir_base using abstract file system fs. Typically
        this might involve downloading, unzipping and rearranging the input files. The intent is
        to retain the data lineage.

        Args:
            download_dir: Path to local working directory for any temporary downloading and
                unzipping prior to copy to source directory.
            force_download: If True, forces download/copy of the source files even if they
                already exist.

        """
        ...

    @abstractmethod
    def is_prepared(self) -> bool:
        """Check if the source files are prepared and available in the source directory.

        This is a pure query: it reports actual on-disk state and takes no arguments to
        override that. A caller that wants to force re-preparation regardless of this
        result should do so itself (see onboard_hazards), rather than asking this method
        to misreport its answer.

        Returns:
            bool: True if the source files are prepared, False otherwise.
        """
        ...

    @abstractmethod
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create maps for the onboarded data."""
        pass
