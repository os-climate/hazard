from abc import ABC, abstractmethod
import logging
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, Optional

from hazard.inventory import HazardResource
from hazard.protocols import WriteDataArray

logger = logging.getLogger(__name__)


class Onboarder(ABC):
    """Onboard a hazard indicator data set into physrisk. Typically this is a pre-existing data set
    which requires a transformation which might be a simple conversion into physrisk conventions
    or something more complex.
    """

    def __init__(self, source_dir_base: str = "", fs: Optional[AbstractFileSystem] = None):
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
    def onboard(self, target: WriteDataArray):
        """Onboard the data, reading from the file source and writing to the target provided.

        Args:
            target (WriteDataArray): Hazard indicators are written to this target.
        """
        ...

    def prepare(self, working_dir: str):
        """Create the source files in source_dir_base using abstract file system fs.

        Args:
            working_dir (str): Path to local working directory for any temporary downloading and unzipping prior to
            copy to source directory.
        """
        ...
