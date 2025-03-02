from abc import ABC, abstractmethod
import logging
from pathlib import Path, PurePath
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, Optional, Union

from hazard.inventory import HazardResource
from hazard.protocols import WriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class Onboarder(ABC):
    """Onboard a hazard indicator data set into physrisk. Typically this is a pre-existing data set
    which requires a transformation which might be a simple conversion into physrisk conventions
    or something more complex.
    """

    def __init__(
        self,
        source_dir_base: Union[str, PurePath] = PurePath(),
        fs: Optional[AbstractFileSystem] = None,
    ):
        """Create Onboarder instance. The original file set is the starting point on the
        onboarding. This is stored in the (abstract) file system specified.

        Args:
            source_dir_base (Union[str, PurePath], optional): Base path of the source files.
            fs (Optional[AbstractFileSystem], optional): File system for storing source files.
            Defaults to None.
        """
        self.source_dir = self.source_dir_from_base(
            Path(source_dir_base)
            if isinstance(source_dir_base, str)
            else source_dir_base
        )
        self.fs = fs if fs else LocalFileSystem()

    def create_maps(self, source: OscZarr, target: OscZarr):
        for resource in self.inventory():
            create_tiles_for_resource(
                source,
                target,
                resource,
                nodata_as_zero=True,
                nodata_as_zero_coarsening=True,
            )

    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Return resource(s) to add to Inventory."""
        ...

    def is_source_dir_populated(self) -> bool:
        """Return True if source_dir is already populated."""
        return self.fs.exists(self.source_dir) and any(self.fs.ls(self.source_dir))

    @abstractmethod
    def onboard(self, target: WriteDataArray):
        """Onboard the data, reading from the file source and writing to the target provided.

        Args:
            target (WriteDataArray): Hazard indicators are written to this target.
        """
        ...

    @abstractmethod
    def prepare(self, working_dir: Path, force_download: bool = True):
        """Create the source files in source_dir_base using abstract file system fs. Typically
        this might involve downloading, unzipping and rearranging the input files. The intent is
        to retain the data lineage.

        Args:
            working_dir (Path): Path to local working directory for any temporary downloading and unzipping prior to
            copy to source directory.
            force_download (bool, optional): If False and feature is supported, files will not be re-downloaded. Defaults to True.
        """
        ...

    @abstractmethod
    def source_dir_from_base(self, source_dir_base: PurePath) -> PurePath:
        """Return source_dir relative to source_dir_base."""
        ...
