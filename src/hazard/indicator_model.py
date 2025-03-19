"""Module for the abstract base class `IndicatorModel` which defines the interface for hazard indicator models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Optional, TypeVar

from dask.distributed import Client, LocalCluster

from hazard.inventory import HazardResource
from hazard.protocols import OpenDataset, ReadWriteDataArray

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Declare type variable


class IndicatorModel(ABC, Generic[T]):
    """Generates a set of hazard indicators."""

    def run_all(
        self,
        source: OpenDataset,
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

    def onboard_all(
        self,
        target: ReadWriteDataArray,
        download_dir: Optional[str] = None,
        force_prepare: bool = False,
        force_download: bool = False,
    ):
        """Onboard a set of hazards."""
        self.onboard_single(
            target,
            force_prepare=force_prepare,
            download_dir=download_dir,
            force_download=force_download,
        )

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get a list of all batch items."""
        ...

    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        ...

    @abstractmethod
    def prepare(
        self,
        force: Optional[bool],
        download_dir: Optional[str],
        force_download: Optional[bool],
    ):
        """Prepare the baseline and future projection data by downloading or extracting it.

        This method ensures that the required data is available in `self.source_dir`. If the data
        does not already exist, is incomplete, or if `force_download` is set to True, the method
        downloads the ZIP file from `self.zip_url`, extracts its contents to `self.source_dir`,
        and removes the ZIP file after extraction.

        Args:
            force (bool, optional): If True, forces the recreation of `self.source_dir` even if it exists.
                Defaults to False.
            download_dir (str, optional): The directory where the ZIP file will be temporarily downloaded
                before extraction. If None, a default location is used. Defaults to None.
                It is not always necessary to provide a `download_dir`.
            force_download (bool, optional): If True, forces downloading the ZIP file even if the data
                already exists in `self.source_dir`. Defaults to False.

        Notes:
            - The 'force_download' parameter is sometimes ignored because the data must
            already be present in the directory.

        Raises:
            PermissionError: If the process does not have permissions to write to `self.source_dir`.
            RuntimeError: If the download or extraction fails due to unexpected issues.

        """
        ...

    @abstractmethod
    def onboard_single(
        self,
        target: ReadWriteDataArray,
        force_prepare: Optional[bool],
        download_dir: Optional[str],
        force_download: Optional[bool],
    ) -> Iterable[HazardResource]:
        """Run onboard for a given hazard."""
        ...

    @abstractmethod
    def run_single(
        self, item: T, source: Any, target: ReadWriteDataArray, client: Client
    ):
        """Run a single item of the batch."""
        ...
