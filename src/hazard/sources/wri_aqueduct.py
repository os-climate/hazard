"""."""

import logging
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing_extensions import Generator, Iterable, Optional

import s3fs  # type: ignore
import xarray as xr
from botocore import UNSIGNED  # type: ignore

logger = logging.getLogger(__name__)


class WRIAqueductSource:  # (OpenDataset):
    """Represents a data source for the WRI Aqueduct Flood Tool.

    This class provides methods for accessing datasets stored in an S3 bucket
    using an anonymous (unsigned) connection.

    Attributes
        fs (s3fs.S3FileSystem): Filesystem object for accessing S3 storage.
        prefix (str): Base path for datasets in the S3 bucket.

    """

    def __init__(self):
        """Initialize the WRI Aqueduct data source.

        Sets up an S3 filesystem connection with unsigned access
        and defines the base path for datasets.
        """
        self.fs = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))
        self.prefix = "wri-projects/AqueductFloodTool/download/v2"

    def gcms(self) -> Iterable[str]:
        """Retrieve the list of general circulation models (GCMs)."""
        raise NotImplementedError()

    @contextmanager
    def open_dataset(self, path: str) -> Generator[Optional[xr.DataArray], None, None]:
        """Open a dataset from the S3 storage and loads it as an xarray DataArray.

        This method opens a `.tif` file from the predefined S3 path and loads it
        as an xarray `DataArray`. It ensures proper handling of file resources.

        Args:
            path (str): The relative path of the dataset within the S3 storage.

        """
        logger.info(f"Opening DataArray, relative path={path}")
        da: Optional[xr.DataArray] = None
        f = None
        try:
            f = self.fs.open(str(PurePosixPath(self.prefix, path)) + ".tif", "rb")
            da = xr.open_dataarray(f, engine="rasterio")  # type: ignore[attr-defined]
            yield da
        finally:
            if da is not None:
                da.close()
            if f is not None:
                f.close()
