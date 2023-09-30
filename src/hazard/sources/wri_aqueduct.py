
from contextlib import contextmanager
import logging
from pathlib import PurePosixPath
from typing import Generator, Iterable, Optional
from hazard.protocols import OpenDataset
from botocore import UNSIGNED # type: ignore
import s3fs # type: ignore
import xarray as xr

logger = logging.getLogger(__name__)

class WRIAqueductSource: #(OpenDataset):
    def __init__(self):
        self.fs = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))
        self.prefix = "wri-projects/AqueductFloodTool/download/v2"

    def gcms(self) -> Iterable[str]:
        raise NotImplementedError()


    @contextmanager
    def open_dataset(self, path: str) -> Generator[Optional[xr.DataArray], None, None]: 
        logger.info(f"Opening DataArray, relative path={path}")
        da: Optional[xr.DataArray] = None
        f = None
        try:
            f = self.fs.open(str(PurePosixPath(self.prefix, path)) + ".tif", 'rb')
            da = xr.open_rasterio(f)
            yield da
        finally:
            if da is not None:
                da.close()
            if f is not None:
                f.close()

    