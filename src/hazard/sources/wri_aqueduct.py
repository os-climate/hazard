
from pathlib import PurePosixPath
from typing import Iterable
from hazard.protocols import OpenDataset
from botocore import UNSIGNED # type: ignore
import s3fs, fsspec # type: ignore
import xarray as xr

class NexGddpCmip6(OpenDataset):
    def __init__(self):
        self.fs = s3fs.S3FileSystem(config_kwargs=dict(signature_version=UNSIGNED))
        self.prefix = "AqueductFloodTool/download/v2"

    def gcms(self) -> Iterable[str]:
        ...
        
    def open_dataset_year(self, gcm: str, scenario: str, quantity: str, year: int, chunks=None) -> xr.Dataset:
        ...

    def path_riverine(self, gcm: str, scenario: str, year: int, return_period: int):
        pp = PurePosixPath(self.prefix, f"inunriver_{scenario}_{gcm}_{year}_rp{return_period:05d}")
        return str(pp)
    