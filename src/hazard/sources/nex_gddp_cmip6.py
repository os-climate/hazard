from contextlib import contextmanager
from dataclasses import dataclass
import logging, os
import posixpath
from typing import Dict, Generator, List, Optional

import s3fs, fsspec  # type: ignore
import xarray as xr

from hazard.protocols import OpenDataset

logger = logging.getLogger(__name__)


@dataclass
class Cmip6Item:
    gcm: str
    variant_id: str


class NexGddpCmip6(OpenDataset):
    """Source class for loading in data from
    NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6)
    https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
    """

    bucket: str = "nex-gddp-cmip6"

    def __init__(
        self,
        root: Optional[str] = None,
        fs: Optional[fsspec.spec.AbstractFileSystem] = None,
    ):
        """
        Args:
            fs: Optional existing filesystem to use for accessing data.
            root: Optional root path (bucket name in case of S3).
        """
        # subset of General Circulation Models (GCMs) and Variant IDs for analysis
        self.subset: Dict[str, Dict[str, str]] = {
            "ACCESS-CM2": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "CMCC-ESM2": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "CNRM-CM6-1": {"variant_label": "r1i1p1f2", "grid_label": "gr"},
            "MPI-ESM1-2-LR": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "MIROC6": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "NorESM2-MM": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
        }
        # <variable_id>_<table_id>_<source_id>_<experiment_id>_<variant_label>_<grid_label>_<time_range>.nc
        self.fs = s3fs.S3FileSystem(anon=True) if fs is None else fs
        self.root = NexGddpCmip6.bucket if root is None else root
        self.quantities = {"tas": {"name": "Daily average temperature"}}

    def path(self, gcm="NorESM2-MM", scenario="ssp585", quantity="tas", year=2030):
        component = self.subset[gcm]
        variant_label = component["variant_label"]
        grid_label = component["grid_label"]
        filename = (
            f"{quantity}_day_{gcm}_{scenario}_{variant_label}_{grid_label}_{year}.nc"
        )
        return (
            posixpath.join(
                self.root,
                f"NEX-GDDP-CMIP6/{gcm}/{scenario}/{variant_label}/{quantity}/",
            )
            + filename,
            filename,
        )

    def gcms(self) -> List[str]:
        return list(self.subset.keys())

    @contextmanager
    def open_dataset_year(
        self,  # type: ignore
        gcm: str,
        scenario: str,
        quantity: str,
        year: int,
        chunks=None,
    ) -> Generator[xr.Dataset, None, None]:
        # use "s3://bucket/root" ?
        path, _ = self.path(gcm, scenario, quantity, year)
        logger.info(f"Opening DataSet, relative path={path}, chunks={chunks}")
        ds: Optional[xr.Dataset] = None
        f = None
        try:
            f = self.fs.open(path, "rb")
            ds = xr.open_dataset(f, chunks=chunks)
            yield ds
        finally:
            if ds is not None:
                ds.close()
            if f is not None:
                f.close()
