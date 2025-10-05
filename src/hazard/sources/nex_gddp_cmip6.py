import logging
import posixpath
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, Iterator, List, Optional

import fsspec
import s3fs  # type: ignore
import xarray as xr
from pystac.item_collection import ItemCollection
from pystac_client import Client

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
            "CNRM-ESM2-1": {"variant_label": "r1i1p1f2", "grid_label": "gr"},
            "EC-Earth3": {"variant_label": "r1i1p1f1", "grid_label": "gr"},
            "GFDL-ESM4": {"variant_label": "r1i1p1f1", "grid_label": "gr1"},
            "HadGEM-GC31-LL": {"variant_label": "r1i1p1f3", "grid_label": "gn"},
            "KACE-1-0-G": {"variant_label": "r1i1p1f1", "grid_label": "gr"},
            "MRI-ESM2-0": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "TaiESM1": {"variant_label": "r1i1p1f1", "grid_label": "gn"},
            "UKESM1-0-LL": {"variant_label": "r1i1p1f2", "grid_label": "gn"},
        }
        # <variable_id>_<table_id>_<source_id>_<experiment_id>_<variant_label>_<grid_label>_<time_range>.nc
        self.fs = s3fs.S3FileSystem(anon=True) if fs is None else fs
        self.quantities = {"tas": {"name": "Daily average temperature"}}
        self.root = NexGddpCmip6.bucket if root is None else root

    def path(self, gcm="NorESM2-MM", scenario="ssp585", quantity="tas", year=2030):
        """directly construct the S3 path to input dataset"""

        component = self.subset[gcm]
        variant_label = component["variant_label"]
        grid_label = component["grid_label"]
        filename = f"{quantity}_day_{gcm}_{scenario}_{variant_label}_{grid_label}_{year}_v2.0.nc"
        return (
            posixpath.join(
                self.root,
                f"NEX-GDDP-CMIP6/{gcm}/{scenario}/{variant_label}/{quantity}/",
            )
            + filename,  # noqa:W503
            filename,
        )

    def path_stac(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection_id: str = "nasa-nex-gddp-cmip6",
        gcm="NorESM2-MM",
        scenario="ssp585",
        quantity="tas",
        year=2030,
    ):
        """Retrieves the path to the input dataset from STAC metadata"""

        items = self.search_stac_items(
            catalog_url=catalog_url,
            search_params={
                "collections": [collection_id],
                "query": {
                    "cmip6:model": {"eq": gcm},
                    "cmip6:scenario": {"eq": scenario},
                    "cmip6:year": {"eq": year},
                },
            },
        )
        if len(items) == 0:
            raise ValueError(
                f"No items found for gcm={gcm}, scenario={scenario}, year={year}"
            )
        elif len(items) > 1:
            raise ValueError(
                f"Multiple items found for gcm={gcm}, scenario={scenario}, year={year}"
            )
        else:
            item = items[0]
        href = item.assets[quantity].href
        href_replaced = href.replace(
            "https://nasagddp.blob.core.windows.net/nex-gddp-cmip6/NEX/GDDP-CMIP6",
            "s3://nex-gddp-cmip6/NEX-GDDP-CMIP6",
        )
        return href_replaced

    def gcms(self) -> List[str]:
        return list(self.subset.keys())

    @contextmanager
    def open_dataset_year(
        self,
        gcm: str,
        scenario: str,
        quantity: str,
        year: int,
        chunks=None,
        catalog_url: Optional[str] = None,
        collection_id: Optional[str] = None,  # type: ignore
    ) -> Iterator[xr.Dataset]:
        # use "s3://bucket/root" ?
        if catalog_url is not None or collection_id is not None:
            assert catalog_url is not None and collection_id is not None
            path = self.path_stac(
                catalog_url, collection_id, gcm, scenario, quantity, year
            )
        else:
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

    def search_stac_items(
        self, catalog_url: str, search_params: Dict
    ) -> ItemCollection:
        client = Client.open(catalog_url)
        search = client.search(**search_params)
        items = search.item_collection()
        return items
