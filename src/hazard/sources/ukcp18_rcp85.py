import io
import logging
import re
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Union

import fsspec
import xarray as xr

from hazard.protocols import OpenDataset

logger = logging.getLogger(__name__)


class Ukcp18Rcp85(OpenDataset):
    def __init__(self, ceda_ftp_url: str, ceda_ftp_username: str, ceda_ftp_password: str):
        self._fs = fsspec.filesystem("ftp", host=ceda_ftp_url, username=ceda_ftp_username, password=ceda_ftp_password)

        self.quantities: Dict[str, Dict[str, str]] = {"tas": {"name": "Daily average temperature"}}

    def gcms(self) -> List[str]:
        return list("ukcp18")

    @contextmanager
    def open_dataset_year(
        self,
        gcm: str = "ukcp18",
        scenario: str = "rcp85",
        quantity: str = "tas",
        year: int = 1981,
        chunks=None,
        catalog_url: Optional[str] = None,
        collection_id: Optional[str] = None,  # type: ignore
    ) -> Generator[xr.Dataset, None, None]:
        files_available_for_quantity: List[str] = self._get_files_available_for_quantity_and_year(
            gcm, scenario, quantity, year
        )
        all_data_from_files = self._combine_all_files_data(files_available_for_quantity)
        only_data_for_year = all_data_from_files.sel(time=str(year))

        only_data_for_year_with_quantity_reprojected = self._reproject_quantity(only_data_for_year, quantity)

        yield only_data_for_year_with_quantity_reprojected

        if only_data_for_year is not None:
            only_data_for_year.close()

    def _combine_all_files_data(self, files_available_for_quantity: List[str]) -> xr.Dataset:
        datasets = []
        for file in files_available_for_quantity:
            with self._fs.open(file, "rb") as f:
                with io.BytesIO(f.read()) as file_in_memory:
                    file_in_memory.seek(0)
                    datasets.append(xr.open_dataset(file_in_memory).load())
        return xr.combine_by_coords(datasets)  # type: ignore[return-value]

    def _get_files_available_for_quantity_and_year(
        self, gcm: str, scenario: str, quantity: str, year: int
    ) -> List[str]:
        ftp_url = f"/badc/{gcm}/data/land-rcm/uk/12km/{scenario}/01/{quantity}/day/v20190731/"
        all_files = self._fs.ls(ftp_url, detail=False)
        files_that_contain_year = []
        start_end_date_regex = re.compile(r"_(\d{8})-(\d{8})\.nc")
        for file in all_files:
            matches = start_end_date_regex.search(file)
            if matches:
                start_date = int(matches.group(1)[:4])
                end_date = int(matches.group(2)[:4])
                if start_date <= year <= end_date:
                    files_that_contain_year.append(f"{ftp_url}{file}")
        return files_that_contain_year

    def _reproject_quantity(self, only_data_for_year: xr.Dataset, quantity: str) -> xr.Dataset:
        squeezed = only_data_for_year[quantity].squeeze()
        no_grid_values = squeezed.drop_vars(["grid_latitude", "grid_longitude"])
        del no_grid_values.attrs["grid_mapping"]
        pre_projection = no_grid_values.rio.write_crs("EPSG:27700")
        reprojected = pre_projection.rio.reproject("EPSG:4326")
        reprojected_and_renamed = reprojected.rename({"x": "lon", "y": "lat"})
        only_data_for_year[quantity] = reprojected_and_renamed
        return only_data_for_year
