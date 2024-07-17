import io
import logging
import ftplib
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Generator

from hazard.protocols import OpenDataset
import xarray as xr

logger = logging.getLogger(__name__)


class Ukcp18Rcp85(OpenDataset):
    def __init__(self, ceda_ftp_url: str, ceda_ftp_username: str, ceda_ftp_password: str):
        self._ftp = ftplib.FTP(ceda_ftp_url)
        self._ftp.login(ceda_ftp_username, ceda_ftp_password)

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
        files_available_for_quantity: List[str] = self._get_files_available_for_quantity_and_year(gcm, scenario,
                                                                                                  quantity, year)
        all_data_from_files: xr.Dataset = self._combine_all_files_data(files_available_for_quantity)
        only_data_for_year = all_data_from_files.sel(time=str(year))

        yield only_data_for_year[quantity]

        if only_data_for_year is not None:
            only_data_for_year.close()

    def _combine_all_files_data(self, files_available_for_quantity: List[str]) -> xr.Dataset:
        datasets = []
        for file in files_available_for_quantity:
            with io.BytesIO() as file_in_memory:
                self._ftp.retrbinary(f'RETR {file}', file_in_memory.write)
                file_in_memory.seek(0)
                datasets.append(xr.open_dataset(file_in_memory).load())
        return xr.combine_by_coords(datasets)

    def _get_files_available_for_quantity_and_year(self, gcm: str, scenario: str, quantity: str, year: int) -> List[
        str]:
        ftp_files_path = f"badc/{gcm}/data/land-rcm/uk/12km/{scenario}/01/{quantity}/day/v20190731"
        all_files = self._ftp.nlst(ftp_files_path)
        files_that_contain_year = []
        start_end_date_regex = re.compile(r'_(\d{8})-(\d{8})\.nc')
        for file in all_files:
            matches = start_end_date_regex.search(file)
            if matches:
                start_date = int(matches.group(1)[:4])
                end_date = int(matches.group(2)[:4])
                if start_date <= year <= end_date:
                    files_that_contain_year.append(file)
        return files_that_contain_year
