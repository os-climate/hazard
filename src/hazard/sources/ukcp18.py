import io
import logging
import os
import re
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

import fsspec
import xarray as xr

from hazard.protocols import OpenDataset

logger = logging.getLogger(__name__)


class Ukcp18(OpenDataset):
    def __init__(
        self,
        dataset_member_id: str = "01",
        dataset_frequency: str = "day",
        dataset_version: str = "v20190731",
    ):
        self._fs = fsspec.filesystem(
            protocol="ftp",
            host=os.environ["CEDA_FTP_URL"],
            username=os.environ["CEDA_FTP_USERNAME"],
            password=os.environ["CEDA_FTP_PASSWORD"],
        )
        self.quantities: Dict[str, Dict[str, str]] = {"tas": {"name": "Daily average temperature"}}

        # Refer to https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance-data-availability-access-and-formats.pdf on what these values refer to # noqa
        self._dataset_member_id = dataset_member_id
        self._dataset_frequency = dataset_frequency
        self._dataset_version = dataset_version

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

        if not files_available_for_quantity:
            raise Exception(
                f"No UKCP18 files available for: gcm:{gcm}, scenario:{scenario}, quantity:{quantity}, year:{year}"
            )

        all_data_from_files = self._combine_all_files_data(files_available_for_quantity)
        only_data_for_year = all_data_from_files.sel(time=str(year))
        reprojected = self._reproject_quantity(only_data_for_year, quantity)
        converted_to_kelvin = self._convert_to_kelvin(reprojected, quantity)

        yield converted_to_kelvin

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
        ftp_url = (
            f"/badc/{gcm}/data/land-rcm/uk/12km/{scenario}/{self._dataset_member_id}/{quantity}"
            f"/{self._dataset_frequency}/{self._dataset_version}/"
        )
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

    def _reproject_quantity(self, dataset: xr.Dataset, quantity: str) -> xr.Dataset:
        squeezed = dataset[quantity].squeeze()
        no_grid_values = squeezed.drop_vars(["grid_latitude", "grid_longitude"])
        del no_grid_values.attrs["grid_mapping"]
        pre_projection = no_grid_values.rio.write_crs("EPSG:27700")
        reprojected = pre_projection.rio.reproject("EPSG:4326")
        reprojected_and_renamed = reprojected.rename({"x": "lon", "y": "lat"})
        dataset[quantity] = reprojected_and_renamed
        return dataset

    def _convert_to_kelvin(self, dataset: xr.Dataset, quantity: str) -> xr.Dataset:
        quantity_data = dataset[quantity]
        converted = quantity_data + 273.15
        converted.attrs["units"] = "K"
        converted.attrs["label_units"] = "K"
        converted.attrs["plot_label"] = "Mean air temperature at 1.5m (K)"
        dataset[quantity] = converted
        return dataset
