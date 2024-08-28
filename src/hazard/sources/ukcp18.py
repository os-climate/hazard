import io
import logging
import os
import re
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, Union

import fsspec
import rasterio
import xarray as xr
from rasterio import CRS

from hazard.protocols import OpenDataset

_WGS84 = "EPSG:4326"
_RESOLUTION_TO_COLLECTION_MAPPINGS = {
    "60km": "land-gcm",
    "12km": "land-rcm",
    "5km": "land-cpm",
    "2.2km": "land-cpm",
}

logger = logging.getLogger(__name__)


class Ukcp18(OpenDataset):
    def __init__(
        self,
        dataset_member_id: str = "01",
        dataset_frequency: str = "day",
        dataset_version: str = "latest",
        domain: str = "uk",
        resolution: str = "12km",
    ):
        self._fs = fsspec.filesystem(
            "filecache",
            target_protocol="ftp",
            target_options={
                "host": os.environ["CEDA_FTP_URL"],
                "username": os.environ["CEDA_FTP_USERNAME"],
                "password": os.environ["CEDA_FTP_PASSWORD"],
            },
            cache_storage="/tmp/ukcp18cache/",
        )
        self.quantities: Dict[str, Dict[str, str]] = {
            "tas": {"name": "Daily average temperature"}
        }

        # Refer to https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance-data-availability-access-and-formats.pdf on what these values refer to # noqa
        self._dataset_member_id = dataset_member_id
        self._dataset_frequency = dataset_frequency
        self._dataset_version = dataset_version

        self._collection = _RESOLUTION_TO_COLLECTION_MAPPINGS[resolution]
        self._domain = domain
        self._resolution = resolution

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
        files_available_for_quantity: List[str] = (
            self._get_files_available_for_quantity_and_year(
                gcm, scenario, quantity, year
            )
        )

        if not files_available_for_quantity:
            raise Exception(
                f"No UKCP18 files available for: gcm:{gcm}, scenario:{scenario}, quantity:{quantity}, year:{year}"
            )

        all_data_from_files, crs = self._combine_all_files_data(
            files_available_for_quantity
        )
        only_data_for_year = all_data_from_files.sel(time=str(year))
        reprojected = self._reproject_quantity(only_data_for_year, quantity, crs)
        converted_to_kelvin = self._convert_to_kelvin(reprojected, quantity)

        yield converted_to_kelvin

    def _combine_all_files_data(
        self, files_available_for_quantity: List[str]
    ) -> Tuple[xr.Dataset, rasterio.CRS]:
        datasets = []
        crs = None
        for file in files_available_for_quantity:
            with self._fs.open(file, "rb") as f:
                with io.BytesIO(f.read()) as file_in_memory:
                    file_in_memory.seek(0)
                    datasets.append(xr.open_dataset(file_in_memory).load())
            if crs is None:
                with self._fs.open(file, "rb") as crs_f:
                    with rasterio.MemoryFile(crs_f.read(), ext=".nc") as crs_mem:
                        crs = crs_mem.open().crs
        return xr.combine_by_coords(datasets, combine_attrs="override"), crs  # type: ignore[return-value]

    def _get_files_available_for_quantity_and_year(
        self, gcm: str, scenario: str, quantity: str, year: int
    ) -> List[str]:
        ftp_url = (
            f"/badc/{gcm}/data/{self._collection}/{self._domain}/{self._resolution}/{scenario}/{self._dataset_member_id}/{quantity}"
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

    def _prepare_data_array(
        self, data_array: xr.DataArray, crs: Union[str, CRS], drop_vars: List[str]
    ) -> xr.DataArray:
        squeezed = data_array.squeeze()
        dropped_vars = squeezed.drop_vars(drop_vars, errors="ignore")
        dropped_vars.attrs.pop("grid_mapping", None)
        return dropped_vars.rio.write_crs(crs)

    def _reproject_and_rename_coordinates(
        self,
        data_array: xr.DataArray,
        target_crs: str,
        to_rename_to_lon: str,
        to_rename_to_lat: str,
    ) -> xr.DataArray:
        reprojected = data_array.rio.reproject(target_crs)
        return reprojected.rename({to_rename_to_lon: "lon", to_rename_to_lat: "lat"})

    def _reproject_quantity(
        self, dataset: xr.Dataset, quantity: str, crs: rasterio.CRS
    ) -> xr.Dataset:
        if self._domain == "uk" and self._resolution in ["5km", "12km", "60km"]:
            prepped_data_array = self._prepare_data_array(
                dataset[quantity],
                crs,
                ["latitude", "longitude", "grid_latitude", "grid_longitude"],
            )
            dataset[quantity] = self._reproject_and_rename_coordinates(
                prepped_data_array, _WGS84, "x", "y"
            )
        elif self._domain == "uk" and self._resolution == "2.2km":
            prepped_data_array = self._prepare_data_array(
                dataset[quantity],
                crs,
                drop_vars=["latitude", "longitude"],
            )
            prepped_data_array.rio.set_spatial_dims(
                "grid_longitude", "grid_latitude", inplace=True
            )
            dataset[quantity] = self._reproject_and_rename_coordinates(
                prepped_data_array, _WGS84, "x", "y"
            )
        elif self._domain == "eur" and self._resolution == "12km":
            prepped_data_array = self._prepare_data_array(
                dataset[quantity],
                crs,
                drop_vars=["latitude", "longitude"],
            )
            prepped_data_array.rio.set_spatial_dims(
                "grid_longitude", "grid_latitude", inplace=True
            )
            dataset[quantity] = self._reproject_and_rename_coordinates(
                prepped_data_array, _WGS84, "x", "y"
            )
        elif self._domain == "global" and self._resolution == "60km":
            prepped_data_array = self._prepare_data_array(dataset[quantity], crs, [])
            dataset[quantity] = self._reproject_and_rename_coordinates(
                prepped_data_array, _WGS84, "x", "y"
            )
        else:
            logger.warning(
                "Didn't find a matching domain and resolution for reprojecting the dataset, returning it untouched"
            )
        return dataset

    def _convert_to_kelvin(self, dataset: xr.Dataset, quantity: str) -> xr.Dataset:
        quantity_data = dataset[quantity]
        converted = quantity_data + 273.15
        converted.attrs["units"] = "K"
        converted.attrs["label_units"] = "K"
        converted.attrs["plot_label"] = "Mean air temperature at 1.5m (K)"
        dataset[quantity] = converted
        return dataset
