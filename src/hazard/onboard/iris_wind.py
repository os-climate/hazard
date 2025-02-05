"""Module for handling the onboarding and processing of IRIS - Imperial College Storm Model data."""

import os
from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional
from typing_extensions import Iterable, override

import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import tiles


@dataclass
class BatchItem:
    """Represents a batch item for hazard processing, including resource, year, and scenario."""

    resource: HazardResource  # type of hazard
    year: int
    scenario: str


class IRISIndicator(IndicatorModel[BatchItem]):
    """On-board returns data set from IRIS - Imperial College Storm Model."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Initialize the IRISIndicator class with the input directory for IRIS data.

        Assumes iris downloaded data is of the form wind/IRIS/return_value_maps/--files and that they are in the downloads folder.

        """
        self.source_dir = PurePath(source_dir_base, "iris_wind").as_posix() + "/"

        self.fs = fs if fs else LocalFileSystem()

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        # source_dir = PurePath(download_dir, "wind", "IRIS", "return_value_maps")

        if download_dir and not os.path.exists(download_dir):
            msg = f"{self.__class__.__name__} requires the file return_value_maps to be in the download_dir.\nThe download_dir was {download_dir}."
            raise FileNotFoundError(msg)

        self.fs.makedirs(self.source_dir, exist_ok=True)
        if download_dir:
            for _, _, files in os.walk(download_dir):
                for file_name in files:
                    dest_file_path = PurePath(self.source_dir, file_name)
                    if force or not os.path.exists(dest_file_path):
                        self.fs.copy(
                            PurePath(download_dir, file_name),
                            self.source_dir,
                        )

    def run_single(
        self, item: BatchItem, source, target: ReadWriteDataArray, client: Client
    ):
        """Process a single batch item and writes the data to the Zarr store."""
        file_name = self._file_name(item.scenario, item.year)
        ds = xr.open_dataset(file_name.format(year=item.year, scenario=item.scenario))
        # dimensions: (rp: 19, latitude: 1200, longitude: 3600)
        da = OscZarr.normalize_dims(ds.vmax)
        # if the coordinates give the left, bottom of each pixel:
        da = da.assign_coords(
            latitude=da.latitude.data + 0.05, longitude=da.longitude.data + 0.05
        )
        target.write(
            item.resource.path.format(scenario=item.scenario, year=item.year),
            da,
            chunks=[len(da.index.data), 250, 250],
        )
        self.generate_single_map(item, target, target)

    def generate_single_map(
        self, item: BatchItem, source: ReadWriteDataArray, target: ReadWriteDataArray
    ):
        """Generate a single map from the batch item and writes it to the target store."""
        source_path = item.resource.path.format(scenario=item.scenario, year=item.year)
        assert item.resource.map is not None
        assert isinstance(source, OscZarr) and isinstance(target, OscZarr)
        target_path = item.resource.map.path.format(
            scenario=item.scenario, year=item.year
        )
        tiles.create_tile_set(source, source_path, target, target_path, check_fill=True)
        # tiles.create_image_set(source, source_path, target, target_path)

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        resource = self._hazard_resource()
        for scenario in resource.scenarios:
            for year in scenario.years:
                yield BatchItem(resource, year, scenario.id)

    def onboard_single(
        self, target, download_dir=None, force_prepare=False, force_download=False
    ):
        """Onboard a single batch of hazard data into the system.

        Args:
            target: Target system for writing the processed data.
            download_dir (str): Directory where downloaded files will be stored.
            force_prepare(bool): Flag to force data preparation. Default is False
            force_download(bool):Flag to force re-download of data. Default is False

        """
        self.prepare(
            force=force_prepare,
            download_dir=download_dir,
            force_download=force_download,
        )
        self.run_all(source=None, target=target, client=None, debug_mode=False)

    def _file_name(self, scenario: str, year: int):
        """Return the file name for a specific scenario and year."""
        # file name for 0.1 degree resolution sets
        # including radiative forcing (1.9 to 8.5 W/m^2): SSP1 = SSP1-1.9, SSP2 = SSP2-4.5, SSP5 = SSP5-8.5
        if scenario == "historical":
            return os.path.join(
                self.source_dir,
                "IRIS_vmax_maps_PRESENT_tenthdeg.nc",
            )
        else:
            scen_lookup = {"ssp119": "SSP1", "ssp245": "SSP2", "ssp585": "SSP5"}
            return os.path.join(
                self.source_dir,
                f"IRIS_vmax_maps_{year}-{scen_lookup[scenario]}_tenthdeg.nc",
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._hazard_resource()]

    def _hazard_resource(self) -> HazardResource:
        """Return the hazard resource details, including metadata and map info, for the IRIS dataset."""
        with open(os.path.join(os.path.dirname(__file__), "iris_wind.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="Wind",
            indicator_id="max_speed",
            indicator_model_id="iris",
            indicator_model_gcm="combined",
            path="wind/iris/v1/max_speed_{scenario}_{year}",
            params={},
            display_name="Max wind speed (IRIS)",
            description=description,
            group_id="iris_osc",
            display_groups=[],
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                bounds=[(-180.0, 60.0), (180.0, 60.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 60.0],
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_index=255,
                    max_value=120.0,
                    units="m/s",
                ),
                index_values=None,
                path="wind/iris/v1/max_speed_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="m/s",
            scenarios=[
                Scenario(id="historical", years=[2010]),
                Scenario(id="ssp119", years=[2050]),
                Scenario(id="ssp245", years=[2050]),
                Scenario(id="ssp585", years=[2050]),
            ],
        )
        return resource
