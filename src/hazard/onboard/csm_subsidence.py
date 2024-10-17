"""Module for handling the onboarding of the Davydzenka et al. land subsidence dataset."""

import logging
import os
from dataclasses import dataclass
from pathlib import PurePath, Path
from typing_extensions import Any, Iterable, Optional, override

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a batch item with a scenario and year for hazard processing."""

    scenario: str
    year: int


class DavydzenkaEtAlLandSubsidence(IndicatorModel[BatchItem]):
    """Handles the onboarding and processing of the Davydzenka et al. land subsidence dataset."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the land subsidence dataset.

        METADATA:
        Link: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104497
        Data type: historical
        Hazard indicator: land subsidence rate
        Region: Global
        Resolution: 30 arc-seconds (or 1km roughly)
        Scenarios: historical
        File type: Map (.tif)

        DATA DESCRIPTION:
        Global map of land subsidence rates covering historically documented and
        new subsiding areas with a spatial resolution of 30 × 30 arc seconds.

        Args:
            source_dir_base (str): directory containing source files.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none,
            a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.resource = list(self.inventory())[0]
        self.source_dir = PurePath(source_dir_base, "csm_subsidence").as_posix() + "/"
        self.download_url = (
            "https://zenodo.org/records/10223637/files/ds03.tif?download=1"
        )

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        if (
            Path(PurePath(self.source_dir + "ds03.tif")).exists()
            and not force
            and not force_download
        ):
            return
        else:  # if source_dir not exists or force:
            self.fs.makedirs(self.source_dir, exist_ok=True)
            download_file(
                url=self.download_url,
                directory=(Path((self.source_dir))),
            )

    def batch_items(self) -> Iterable[BatchItem]:
        """Return the batch items for processing the land subsidence dataset."""
        return [
            BatchItem(
                scenario=self.resource.scenarios[0].id,
                year=self.resource.scenarios[0].years[0],
            ),
        ]

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
        self.create_maps(target, target)

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Any
    ):
        """Process a single batch item, writing data to the Zarr store."""
        assert target is None or isinstance(target, OscZarr)
        da = xr.open_rasterio(self.resource).isel(band=0)  # type: ignore[attr-defined]
        z = target.create_empty(
            self.resource.path.format(scenario=item.scenario, year=item.year),
            len(da.x),
            len(da.y),
            da.rio.transform(),
            str(da.crs.replace("+init=", "")),
        )
        values = (
            da.values
        )  # will load into memory; assume source not chunked efficiently
        values[values == -9999.0] = float("nan")
        z[0, :, :] = values

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self.resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        with open(
            os.path.join(os.path.dirname(__file__), "csm_subsidence.md"), "r"
        ) as f:
            description = f.read()
        return [
            HazardResource(
                hazard_type="Subsidence",
                indicator_id="land_subsidence_rate",
                indicator_model_id=None,
                indicator_model_gcm="historical",
                path="subsidence/csm/v1/land_subsidence_rate_{scenario}_{year}",
                params={},
                display_name="Land subsidence rate (Davydzenka et Al (2024))",
                description=description,
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -60.0),
                        (-180.0, -60.0),
                    ],
                    bbox=[-180.0, -60.0, 180.0, 85.0],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="heating",
                        min_value=0.0,
                        max_value=100.0,
                        units="millimetres/year",
                    ),
                    index_values=None,
                    path="maps/subsidence/csm/v1/land_subsidence_rate_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="millimetres/year",
                store_netcdf_coords=False,
                scenarios=[Scenario(id="historical", years=[2021])],
            )
        ]
