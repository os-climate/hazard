import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import xarray as xr

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    scenario: str
    year: int


class DavydzenkaEtAlLandSubsidence(IndicatorModel[BatchItem]):
    def __init__(self, source_dir: Optional[str]):
        """
        Define every attribute of the onboarding class for the land subsidence dataset.

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
            source_dir (str): directory containing source files.
        """
        self.resource = list(self.inventory())[0]
        if source_dir is not None:
            self.source = pathlib.PurePosixPath(
                pathlib.Path(source_dir).as_posix(), "ds03.tif"
            )
            if not os.path.exists(source_dir):
                os.makedirs(source_dir)
            if not os.path.exists(str(self.source)):
                # Download source data
                url = "https://zenodo.org/records/10223637/files/ds03.tif?download=1"
                download_file(
                    url, str(self.source.parent), filename=self.source.parts[-1]
                )

    def batch_items(self) -> Iterable[BatchItem]:
        return [
            BatchItem(
                scenario=self.resource.scenarios[0].id,
                year=self.resource.scenarios[0].years[0],
            ),
        ]

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Any
    ):
        assert target is None or isinstance(target, OscZarr)
        da = xr.open_rasterio(self.source).isel(band=0)
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
        """
        Create map images.
        """
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
                save_netcdf_coords=False,
                scenarios=[Scenario(id="historical", years=[2021])],
            )
        ]
