import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

import rioxarray  # noqa: F401
import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_and_unzip
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    scenario: str
    central_year: int
    input_dataset_filename: str


class TUDelftRiverFlood(IndicatorModel[BatchItem]):

    def __init__(self, source_dir: str, fs: Optional[AbstractFileSystem] = None):
        """
        Define every attribute of the onboarding class for the Delft University of Technology
        river flood data.

        METADATA:
        Link: https://data.4tu.nl/articles/dataset/Pan-European_data_sets_of_river_flood_probability_of_occurrence_under_present_and_future_climate/12708122 # noqa: E501
        Data type: historical and scenario return period
        Hazard indicator: flood depth
        Region: Pan-Europe
        Resolution: 100m
        Return periods: 10, 30, 100, 300, 1000
        Scenarios: RCP4.5, 8.5
        Time range: 1971-2000, 2050, 2070, 2100
        File type: Map (.tif)

        DATA DESCRIPTION:
        GIS-compatible files containing data related to the probability of
        river floods occurring in Europe under present and future climate.
        Includes gridded (GeoTIFF) datasets of river flood extents (in two
        variants, with or without flood protection) and water depths.
        Additionally includes extreme river discharge estimates in ESRI
        Shapefile format. Based upon CLMcom-CCLM4-8-17-EC-EARTH regional
        climate simulation (EURO-CORDEX).

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir

        # Download source data
        self.return_periods = [10, 30, 100, 300, 1000]
        self.return_period_str = {10: "0010y", 30: "0030y", 100: "0100y", 300: "0300y", 1000: "1000y"}
        self.zip_url = (
            "https://data.4tu.nl/file/df7b63b0-1114-4515-a562-117ca165dc5b/5e6e4334-15b5-4721-a88d-0c8ca34aee17"
        )

        self._resource = list(self.inventory())[0]

    def batch_items(self) -> Iterable[BatchItem]:
        return [
            BatchItem(
                scenario="historical",
                central_year=1971,
                input_dataset_filename="River_flood_depth_1971_2000_hist_{return_period}.tif",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2035,
                input_dataset_filename="River_flood_depth_2021_2050_RCP45_{return_period}.tif",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2035,
                input_dataset_filename="River_flood_depth_2021_2050_RCP85_{return_period}.tif",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2085,
                input_dataset_filename="River_flood_depth_2071_2100_RCP45_{return_period}.tif",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2085,
                input_dataset_filename="River_flood_depth_2071_2100_RCP85_{return_period}.tif",
            ),
        ]

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            assert working_dir is not None
            download_and_unzip(self.zip_url, working_dir, "tudelft_river")
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            source = PurePosixPath(self.source_dir)
            download_and_unzip(self.zip_url, str(source.parent), source.parts[-1])

    def run_single(self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client):
        assert isinstance(target, OscZarr)
        input = PurePosixPath(self.source_dir, item.input_dataset_filename)
        assert target is None or isinstance(target, OscZarr)
        shape = [39420, 38374]  # y, x not all returns have same size (first one smaller at 38371)
        for i, return_period in enumerate(self.return_periods):
            filename = str(input).format(return_period=self.return_period_str[return_period])
            with self.fs.open(filename, "rb") as f:
                da = xr.open_rasterio(f).isel(band=0)
                # bounds = da.rio.bounds()
                if return_period == self.return_periods[0]:
                    z = target.create_empty(
                        self._resource.path.format(scenario=item.scenario, year=item.central_year),
                        shape[1],
                        shape[0],
                        da.rio.transform(),
                        str(da.crs),
                        indexes=self.return_periods,
                    )
                values = da.data
                no_data = da.attrs["nodatavals"]
                values[values == no_data] = float("nan")
                z[i, 0 : len(da.y), 0 : len(da.x)] = values[:, :]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        ...
        create_tiles_for_resource(source, target, self._resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""

        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_id="tudelft",
                indicator_model_gcm="CLMcom-CCLM4-8-17-EC-EARTH",
                path="inundation/river_tudelft/v2/flood_depth_{scenario}_{year}",
                params={},
                display_name="Flood depth (TUDelft)",
                description="""
Flood water depth, part of data set containing data related to the probability of
river floods occurring in Europe under present and future climate.
Based upon CLMcom-CCLM4-8-17-EC-EARTH regional
climate simulation (EURO-CORDEX).
                """,
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=5.0,
                        units="metres",
                    ),
                    index_values=None,
                    path="maps/inundation/river_tudelft/v2/flood_depth_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="metres",
                scenarios=[
                    Scenario(id="historical", years=[1971]),
                    Scenario(id="rcp45", years=[2050, 2070]),
                    Scenario(id="rcp85", years=[2050, 2070]),
                ],
            )
        ]
