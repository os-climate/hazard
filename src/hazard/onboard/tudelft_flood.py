import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

import numpy as np
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
    flood_depth_filename: str
    extent_protected_filename: str


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
                flood_depth_filename="River_flood_depth_1971_2000_hist_{return_period}.tif",
                extent_protected_filename="River_flood_extent_1971_2000_hist_with_protection.tif",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2035,
                flood_depth_filename="River_flood_depth_2021_2050_RCP45_{return_period}.tif",
                extent_protected_filename="River_flood_extent_2021_2050_RCP45_with_protection.tif",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2035,
                flood_depth_filename="River_flood_depth_2021_2050_RCP85_{return_period}.tif",
                extent_protected_filename="River_flood_extent_2021_2050_RCP85_with_protection.tif",
            ),
            BatchItem(
                scenario="rcp4p5",
                central_year=2085,
                flood_depth_filename="River_flood_depth_2071_2100_RCP45_{return_period}.tif",
                extent_protected_filename="River_flood_extent_2071_2100_RCP45_with_protection.tif",
            ),
            BatchItem(
                scenario="rcp8p5",
                central_year=2085,
                flood_depth_filename="River_flood_depth_2071_2100_RCP85_{return_period}.tif",
                extent_protected_filename="River_flood_extent_2071_2100_RCP85_with_protection.tif",
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
        full_path_depth_format = PurePosixPath(self.source_dir, item.flood_depth_filename)
        full_path_extent = PurePosixPath(self.source_dir, item.extent_protected_filename)
        assert target is None or isinstance(target, OscZarr)
        shape = [39420, 38374]  # y, x not all returns have same size (first one smaller at 38371)
        for i, return_period in enumerate(self.return_periods):
            full_path_depth = str(full_path_depth_format).format(return_period=self.return_period_str[return_period])
            with self.fs.open(full_path_depth, "rb") as fd:
                dad = xr.open_rasterio(fd).isel(band=0)
                with self.fs.open(full_path_extent, "rb") as fe:
                    dae = xr.open_rasterio(fe).isel(band=0)
                    # bounds = da.rio.bounds()
                    if return_period == self.return_periods[0]:
                        z = target.create_empty(
                            self._resource.path.format(scenario=item.scenario, year=item.central_year),
                            shape[1],
                            shape[0],
                            dad.rio.transform(),
                            str(dad.crs),
                            indexes=self.return_periods,
                        )
                # dad_nodata = 65535
                if (
                    dad.shape[1] == 38371
                ):  # corrections for various possible errors whereby coordinates are missing for certain files:
                    dae = dae[:, 0:38371]
                if dae.shape[1] == 38375:
                    dae = dae[:, 0:38374]
                if dad.shape[1] == 38375:
                    dad = dad[:, 0:38374]
                if dae.shape[0] == 39385:
                    dad = dad[35:, :]

                # not quite the same coordinates: check if close, within rounding error, and align exactly
                assert np.abs(np.array(dae.x) - np.array(dad.x)).max() < 1e-4
                assert np.abs(np.array(dae.y) - np.array(dad.y)).max() < 1e-4
                dae = dae.assign_coords({"x": dad.x, "y": dad.y})
                da_combined = xr.where(dae <= return_period, dad, 0)
                values = da_combined.data
                depth_no_data = dad.attrs["nodatavals"]
                values[values == depth_no_data] = float("nan")
                z[i, 0 : len(da_combined.y), 0 : len(da_combined.x)] = values[:, :]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        ...
        create_tiles_for_resource(source, target, self._resource, max_zoom=10)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        with open(os.path.join(os.path.dirname(__file__), "tudelft_flood.md"), "r") as f:
            description = f.read()
        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_id="tudelft",
                indicator_model_gcm="CLMcom-CCLM4-8-17-EC-EARTH",
                path="inundation/river_tudelft/v2/flood_depth_{scenario}_{year}",
                params={},
                display_name="Flood depth (TUDelft)",
                description=description,
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
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            )
        ]
