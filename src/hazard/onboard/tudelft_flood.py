"""Module for handling the onboarding and processing of TUDelft riverine and coastal inundations data."""

import logging
import os
from pathlib import PurePath
from typing_extensions import Iterable, Optional, override
import zipfile

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class TUDelftRiverFlood(Onboarder):
    """On-board returns data set from TUDelft for river inundation."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the Delft University of Technology river flood data.

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
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = (
            PurePath(source_dir_base, "tudelft_river_flood").as_posix() + "/"
        )

        # Download source data
        self.return_periods = [10, 30, 100, 300, 1000]
        self.return_period_str = {
            10: "0010y",
            30: "0030y",
            100: "0100y",
            300: "0300y",
            1000: "1000y",
        }
        self.zip_url = "https://data.4tu.nl/file/df7b63b0-1114-4515-a562-117ca165dc5b/5e6e4334-15b5-4721-a88d-0c8ca34aee17"
        self.dataset_filename = "data_river_flood.zip"

        self.urls = {
            "data_river_flood.zip": "https://data.4tu.nl/file/df7b63b0-1114-4515-a562-117ca165dc5b/5e6e4334-15b5-4721-a88d-0c8ca34aee17"
        }

        """
        This is a comment on the understanding of the data that underlies the processing herein.
        The files comprise flood depth and flood extent, protected and unprotected.
        Flood extent is understood to contain the minimum return period for which a flood can occur.
        This implies that *unprotected* flood extent is simply inferred from flood depth, presenting the
        minimum return period for which flood depth is non-zero. This can be (and was for some examples) checked, e.g. for
        pixels where the 30 year return flood depth is non-zero and the 10 year return flood depth is zero, which
        should then have a 30 year flood extent. Protected flood extent is defined in the same way, but some caution required.
        The data set takes into account a minimum and maximum standard of protection (SoP) from FLOPROS as shown
        in Fig 3.2 of http://rain-project.eu/wp-content/uploads/2016/09/D2.5_REPORT_final.pdf
        The protected extent (assuming flood depth curve all non-zero) would then be the maximum SoP. In the UK,
        for example, the SoP is between 100 years and 300 years. It it tempting to set protected flood depths which are less
        than the maximum SoP to zero, however this will not give us the behaviour we want in calculations.
        Say we have 30, 100, 300 and 1000 year flood depths of 0.2, 0.4, 0.5 and 0.6m and we know the SoP is between 100 years and 300 years.
        Unprotected we would have a probability of (1/100 - 1/300) of a depth between 0.4 and 0.5m and (1/300 - 1/1000) of a depth between 0.5 and 0.6m.
        Protected, we either want to set our probability of depth between 0.4 and 0.5m to zero or - more likely - some
        lower value to reflect the uncertainty of the SoP. In neither case do we get the desired result by setting the flood depth to zero.
        In summary, we think it's best to provide data sets of
        - unprotected depth
        - SoP
        """

        self._depth_resource = self.inventory()

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)

        if (
            not os.path.exists(PurePath(download_dir, self.dataset_filename))
            or force_download
        ):
            download_file(
                url=self.urls[self.dataset_filename],
                directory=download_dir,
                filename=self.dataset_filename,
                force_download=force_download,
            )

        with zipfile.ZipFile(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as z:
            zip2source = {
                name: PurePath(self.source_dir, os.path.basename(name)).as_posix()
                for name in z.namelist()
            }
            for zname, target_file in zip2source.items():
                with self.fs.open(target_file, mode="wb") as mifi:
                    mifi.write(z.read(name=zname))

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if not os.path.exists(self.source_dir) or force or force_download:
            return False

        try:
            elements = os.listdir(self.source_dir)
        except PermissionError:
            return False

        return len(elements) == 124

    @override
    def onboard(self, target):
        items = self._get_items_to_process()
        for item in items:
            """Process a single item and writes the data to the Zarr store."""
            assert isinstance(target, OscZarr)
            full_path_depth_format = os.path.join(
                self.source_dir, "river_tudelft", item["flood_depth_filename"]
            )
            full_path_extent = os.path.join(
                self.source_dir, "river_tudelft", item["extent_protected_filename"]
            )
            assert target is None or isinstance(target, OscZarr)
            shape = [
                39420,
                38374,
            ]  # y, x not all returns have same size (first one smaller at 38371)
            # not all return periods have the same size. We pick the
            i, return_period = 1, self.return_periods[1]
            full_path_depth = str(full_path_depth_format).format(
                return_period=self.return_period_str[return_period]
            )
            with self.fs.open(full_path_depth, "rb") as fd:
                da_depth = xr.open_dataarray(fd, engine="rasterio").isel(band=0)  # type: ignore[attr-defined]
                coords_x, coords_y = np.array(da_depth.x), np.array(da_depth.y)

            # only create SoP for historical
            if item["scenario"] == "historical":
                with self.fs.open(full_path_extent, "rb") as fe:
                    da_sop = xr.open_dataarray(fe, engine="rasterio").isel(band=0)  # type: ignore[attr-defined]
                    # bounds = da.rio.bounds()
                    z_sop = target.create_empty(
                        self._sop_resource.path.format(
                            scenario=item["scenario"], year=item["central_year"]
                        ),
                        shape[1],
                        shape[0],
                        da_sop.rio.transform(),
                        str(da_sop.crs),
                        index_name="standard of protection (years)",
                        index_values=["min", "max"],
                    )
                    values_max_sop = np.array(da_sop.data, dtype="float32")
                    sop_no_data = da_sop.attrs["nodatavals"]
                    values_max_sop[values_max_sop == sop_no_data] = float("nan")
                    values_min_sop = self._get_mins(values_max_sop)
                    z_sop[0, 0 : len(da_sop.y), 0 : len(da_sop.x)] = values_min_sop[
                        :, :
                    ]
                    z_sop[1, 0 : len(da_sop.y), 0 : len(da_sop.x)] = values_max_sop[
                        :, :
                    ]
                    del values_min_sop, values_max_sop

            for i, return_period in enumerate(self.return_periods):
                full_path_depth = str(full_path_depth_format).format(
                    return_period=self.return_period_str[return_period]
                )
                with self.fs.open(full_path_depth, "rb") as fd:
                    da_depth = xr.open_dataarray(fd, engine="rasterio").isel(band=0)  # type: ignore[attr-defined]
                    if return_period == self.return_periods[0]:
                        z_depth = target.create_empty(
                            self._depth_resource.path.format(
                                scenario=item["scenario"], year=item["central_year"]
                            ),
                            shape[1],
                            shape[0],
                            da_depth.rio.transform(),
                            str(da_depth.crs),
                            index_values=self.return_periods,
                        )
                    if da_depth.shape[1] == 38375:
                        da_depth = da_depth[:, 0:38374]
                    # not quite the same coordinates: check if close, within rounding error, and align exactly
                    lenx, leny = (
                        min(len(da_depth.x), len(coords_x)),
                        min(len(da_depth.y), len(coords_y)),
                    )
                    assert (
                        np.abs(np.array(da_depth.x[0:lenx]) - coords_x[0:lenx]).max()
                        < 1e-4
                    )
                    assert (
                        np.abs(np.array(da_depth.y[0:leny]) - coords_y[0:leny]).max()
                        < 1e-4
                    )
                    # da_depth = da_depth.assign_coords({"x": da_depth.x, "y": da_depth.y})
                    values_depth = da_depth.data
                    depth_no_data = da_depth.attrs["nodatavals"]
                    values_depth[values_depth == depth_no_data] = float("nan")
                    z_depth[i, 0 : len(da_depth.y), 0 : len(da_depth.x)] = values_depth[
                        :, :
                    ]
                    del values_depth

    def _get_items_to_process(self):
        """Get a list of all items to process as dictionaries."""
        return [
            {
                "scenario": "historical",
                "central_year": 1985,
                "flood_depth_filename": "River_flood_depth_1971_2000_hist_{return_period}.tif",
                "extent_protected_filename": "River_flood_extent_1971_2000_hist_with_protection.tif",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "flood_depth_filename": "River_flood_depth_2021_2050_RCP45_{return_period}.tif",
                "extent_protected_filename": "River_flood_extent_2021_2050_RCP45_with_protection.tif",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "flood_depth_filename": "River_flood_depth_2021_2050_RCP85_{return_period}.tif",
                "extent_protected_filename": "River_flood_extent_2021_2050_RCP85_with_protection.tif",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "flood_depth_filename": "River_flood_depth_2071_2100_RCP45_{return_period}.tif",
                "extent_protected_filename": "River_flood_extent_2071_2100_RCP45_with_protection.tif",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "flood_depth_filename": "River_flood_depth_2071_2100_RCP85_{return_period}.tif",
                "extent_protected_filename": "River_flood_extent_2071_2100_RCP85_with_protection.tif",
            },
        ]

    def _get_mins(self, maxes: np.ndarray):
        mins = np.empty_like(maxes)
        for min, max in [
            (2, 10),
            (10, 30),
            (30, 100),
            (100, 300),
            (300, 1000),
            (1000, 10000),
        ]:
            mins = np.where(maxes == max, min, mins)
        return mins

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        # for TUDelft data, zero risk of flooding seems to be NaN
        # this presents a problem when creating lower resolution images where we might want to see an
        create_tiles_for_resource(
            source,
            target,
            self._depth_resource,
            max_zoom=10,
            nodata_as_zero_coarsening=True,
        )
        create_tiles_for_resource(
            source,
            target,
            self._sop_resource,
            max_zoom=10,
            nodata_as_zero_coarsening=True,
        )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        with open(
            os.path.join(os.path.dirname(__file__), "tudelft_flood.md"), "r"
        ) as f:
            description = f.read()
        with open(
            os.path.join(os.path.dirname(__file__), "tudelft_flood_sop.md"), "r"
        ) as f:
            description_sop = f.read()
        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_id="tudelft",
                indicator_model_gcm="CLMcom-CCLM4-8-17-EC-EARTH",
                path="inundation/river_tudelft/v2/flood_depth_unprot_{scenario}_{year}",
                params={},
                display_name="Flood depth (TUDelft)",
                description=description,
                group_id="",
                display_groups=[],
                version="",
                license="General terms of use for 4TU.Centre for Research Data. https://data.4tu.nl/articles/dataset/Pan-European_data_sets_of_river_flood_probability_of_occurrence_under_present_and_future_climate/12708122",
                source="",
                resolution="100 m",
                map=MapInfo(
                    bbox=[],
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
                    index_values=[1000],
                    path="maps/inundation/river_tudelft/v2/flood_depth_unprot_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="metres",
                store_netcdf_coords=False,
                scenarios=[
                    Scenario(id="historical", years=[1985]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            ),
            # HazardResource(
            #     hazard_type="RiverineInundation",
            #     indicator_id="flood_sop",
            #     indicator_model_id="tudelft",
            #     indicator_model_gcm="CLMcom-CCLM4-8-17-EC-EARTH",
            #     resolution="1000 m",
            #     path="inundation/river_tudelft/v2/flood_sop_{scenario}_{year}",
            #     params={},
            #     display_name="Standard of protection (TUDelft)",
            #     description=description_sop,
            #     group_id="",
            #     version="",
            #     attribution="“This product incorporates FLOPROS data © Authors 2016, licensed under CC BY 3.0.”    , Scussolini, P., Aerts, J. C. J. H., Jongman, B., Bouwer, L. M., Winsemius, H. C., de Moel, H., & Ward, P. J. (2016). FLOPROS: an evolving global database of flood protection standards. Natural Hazards and Earth System Sciences, 16, 1049–1061. https://doi.org/10.5194/nhess-16-1049-2016",
            #     license="CC BY 3.0.",
            #     source="4TU Research Data: https://data.4tu.nl/articles/dataset/Pan-European_data_sets_of_river_flood_probability_of_occurrence_under_present_and_future_climate/12708122",
            #     display_groups=[],
            #     map=MapInfo(
            #         bbox=[],
            #         bounds=[],
            #         colormap=Colormap(
            #             max_index=255,
            #             min_index=1,
            #             nodata_index=0,
            #             name="viridis",
            #             min_value=0.0,
            #             max_value=1500.0,
            #             units="years",
            #         ),
            #         index_values=None,
            #         path="maps/inundation/river_tudelft/v2/flood_sop_{scenario}_{year}_map",
            #         source="map_array_pyramid",
            #     ),
            #     units="years",
            #     store_netcdf_coords=False,
            #     scenarios=[
            #         Scenario(id="historical", years=[1985]),
            #         Scenario(id="rcp4p5", years=[2035, 2085]),
            #         Scenario(id="rcp8p5", years=[2035, 2085]),
            #     ],
            # ),
        ]


class TUDelftCoastalFlood(Onboarder):
    """On-board returns data set from TUDelft for coastal inundation."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Define every attribute of the onboarding class for the Delft University of Technology coastal flood data.

        METADATA:
        Link: https://data.4tu.nl/articles/Pan-European_data_sets_of_coastal_flood_probability_of_occurrence_under_present_and_future_climate/12717446
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
        coastal floods occurring in Europe under present and future climate.
        Includes gridded (GeoTIFF) datasets of coastal flood extents (in two
        variants, with or without flood protection) and water depths.
        Additionally includes extreme water level and storm surge heights
        estimates in ESRI Shapefile format. Based upon SMHI-RCA4-EC-EARTH
        regional climate simulation (EURO-CORDEX).

        Args:
            source_dir_base (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = (
            PurePath(source_dir_base, "tudelft_costal_flood").as_posix() + "/"
        )

        # Download source data
        self.return_periods = [10, 30, 100, 300, 1000]
        self.return_period_str = {
            10: "0010y",
            30: "0030y",
            100: "0100y",
            300: "0300y",
            1000: "1000y",
        }
        self.zip_url = "https://data.4tu.nl/file/57254e09-82ce-4257-8907-87a7be45bb21/f076ed16-066d-4d5a-a2a9-c2ed4b1a0e76"

        self.dataset_filename = "data_coastal_flood.zip"

        self.urls = {
            "data_coastal_flood.zip": "https://data.4tu.nl/file/df7b63b0-1114-4515-a562-117ca165dc5b/5e6e4334-15b5-4721-a88d-0c8ca34aee17"
        }

        self._resource = list(self.inventory())[0]

    def _get_items_to_process(self):
        """Get a list of all items to process."""
        return [
            {
                "scenario": "historical",
                "central_year": 1971,
                "input_dataset_filename": "Coastal_flood_depth_1971_2000_hist_{return_period}.tif",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2035,
                "input_dataset_filename": "Coastal_flood_depth_2021_2050_RCP45_{return_period}.tif",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2035,
                "input_dataset_filename": "Coastal_flood_depth_2021_2050_RCP85_{return_period}.tif",
            },
            {
                "scenario": "rcp4p5",
                "central_year": 2085,
                "input_dataset_filename": "Coastal_flood_depth_2071_2100_RCP45_{return_period}.tif",
            },
            {
                "scenario": "rcp8p5",
                "central_year": 2085,
                "input_dataset_filename": "Coastal_flood_depth_2071_2100_RCP85_{return_period}.tif",
            },
        ]

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)
        if (
            not os.path.exists(PurePath(download_dir, self.dataset_filename))
            or force_download
        ):
            download_file(
                url=self.urls[self.dataset_filename],
                directory=download_dir,
                filename=self.dataset_filename,
                force_download=force_download,
            )

        with zipfile.ZipFile(
            os.path.join(download_dir, self.dataset_filename), "r"
        ) as z:
            zip2source = {
                name: PurePath(self.source_dir, os.path.basename(name)).as_posix()
                for name in z.namelist()
            }
            for zname, target_file in zip2source.items():
                with self.fs.open(target_file, mode="wb") as mifi:
                    mifi.write(z.read(name=zname))

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if not os.path.exists(self.source_dir) or force or force_download:
            return False

        try:
            elements = os.listdir(self.source_dir)
        except PermissionError:
            return False

        return len(elements) == 124

    @override
    def onboard(self, target):
        """Process a single item and writes the data to the Zarr store."""
        items = self._get_items_to_process()
        for item in items:
            input = os.path.join(
                self.source_dir,
                "coastal_tudelft",
                "data",
                item["input_dataset_filename"],
            )
            assert target is None or isinstance(target, OscZarr)
            shape = [
                40300,
                38897,
            ]  # y, x not all returns have same size. Use max shapes
            for i, return_period in enumerate(self.return_periods):
                filename = str(input).format(
                    return_period=self.return_period_str[return_period]
                )
                with self.fs.open(filename, "rb") as f:
                    da = xr.open_dataarray(f, engine="rasterio").isel(band=0)  # type: ignore[attr-defined]
                    # bounds = da.rio.bounds()
                    if return_period == self.return_periods[0]:
                        z = target.create_empty(
                            self._resource.path.format(
                                scenario=item["scenario"], year=item["central_year"]
                            ),
                            shape[1],
                            shape[0],
                            da.rio.transform(),
                            str(da.crs),
                            index_values=self.return_periods,
                        )
                    values = da.data
                    no_data = da.attrs["nodatavals"]
                    values[values == no_data] = float("nan")
                    z[i, 0 : len(da.y), 0 : len(da.x)] = values[:, :]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self._resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="CoastalInundation",
                indicator_id="flood_depth",
                indicator_model_gcm="CLMcom-CCLM4-8-17-EC-EARTH",
                path="inundation/coastal_tudelft/v2/flood_depth_{scenario}_{year}",
                params={},
                display_name="Flood depth (TUDelft)",
                description="""
                Flood water depth, part of data set containing data related to the probability of
                coastal floods occurring in Europe under present and future climate.
                Based upon CLMcom-CCLM4-8-17-EC-EARTH regional
                climate simulation (EURO-CORDEX).
                """,
                version="",
                license="CC0 (Creative Commons Zero)",
                source="4TU Research Data: https://data.4tu.nl/articles/dataset/Pan-European_data_sets_of_river_flood_probability_of_occurrence_under_present_and_future_climate/12708122",
                resolution="100 m",
                group_id="coastal_tudelft",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
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
                    index_values=[1000],
                    path="maps/inundation/coastal_tudelft/v2/flood_depth_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                units="metres",
                store_netcdf_coords=False,
                scenarios=[
                    Scenario(id="historical", years=[1971]),
                    Scenario(id="rcp4p5", years=[2035, 2085]),
                    Scenario(id="rcp8p5", years=[2035, 2085]),
                ],
            )
        ]
