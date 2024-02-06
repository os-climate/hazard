import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from rasterio import features
from rasterio.crs import CRS  # type: ignore
from rasterio.enums import MergeAlg
from shapely import union_all

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_and_unzip
from hazard.utilities.xarray_utilities import (affine_to_coords,
                                               enforce_conventions_lat_lon,
                                               global_crs_transform)

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    resource: HazardResource
    scenario: str
    year: str
    indicator: str


class WRIAqueductWaterRiskSource(OpenDataset):
    def __init__(self, source_dir, fs: Optional[AbstractFileSystem] = None):
        """
        Define every attribute of the onboarding class for the Water Resources Institute (WRI)
        water-related risk data.

        METADATA:
        Link: https://www.wri.org/data/aqueduct-global-maps-40-data
        Data type: baseline and future projection indicators of water-related risk
        Hazard indicator: water-related risk
        Region: Global
        Resolution: 10km
        Scenarios: business-as-usual SSP 3 RCP 7.0, optimistic SSP 1 RCP 2.6, and pessimistic SSP 5 RCP 8.5
        Time range: 1979-2019, 2030, 2050, 2080
        File type: CSV

        DATA DESCRIPTION:
        The Aqueduct Water Stress Projections Data include indicators of change in water supply,
        water demand, water stress, and seasonal variability, projected for the coming decades
        under scenarios of climate and economic growth.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance. If none, a LocalFileSystem is used.
        """

        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.zip_url = "https://files.wri.org/aqueduct/aqueduct-4-0-water-risk-data.zip"
        self.prepare()

        self.indicator_map = {
            "baseline": {
                "water_stress": "bws",
                "water_depletion": "bwd",
                "seasonal_variability": "sev",
                "interannual_variability": "iav",
                "groundwater_table_decline": "gtd",
                "riverine_flood_risk": "rfr",
                "coastal_flood_risk": "cfr",
                "drought_risk": "drr",
                "untreated_connected_wastewater": "ucw",
                "coastal_eutrophication_potention": "uep",
                "unimproved_drinking_water": "udw",
                "unimproved_sanitation": "usa",
            },
            "future": {
                "water_demand": "ww",
                "water_supply": "ba",
                "water_stress": "ws",
                "water_depletion": "wd",
                "seasonal_variability": "sv",
                "interannual_variability": "iv",
            },
        }
        self.scenario_map = {"ssp126": "opt", "ssp370": "bau", "ssp585": "pes"}
        self.file_dir = os.path.join(
            os.path.join(self.source_dir, self.zip_url.split("/")[-1].split(".")[0]),
            "Aqueduct40_waterrisk_download_Y2023M07D05",
        )
        self.geometry = gpd.read_file(
            os.path.join(os.path.join(self.file_dir, "GDB"), "Aq40_Y2023D07M05.gdb"),
            include_fields=["pfaf_id", "geometry"],
        )
        self.geometry = (
            self.geometry[self.geometry["pfaf_id"] != -9999]
            .groupby("pfaf_id")
            .apply(lambda x: union_all(x))
            .reset_index()
            .rename(columns={0: "geometry"})
        )

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            download_and_unzip(
                self.zip_url, working_dir, self.zip_url.split("/")[-1].split(".")[0]
            )
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            download_and_unzip(
                self.zip_url, self.source_dir, self.zip_url.split("/")[-1].split(".")[0]
            )

    def open_dataset_year(
        self, _: str, scenario: str, indicator: str, year: int
    ) -> Optional[xr.Dataset]:
        key = "baseline" if scenario == "historical" else "future"
        filename = os.path.join(
            os.path.join(self.file_dir, "CVS"),
            "Aqueduct40_{key}_annual_y2023m07d05.csv".format(key=key),
        )
        if indicator not in self.indicator_map[key]:
            raise ValueError(
                "unexpected indicator {indicator}".format(indicator=indicator)
            )
        column = self.indicator_map[key][indicator]
        if key == "future":
            if scenario not in self.scenario_map:
                raise ValueError(
                    "unexpected scenario {scenario}".format(scenario=scenario)
                )
            column = "_".join(
                [self.scenario_map[scenario] + str(year)[-2:], column, "x", "r"]
            )
        else:
            column = "_".join([column, "raw"])

        if indicator in ["water_stress", "water_depletion"]:
            extra_column = column.replace("_raw", "_cat").replace("_x_r", "_x_c")
            df = pd.read_csv(filename, usecols=["pfaf_id", column, extra_column])
            df[column] = df[[column, extra_column]].apply(
                lambda x: -x[0] if x[1] == -1 else x[0], axis=1
            )
        else:
            df = pd.read_csv(filename, usecols=["pfaf_id", column])

        df = df.merge(self.geometry, how="left", on="pfaf_id", validate="one_to_one")
        width, height = 12 * 360, 12 * 180
        _, transform = global_crs_transform(width, height)
        coords = affine_to_coords(transform, width, height, x_dim="lon", y_dim="lat")
        shapes = [
            (geometry, value) for geometry, value in zip(df["geometry"], df[column])
        ]
        rasterized = features.rasterize(
            shapes,
            out_shape=[height, width],
            transform=transform,
            all_touched=True,
            fill=float("nan"),  # background value
            merge_alg=MergeAlg.replace,
        )
        da = xr.DataArray(rasterized, coords=coords)
        dataset = xr.Dataset({indicator: da})
        return dataset


class WRIAqueductWaterRisk(IndicatorModel[BatchItem]):
    def __init__(
        self,
        scenarios: Iterable[str] = [
            "historical",
            "ssp126",
            "ssp370",
            "ssp585",
        ],
        central_year_historical: int = 1999,
        central_years: Iterable[int] = [2030, 2050, 2080],
        indicators: Iterable[str] = [
            "water_supply",
            "water_demand",
            "water_stress",
            "water_depletion",
        ],
    ):
        self.indicators = indicators
        self.scenarios = scenarios
        self.central_years = central_years
        self.central_year_historical = central_year_historical

    def run_single(
        self, item: BatchItem, source, target: ReadWriteDataArray, _: Client
    ):
        """Run a single item of the batch."""
        logger.info(
            f"Starting calculation for scenario {item.scenario}, indicator {item.indicator} and year {item.year}"
        )
        assert target == None or isinstance(target, OscZarr)
        with ExitStack() as stack:
            dataset = stack.enter_context(
                source.open_dataset_year("", item.scenario, item.indicator, item.year)
            )
            da = enforce_conventions_lat_lon(dataset[item.indicator].array)
            da.attrs["crs"] = CRS.from_epsg(4326)
            logger.info(f"Writing array to {str(item.resource.path)}")
            target.write(str(item.resource.path), da)
        logger.info(
            f"Calculation complete for scenario {item.scenario}, indicator {item.indicator} and year {item.year}"
        )

    def batch_items(self) -> Iterable[BatchItem]:
        items: List[BatchItem] = []
        resources = self._resource()
        for indicator in self.indicators:
            resource = resources[indicator]
            for scenario in self.scenarios:
                for year in (
                    [self.central_year_historical]
                    if scenario == "historical"
                    else self.central_years
                ):
                    items.append(BatchItem(resource, scenario, str(year), indicator))
        return items

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return self._resource().values()  # .expand()

    def _resource(self) -> Dict[str, HazardResource]:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "wri_aqueduct_waterrisk.md"), "r"
        ) as f:
            description = f.read()
        info_map = {
            "water_demand": ("cm/year", "Measure of the total water withdrawals"),
            "water_supply": (
                "cm/year",
                "Measure of the total available renewable surface and ground water supplies",
            ),
            "water_stress": (
                "",
                "Measure of the ratio of total water withdrawals to available renewable surface and ground water supplies "
                + "(-1: Arid and low water use, 0 : Low (<10%), 1: Low-medium (10-20%), 2 : Medium-high (20-40%), 3: High (40-80%), 4 : Extremely high (>80%))",
            ),
            "water_depletion": (
                "",
                "Measure of the ratio of total water consumption to available renewable water supplies "
                + "(-1: Arid and low water use, 0 : Low (<5%), 1: Low-medium (5-25%), 2 : Medium-high (25-50%), 3: High (50-75%), 4 : Extremely high (>75%))",
            ),
        }
        resources: Dict[str, HazardResource] = dict()
        for indicator in self.indicators:
            path = (
                "water_risk/wri/v4/{indicator}".format(indicator=indicator)
                + "_{scenario}_{year}"
            )
            units, display_name = (
                info_map[indicator] if indicator in info_map else ("", "")
            )
            resources[indicator] = HazardResource(
                hazard_type="WaterRisk",
                indicator_id=indicator,
                indicator_model_id=None,
                indicator_model_gcm="combined",
                params={},
                path=path,
                display_name=display_name,
                description=description,
                display_groups=[display_name],
                group_id="",
                map=MapInfo(
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=-1.0,
                        max_value=1.0,
                        max_index=255,
                        units=units,
                    ),
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    path=os.path.join("maps", path + "_map"),
                    source="map_array",
                ),
                units=units,
                scenarios=[
                    Scenario(id="historical", years=[self.central_year_historical]),
                    Scenario(id="ssp126", years=list(self.central_years)),
                    Scenario(id="ssp370", years=list(self.central_years)),
                    Scenario(id="ssp585", years=list(self.central_years)),
                ],
            )
        return resources
