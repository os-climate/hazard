# https://www.wri.org/data/aqueduct-global-maps-40-data

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests  # type: ignore
import xarray as xr
from dask.distributed import Client
from rasterio import features
from rasterio.crs import CRS  # type: ignore
from rasterio.enums import MergeAlg
from rasterio.plot import show
from shapely import union_all

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, ReadWriteDataArray
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
    def __init__(self, working_directory):
        self.path = os.path.join(
            working_directory, "Aqueduct40_waterrisk_download_Y2023M07D05"
        )
        self.baseline_indicator_map = {
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
        }
        self.projection_indicator_map = {
            "water_demand": "ww",
            "water_supply": "ba",
            "water_stress": "ws",
            "water_depletion": "wd",
            "seasonal_variability": "sv",
            "interannual_variability": "iv",
        }
        self.scenario_map = {"ssp245": "opt", "ssp370": "bau", "ssp585": "pes"}
        if not os.path.exists(self.path):
            url = "https://files.wri.org/aqueduct/aqueduct-4-0-water-risk-data.zip"
            file = os.path.join(working_directory, url.split("/")[-1])
            self._download_and_unzip(url, file, working_directory)
        self.geometry = gpd.read_file(
            os.path.join(os.path.join(self.path, "GDB"), "Aq40_Y2023D07M05.gdb"),
            include_fields=["pfaf_id", "geometry"],
        )
        self.geometry = (
            self.geometry[self.geometry["pfaf_id"] != -9999]
            .groupby("pfaf_id")
            .apply(lambda x: union_all(x))
            .reset_index()
            .rename(columns={0: "geometry"})
        )

    def open_dataset_year(
        self, _: str, scenario: str, indicator: str, year: int
    ) -> Optional[xr.Dataset]:
        scenario_id = None
        if scenario == "historical":
            path = os.path.join(
                os.path.join(self.path, "CVS"),
                "Aqueduct40_{}_annual_y2023m07d05.csv".format("baseline"),
            )
            if indicator in self.baseline_indicator_map:
                scenario_id = "_".join([self.baseline_indicator_map[indicator], "raw"])
        else:
            path = os.path.join(
                os.path.join(self.path, "CVS"),
                "Aqueduct40_{}_annual_y2023m07d05.csv".format("future"),
            )
            if indicator in self.projection_indicator_map:
                if scenario in self.scenario_map:
                    scenario_id = "_".join(
                        [
                            self.scenario_map[scenario] + str(year)[-2:],
                            self.projection_indicator_map[indicator],
                            "x",
                            "r",
                        ]
                    )

        if scenario_id is not None:
            df = pd.read_csv(path, usecols=["pfaf_id", scenario_id])
            df = df.merge(
                self.geometry, how="left", on="pfaf_id", validate="one_to_one"
            )
            shapes = [
                (geometry, value)
                for geometry, value in zip(df["geometry"], df[scenario_id])
            ]
            width, height = 3600, 1800
            _, transform = global_crs_transform(width, height)
            rasterized = features.rasterize(
                shapes,
                out_shape=[height, width],
                transform=transform,
                all_touched=True,
                fill=float("nan"),  # background value
                merge_alg=MergeAlg.replace,
            )
            _, ax = plt.subplots(1, figsize=(10, 10))
            show(rasterized, ax=ax)
            coords = affine_to_coords(
                transform, width, height, x_dim="lon", y_dim="lat"
            )
            da = xr.DataArray(rasterized, coords=coords)
            dataset = xr.Dataset({indicator: da})
            return dataset
        return None

    @staticmethod
    def _download_and_unzip(url: str, file: str, dir: str):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)
            shutil.unpack_archive(file, dir)


class WRIAqueductWaterRisk(IndicatorModel[BatchItem]):
    def __init__(
        self,
        scenarios: Iterable[str] = [
            "historical",
            "ssp245",
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
        dataset = source.open_dataset_year("", item.scenario, item.indicator, item.year)
        da = enforce_conventions_lat_lon(dataset.array)
        da.attrs["crs"] = CRS.from_epsg(4326)
        logger.info(f"Writing array to {str(item.resource.path)}")
        target.write(str(item.resource.path), da)

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
                "Measure of the ratio of total water withdrawals to available renewable surface and ground water supplies",
            ),
            "water_depletion": (
                "",
                "Measure of the ratio of total water consumption to available renewable water supplies",
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
                        min_value=0.0,
                        max_value=100,
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
                    Scenario(id="ssp245", years=list(self.central_years)),
                    Scenario(id="ssp370", years=list(self.central_years)),
                    Scenario(id="ssp585", years=list(self.central_years)),
                ],
            )
        return resources
