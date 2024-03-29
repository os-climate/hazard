import logging
import os
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
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import affine_to_coords, enforce_conventions_lat_lon, global_crs_transform

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    indicator: str
    scenario: str
    year: int


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
        The Aqueduct 4.0 Global Water Risk Indicators include indicators of change in water supply,
        water demand, water stress, water depletion as well as interannual and seasonal variability,
        projected for the coming decades under scenarios of climate and economic growth.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
            If none, a LocalFileSystem is used.
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

        geometry = gpd.read_file(
            os.path.join(os.path.join(self.file_dir, "GDB"), "Aq40_Y2023D07M05.gdb"),
            include_fields=["aq30_id", "pfaf_id", "geometry"],
        )

        self.geometry: Dict[str, pd.DataFrame] = dict()

        # The jointure is based on aq30_id for the baseline:
        self.geometry["aq30_id"] = (
            geometry.drop(columns=["pfaf_id"])
            .groupby("aq30_id")
            .apply(lambda x: union_all(x))
            .reset_index()
            .rename(columns={0: "geometry"})
        )

        # The jointure is based on pfaf_id for future projections:
        self.geometry["pfaf_id"] = (
            geometry[geometry["pfaf_id"] != -9999]
            .drop(columns=["aq30_id"])
            .groupby("pfaf_id")
            .apply(lambda x: union_all(x))
            .reset_index()
            .rename(columns={0: "geometry"})
        )

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            assert working_dir is not None
            download_and_unzip(self.zip_url, working_dir, self.zip_url.split("/")[-1].split(".")[0])
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            download_and_unzip(self.zip_url, self.source_dir, self.zip_url.split("/")[-1].split(".")[0])

    def open_dataset_year(
        self,
        gcm: str,
        scenario: str,
        indicator: str,
        year: int,
        chunks=None,
    ):
        key = "baseline" if scenario == "historical" else "future"
        joint_on = "aq30_id" if scenario == "historical" else "pfaf_id"
        filename = os.path.join(
            os.path.join(self.file_dir, "CVS"),
            "Aqueduct40_{key}_annual_y2023m07d05.csv".format(key=key),
        )
        if indicator not in self.indicator_map[key]:
            raise ValueError("unexpected indicator {indicator}".format(indicator=indicator))
        label = self.indicator_map[key][indicator]
        if key == "future":
            if scenario not in self.scenario_map:
                raise ValueError("unexpected scenario {scenario}".format(scenario=scenario))
            label = "_".join([self.scenario_map[scenario] + str(year)[-2:], label, "x", "r"])
        else:
            label = "_".join([label, "raw"])

        if indicator in ["water_stress", "water_depletion"]:
            category = label.replace("_raw", "_cat").replace("_x_r", "_x_c")
            df = pd.read_csv(filename, usecols=[joint_on, label, category]).rename(
                columns={label: indicator, category: "_".join([indicator, "category"])}
            )
        else:
            df = pd.read_csv(filename, usecols=[joint_on, label]).rename(columns={label: indicator})
        df = (
            df[df[indicator] != -9999]
            .drop_duplicates()
            .merge(self.geometry[joint_on], how="left", on=joint_on, validate="one_to_one")
            .drop(columns=[joint_on])
        )

        # The grid resolution is 5 × 5 arc minutes:
        width, height = 12 * 360, 12 * 180
        _, transform = global_crs_transform(width, height)
        coords = affine_to_coords(transform, width, height, x_dim="lon", y_dim="lat")

        da: Dict[str, xr.DataArray] = dict()
        for column in [column for column in df.columns if column != "geometry"]:
            shapes = [(geometry, value) for geometry, value in zip(df["geometry"], df[column])]
            rasterized = features.rasterize(
                shapes,
                out_shape=[height, width],
                transform=transform,
                all_touched=True,
                fill=float("nan"),  # background value
                merge_alg=MergeAlg.replace,
            )
            da[column] = xr.DataArray(rasterized, coords=coords)
            da[column] = enforce_conventions_lat_lon(da[column])
            da[column].attrs["crs"] = CRS.from_epsg(4326)
        dataset = xr.Dataset(da)
        return dataset


class WRIAqueductWaterSupplyDemandBaselineSource(OpenDataset):
    def __init__(self, source_dir, fs: Optional[AbstractFileSystem] = None):
        """
        Built baseline water demand/supply from the future values as well as their change from baseline.

        METADATA:
        Link: https://www.wri.org/data/aqueduct-water-stress-projections-data
        Data type: baseline and future projection indicators of water-related risk
        Hazard indicator: water-related risk
        Region: Global
        Resolution: 22km
        Scenarios: optimistic SSP 2 RCP 4.5, business-as-usual SSP 2 RCP 8.5, and pessimistic SSP 3 RCP 8.5
        Time range: 2020, 2030, 2040
        File type: Shape File (.shx)

        DATA DESCRIPTION:
        The Aqueduct Global Maps 2.1 include indicators of water stress, seasonal variability, water demand
        and water supply, projected for the coming decades under scenarios of climate and economic growth.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
            If none, a LocalFileSystem is used.
        """

        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.zip_url = "https://files.wri.org/d8/s3fs-public/aqueduct_projections_20150309_shp.zip"
        self.prepare()

        self.indicator_map = {
            "water_demand": "ut",
            "water_supply": "bt",
            "water_stress": "ws",
            "seasonal_variability": "sv",
        }
        self.scenario_map = {"ssp245": "24", "ssp285": "28", "ssp385": "38"}
        self.years = [2020, 2030, 2040]
        self.filename = os.path.join(
            self.source_dir,
            self.zip_url.split("/")[-1].split(".")[0],
            "aqueduct_projections_20150309.shx",
        )

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            assert working_dir is not None
            download_and_unzip(self.zip_url, working_dir, self.zip_url.split("/")[-1].split(".")[0])
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            download_and_unzip(self.zip_url, self.source_dir, self.zip_url.split("/")[-1].split(".")[0])

    def open_dataset_year(self, _: str, scenario: str, indicator: str, year: int, chunks=None) -> Optional[xr.Dataset]:
        if indicator.replace("_multiplier", "") not in self.indicator_map:
            raise ValueError("unexpected indicator {indicator}".format(indicator=indicator.replace("_multiplier", "")))

        df = gpd.read_file(self.filename)
        if "_multiplier" in indicator:
            df[indicator] = df[
                [self.indicator_map[indicator.replace("_multiplier", "")] + "3024cr"]  # hard-coded
            ].apply(lambda x: 1.0 if x[0] == 0.0 else 1.0 / x[0], axis=1)
        else:
            keys = [
                self.indicator_map[indicator] + str(year)[-2:] + self.scenario_map[scenario_key]
                for year in self.years
                for scenario_key in self.scenario_map
            ]
            for key in keys:
                df[key] = df[[key + "tr", key + "cr"]].apply(
                    lambda x: 0.0 if x[1] == 0 else 100.0 * x[0] / x[1], axis=1
                )
            df[indicator] = df[keys].mean(axis=1)

        width, height = 12 * 360, 12 * 180
        _, transform = global_crs_transform(width, height)
        coords = affine_to_coords(transform, width, height, x_dim="lon", y_dim="lat")

        shapes = [(geometry, value) for geometry, value in zip(df["geometry"], df[indicator])]

        rasterized = features.rasterize(
            shapes,
            out_shape=[height, width],
            transform=transform,
            all_touched=True,
            fill=float("nan"),  # background value
            merge_alg=MergeAlg.replace,
        )
        da = xr.DataArray(rasterized, coords=coords)
        da = enforce_conventions_lat_lon(da)
        da.attrs["crs"] = CRS.from_epsg(4326)
        return xr.Dataset({indicator: da})


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
        self.resources = self._resources()

    def run_single(self, item: BatchItem, source, target: ReadWriteDataArray, _: Client):
        """Run a single item of the batch."""

        indicator_suffix = ""
        if type(source).__name__ == "WRIAqueductWaterSupplyDemandBaselineSource":
            indicator_suffix = "_multiplier"
            if item.indicator not in ["water_demand", "water_supply"] or item.scenario != "historical":
                return

        if type(source).__name__ == "WRIAqueductWaterRiskSource":
            if item.indicator in ["water_demand", "water_supply"] and item.scenario == "historical":
                return

        logger.info(
            "Starting calculation for {indicator}/{scenario}/{year}".format(
                indicator=item.indicator, scenario=item.scenario, year=str(item.year)
            )
        )

        assert target is None or isinstance(target, OscZarr)

        dataset = source.open_dataset_year("", item.scenario, item.indicator + indicator_suffix, item.year)
        for key in dataset:
            if key.replace("_multiplier", "") in self.resources:
                path = self.resources[key.replace("_multiplier", "")].path.format(
                    scenario=item.scenario, year=item.year
                )
                logger.info(f"Writing array to {path}")
                if "_multiplier" in key:
                    # hard-coded
                    reference_path = path.replace(item.scenario, "ssp126").replace(str(item.year), "2030")
                    reference_data = target.read(reference_path)
                    dataset[key].values *= reference_data[0].values
                target.write(path, dataset[key])
        logger.info(
            "Calculation complete for {indicator}/{scenario}/{year}".format(
                indicator=item.indicator, scenario=item.scenario, year=str(item.year)
            )
        )

    def batch_items(self) -> Iterable[BatchItem]:
        items: List[BatchItem] = []
        for indicator in self.indicators:
            for scenario in self.scenarios:
                for year in [self.central_year_historical] if scenario == "historical" else self.central_years:
                    items.append(BatchItem(indicator, scenario, year))
        return items

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        for key in self.resources:
            create_tiles_for_resource(source, target, self.resources[key])

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return self.resources.values()  # .expand()

    def _resources(self) -> Dict[str, HazardResource]:
        """Create resource."""
        resource_map = {
            "water_demand": {
                "units": "centimeters/year",
                "display": "Water demand in centimeters/year (Aqueduct 4.0)",
                "min_value": 0.0,
                "max_value": 100,
            },
            "water_supply": {
                "units": "centimeters/year",
                "display": "Water supply in centimeters/year (Aqueduct 4.0)",
                "min_value": 0.0,
                "max_value": 2000,
            },
            "water_stress": {
                "units": "",
                "display": "Water stress (Aqueduct 4.0)",
                "min_value": 0.0,
                "max_value": 2.0,
            },
            "water_depletion": {
                "units": "",
                "display": "Water depletion (Aqueduct 4.0)",
                "min_value": 0.0,
                "max_value": 2.0,
            },
            "water_stress_category": {
                "units": "",
                "display": "Water stress category (Aqueduct 4.0)",
                "min_value": -5,
                "max_value": 5,
            },
            "water_depletion_category": {
                "units": "",
                "display": "Water depletion category (Aqueduct 4.0)",
                "min_value": -5,
                "max_value": 5,
            },
        }

        with open(os.path.join(os.path.dirname(__file__), "wri_aqueduct_water_risk.md"), "r") as f:
            description = f.read()

        resources: Dict[str, HazardResource] = dict()
        for key in resource_map:
            indicator = key.replace("_category", "")
            if indicator in self.indicators:
                path = "water_risk/wri/v2/{key}".format(key=key) + "_{scenario}_{year}"
                resources[key] = HazardResource(
                    hazard_type="WaterRisk",
                    indicator_id=key,
                    indicator_model_id=None,
                    indicator_model_gcm="combined",
                    params={},
                    path=path,
                    display_name=str(resource_map[key]["display"]),
                    description=description,
                    display_groups=[str(resource_map[key]["display"])],
                    group_id="",
                    map=MapInfo(
                        colormap=Colormap(
                            name="heating",
                            min_value=resource_map[key]["min_value"],  # type:ignore
                            max_value=resource_map[key]["max_value"],  # type:ignore
                            min_index=1,
                            max_index=255,
                            nodata_index=0,
                            units=resource_map[key]["units"],  # type:ignore
                        ),
                        bounds=[
                            (-180.0, 85.0),
                            (180.0, 85.0),
                            (180.0, -85.0),
                            (-180.0, -85.0),
                        ],
                        index_values=None,
                        path="maps/" + path + "_map",
                        source="map_array_pyramid",
                    ),
                    units=resource_map[key]["units"],  # type:ignore
                    scenarios=(
                        [
                            Scenario(id="historical", years=[self.central_year_historical]),
                            Scenario(id="ssp126", years=list(self.central_years)),
                            Scenario(id="ssp370", years=list(self.central_years)),
                            Scenario(id="ssp585", years=list(self.central_years)),
                        ]
                    ),
                )
        return resources
