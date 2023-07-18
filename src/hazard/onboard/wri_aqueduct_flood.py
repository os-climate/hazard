from dataclasses import dataclass
import os
from pathlib import PosixPath
from dask.distributed import Client
from fsspec.spec import AbstractFileSystem # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from pydantic import parse_obj_as # type: ignore
import rasterio # type: ignore
from rasterio.crs import CRS # type: ignore
import rasterio.enums # type: ignore
import rioxarray
import xarray as xr
from typing import Dict, Iterable, List
from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Period, Scenario
from hazard.protocols import OpenDataset, ReadDataArray, WriteDataArray, WriteDataset
from hazard.utilities import xarray_utilities
from hazard.utilities.map_utilities import alphanumeric, check_map_bounds, transform_epsg4326_to_epsg3857

@dataclass
class BatchItem:
    resource: HazardResource

class WriAqueductFlood(IndicatorModel):
    """On-board the WRI Aqueduct flood model data set.
    """
    
    def batch_items(self) -> Iterable[BatchItem]:
        raise NotImplementedError()

    def inventory(self) -> Iterable[HazardResource]:
        """Here we create the JSON directly, as a demonstration and for the sake of variety."""
        with open(os.path.join(os.path.dirname(__file__), "wri_aqueduct_flood.md"), "r") as f:
            aqueduct_description = f.read()

        wri_colormap = {
            "name": "flare",
            "nodata_index": 0,
            "min_index": 1,
            "min_value": 0.0,
            "max_index": 255,
            "max_value": 2.0,
            "units": "m",
        }

        wri_riverine_inundation_models = [
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "000000000WATCH",
                "display_name": "WRI/Baseline",
                "description": """
World Resources Institute Aqueduct Floods baseline riverine model using historical data.

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980], "periods": [{"year": 1980, "map_id": "gw4vgq"}]}],
            },
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "00000NorESM1-M",
                "display_name": "WRI/NorESM1-M",
                "description": """
World Resources Institute Aqueduct Floods riverine model using GCM model from
Bjerknes Centre for Climate Research, Norwegian Meteorological Institute.

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "0000GFDL-ESM2M",
                "display_name": "WRI/GFDL-ESM2M",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Geophysical Fluid Dynamics Laboratory (NOAA).

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "0000HadGEM2-ES",
                "display_name": "WRI/HadGEM2-ES",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model:
Met Office Hadley Centre.

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "00IPSL-CM5A-LR",
                "display_name": "WRI/IPSL-CM5A-LR",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Institut Pierre Simon Laplace

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "RiverineInundation",
                "path": "inundation/wri/v2",
                "id": "MIROC-ESM-CHEM",
                "display_name": "WRI/MIROC-ESM-CHEM",
                "description": """World Resource Institute Aqueduct Floods riverine model using
 GCM model from Atmosphere and Ocean Research Institute
 (The University of Tokyo), National Institute for Environmental Studies, and Japan Agency
 for Marine-Earth Science and Technology.

                """
                + aqueduct_description,
                "array_name": "inunriver_{scenario}_{id}_{year}",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inunriver_{scenario}_{id}_{year}_rp{return_period:05d}",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {
                        "id": "rcp4p5",
                        "years": [2030, 2050, 2080],
                        "periods": [
                            {"year": 2030, "map_id": "ht2kn3"},
                            {"year": 2050, "map_id": "1k4boi"},
                            {"year": 2080, "map_id": "3rok7b"},
                        ],
                    },
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]

        wri_coastal_inundation_models = [
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "nosub",
                "display_name": "WRI/Baseline no subsidence",
                "description": """
World Resources Institute Aqueduct Floods baseline coastal model using historical data. Model excludes subsidence.

                """
                + aqueduct_description,
                "array_name": "inuncoast_historical_nosub_hist_0",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_historical_nosub_hist_rp{return_period:04d}_0",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "nosub/95",
                "display_name": "WRI/95% no subsidence",
                "description": """
World Resource Institute Aqueduct Floods coastal model, exclusing subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_nosub_{year}_0",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "nosub/5",
                "display_name": "WRI/5% no subsidence",
                "description": """
World Resource Institute Aqueduct Floods coastal model, excluding subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_nosub_{year}_0_perc_05",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0_perc_05",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "nosub/50",
                "display_name": "WRI/50% no subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_nosub_{year}_0_perc_50",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_nosub_{year}_rp{return_period:04d}_0_perc_50",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "wtsub",
                "display_name": "WRI/Baseline with subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; baseline (based on historical data).

                """
                + aqueduct_description,
                "array_name": "inuncoast_historical_wtsub_hist_0",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_historical_wtsub_hist_rp{return_period:04d}_0",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "wtsub/95",
                "display_name": "WRI/95% with subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_wtsub_{year}_0",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "wtsub/5",
                "display_name": "WRI/5% with subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_wtsub_{year}_0_perc_05",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0_perc_05",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "type": "CoastalInundation",
                "path": "inundation/wri/v2",
                "id": "wtsub/50",
                "display_name": "WRI/50% with subsidence",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "array_name": "inuncoast_{scenario}_wtsub_{year}_0_perc_50",
                "map": {
                    "colormap": wri_colormap,
                    "array_name": "inuncoast_{scenario}_wtsub_{year}_rp{return_period:04d}_0_perc_50",
                    "source": "mapbox",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]
        resources = parse_obj_as(List[HazardResource], wri_riverine_inundation_models + wri_coastal_inundation_models)
        return self._expand_resources(resources)
        

    def _expand_resources(self, models: List[HazardResource]) -> List[HazardResource]:
        expanded_models = [e for model in models for e in model.expand()]
        # we populate map_id hashes programmatically
        for model in expanded_models:
            for scenario in model.scenarios:
                test_periods = scenario.periods
                scenario.periods = []
                for year in scenario.years:
                    if model.map and model.map.array_name:
                        name_format = model.map.array_name
                        array_name = name_format.format(
                            scenario=scenario.id, year=year, id=model.id, return_period=1000
                        )
                        id = alphanumeric(array_name)[0:6]
                    else:
                        id = ""
                    scenario.periods.append(Period(year=year, map_id=id))
                # if a period was specified explicitly, we check that hash is the same: a build-in check
                if test_periods is not None:
                    for period, test_period in zip(scenario.periods, test_periods):
                        if period.map_id != test_period.map_id:
                            raise Exception(
                                f"validation error: hash {period.map_id} different to specified hash {test_period.map_id}"  # noqa: E501
                            )

        return expanded_models


    def run_single(self, item: BatchItem, source: ReadDataArray, target: WriteDataArray, client: Client):
        raise NotImplementedError()
