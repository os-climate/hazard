import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

from affine import Affine
from dask.distributed import Client
from pydantic import parse_obj_as  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource, Period
from hazard.protocols import WriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.wri_aqueduct import WRIAqueductSource
from hazard.utilities.map_utilities import alphanumeric
from hazard.utilities.tiles import create_tile_set

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    resource: HazardResource
    path: str
    scenario: str
    year: str
    filename_return_period: str  # the filename of the input


class WRIAqueductFlood(IndicatorModel):
    """On-board the WRI Aqueduct flood model data set from
    http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html
    """

    def __init__(self):
        self.resources = {}
        for res in self.inventory():
            for scen in res.scenarios:
                for year in scen.years:
                    path = res.path.format(scenario=scen.id, year=year)
                    self.resources[path] = res
        self.return_periods = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    def _resource(self, path):
        return self.resources[path]

    def batch_items(self) -> Iterable[BatchItem]:
        items = self.batch_items_riverine() + self.batch_items_coastal()
        # filtered = [i for i in items if i.resource.path in \
        # ["inundation/wri/v2/inuncoast_historical_nosub_hist_0",
        # "inundation/wri/v2/inuncoast_historical_wtsub_hist_0"]]
        return items

    def batch_items_riverine(self):
        gcms = [
            "00000NorESM1-M",
            "0000GFDL-ESM2M",
            "0000HadGEM2-ES",
            "00IPSL-CM5A-LR",
            "MIROC-ESM-CHEM",
        ]
        years = [2030, 2050, 2080]
        scenarios = ["rcp4p5", "rcp8p5"]
        items = []
        for gcm in gcms:
            for year in years:
                for scenario in scenarios:
                    path, filename_return_period = self.path_riverine(scenario, gcm, year)
                    items.append(
                        BatchItem(
                            self._resource(path),
                            path,
                            scenario,
                            str(year),
                            filename_return_period,
                        )
                    )
        # plus one extra historical/baseline item
        scenario, year = "historical", 1980
        path, filename_return_period = self.path_riverine(scenario, "000000000WATCH", year)
        items.append(BatchItem(self._resource(path), path, scenario, str(year), filename_return_period))
        return items

    def batch_items_coastal(self):
        models = ["0", "0_perc_05", "0_perc_50"]
        subs = ["wtsub", "nosub"]
        years = [2030, 2050, 2080]
        scenarios = ["rcp4p5", "rcp8p5"]
        items = []
        for model in models:
            for sub in subs:
                for year in years:
                    for scenario in scenarios:
                        path, filename_return_period = self.path_coastal(scenario, sub, str(year), model)
                        items.append(
                            BatchItem(
                                self._resource(path),
                                path,
                                scenario,
                                str(year),
                                filename_return_period,
                            )
                        )
        # plus two extra historical/baseline items
        for sub in subs:
            scenario, year = "historical", "hist"
            path, filename_return_period = self.path_coastal(scenario, sub, year, "0")
            items.append(
                BatchItem(
                    self._resource(path),
                    path,
                    scenario,
                    str(year),
                    filename_return_period,
                )
            )
        return items

    def path_riverine(self, scenario: str, gcm: str, year: int):
        path = "inundation/wri/v2/" + f"inunriver_{scenario}_{gcm}_{year}"
        return path, f"inunriver_{scenario}_{gcm}_{year}_rp{{return_period:04d}}"

    def path_coastal(self, scenario: str, sub: str, year: str, model: str):
        path = "inundation/wri/v2/" + f"inuncoast_{scenario}_{sub}_{year}_{model}"
        return (
            path,
            f"inuncoast_{scenario}_{sub}_{year}_rp{{return_period:04d}}_{model}",
        )

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
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_000000000WATCH_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "historical",
                "display_name": "Flood depth/baseline (WRI)",
                "description": """
World Resources Institute Aqueduct Floods baseline riverine model using historical data.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "path": "inundation/wri/v2/inunriver_{scenario}_000000000WATCH_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {
                        "id": "historical",
                        "years": [1980],
                        "periods": [{"year": 1980, "map_id": "gw4vgq"}],
                    }
                ],
            },
            {
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_00000NorESM1-M_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "NorESM1-M",
                "display_name": "Flood depth/NorESM1-M (WRI)",
                "description": """
World Resources Institute Aqueduct Floods riverine model using GCM model from
Bjerknes Centre for Climate Research, Norwegian Meteorological Institute.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_00000NorESM1-M_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_0000GFDL-ESM2M_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "GFDL-ESM2M",
                "display_name": "Flood depth/GFDL-ESM2M (WRI)",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Geophysical Fluid Dynamics Laboratory (NOAA).

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_0000GFDL-ESM2M_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_0000HadGEM2-ES_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "HadGEM2-ES",
                "display_name": "Flood depth/HadGEM2-ES (WRI)",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model:
Met Office Hadley Centre.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_0000HadGEM2-ES_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_00IPSL-CM5A-LR_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "IPSL-CM5A-LR",
                "display_name": "Flood depth/IPSL-CM5A-LR (WRI)",
                "description": """
World Resource Institute Aqueduct Floods riverine model using GCM model from
Institut Pierre Simon Laplace

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_00IPSL-CM5A-LR_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "RiverineInundation",
                "path": "inundation/wri/v2/inunriver_{scenario}_MIROC-ESM-CHEM_{year}",
                "indicator_id": "flood_depth",
                "indicator_model_gcm": "MIROC-ESM-CHEM",
                "display_name": "Flood depth/MIROC-ESM-CHEM (WRI)",
                "description": """World Resource Institute Aqueduct Floods riverine model using
 GCM model from Atmosphere and Ocean Research Institute
 (The University of Tokyo), National Institute for Environmental Studies, and Japan Agency
 for Marine-Earth Science and Technology.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_MIROC-ESM-CHEM_{year}_map",
                    "source": "map_array_pyramid",
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
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_historical_nosub_hist_0",
                "indicator_id": "flood_depth",
                "indicator_model_id": "nosub",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/baseline, no subsidence (WRI)",
                "description": """
World Resources Institute Aqueduct Floods baseline coastal model using historical data. Model excludes subsidence.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_historical_nosub_hist_0_map",
                    # "inuncoast_historical_nosub_hist_rp{return_period:04d}_0",
                    "source": "map_array_pyramid",  # "mapbox",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0",
                "indicator_id": "flood_depth",
                "indicator_model_id": "nosub/95",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/95%, no subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods coastal model, excluding subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_05",
                "indicator_id": "flood_depth/nosub/5",
                "indicator_model_id": "nosub/5",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/5%, no subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods coastal model, excluding subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_05_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_50",
                "indicator_id": "flood_depth",
                "indicator_model_id": "nosub/50",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/50%, no subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_50_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_historical_wtsub_hist_0",
                "indicator_id": "flood_depth",
                "indicator_model_id": "wtsub",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/baseline, with subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods model, excluding subsidence; baseline (based on historical data).

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_historical_wtsub_hist_0_map",
                    "source": "map_array_pyramid",  # "mapbox",
                },
                "units": "metres",
                "scenarios": [{"id": "historical", "years": [1980]}],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0",
                "indicator_id": "flood_depth",
                "indicator_model_id": "wtsub/95",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/95%, with subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 95th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_05",
                "indicator_id": "flood_depth",
                "indicator_model_id": "wtsub/5",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/5%, with subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 5th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_05_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
            {
                "hazard_type": "CoastalInundation",
                "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_50",
                "indicator_id": "flood_depth",
                "indicator_model_id": "wtsub/50",
                "indicator_model_gcm": "unknown",
                "display_name": "Flood depth/50%, with subsidence (WRI)",
                "description": """
World Resource Institute Aqueduct Floods model, including subsidence; 50th percentile sea level rise.

                """
                + aqueduct_description,
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_50_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]
        resources = parse_obj_as(
            List[HazardResource],
            wri_riverine_inundation_models + wri_coastal_inundation_models,
        )
        return resources  # self._expand_resources(resources)

    def _expand_resources(self, models: List[HazardResource]) -> List[HazardResource]:
        expanded_models = [e for model in models for e in model.expand()]
        # we populate map_id hashes programmatically
        for model in expanded_models:
            for scenario in model.scenarios:
                test_periods = scenario.periods
                scenario.periods = []
                for year in scenario.years:
                    if model.map and model.map.path:
                        name_format = model.map.path
                        path = name_format.format(
                            scenario=scenario.id,
                            year=year,
                            id=model.indicator_id,
                            return_period=1000,
                        )
                        id = alphanumeric(path)[0:6]
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

    def run_single(
        self,
        item: BatchItem,
        source: WRIAqueductSource,
        target: WriteDataArray,
        client: Client,
    ):
        assert isinstance(target, OscZarr)
        logger.info(f"Running batch item with path {item.path}")
        for i, ret in enumerate(self.return_periods):
            logger.info(f"Copying return period {i + 1}/{len(self.return_periods)}")
            with source.open_dataset(item.filename_return_period.format(return_period=ret)) as da:
                assert da is not None
                if ret == self.return_periods[0]:
                    z = target.create_empty(
                        item.resource.path,
                        len(da.x),
                        len(da.y),
                        Affine(
                            da.transform[0],
                            da.transform[1],
                            da.transform[2],
                            da.transform[3],
                            da.transform[4],
                            da.transform[5],
                        ),
                        str(da.crs),
                        indexes=self.return_periods,
                    )
                # ('band', 'y', 'x')
                values = da[0, :, :].data  # will load into memory; assume source not chunked efficiently
                values[values == -9999.0] = float("nan")
                z[i, :, :] = values
        print("done")

    def generate_tiles_single(self, item: BatchItem, source: OscZarr, target: OscZarr):
        logger.info(f"Generating tile-set for batch item with path {item.path})")
        source_path = item.path
        assert item.resource.map is not None
        target_path = item.resource.map.path.format(scenario=item.scenario, year=item.year)
        if target_path != source_path + "_map":
            raise ValueError(f"unexpected target path {target_path}")
        create_tile_set(
            source,
            source_path,
            target,
            target_path,
            nodata=-9999.0,
            nodata_as_zero=True,
        )
        # create_tiles_for_resource(source, target, resource)
