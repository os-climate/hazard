"""Module for handling the onboarding and processing of WRI - World Resources Institute data."""

import logging
import os
from typing_extensions import Iterable, List, override

from affine import Affine
from pydantic import TypeAdapter

from hazard.onboarder import Onboarder
from hazard.inventory import HazardResource, Period
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.wri_aqueduct import WRIAqueductSource
from hazard.utilities.map_utilities import alphanumeric
from hazard.utilities.tiles import create_tile_set

logger = logging.getLogger(__name__)


# @dataclass
# class BatchItem:
#     """Represent a batch item for hazard processing.

#     It includes scenario, central_year and input_dataset_filename.

#     """

#     resource: HazardResource
#     path: str
#     scenario: str
#     year: str
#     filename_return_period: str  # the filename of the input


class WRIAqueductFlood(Onboarder):
    """On-board the WRI Aqueduct flood model data set from http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html."""

    def __init__(self):
        """WRI Aqueduct Floods model of acute riverine and coastal flood hazards, providing flood intensities as return period maps.

        METADATA:
        Link: https://www.wri.org/aqueduct
        Data type: Riverine and coastal flood hazard probabilities
        Hazard indicator: Flood depth at various return periods
        Region: Global coverage with spatial resolution of 30 × 30 arc seconds (~1 km at the equator)
        Resolution: ~1 km at the equator
        Return periods: 2, 5, 10, 25, 50, 100, 250, 500, 1000 years
        Scenarios: Historical, RCP4.5, RCP8.5
        Data Source: GLOFRIS model

        DATA DESCRIPTION:
        Provides inundation depth data for 9 return periods (reoccurrence intervals) at each point, representing the probability of flood events exceeding a given depth each year (exceedance probability). The model builds on the Global Flood Risk with IMAGE Scenarios (GLOFRIS) and uses multi-scenario inputs to simulate riverine and coastal flood risk.

        IMPORTANT NOTES:
        - Transform coordinates from rotated longitude and latitude for compatibility.
        - Refer to OS-Climate Physical Climate Risk Methodology for methodology details and usage guidelines.

        Args:
            None

        """
        self.resources = {}
        for res in self.inventory():
            for scen in res.scenarios:
                for year in scen.years:
                    path = res.path.format(scenario=scen.id, year=year)
                    self.resources[path] = res
        self.return_periods = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    def _resource(self, path):
        return self.resources[path]

    @override
    def prepare(self, force, download_dir, force_download):
        return super().prepare(force, download_dir, force_download)

    @override
    def is_prepared(self, force=False, force_download=False):
        return super().is_prepared(force, force_download)

    def _get_items_to_process(self) -> Iterable[dict]:
        """Get a list of all batch items."""
        items = (
            self._get_items_to_process_riverine() + self._get_items_to_process_coastal()
        )
        # filtered = [i for i in items if i.resource.path in \
        # ["inundation/wri/v2/inuncoast_historical_nosub_hist_0",
        # "inundation/wri/v2/inuncoast_historical_wtsub_hist_0"]]
        return items

    def _get_items_to_process_riverine(self) -> List[dict]:
        """Get a list of all riverine items."""
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
                    path, filename_return_period = self.path_riverine(
                        scenario, gcm, year
                    )
                    items.append(
                        {
                            "resource": self._resource(path),
                            "path": path,
                            "scenario": scenario,
                            "year": str(year),
                            "filename_return_period": filename_return_period,
                        }
                    )
        # plus one extra historical/baseline item
        scenario, year = "historical", 1980
        path, filename_return_period = self.path_riverine(
            scenario, "000000000WATCH", year
        )
        items.append(
            {
                "resource": self._resource(path),
                "path": path,
                "scenario": scenario,
                "year": str(year),
                "filename_return_period": filename_return_period,
            }
        )
        return items

    def _get_items_to_process_coastal(self) -> List[dict]:
        """Get a list of all coastal items."""
        models = ["0", "0_perc_05", "0_perc_50"]
        subs = ["wtsub", "nosub"]
        years = [2030, 2050, 2080]
        scenarios = ["rcp4p5", "rcp8p5"]
        items = []
        for model in models:
            for sub in subs:
                for year in years:
                    for scenario in scenarios:
                        path, filename_return_period = self.path_coastal(
                            scenario, sub, str(year), model
                        )
                        items.append(
                            {
                                "resource": self._resource(path),
                                "path": path,
                                "scenario": scenario,
                                "year": str(year),
                                "filename_return_period": filename_return_period,
                            }
                        )
        # plus two extra historical/baseline items
        for sub in subs:
            hist_scenario: str = "historical"
            hist_year: str = "hist"
            path, filename_return_period = self.path_coastal(
                hist_scenario, sub, hist_year, "0"
            )
            items.append(
                {
                    "resource": self._resource(path),
                    "path": path,
                    "scenario": hist_scenario,
                    "year": hist_year,
                    "filename_return_period": filename_return_period,
                }
            )
        return items

    def path_riverine(self, scenario: str, gcm: str, year: int):
        """Get riverine paths."""
        path = "inundation/wri/v2/" + f"inunriver_{scenario}_{gcm}_{year}"
        return path, f"inunriver_{scenario}_{gcm}_{year}_rp{{return_period:05d}}"

    def path_coastal(self, scenario: str, sub: str, year: str, model: str):
        """Get coastal paths."""
        path = "inundation/wri/v2/" + f"inuncoast_{scenario}_{sub}_{year}_{model}"
        return (
            path,
            f"inuncoast_{scenario}_{sub}_{year}_rp{{return_period:05d}}_{model}",
        )

    def onboard(self, target, download_dir):
        """Onboards a single hazard resource by running all batch items."""
        source = WRIAqueductSource()
        items = self._get_items_to_process()
        for item in items:
            resource = item["resource"]
            scenario = item["scenario"]
            year = item["year"]
            path = item["path"]
            filename_template = item["filename_return_period"]

            map_path = resource["map"]["path"].format(scenario=scenario, year=year)
            if map_path != (path + "_map"):
                raise ValueError(f"unexpected map path {map_path}")
            self.run_single(item, source, target, None)
            self.generate_tiles_single(item, target, target)
            assert isinstance(target, OscZarr)
            logger.info(f"Running batch item with path {item.path}")
            for i, ret in enumerate(self.return_periods):
                logger.info(f"Copying return period {i + 1}/{len(self.return_periods)}")
                with source.open_dataset(
                    filename_template.format(return_period=ret)
                ) as da:
                    assert da is not None
                    if ret == self.return_periods[0]:
                        z = target.create_empty(
                            resource["path"],
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
                            index_values=self.return_periods,
                        )
                    # ('band', 'y', 'x')
                    values = (
                        da[0, :, :].data
                    )  # will load into memory; assume source not chunked efficiently
                    values[values == -9999.0] = float("nan")
                    z[i, :, :] = values

    def create_maps(self, source: OscZarr, target: OscZarr):
        items = self._get_items_to_process()
        for item in items:
            resource = item["resource"]
            scenario = item["scenario"]
            year = item["year"]
            path = item["path"]

            map_path = resource["map"]["path"].format(scenario=scenario, year=year)
            if map_path != (path + "_map"):
                raise ValueError(f"unexpected map path {map_path}")

        self.generate_tiles_single(item, target, target)

    def generate_tiles_single(self, item, source: OscZarr, target: OscZarr):
        """Generate a tile set for a single batch."""
        resource = item["resource"]
        scenario = item["scenario"]
        year = item["year"]
        path = item["path"]
        logger.info(f"Generating tile-set for batch item with path {path})")
        source_path = item["path"]
        assert resource["map"] is not None
        target_path = resource["map"]["path"].format(scenario=scenario, year=year)
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

    def inventory(self) -> Iterable[HazardResource]:
        """Create the JSON directly, as a demonstration and for the sake of variety."""
        with open(
            os.path.join(os.path.dirname(__file__), "wri_aqueduct_flood.md"), "r"
        ) as f:
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
        license = (
            "WRIs Open Data Commitment: https://www.wri.org/data/open-data-commitment"
        )
        source = "WRI"
        attribution = "Source © WRI — Aqueduct Floods v2 (2020), CC-BY-4.0"

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
                # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "path": "inundation/wri/v2/inunriver_{scenario}_000000000WATCH_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_00000NorESM1-M_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_0000GFDL-ESM2M_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_0000HadGEM2-ES_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_00IPSL-CM5A-LR_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inunriver_{scenario}_MIROC-ESM-CHEM_{year}_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_historical_nosub_hist_0_map",
                    # "inuncoast_historical_nosub_hist_rp{return_period:04d}_0",
                    "source": "map_array_pyramid",  # "mapbox",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_05_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_nosub_{year}_0_perc_50_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_historical_wtsub_hist_0_map",
                    "source": "map_array_pyramid",  # "mapbox",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_05_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
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
                + aqueduct_description,  # noqa:W503
                "map": {
                    "colormap": wri_colormap,
                    "index_values": [8],
                    "path": "inundation/wri/v2/inuncoast_{scenario}_wtsub_{year}_0_perc_50_map",
                    "source": "map_array_pyramid",
                },
                "units": "metres",
                "resolution": "1000 m",
                "license": license,
                "source": source,
                "attribution": attribution,
                "version": "",
                "scenarios": [
                    {"id": "rcp4p5", "years": [2030, 2050, 2080]},
                    {"id": "rcp8p5", "years": [2030, 2050, 2080]},
                ],
            },
        ]
        resources = TypeAdapter(List[HazardResource]).validate_python(
            wri_riverine_inundation_models + wri_coastal_inundation_models
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
                    for period, test_period in zip(
                        scenario.periods, test_periods, strict=False
                    ):
                        if period.map_id != test_period.map_id:
                            raise Exception(
                                f"validation error: hash {period.map_id} different to specified hash {test_period.map_id}"  # noqa: E501
                            )

        return expanded_models
