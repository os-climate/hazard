"""Module for IPCC Drought Indicator model.

This module defines the IPCC Drought Indicator model, including classes and methods
for preparing, processing, and generating maps for drought-related indicators such as
Standardized Precipitation Index (SPI6) and Consecutive Dry Days (CDD). It handles
data management, file processing, and integration with the hazard resource inventory.
"""

import io
import requests
import logging
import os
from pathlib import Path, PurePath, PurePosixPath
import zipfile
from typing_extensions import Iterable, Optional, override

import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class IPCCDrought(Onboarder):
    """IPCC Drought Indicator model class.

    This class handles the preparation, processing, and map generation
    for IPCC drought indicators like SPI6 and CDD.

    Attributes:
        source_dir (str): Directory containing source data.
        fs (AbstractFileSystem, optional): Filesystem for managing files.

    """

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Initialize the IPCCDrought model.

        Args:
            source_dir_base (str): Directory for the source files.
            fs (AbstractFileSystem, optional): Filesystem interface (default is LocalFileSystem).

        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = PurePath(source_dir_base, "ipcc_drought").as_posix() + "/"
        os.makedirs(self.source_dir, exist_ok=True)

        # Nuevas URLs para la descarga (se obtendrán archivos ZIP que contienen "map.nc")
        self.urls = {
            "SPI6": {
                "historical": {
                    "1995": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=1986-2005&DIM_scenario=historical&DIM_time_filter=Annual&DIM_domain=global&layers=spi6&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6.cryo_div_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148"
                },
                "rcp2p6": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp26&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp26&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
                "rcp4p5": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp45&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp45&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
                "rcp8p5": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp85&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp85&DIM_time_filter=Annual&DIM_domain=global&DIM_baseline=1986-2005&layers=spi6_anom%2Cspi6_anom_hatching_advanced&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=spi6_anom.cryo_div_19%2Chatching_advanced&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
            },
            "CDD": {
                "historical": {
                    "1995": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=1986-2005&DIM_scenario=historical&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148"
                },
                "rcp2p6": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp26&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp26&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
                "rcp4p5": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp45&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp45&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
                "rcp8p5": {
                    "2030": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2021-2040&DIM_scenario=rcp85&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                    "2050": r"https://atlas.climate.copernicus.eu/export/map/netcdf?DIM_project=CORDEX-EUR-11&DIM_period=2041-2060&DIM_scenario=rcp85&DIM_time_filter=Annual&DIM_domain=global&layers=cdd&FORMAT=image%2Fpng&SERVICE=WMS&VERSION=1.1.1&styles=cdd.temp_seq_19&SRS=ESRI%3A54030&hatching=PROJECTIONS&BBOX=-18016851.304543436%2C-15411389.820994148%2C18016851.304543436%2C15411389.820994148",
                },
            },
        }
        self._resources = list(self.inventory())

    @override
    def prepare(self, download_dir=None):
        """Prepare the source data for the IPCC Drought model."""
        self.fs.makedirs(self.source_dir, exist_ok=True)
        for variable_name, scenarios in self.urls.items():
            for scenario, years in scenarios.items():
                for year, url in years.items():
                    # Descargar el archivo ZIP en memoria
                    response = requests.get(url)
                    response.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for file in z.namelist():
                            if file == "map.nc":
                                output_filename = (
                                    f"{variable_name}_{scenario}_{year}.nc"
                                )
                                output_path = os.path.join(
                                    self.source_dir, output_filename
                                )
                                with (
                                    z.open(file) as source,
                                    open(output_path, "wb") as dest,
                                ):
                                    dest.write(source.read())

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        if force or force_download:
            return False
        return (
            self.fs.exists(self.source_dir)
            and len(self.fs.listdir(self.source_dir)) > 0
        )

    @override
    def onboard(self, target):
        items_to_process = self._get_items_to_process()
        for item in items_to_process:
            source_dir = PurePosixPath(self.source_dir)
            filename = os.path.basename(item["input_dataset_filename"])
            local_file = source_dir / filename
            variable_name = item["indicator"].lower()
            try:
                ds = xr.open_dataset(
                    str(local_file), decode_times=True, mask_and_scale=True
                )
                da = ds[variable_name]
                logger.info(
                    f"Dataset {filename} opened with variable '{variable_name}'."
                )
            except KeyError:
                logger.error(
                    f"Variable '{variable_name}' not found in dataset '{filename}'."
                )
                return
            except Exception as e:
                logger.error(f"Error opening dataset {filename}: {e}")
                return
            # Lógica de escalado y combinación para SPI6
            if variable_name == "spi6":
                if item["scenario"].lower() == "historical":
                    da.values = da.values * 100
            elif variable_name == "spi6_anom":
                # Escenario futuro: combinar la data histórica de referencia con la anomalía
                hist_file = Path(source_dir) / "SPI6_historical_1995.nc"
                if not hist_file.exists():
                    logger.error(f"Historical file {hist_file} not found.")
                    return
                ds_hist = xr.open_dataset(hist_file)
                # Escalar el dataset histórico a 100
                ds_hist["spi6"] = ds_hist["spi6"] * 100
                # Se asume que en el dataset futuro la anomalía se llama "spi6_anom"
                if "spi6_anom" not in ds.variables:
                    logger.error(f"'spi6_anom' not found in dataset '{filename}'.")
                    return
                da = ds_hist["spi6"] + ds_hist["spi6"] * np.abs(ds["spi6_anom"])

            # Eliminar dimensiones redundantes y reordenar a (lat, lon) si es necesario
            da = da.squeeze().reset_coords(drop=True)
            if set(da.dims) != {"lat", "lon"}:
                try:
                    da = da.transpose("lat", "lon")
                    logger.info(f"Transposed dimensions to: {da.dims}.")
                except ValueError:
                    logger.warning(
                        f"Could not transpose to (lat, lon). Dimensions are {da.dims}."
                    )
            if variable_name == "spi6_anom":
                variable_name = "spi6"
            # Ajuste de límites (bounds) del recurso
            try:
                resource = next(
                    res
                    for res in self._resources
                    if res.indicator_id.lower() == variable_name
                )
            except StopIteration:
                logger.error(f"No resource found for variable '{variable_name}'.")
                return
            if resource.map is not None and not resource.map.bounds:
                lat_bounds = [float(da.lat.min()), float(da.lat.max())]
                lon_bounds = [float(da.lon.min()), float(da.lon.max())]
                resource.map.bounds = [
                    (lon_bounds[0], lat_bounds[0]),  # (lon_min, lat_min)
                    (lon_bounds[1], lat_bounds[1]),
                ]
                logger.info(
                    f"Set map bounds for '{variable_name}': {resource.map.bounds}"
                )

            path_ = resource.path.format(
                scenario=item["scenario"].lower(), year=item["central_year"]
            )
            try:
                target.write(path_, da)
                logger.info(
                    f"Successfully wrote data for '{variable_name}' to '{path_}'."
                )
            except Exception as e:
                logger.error(
                    f"Failed to write data for '{variable_name}' to '{path_}': {e}"
                )

    def _get_items_to_process(self):
        return [
            {
                "indicator": "SPI6",
                "scenario": "historical",
                "central_year": 1995,
                "input_dataset_filename": "SPI6_historical_1995.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp2p6",
                "central_year": 2030,
                "input_dataset_filename": "SPI6_rcp2p6_2030.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp2p6",
                "central_year": 2050,
                "input_dataset_filename": "SPI6_rcp2p6_2050.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp4p5",
                "central_year": 2030,
                "input_dataset_filename": "SPI6_rcp4p5_2030.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp4p5",
                "central_year": 2050,
                "input_dataset_filename": "SPI6_rcp4p5_2050.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp8p5",
                "central_year": 2030,
                "input_dataset_filename": "SPI6_rcp8p5_2030.nc",
            },
            {
                "indicator": "SPI6_ANOM",
                "scenario": "rcp8p5",
                "central_year": 2050,
                "input_dataset_filename": "SPI6_rcp8p5_2050.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "historical",
                "central_year": 1995,
                "input_dataset_filename": "CDD_historical_1995.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp2p6",
                "central_year": 2030,
                "input_dataset_filename": "CDD_rcp2p6_2030.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp2p6",
                "central_year": 2050,
                "input_dataset_filename": "CDD_rcp2p6_2050.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp4p5",
                "central_year": 2030,
                "input_dataset_filename": "CDD_rcp4p5_2030.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp4p5",
                "central_year": 2050,
                "input_dataset_filename": "CDD_rcp4p5_2050.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp8p5",
                "central_year": 2030,
                "input_dataset_filename": "CDD_rcp8p5_2030.nc",
            },
            {
                "indicator": "CDD",
                "scenario": "rcp8p5",
                "central_year": 2050,
                "input_dataset_filename": "CDD_rcp8p5_2050.nc",
            },
        ]

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Generate map images for each hazard resource in the dataset.

        Args:
            source (OscZarr): Source data object containing hazard data.
            target (OscZarr): Target data object where the output maps will be written.

        """
        for resource in self.inventory():
            try:
                create_tiles_for_resource(source, target, resource)
                logger.info(
                    f"Successfully created map tiles for resource '{resource.indicator_id}'."
                )
            except Exception as e:
                logger.error(
                    f"Failed to create map tiles for resource '{resource.indicator_id}': {e}"
                )

    def inventory(self) -> Iterable[HazardResource]:
        return [
            # SPI6 resource
            HazardResource(
                hazard_type="Drought",
                indicator_id="spi6",
                indicator_model_id="ipcc",
                indicator_model_gcm="IPCC",
                path="drought/ipcc/v1/spi6_{scenario}_{year}",
                params={},
                display_name="Standardized Precipitation Index (SPI-6)",
                resolution="30400 m",
                description=r"""
                   ## Standardized Precipitation Index (SPI-6)

                    The **SPI-6** is a monthly index that compares accumulated precipitation over a 6-month period with the long-term precipitation distribution for the same location and accumulation period. The index is expressed in terms of the number of standard deviations from the median (reference period: 1971–2010), providing a standardized measure of drought severity.

                    **Calculation & Drought Definition:**
                    The **SPI** is calculated by comparing cumulated precipitation over \( n \) months (commonly \( n=6 \) or \( n=12 \)) with the long-term precipitation distribution. Following Spinoni et al. (2014), a drought event is defined to begin in the month when the SPI drops below \(-1\) and to end when the SPI becomes positive for **at least two consecutive months**.

                    **Dataset Information:**
                    - **Dataset:** CORDEX Coordinated Output for Regional Evaluations
                    - **CORDEX-CORE:**
                      CORDEX-CORE is an initiative developed to produce homogeneous regional climate model projections at approximately 25 km resolution for all major inhabited regions worldwide. This sub-ensemble of the global CORDEX dataset consists of a minimum homogeneous ensemble of six members, generated by two regional models driven by three CMIP5 global models (representing high, medium, and low climate sensitivities). It serves as the primary source for high-resolution regional climate projections used in climate change impact and adaptation studies and is catalogued in the CORDEX CDS catalogue.

                    **References:**
                    - [IPCC AR6 WGI Annex VI](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Annex_VI.pdf)
                    - [CORDEX-CORE Information](https://cds.climate.copernicus.eu/cdsapp#!/dataset/corhype-cordex)

                    """,
                group_id="",
                source="https://atlas.climate.copernicus.eu/atlas",
                license="Licence to use Copernicus products",
                version="",
                display_groups=[],
                map=MapInfo(
                    bounds=[],  # Se asigna dinámicamente
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="coolwarm_r",
                        min_value=-10.0,
                        max_value=10.0,
                        units="SPI-6",
                    ),
                    path="maps/drought/ipcc/v1/spi6_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                store_netcdf_coords=False,
                units="dimensionless",
                scenarios=[
                    Scenario(id="historical", years=[1995]),
                    Scenario(id="rcp2p6", years=[2030, 2050]),
                    Scenario(id="rcp4p5", years=[2030, 2050]),
                    Scenario(id="rcp8p5", years=[2030, 2050]),
                ],
            ),
            # CDD resource
            HazardResource(
                hazard_type="Drought",
                indicator_id="cdd",
                indicator_model_id="ipcc",
                indicator_model_gcm="IPCC",
                path="drought/ipcc/v1/cdd_{scenario}_{year}",
                params={},
                display_name="Consecutive Dry Days",
                resolution="47100 m",
                description="""
                    ## Consecutive Dry Days (CDD)

                    A **dry day** is defined as any day with precipitation lower than 1 mm. The **Consecutive Dry Days (CDD)** metric calculates the maximum number of consecutive days during which the daily accumulated precipitation remains below 1 mm.

                    **Dataset Information:**
                    - **Variable:** Consecutive Dry Days – maximum consecutive days with daily precipitation < 1 mm.
                    - **Dataset:** CORDEX Coordinated Output for Regional Evaluations
                    - **CORDEX-CORE:**
                    CORDEX-CORE is an initiative developed to produce homogeneous regional climate model projections at a ~25 km resolution for all major inhabited regions worldwide. This sub-ensemble of the global CORDEX dataset consists of a minimum homogeneous ensemble of six members, generated by two regional models driven by three CMIP5 global models (representing high, medium, and low climate sensitivities). It serves as the primary source of high-resolution regional climate projections for climate change impact and adaptation studies and is catalogued in the CORDEX CDS catalogue (0.22 resolution).

                    **Reference:**
                    - [IPCC AR6 WGI Annex VI](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Annex_VI.pdf)
                    """,
                source="Copernicus Interactive Climate Atlas: https://atlas.climate.copernicus.eu/atlas",
                version="v2",
                license="Licence to use Copernicus products",
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bounds=[],  # Se asigna dinámicamente
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="Reds",
                        min_value=0.0,
                        max_value=100.0,
                        units="days",
                    ),
                    path="maps/drought/ipcc/v1/cdd_{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                store_netcdf_coords=False,
                units="days",
                scenarios=[
                    Scenario(id="historical", years=[1995]),
                    Scenario(id="rcp2p6", years=[2030, 2050]),
                    Scenario(id="rcp4p5", years=[2030, 2050]),
                    Scenario(id="rcp8p5", years=[2030, 2050]),
                ],
            ),
        ]
