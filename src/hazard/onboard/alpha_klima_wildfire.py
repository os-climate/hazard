"""Module for handling the onboarding and processing of Fire Risk Indicators data."""

import logging
import os
from dataclasses import dataclass
from pathlib import PurePath
from typing_extensions import Any, Iterable, Optional, override

import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class FireRiskIndicators(Onboarder):
    """On-board returns data set from Alpha Klima Wildfire hazard."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """On-board dataset for fire risk indicators replicating the methodology from the paper:
        "Climate Change Risk Indicators for Central Banking" by Burger, Csaba; Herzberg, Julika;
        and Nuvoli, Thaïs. Climate Change Risk Indicators for Central Banking: Explainable AI
        in Fire Risk Estimations (February 01, 2024).

        Available at SSRN: https://ssrn.com/abstract=4865384 or
        http://dx.doi.org/10.2139/ssrn.4865384.

        This dataset has been replicated using the following data sources:

        - **Land Cover Data**:
          MCD12Q1.061 MODIS Land Cover Type Yearly Global 500m,
          https://doi.org/10.5067/MODIS/MCD12Q1.061

        - **Burned Area Data**:
          MCD64A1.061 MODIS Burned Area Monthly Global 500m,
          https://doi.org/10.5067/MODIS/MCD64A1.061

        - **Fire Weather Index Data**:
          Copernicus Climate Change Service, Climate Data Store (2020):
          Fire danger indicators for Europe from 1970 to 2098 derived from climate projections.
          DOI: https://doi.org/10.24381/cds.ca755de7
          (Accessed on 12-1-24)

        - **Critical Infrastructure Data**:
          https://publications.jrc.ec.europa.eu/repository/handle/JRC116012

        - **City and Population Data**:
          Collected from the OpenStreetMap database via Geoapify,
          https://www.geoapify.com/data-share/localities/

        The `.zarr` dataset has been created using monotonic constraints on the features mentioned in the paper.

        METADATA:
        - **Data type**: Historical and scenario annual probability
        - **Hazard indicator**: Probability of fire occurance
        - **Region**: Pan-Europe
        - **Resolution**: 2.5km
        - **Time range**: 2000-2020, 2021-2050
        - **Scenarios**: Historical, RCP4.5, RCP8.5
        - **File type**: Map (.zarr)

        DATA DESCRIPTION:
        .zarr files containing annual probabilities of fire occurance in
        EU under present and projected future climate under the historical,
        RCP4.5 and RCP8.5 scenarios (periods 2000-2020, 2021-2050).

        Args:
            source_dir_base (str): Directory containing source files. If `fs` is an
                S3FileSystem instance, <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
                If None, a LocalFileSystem is used.

        For more information, contact:
        - Author: Carlos San Millán https://orcid.org/0000-0001-7506-5552
        - Email: csanmillan@arfimaconsulting.com
        - GitHub: https://github.com/csanmillan

        """  # noqa: D205
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = (
            PurePath(source_dir_base, "Alpha_Klima_Hazard_Datasets").as_posix() + "/"
        )

        # Define the resource
        self._resource = list(self.inventory())[0]

    @override
    def prepare(self, download_dir=None):
        """Prepare the data for processing."""
        self.fs.makedirs(self.source_dir, exist_ok=True)
        items_to_process = self._get_items_to_process()
        for item in items_to_process:
            if not os.path.exists(
                os.path.join(self.source_dir, item["input_dataset_filename"])
            ):
                msg = f"{self.__class__.__name__} requires the file {item['input_dataset_filename']} to be in the download_dir.\nThe download_dir was {download_dir}."
                raise FileNotFoundError(msg)

    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if force or force_download:
            return False
        items_to_process = self._get_items_to_process()
        return all(
            [
                os.path.exists(
                    os.path.join(self.source_dir, item["input_dataset_filename"])
                )
                for item in items_to_process
            ]
        )

    def onboard(self, target: ReadWriteDataArray):
        """Process a single batch item and write the data to the Zarr store."""
        items_to_process = self._get_items_to_process()
        for item in items_to_process:
            input_path = os.path.join(self.source_dir, item["input_dataset_filename"])
            assert target is None or isinstance(target, OscZarr)

            path_ = self._resource.path.format(
                scenario=item["scenario"], year=item["year"]
            )

            # Open the Zarr dataset
            ds = xr.open_zarr(input_path).unify_chunks()
            da = ds["fire_probability"].load()
            target.write(path_, da)

    def _get_items_to_process(self):
        """Get the list of items to process."""
        return [
            {
                "scenario": "historical",
                "year": 2010,
                "input_dataset_filename": "alpha_klima_historical_fire_probability_2010.zarr",
            },
            {
                "scenario": "rcp4p5",
                "year": 2035,
                "input_dataset_filename": "alpha_klima_rcp_4p5_fire_probability_2035.zarr",
            },
            {
                "scenario": "rcp8p5",
                "year": 2035,
                "input_dataset_filename": "alpha_klima_rcp_8p5_fire_probability_2035.zarr",
            },
        ]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        create_tiles_for_resource(source, target, self._resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="fire_probability",
                indicator_model_gcm="FireRiskAlphaKlima",
                path="fire/ECB_fire_risk_indicators/v1/{scenario}_{year}",
                params={},
                display_name="Fire Probability (Alpha-Klima)",
                description="""
                This spatial dataset provides annual probabilities of fire occurrence across Europe under both historical conditions and
                projected future climates (RCP4.5 and RCP8.5). It builds upon the methodology from the paper "Climate Change Risk Indicators
                for Central Banking: Explainable AI in Fire Risk Estimations" https://dx.doi.org/10.2139/ssrn.4865384. and integrates a range of high-quality data
                sources—such as MODIS land cover and burned areas, Fire Weather Index data from Copernicus, critical infrastructure from HARCI-EU, and population data
                from OpenStreetMap. Delivered as .zarr files at a resolution of approximately 2.5 km, this resource helps researchers, policymakers,
                and risk analysts understand changing fire regimes and plan more effectively for climate-related risks.
                """,
                group_id="fire_risk_indicators",
                display_groups=[],
                map=MapInfo(
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="colorbrewer:reds_3",
                        min_value=0.0,
                        max_value=1.0,
                        units="probability",
                    ),
                    path="maps/fire/ECB_fire_risk_indicators/v1/{scenario}_{year}_map",
                    source="map_array_pyramid",
                ),
                store_netcdf_coords=False,
                units="probability",
                scenarios=[
                    Scenario(id="historical", years=[2010]),
                    Scenario(id="rcp4p5", years=[2035]),
                    Scenario(id="rcp8p5", years=[2035]),
                ],
            )
        ]

    def run_all(self, source, target, client=None, debug_mode=False):
        return super().run_all(source, target, client, debug_mode)

    def run_single(self, item, source, target, client):
        return super().run_single(item, source, target, client)

    def batch_items(self):
        return super().batch_items()
