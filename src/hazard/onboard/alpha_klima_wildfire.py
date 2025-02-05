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

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represent a batch item for hazard processing.

    It includes scenario, year, and input_dataset_filename.
    """

    scenario: str
    year: int
    input_dataset_filename: str


class FireRiskIndicators(IndicatorModel[BatchItem]):
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
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)
        for batch_item in self.batch_items():
            if not os.path.exists(
                os.path.join(self.source_dir, batch_item.input_dataset_filename)
            ):
                msg = f"{self.__class__.__name__} requires the file {batch_item.input_dataset_filename} to be in the download_dir.\nThe download_dir was {download_dir}."
                raise FileNotFoundError(msg)

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        return [
            BatchItem(
                scenario="historical",
                year=2010,
                input_dataset_filename="alpha_klima_historical_fire_probability_2010.zarr",
            ),
            BatchItem(
                scenario="rcp4p5",
                year=2035,
                input_dataset_filename="alpha_klima_rcp_4p5_fire_probability_2035.zarr",
            ),
            BatchItem(
                scenario="rcp8p5",
                year=2035,
                input_dataset_filename="alpha_klima_rcp_8p5_fire_probability_2035.zarr",
            ),
        ]

    def onboard_single(
        self, target, download_dir=None, force_prepare=False, force_download=False
    ):
        """Onboard a single batch of hazard data into the system.

        Args:
            target: Target system for writing the processed data.
            download_dir (str): Directory where downloaded files will be stored.
            force_prepare(bool): Flag to force data preparation. Default is False
            force_download(bool):Flag to force re-download of data. Default is False

        """
        self.prepare(
            force=force_prepare,
            download_dir=download_dir,
            force_download=force_download,
        )
        self.run_all(source=None, target=target, client=None, debug_mode=False)
        self.create_maps(target, target)

    def run_single(
        self, item: BatchItem, source: Any, target: ReadWriteDataArray, client: Client
    ):
        """Process a single batch item and write the data to the Zarr store."""
        input_path = os.path.join(self.source_dir, item.input_dataset_filename)
        assert target is None or isinstance(target, OscZarr)

        path_ = self._resource.path.format(scenario=item.scenario, year=item.year)

        # Open the Zarr dataset
        ds = xr.open_zarr(input_path).unify_chunks()
        da = ds["fire_probability"].load()
        target.write(path_, da)

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
