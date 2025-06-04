"""Module for handling the onboarding and processing of Fire Risk Indicators data."""

import logging
import os
from pathlib import PurePath
from typing_extensions import Iterable, Optional, override

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.onboarder import Onboarder
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

        For more information, contact:
        - Authors: Carlos San Millán https://orcid.org/0000-0001-7506-5552,  Virginia Morales, https://orcid.org/0009-0008-5728-5656"
        - Email: csanmillan@arfimaconsulting.com, vmorales@arfimaconsulting.com
        - GitHub: https://github.com/csanmillan, https://github.com/vmorales

        """  # noqa: D205
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = (
            PurePath(source_dir_base, "European_Wildfire_Hazard_Datasets").as_posix()
            + "/"
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

    @override
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

    @override
    def onboard(self, target: ReadWriteDataArray):
        """Process a single item and write the data to the Zarr store."""
        items_to_process = self._get_items_to_process()
        for item in items_to_process:
            input_path = os.path.join(self.source_dir, item["input_dataset_filename"])
            assert target is None or isinstance(target, OscZarr)

            path_ = self._resource.path.format(
                scenario=item["scenario"], year=item["year"]
            )

            # Open the Zarr dataset
            ds = xr.open_dataset(input_path, engine="netcdf4").unify_chunks()
            da = ds["fire_probability"].load()
            target.write(path_, da)

    def _get_items_to_process(self):
        """Get the list of items to process."""
        return [
            {
                "scenario": "historical",
                "year": 2010,
                "input_dataset_filename": "historical_probabilistic_pan_european_wildfire_2010.nc",
            },
            {
                "scenario": "rcp4p5",
                "year": 2035,
                "input_dataset_filename": "rcp_4p5_probabilistic_pan_european_wildfire_2035.nc",
            },
            {
                "scenario": "rcp8p5",
                "year": 2035,
                "input_dataset_filename": "rcp_8p5_probabilistic_pan_european_wildfire_2035.nc",
            },
        ]

    @override
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
                path="fire/probabilistic_pan_european_wildfire/{scenario}_{year}",
                params={},
                display_name="Probabilistic Pan-European Wildfire Map",
                resolution="2500 m",
                description="""


[![Logo](https://alpha-klima.com/wp-content/uploads/2025/06/Logo-ALPHA-KLIMA-con-texto.png)](https://alpha-klima.com/)


This spatial dataset provides annual probabilities of fire occurrence across Europe under both historical
conditions (2002–2022) and projected future climates (2023–2050) for two emissions pathways (RCP 4.5 and
RCP 8.5). It builds upon the methodology presented in **“Climate Change Risk Indicators for Central Banking:
Explainable AI in Fire Risk Estimations”** (https://dx.doi.org/10.2139/ssrn.4865384) and integrates a range of
high-quality geospatial layers to deliver wall-to-wall coverage at ~2.5 km resolution.

Annual probabilities were estimated by training XGBoost models with monotonic constraints on FWI-related features,
using historical fire occurrence as the target, followed by probability calibration and spatial application
to each year's input layers.

**Key input layers**

* **Fire Weather Index (FWI) v1.0** — [Copernicus Climate Data Store](
  https://cds.climate.copernicus.eu/datasets/sis-tourism-fire-danger-indicators?tab=overview)
* **Burned-area product (MCD64A1 v6.1)** — [NASA/USGS LP DAAC](https://lpdaac.usgs.gov/products/mcd64a1v061/)
* **Land-cover product (MCD12Q1 v6)** — [NASA/USGS LP DAAC](https://lpdaac.usgs.gov/products/mcd12q1v006/)
* **Critical infrastructure (HARCI-EU)** — Koks et al. (2019), *Scientific Data*
  <https://www.nature.com/articles/s41597-019-0135-1>
* **Urban centres & population** — [Geoapify](https://www.geoapify.com/)

For implementing the processing pipeline, we also leveraged the [CLAIMED library](https://github.com/claimed-framework/component-library), developed by IBM. CLAIMED provides a fully modular,
scalable architecture in which each stage of the workflow is encapsulated as a decoupled component; this greatly simplifies maintenance,
parallel execution, and automatic versioning. By using CLAIMED, we can natively integrate distributed-processing engines (e.g. Dask)
and orchestrators like Airflow, ensuring full transparency, reproducibility, and rapid adaptability whenever data sources
or model requirements evolve.

**Countries covered (ISO alpha-2 codes)**
AL, AT, BE, BG, CH, CY, CZ, DE, DK, EE, ES, FI, FR, GB, GR, HR, HU, IE, IS, IT, LI, LT, LU, LV, ME, MK, MT, NL, NO, PL, PT, RO, RS, SE, SI, SK, TR

Authors: [Carlos San Millán](https://github.com/csanmillan) and [Virginia Morales](https://github.com/VMarfima)
Property of Arfima Financial Services (https://afs-services.com/)
License: CC BY-NC 4.0
(Arfima: https://afs-services.com/; CLAIMED: https://github.com/claimed-framework/component-library; Alpha Klima: https://alpha-klima.com/)

Disclaimer: These data are not intended for use in emergency response, public safety operations, or any context where physical safety or property risk is involved. If you require higher-resolution datasets built with the latest methodologies, subject to rigorous validation, and designed for detailed bottom-up risk analysis, please contact [Alpha-Klima](https://alpha-klima.com/).
""",
                group_id="fire_risk_indicators",
                source="Alpha Klima",
                version="v1_0",
                license="Commercial",
                display_groups=[],
                map=MapInfo(
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="Reds",
                        min_value=0.0,
                        max_value=0.7,
                        units="probability",
                    ),
                    path="maps/fire/probabilistic_pan_european_wildfire/{scenario}_{year}_map",
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
