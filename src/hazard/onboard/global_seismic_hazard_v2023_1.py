"""Module for handling the onboarding and processing of GEM Global Seismic Hazard Map data."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import PurePath
from typing_extensions import Iterable, Optional, override, cast

import numpy as np
import xarray as xr

from xarray import DataArray
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel  # noqa: F401  (required by Onboarder parent)
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.onboarder import Onboarder
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class GEMSeismicHazard(Onboarder):
    """On-boards the GEM Global Seismic Hazard Map (v2023.1, PGA 10 % in 50 years)."""

    def __init__(
        self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None
    ) -> None:
        """Args:
        ----
        source_dir_base
            Directory (local or remote) that will contain the source files.
            For S3FileSystem use “<bucket>/<prefix>”.
        fs
            Instance of `AbstractFileSystem`.  If `None`, a local file system is assumed.

        """
        self.fs: AbstractFileSystem = fs or LocalFileSystem()
        self.source_dir: str = (
            PurePath(source_dir_base, "gem_seismic_hazard").as_posix() + "/"
        )
        self.return_period = 475

        self.url: str = "https://zenodo.org/records/8409647/files/GEM-GSHM_PGA-475y-rock_v2023.zip?download=1"

        self._resource: HazardResource = list(self.inventory())[0]

    @override
    def prepare(self, download_dir: str | None = None) -> None:
        """Download and unzip the GeoTIFF if it is not already available."""
        self.fs.makedirs(self.source_dir, exist_ok=True)

        zip_name = "GEM-GSHM_PGA-475y-rock_v2023.zip"
        zip_path = os.path.join(self.source_dir, zip_name)
        tif_name = "v2023_1_pga_475_rock_3min.tif"
        tif_path = os.path.join(self.source_dir, tif_name)

        # Download
        if not self.fs.exists(tif_path):
            logger.info("Downloading GEM GSHM dataset…")
            download_file(url=self.url, directory=self.source_dir, filename=zip_name)

            # Unzip
            logger.info("Extracting GeoTIFF…")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.source_dir)

            # (Optional) delete the ZIP file to save space
            try:
                self.fs.rm(zip_path)
            except Exception:  # pragma: no cover
                logger.warning("Temporary ZIP file could not be deleted.")

    @override
    def is_prepared(self, force: bool = False, force_download: bool = False) -> bool:
        """Check whether the GeoTIFF is ready."""
        if force or force_download:
            return False
        tif_path = os.path.join(self.source_dir, "v2023_1_pga_475_rock_3min.tif")
        return self.fs.exists(tif_path)

    @override
    def onboard(self, target) -> None:
        assert target is None or isinstance(target, OscZarr)

        tif_path = os.path.join(self.source_dir, "v2023_1_pga_475_rock_3min.tif")

        da_raw = xr.open_dataarray(tif_path, engine="rasterio")

        da_raw = cast(DataArray, da_raw)

        da = da_raw.squeeze(drop=True)

        # Normalize coordinate names and convert nodata to NaN
        da = da.rename({"y": "lat", "x": "lon"})
        da = da.where(~np.isnan(da), other=np.nan)

        # Set the 'index' dimension and coordinate to 475
        da = da.expand_dims(index=[475])

        # Save
        path_ = self._resource.path.format(year=2023)
        if target is not None:
            target.write(path_, da)

    @override
    def create_maps(self, source: OscZarr, target: OscZarr) -> None:
        """Create tile pyramids from the PGA resource."""
        create_tiles_for_resource(
            source,
            target,
            self._resource,
            max_zoom=4,
            nodata_as_zero_coarsening=True,
        )

    def inventory(self) -> Iterable[HazardResource]:
        """Devuelve la definición (no expandida) del recurso de peligro sísmico."""
        return [
            HazardResource(
                hazard_type="GroundShaking",
                indicator_id="pga_10pc50yr",
                indicator_model_id="",
                indicator_model_gcm="",
                path="earthquake/gem_gshm/v2023_1/pga_475y_{year}",
                params={},
                display_name=(
                    "Peak Ground Acceleration (10 % prob. exceedance in 50 yrs) – GEM GSHM v2023.1"
                ),
                resolution="9800 m",
                description=(
                    """
                    ### GEM Global Seismic Hazard Map (v 2023.1) — Peak Ground Acceleration, 10 % Exceedance in 50 Years
                        A global, ~3 arc-minute (≈ 5.5 km at the equator) raster of **peak ground acceleration (PGA)** expressed as a fraction of *g* (acceleration of gravity). Each cell gives the ground-motion level that has **a 10 % probability of being exceeded in a 50-year time window**—equivalent to an average annual exceedance probability of 0.0021 yr⁻¹ or a 475-year return period—**for reference rock conditions** *(Vs₃₀ = 760–800 m s⁻¹)*.


                        #### Typical uses
                        * **Seismic risk assessments** – combine with vulnerability curves to estimate direct economic losses for the 475-year design event or to approximate annualised loss (AAL).
                        * **Building-code benchmarking** – contrast local code hazard levels against the GEM global model.
                        * **Portfolio screening & parametric insurance** – identify high-hazard hotspots across global asset portfolios.
                        * **Education & outreach** – illustrate relative seismic hazard worldwide for non-specialist audiences.

                        #### Limitations & best practice
                        * A single exceedance level does **not** capture the full frequency–intensity relationship; for rigorous AAL calculations you need a full hazard curve (multiple probabilities or annual rates).
                        * Local site effects (soil amplification, topography, basin effects) are **not** included; adjust with appropriate amplification factors if working at asset scale.
                        * Always cite the GEM Foundation and the DOI above when using the dataset in analyses or publications.


                        #### Citation
                        > GEM Foundation (2023). *Global Seismic Hazard Map v 2023.1 – Peak Ground Acceleration, 10 % Exceedance in 50 Years (Reference Rock, 3-arc-min)* [Data set]. https://doi.org/10.5281/zenodo.8409647


                    """
                ),
                version="v2023.1",
                license="CC BY-NC-SA 4.0",
                source="GEM Foundation: https://doi.org/10.5281/zenodo.8409647",
                group_id="earthquake_gem_gshm",
                display_groups=[],
                map=MapInfo(
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="magma",
                        min_value=0.0,
                        max_value=1.0,
                        units="g",
                    ),
                    path="maps/earthquake/gem_gshm/v2023_1/pga_475y_{year}_map",
                    source="map_array_pyramid",
                ),
                units="g",
                store_netcdf_coords=False,
                scenarios=[Scenario(id="historical", years=[2023])],
            ),
        ]

    def _get_items_to_process(self) -> list[dict[str, str | int]]:
        return [
            {
                "central_year": 2023,
                "input_dataset_filename": "v2023_1_pga_475_rock_3min.tif",
            }
        ]
