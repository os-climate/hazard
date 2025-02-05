"""Module for IPCC Drought Indicator model.

This module defines the IPCC Drought Indicator model, including classes and methods
for preparing, processing, and generating maps for drought-related indicators such as
Standardized Precipitation Index (SPI6) and Consecutive Dry Days (CDD). It handles
data management, file processing, and integration with the hazard resource inventory.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing_extensions import Any, Iterable, Optional, override
from urllib.parse import urlparse

import numpy as np
import xarray as xr
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_file
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Data container for batching IPCC drought model processing.

    Attributes
        indicator (str): The type of indicator, e.g., "SPI6", "CDD".
        scenario (str): Scenario under which the indicator is calculated.
        central_year (int): The central year for the data batch.
        input_dataset_filename (str): Name of the input dataset file.

    """

    indicator: str
    scenario: str
    central_year: int
    input_dataset_filename: str


class IPCCDrought(IndicatorModel[BatchItem]):
    """IPCC Drought Indicator model class.

    This class handles the preparation, processing, and map generation
    for IPCC drought indicators like SPI6 and CDD.

    Attributes
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

        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir, exist_ok=True)
        self._url = "https://digital.csic.es/bitstream/10261/332721/"
        self._resources = list(self.inventory())

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        self.fs.makedirs(self.source_dir, exist_ok=True)
        if (
            not self.fs.exists(self.source_dir)
            or len(self.fs.listdir(self.source_dir)) == 0
            or force_download
        ):
            for batch_item in self.batch_items():
                nc_url = self._url + batch_item.input_dataset_filename
                download_file(
                    url=nc_url,
                    directory=self.source_dir,
                    filename=os.path.basename(batch_item.input_dataset_filename),
                    force_download=force_download,
                )

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
        """Process a single batch of data, including reading, aggregating, and writing the results.

        Args:
            item (BatchItem): The batch item to process.
            source (Any): Source data object to read from.
            target (ReadWriteDataArray): Target array where the processed data will be written.
            client (Client): Dask client for parallel processing.

        """
        source_dir = PurePosixPath(self.source_dir)
        filename = os.path.basename(urlparse(item.input_dataset_filename).path)
        local_file = source_dir / filename
        variable_name = item.indicator.lower()

        try:
            # Adjusted chunks to avoid HDF error
            da = xr.open_dataset(
                str(local_file),
                decode_times=True,
                chunks={"time": "auto"},
                mask_and_scale=True,
            )[variable_name]
            logger.info(f"Dataset {filename} opened with variable '{variable_name}'.")
        except KeyError:
            logger.error(
                f"Variable '{variable_name}' not found in dataset '{filename}'."
            )
            return
        except Exception as e:
            logger.error(f"Error opening dataset {filename}: {e}")
            return

        # Convert timedelta64 data to float days if necessary
        if np.issubdtype(da.dtype, np.timedelta64):
            try:
                days = da.dt.days.astype(float)

                seconds = da.dt.seconds.astype(float) / 86400

                da = days + seconds

                logger.info(
                    f"Converted '{variable_name}' from timedelta64 to float (days with hours as decimal)."
                )
            except Exception as e:
                logger.error(f"Failed to convert '{variable_name}' to float: {e}")
                return

        # Select the desired time range
        if item.scenario.lower() == "historical":
            start_date = "1986-01-01"
            end_date = "2005-12-31"
        else:
            start_date = "2021-01-01"
            end_date = "2060-12-31"

        try:
            da = da.sel(time=slice(start_date, end_date))
            logger.info(
                f"Selected time range for '{variable_name}': {start_date} to {end_date}"
            )
        except Exception as e:
            logger.error(f"Failed to select time range for '{variable_name}': {e}")
            return

        # Aggregate over available dimensions
        dims_to_average = [dim for dim in ["member", "time"] if dim in da.dims]
        if dims_to_average:
            try:
                da = da.mean(dim=dims_to_average, skipna=True)
                logger.info(
                    f"Averaged '{variable_name}' over dimensions: {dims_to_average}"
                )
            except Exception as e:
                logger.error(f"Failed to aggregate data for '{variable_name}': {e}")
                return

        # Multiply by 100 to scale 'spi6' values correctly
        if variable_name == "spi6":
            if item.scenario.lower() == "historical":
                da.values = da.values * 100
            else:
                da.values = da.values * 10

        # Ensure DataArray has only 'lat' and 'lon' dimensions
        da = da.squeeze().reset_coords(drop=True)
        if set(da.dims) != {"lat", "lon"}:
            da = da.transpose("lat", "lon")
            logger.info(f"Transposed dimensions to: {da.dims}")

        if set(da.dims) != {"lat", "lon"}:
            logger.error(
                f"Final dimensions for '{variable_name}' are {da.dims}, expected ('lat', 'lon')."
            )
            return

        # Prepare the path for writing
        try:
            resource = next(
                res
                for res in self._resources
                if res.indicator_id.lower() == variable_name
            )
        except StopIteration:
            logger.error(f"No resource found for variable '{variable_name}'.")
            return

        path_ = resource.path.format(
            scenario=item.scenario.lower(), year=item.central_year
        )

        # Set map bounds if not already set
        if resource.map is not None and not resource.map.bounds:
            try:
                lat_bounds = [float(da.lat.min()), float(da.lat.max())]
                lon_bounds = [float(da.lon.min()), float(da.lon.max())]
                resource.map.bounds = [
                    (lon_bounds[0], lat_bounds[1]),
                    (lon_bounds[1], lat_bounds[1]),
                    (lon_bounds[1], lat_bounds[0]),
                    (lon_bounds[0], lat_bounds[0]),
                ]
                logger.info(
                    f"Set map bounds for '{variable_name}': {resource.map.bounds}"
                )
            except Exception as e:
                logger.error(f"Failed to set map bounds for '{variable_name}': {e}")
                return

        # Write the data to the target
        try:
            target.write(path_, da)
            logger.info(f"Successfully wrote data for '{variable_name}' to '{path_}'.")
        except Exception as e:
            logger.error(
                f"Failed to write data for '{variable_name}' to '{path_}': {e}"
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Define the hazard resources for SPI6 and CDD under the Drought hazard.

        Returns
            Iterable[HazardResource]: A list of hazard resources for the model.

        """
        return [
            # SPI6
            HazardResource(
                hazard_type="Drought",
                indicator_id="spi6",
                indicator_model_id="ipcc",
                indicator_model_gcm="IPCC",
                path="drought/ipcc/v1/spi6_{scenario}_{year}",
                params={},
                display_name="Standardized Precipitation Index (SPI-6)",
                description="""
                    ## Standardized Precipitation Index (SPI)

                    The **SPI** is a statistical index that compares cumulated precipitation over \( n \) months (commonly \( n=6 \) or \( n=12 \)) with the long-term precipitation distribution for the same location and accumulation period. The chosen \( n \)-month window (e.g., 6 months) represents a medium-term cumulative value that can help measure impacts on river flow and reservoir storage (McKee et al., 1993).

                    Following Spinoni et al. (2014), a drought event starts in the month when SPI drops below \(-1\) and ends once the SPI value is positive for **at least two consecutive months**.

                    ### Considerations and Limitations

                    - **High Latitudes & Arid Areas**
                      The SPI may be difficult to interpret in high-latitude or arid regions due to statistical inaccuracies in estimating the Gamma function (Spinoni et al., 2014).

                    - **SPI Variants**
                      The Intergovernmental Panel on Climate Change (IPCC) Interactive Atlas includes both SPI-6 and SPI-12 versions.


                    **Reference**
                    [IPCC AR6 WGI Annex VI](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Annex_VI.pdf)
                    """,
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bounds=[],  # Will be set dynamically
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="matplotlib:coolwarm_r",
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
                    Scenario(id="historical", years=[1994]),
                    Scenario(id="rcp2p6", years=[2040]),
                    Scenario(id="rcp4p5", years=[2040]),
                    Scenario(id="rcp8p5", years=[2040]),
                ],
            ),
            # CDD
            HazardResource(
                hazard_type="Drought",
                indicator_id="cdd",
                indicator_model_id="ipcc",
                indicator_model_gcm="IPCC",
                path="drought/ipcc/v1/cdd_{scenario}_{year}",
                params={},
                display_name="Consecutive Dry Days",
                description="""
                    ## Consecutive Dry Days (CDD)
                    A **dry day** for a model grid box is defined as any day with precipitation lower than 1 mm. The **Consecutive Dry Days (CDD)** metric counts how many dry days occur in a row. Specifically:

                    1. **Dry Day Criterion**
                       For each grid box, precipitation must be less than 1 mm on a given day for it to be considered a dry day.

                    2. **Counting Consecutive Dry Days**
                       - The count starts on the first dry day of the year (or directly on the first day of the year, if it is already dry).
                       - The count continues until a wet day (precipitation ≥ 1 mm) occurs.
                       - If no wet day occurs, the count stops at the year’s end.

                    3. **Annual CDD**
                       The **annual maximum number of consecutive dry days** (CDD) is the longest run of dry days that occurs during the year.

                    In practice, you would calculate the yearly CDD values by reviewing the daily precipitation for each grid box, identifying continuous stretches of days where precipitation remains below 1 mm, and retaining the **maximum** length of these stretches over the full year.
                    **Reference**
                    [IPCC AR6 WGI Annex VI](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Annex_VI.pdf)
                    """,
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bounds=[],  # Will be set dynamically
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="colorbrewer:reds",
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
                    Scenario(id="historical", years=[1994]),
                    Scenario(id="rcp2p6", years=[2040]),
                    Scenario(id="rcp4p5", years=[2040]),
                    Scenario(id="rcp8p5", years=[2040]),
                ],
            ),
        ]

    def batch_items(self) -> Iterable[BatchItem]:
        """Define the batch items for SPI6 and CDD data processing.

        Returns
            Iterable[BatchItem]: A list of batch items to process.

        """
        return [
            # SPI6
            BatchItem(
                indicator="SPI6",
                scenario="historical",
                central_year=1994,
                input_dataset_filename="34/spi6_CORDEX-EUR_historical_mon_197001-200512.nc",
            ),
            BatchItem(
                indicator="SPI6",
                scenario="rcp2p6",
                central_year=2040,
                input_dataset_filename="35/spi6_CORDEX-EUR_rcp26_mon_200601-210012.nc",
            ),
            BatchItem(
                indicator="SPI6",
                scenario="rcp4p5",
                central_year=2040,
                input_dataset_filename="36/spi6_CORDEX-EUR_rcp45_mon_200601-210012.nc",
            ),
            BatchItem(
                indicator="SPI6",
                scenario="rcp8p5",
                central_year=2040,
                input_dataset_filename="37/spi6_CORDEX-EUR_rcp85_mon_200601-210012.nc",
            ),
            # CDD
            BatchItem(
                indicator="CDD",
                scenario="historical",
                central_year=1994,
                input_dataset_filename="2/cdd_CORDEX-EUR_historical_yr_1970-2005.nc",
            ),
            BatchItem(
                indicator="CDD",
                scenario="rcp2p6",
                central_year=2040,
                input_dataset_filename="3/cdd_CORDEX-EUR_rcp26_yr_2006-2100.nc",
            ),
            BatchItem(
                indicator="CDD",
                scenario="rcp4p5",
                central_year=2040,
                input_dataset_filename="4/cdd_CORDEX-EUR_rcp45_yr_2006-2100.nc",
            ),
            BatchItem(
                indicator="CDD",
                scenario="rcp8p5",
                central_year=2040,
                input_dataset_filename="5/cdd_CORDEX-EUR_rcp85_yr_2006-2100.nc",
            ),
        ]
