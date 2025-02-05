"""."""

import concurrent.futures
import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import PurePosixPath
from typing_extensions import Iterable, List, Optional, Protocol, Sequence, override

import dask.array as da
import numpy as np  # type: ignore
import s3fs  # type: ignore
import xarray as xr
import zarr  # type: ignore
import zarr.hierarchy
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic.type_adapter import TypeAdapter
from zarr.errors import GroupNotFoundError  # type: ignore

from hazard.indicator_model import IndicatorModel  # type: ignore
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import MultiYearAverageIndicatorBase  # type: ignore
from hazard.protocols import ReadWriteDataArray
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a batch item for processing climate data with specific model, scenario, and year parameters."""

    gcm: str
    scenario: str
    central_years: list[int]


class ZarrWorkingStore(Protocol):
    """Protocol for a Zarr working store.

    Defines the interface for retrieving a storage backend for a given path.
    """

    def get_store(self, path: str):
        """Retrieve the storage backend for the specified path."""
        ...


class S3ZarrWorkingStore(ZarrWorkingStore):
    """Implementation of ZarrWorkingStore that interacts with an S3 storage backend.

    This class provides a mechanism to retrieve a Zarr-compatible storage backend
    using AWS S3 as the underlying storage system.
    """

    def __init__(self):
        """Initialize the S3ZarrWorkingStore with an S3 filesystem.

        Retrieves S3 credentials and bucket information from environment variables
        and sets up an S3 file system for accessing Zarr stores.

        """
        s3 = s3fs.S3FileSystem(
            anon=False,
            key=os.environ["OSC_S3_ACCESS_KEY_DEV"],
            secret=os.environ["OSC_S3_SECRET_KEY_DEV"],
        )
        base_path = os.environ["OSC_S3_BUCKET_DEV"] + "/drought/osc/v01"
        self._base_path = base_path
        self._s3 = s3

    def get_store(self, path: str):
        """Retrieve an S3-backed Zarr store for the specified path.

        Args:
            path (str): The relative path to the Zarr store within the S3 bucket.

        Returns:
            s3fs.S3Map: A mapping object that represents the S3-backed Zarr store.

        """
        return s3fs.S3Map(
            root=str(PurePosixPath(self._base_path, path)), s3=self._s3, check=False
        )


class LocalZarrWorkingStore(ZarrWorkingStore):
    """Implementation of ZarrWorkingStore for local file system storage.

    This class provides a local directory-based backend for storing and retrieving
    Zarr stores within a specified working directory.
    """

    def __init__(self, working_dir: str):
        """Initialize the local Zarr working store.

        Args:
            working_dir (str): The base directory where Zarr stores will be saved.

        """
        self._base_path = os.path.join(working_dir, "/drought/osc/v01")

    def get_store(self, path: str):
        """Retrieve a local directory-backed Zarr store for the specified path.

        Args:
            path (str): The relative path to the Zarr store within the working directory.

        Returns:
            zarr.DirectoryStore: A directory-based Zarr store.

        """
        return zarr.DirectoryStore(os.path.join(self._base_path, path))


class ChunkIndicesComplete(BaseModel):
    """Represents a set of completed chunk indices.

    This model stores a list of indices that have been fully processed or completed.
    """

    complete_indices: List[int]


class ProgressStore:
    """Simple persistent JSON store tracking completed indices of a chunked calculation."""

    def __init__(self, dir: str, id: str):
        """Initialize the ProgressStore.

        Args:
            dir (str): Path to directory for storing progress files.
            id (str): Identifier for the calculation (for use in filename).

        """
        self.dir = dir
        self.id = id
        if not os.path.exists(self._filename()):
            self.reset()

    def reset(self):
        """Reset the progress by clearing all completed indices."""
        self._write(ChunkIndicesComplete(complete_indices=[]))

    def add_completed(self, indices: Sequence[int]):
        """Add indices to the set of completed calculations.

        Args:
            indices (Sequence[int]): List of indices that have been completed.

        """
        existing = set(self._read().complete_indices)
        union = existing.union(set(indices))
        self._write(ChunkIndicesComplete(complete_indices=list(union)))

    def completed(self):
        """Retrieve the list of completed indices.

        Returns
            Sequence[int]: List of indices that have been marked as completed.

        """
        return self._read().complete_indices

    def remaining(self, n_indices: int):
        """Get the list of indices that have not been completed.

        Args:
            n_indices (int): Total number of indices in the calculation.

        Returns:
            np.ndarray: Array of indices that are yet to be completed.

        """
        existing = self._read().complete_indices
        return np.setdiff1d(np.arange(0, n_indices, dtype=int), existing)

    def _filename(self):
        """Construct the filename for storing progress.

        Returns
            str: Full path to the progress file.

        """
        return os.path.join(self.dir, self.id + ".json")

    def _read(self):
        """Read the progress file and return the stored indices.

        Returns
            ChunkIndicesComplete: Object containing completed indices.

        """
        with open(self._filename(), "r") as f:
            indices = TypeAdapter(ChunkIndicesComplete).validate_json(f.read())
            return indices

    def _write(self, indices: ChunkIndicesComplete):
        """Write the completed indices to the progress file.

        Args:
            indices (ChunkIndicesComplete): Object containing completed indices.

        """
        with open(self._filename(), "w") as f:
            f.write(json.dumps(indices.model_dump()))


class DroughtIndicator(IndicatorModel[BatchItem]):
    """A drought indicator model that calculates the Standardized Precipitation-Evapotranspiration Index (SPEI) and related metrics based on climate model projections."""

    def __init__(
        self,
        working_zarr_store: ZarrWorkingStore,
        window_years: int = MultiYearAverageIndicatorBase._default_window_years,
        gcms: Iterable[str] = MultiYearAverageIndicatorBase._default_gcms,
        scenarios: Iterable[str] = MultiYearAverageIndicatorBase._default_scenarios,
        central_years: Optional[Sequence[int]] = None,
    ):
        """Initialize the DroughtIndicator with the required configuration and datasets.

        Args:
            working_zarr_store (ZarrWorkingStore): Storage location for chunked datasets.
            window_years (int, optional): Number of years in the averaging window. Defaults to MultiYearAverageIndicatorBase._default_window_years.
            gcms (Iterable[str], optional): List of General Circulation Models (GCMs) to process. Defaults to MultiYearAverageIndicatorBase._default_gcms.
            scenarios (Iterable[str], optional): Climate scenarios for analysis. Defaults to MultiYearAverageIndicatorBase._default_scenarios.
            central_years (Sequence[int], optional): List of years around which data is averaged. Defaults to [2005, 2030, 2040, 2050, 2080].

        """
        if central_years is None:
            central_years = [2005, 2030, 2040, 2050, 2080]
        self.calib_start, self.calib_end = datetime(1985, 1, 1), datetime(2015, 1, 1)
        self.calc_start, self.calc_end = datetime(1985, 1, 1), datetime(2100, 12, 31)
        self.freq, self.window, self.dist, self.method = "MS", 12, "gamma", "APP"
        self.lat_min, self.lat_max = -60.0, 90.0
        self.lon_min, self.lon_max = 0, 360.0
        self.central_years = central_years
        self.spei_threshold = [0, -1, -1.5, -2, -2.5, -3, -3.6]
        self.gcms = gcms
        self.scenarios = scenarios
        self.working_zarr_store = working_zarr_store
        self.window_years = window_years
        self.resource = self._resource()

    def pre_chunk(
        self,
        item: BatchItem,
        years=None,
        quantities=None,
        lat_chunk_size=40,
        lon_chunk_size=40,
        datasource=None,
    ):
        """Create a chunked data set for the given quantities and data source.

        This is for when the data
        source is either unchunked or unsuitably chunked for the calculation in hand. An SPEI index is an
        example since the calculation requires a long time series but for a limited spatial extent. The calculation
        therefore runs

        Args:
            item (BatchItem): The batch item containing GCM and scenario details.
            years (_type_, optional): Years included in chunked data. Defaults to np.arange(1950, 2101).
            quantities (list, optional): Climate variables to process. Defaults to ["tas", "pr"].
            variables (list, optional): Quantities included in chunked data. Defaults to ['tas', 'pr'].
            lat_chunk_size (int, optional): Latitude chunks. Defaults to 40.
            lon_chunk_size (int, optional): Longitude chunks. Defaults to 40.
            datasource (_type_, optional): Source for building chunked data. Defaults to NexGddpCmip6().

        """
        if years is None:
            years = np.arange(1950, 2101)
        if quantities is None:
            quantities = ["tas", "pr"]
        if datasource is None:
            datasource = NexGddpCmip6()

        def download_dataset(variable, year, gcm, scenario, datasource=datasource):
            scenario_ = "historical" if year < 2015 else scenario
            with datasource.open_dataset_year(
                gcm, scenario_, variable, year
            ) as ds_temp:
                ds = ds_temp.astype("float32").compute()
                return ds

        for quantity in quantities:
            group = quantity + "_" + item.gcm + "_" + item.scenario
            for year in years:
                ds = download_dataset(quantity, year, item.gcm, item.scenario).chunk(
                    {"time": 365, "lat": lat_chunk_size, "lon": lon_chunk_size}
                )
                if year == years[0]:
                    ds.to_zarr(store=self.working_zarr_store.get_store(group), mode="w")
                else:
                    ds.to_zarr(
                        store=self.working_zarr_store.get_store(group),
                        append_dim="time",
                    )
                logger.info(f"completed processing: variable={quantity}, year={year}.")

    def read_quantity_from_s3_store(
        self, gcm, scenario, quantity, lat_min, lat_max, lon_min, lon_max
    ) -> xr.Dataset:
        """Read a specified climate variable from an S3-backed Zarr store.

        Args:
            gcm (str): The climate model identifier.
            scenario (str): The climate scenario identifier.
            quantity (str): The climate variable (e.g., temperature or precipitation).
            lat_min (float): Minimum latitude of the selection.
            lat_max (float): Maximum latitude of the selection.
            lon_min (float): Minimum longitude of the selection.
            lon_max (float): Maximum longitude of the selection.

        Returns:
            xr.Dataset: A dataset containing the requested climate data.

        """
        ds = self.chunked_dataset(gcm, scenario, quantity).sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        return ds

    def chunked_dataset(self, gcm, scenario, quantity) -> xr.Dataset:
        """Load a chunked dataset from the Zarr store.

        Args:
            gcm (str): The General Circulation Model (GCM) name.
            scenario (str): The climate scenario (e.g., RCP or SSP).
            quantity (str): The data quantity type (e.g., precipitation, temperature).

        Returns:
            xr.Dataset: The loaded dataset as an xarray object.

        """
        ds = xr.open_zarr(
            store=self.working_zarr_store.get_store(
                quantity + "_" + gcm + "_" + scenario
            )
        )
        return ds

    def get_datachunks(self):
        """Generate spatial data chunks based on latitude and longitude bins.

        The function divides the dataset into grid-based chunks, each covering
        a defined range of latitudes and longitudes.

        Returns
            dict[str, dict[str, float]]: A dictionary where keys are chunk identifiers
            (e.g., "Chunk_0001") and values are dictionaries containing `lat_min`,
            `lat_max`, `lon_min`, and `lon_max` boundaries.

        """
        lat_delta = 10.0
        lon_delta = 10.0
        lat_bins = np.arange(self.lat_min, self.lat_max + 0.1 * lat_delta, lat_delta)
        lon_bins = np.arange(self.lon_min, self.lon_max + 0.1 * lon_delta, lon_delta)
        data_chunks = {
            "Chunk_" + str(i).zfill(4): dict(list(d[0].items()) + list(d[1].items()))
            for i, d in enumerate(
                itertools.product(
                    [
                        {"lat_min": x[0], "lat_max": x[1]}
                        for x in zip(lat_bins[:-1], lat_bins[1:])
                    ],
                    [
                        {"lon_min": x[0], "lon_max": x[1]}
                        for x in zip(lon_bins[:-1], lon_bins[1:])
                    ],
                )
            )
        }
        return data_chunks

    def calculate_spei(
        self, gcm, scenario, progress_store: Optional[ProgressStore] = None
    ):
        """Calculate SPEI for the given GCM and scenario, storing."""
        # we infer the lats and lons from the source dataset:
        ds_chunked = self.chunked_dataset(gcm, scenario, "tas")
        data_chunks = self.get_datachunks()
        chunk_names = list(data_chunks.keys())
        if progress_store:
            chunk_names_remaining = list(
                np.array(chunk_names)[progress_store.remaining(len(chunk_names))]
            )
        else:
            chunk_names_remaining = chunk_names

        # do the first chunk in isolation to ensure that dataset is written without contention
        if chunk_names[0] in chunk_names_remaining:
            self._calculate_spei_chunk(
                chunk_names[0], data_chunks, ds_chunked, gcm, scenario
            )
            chunk_names_remaining.remove(chunk_names[0])
            if progress_store:
                progress_store.add_completed([0])

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(
                    self._calculate_spei_chunk,
                    chunk_name,
                    data_chunks,
                    ds_chunked,
                    gcm,
                    scenario,
                ): chunk_name
                for chunk_name in chunk_names_remaining
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_name = future.result()
                    logger.info(f"chunk {chunk_name} complete.")
                    if progress_store:
                        progress_store.add_completed([chunk_names.index(chunk_name)])
                except Exception:
                    logger.info(f"chunk {futures[future]} failed.")

    def _calculate_spei_chunk(
        self, chunk_name, data_chunks, ds_chunked: xr.Dataset, gcm, scenario
    ):
        data_chunk = data_chunks[chunk_name]
        lat_min, lat_max = data_chunk["lat_min"], data_chunk["lat_max"]
        lon_min, lon_max = data_chunk["lon_min"], data_chunk["lon_max"]
        ds_spei_slice = self._calculate_spei_for_slice(
            lat_min, lat_max, lon_min, lon_max, gcm=gcm, scenario=scenario
        )
        lats_all, lons_all = ds_chunked["lat"].values, ds_chunked["lon"].values
        path = os.path.join("spei", gcm + "_" + scenario)
        store = self.working_zarr_store.get_store(path)
        # consider refactoring data_chunks to give both slice and values?
        try:
            # if dataset does not already exist then create
            zarr.hierarchy.open_group(store=store, mode="r")
        except GroupNotFoundError:
            # must use deferred dask array to avoid allocating memory for whole array
            data = da.empty(
                [len(ds_spei_slice["time"].values), len(lats_all), len(lons_all)]
            )
            ds_spei = (
                xr.DataArray(
                    data=data,
                    coords={
                        "time": ds_spei_slice["time"].values,
                        "lat": lats_all,
                        "lon": lons_all,
                    },
                    dims=["time", "lat", "lon"],
                )
                .chunk(chunks={"lat": 40, "lon": 40, "time": 100000})
                .to_dataset(name="spei")
            )
            # compute=False to avoid calculating array
            ds_spei.to_zarr(store=store, mode="w", compute=False)
            logger.info("created new zarr array.")
            # see https://docs.xarray.dev/en/stable/user-guide/io.html?appending-to-existing-zarr-stores=#appending-to-existing-zarr-stores

        lat_indexes = np.where(
            np.logical_and(lats_all >= lat_min, lats_all <= lat_max)
        )[0]
        lon_indexes = np.where(
            np.logical_and(lons_all >= lon_min, lons_all <= lon_max)
        )[0]
        time_indexes = np.arange(0, len(ds_spei_slice["time"].values))
        ds_spei_slice.to_zarr(
            store=store,
            mode="r+",
            region={
                "lat": slice(lat_indexes[0], lat_indexes[-1] + 1),
                "lon": slice(lon_indexes[0], lon_indexes[-1] + 1),
                "time": slice(time_indexes[0], time_indexes[-1] + 1),
            },
        )
        logger.info(f"written chunk {chunk_name} to zarr array.")
        return chunk_name

    def _calculate_spei_for_slice(
        self, lat_min, lat_max, lon_min, lon_max, *, gcm, scenario, num_workers=2
    ):
        # delay import because these take a bit to load.
        from xclim.indices import (
            potential_evapotranspiration as xclim_potential_evapotranspiration,
            water_budget as xclim_water_budget,
            standardized_precipitation_evapotranspiration_index as xclim_std_prec_evap_indx,
        )

        ds_tas = (
            self.read_quantity_from_s3_store(
                gcm, scenario, "tas", lat_min, lat_max, lon_min, lon_max
            )
            .chunk({"time": 100000})
            .compute()
        )
        ds_pr = (
            self.read_quantity_from_s3_store(
                gcm, scenario, "pr", lat_min, lat_max, lon_min, lon_max
            )
            .chunk({"time": 100000})
            .compute()
        )
        ds_tas = ds_tas.drop_duplicates(dim=..., keep="last").sortby("time")
        ds_pr = ds_pr.drop_duplicates(dim=..., keep="last").sortby("time")
        ds_pet = (
            xclim_potential_evapotranspiration(tas=ds_tas["tas"], method="MB05")
            .astype("float32")
            .to_dataset(name="pet")
        )
        da_wb = xclim_water_budget(pr=ds_pr["pr"], evspsblpot=ds_pet["pet"])
        with xr.set_options(keep_attrs=True):
            da_wb = da_wb - 1.01 * da_wb.min()
        da_wb_calib = da_wb.sel(
            time=slice(
                self.calib_start.strftime("%Y-%m-%d"),
                self.calib_end.strftime("%Y-%m-%d"),
            )
        )
        da_wb_calc = da_wb.sel(
            time=slice(
                self.calc_start.strftime("%Y-%m-%d"), self.calc_end.strftime("%Y-%m-%d")
            )
        )
        ds_spei = (
            xclim_std_prec_evap_indx(
                da_wb_calc,
                da_wb_calib,
                freq=self.freq,
                window=self.window,
                dist=self.dist,
                method=self.method,
            )
            .astype("float32")
            .to_dataset(name="spei")
            .compute(scheduler="processes", num_workers=num_workers)
        )
        logger.info(
            f"calculated SPEI for gcm={gcm}, lats=[{lat_min, lat_max}], lons=[{lon_min, lon_max}]"
        )
        return ds_spei

    def calculate_annual_average_spei(
        self, gcm: str, scenario: str, central_year: int, target: OscZarr
    ):
        """Calculate average number of months where 12-month SPEI index is below thresholds [0, -1, -1.5, -2, -2.5, -3.6] for 20 years period.

        Args:
            gcm (str): Global Circulation Model ID.
            scenario (str): Scenario ID.
            central_year (int): The central year of the 20-year period.
            year (int): Year.
            target (OscZarr): Target to write result to.

        """

        def get_spei_full_results(gcm, scenario):
            path = os.path.join("spei", gcm + "_" + scenario)
            ds_spei = xr.open_zarr(self.working_zarr_store.get_store(path))
            return ds_spei

        period = [
            datetime(central_year - self.window_years // 2, 1, 1),
            datetime(central_year + self.window_years // 2 - 1, 12, 31),
        ]
        print(
            gcm
            + " "
            + scenario
            + " "
            + str(central_year)
            + " period:   "
            + str(period[0])
            + "---"
            + str(period[1])
        )
        ds_spei = get_spei_full_results(gcm, scenario)
        lats_all = ds_spei["lat"].values
        lons_all = ds_spei["lon"].values
        spei_annual = np.nan * np.zeros(
            [len(self.spei_threshold), len(lats_all), len(lons_all)]
        )
        spei_temp = ds_spei.sel(time=slice(period[0], period[1]))
        spei_temp = spei_temp.compute()
        spei_temp = spei_temp["spei"]
        for i in range(len(self.spei_threshold)):
            spei_ext = xr.where((spei_temp <= self.spei_threshold[i]), 1, 0)
            spei_ext_sum = spei_ext.mean("time") * 12
            spei_annual[i, :, :] = spei_ext_sum
        spei_annual_all = xr.DataArray(
            spei_annual,
            coords={
                "spei_index": self.spei_threshold,
                "lat": lats_all,
                "lon": lons_all,
            },
            dims=["spei_index", "lat", "lon"],
        )
        path = self.resource.path.format(gcm=gcm, scenario=scenario, year=central_year)
        target.write(path, spei_annual_all)
        return spei_annual_all

    def run_single(
        self,
        item: BatchItem,
        source,
        target: ReadWriteDataArray,
        client,
        progress_store: Optional[ProgressStore] = None,
    ):
        """Process a single batch item and write the data to the Zarr store."""
        assert isinstance(target, OscZarr)
        calculate_spei = True
        calculate_average_spei = True
        if calculate_spei:
            self.calculate_spei(item.gcm, item.scenario, progress_store)
        if calculate_average_spei:
            for central_year in item.central_years:
                self.calculate_annual_average_spei(
                    item.gcm, item.scenario, central_year, target
                )

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        return []

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self.resource)

    @override
    def prepare(self, force, download_dir, force_download):
        return super().prepare(force, download_dir, force_download)

    def onboard_single(self, target, download_dir):
        """Run onboard for a given hazard."""
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._resource()]

    def _resource(self) -> HazardResource:
        # with open(os.path.join(os.path.dirname(__file__), "days_tas_above.md"), "r") as f:
        #    description = f.read()
        resource = HazardResource(
            hazard_type="Drought",
            indicator_id="months/spei12m/below/index",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": list(self.gcms)},
            path="drought/osc/v1/months_spei12m_below_index_{gcm}_{scenario}_{year}",
            display_name="Drought SPEI index",
            description="",
            display_groups=["Drought SPEI index"],  # display names of groupings
            group_id="",
            map=MapInfo(
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=12.0,
                    max_index=255,
                    units="months/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 85.0],
                index_values=self.spei_threshold,
                path="maps/drought/osc/v1/months_spei12m_below_index_{gcm}_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="months/year",
            store_netcdf_coords=False,
            scenarios=[
                # Scenario(
                #     id="historical",
                #     years=[self.central_years[0]]),
                # Scenario(
                #     id="ssp126",
                #     years=list(self.central_years)),
                # Scenario(
                #     id="ssp245",
                #     years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource
