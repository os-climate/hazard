import concurrent.futures
import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Sequence, Union

import cftime
import dask.array as da
from distributed import Client
import numpy as np  # type: ignore
import pandas as pd
import s3fs  # type: ignore
import xarray as xr
import xclim.indices  # type: ignore
import zarr  # type: ignore
import zarr.hierarchy
from pydantic import BaseModel
from pydantic.type_adapter import TypeAdapter
from zarr.errors import GroupNotFoundError  # type: ignore

from hazard.indicator_model import IndicatorModel  # type: ignore
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import MultiYearAverageIndicatorBase  # type: ignore
from hazard.models.prechunker import Prechunker
from hazard.protocols import ReadWriteDataArray
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)
DEFAULT_DATASOURCE = NexGddpCmip6()
DEFAULT_YEARS = np.arange(1980, 2101)
MULTI_MODEL_ID = "multi_model_0"


class BatchItem:
    def __init__(self, gcm: str, scenario: str, central_years: List[int]):
        self.gcm = gcm
        self.scenario = scenario
        self.central_years = central_years

    def __repr__(self):
        return f"{self.gcm}, {self.scenario}, years: {','.join(str(y) for y in self.central_years)}"

    def __str__(self):
        return f"{self.gcm}, {self.scenario}, years: {','.join(str(y) for y in self.central_years)}"


def s3_zarr_working_store():
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=os.environ["OSC_S3_ACCESS_KEY_DEV"],
        secret=os.environ["OSC_S3_SECRET_KEY_DEV"],
    )
    base_path = os.environ["OSC_S3_BUCKET_DEV"] + "/drought/osc/v1"
    return s3fs.S3Map(root=base_path, s3=s3, check=False)


def local_zarr_working_store(working_dir: str):
    base_path = os.path.join(working_dir)
    return zarr.DirectoryStore(base_path)


def in_memory_zarr_working_store():
    return zarr.MemoryStore()


class ChunkIndicesComplete(BaseModel):
    complete_indices: List[int]


class ProgressStore:
    def __init__(self, dir: str, id: str):
        """Simple persistent JSON store of the indices of a chunked calculation
        that are complete.

        Args:
            dir (str): Path to directory for storing progress files.
            id (str): Identifier for the calculation (for use in filename).
        """
        self.dir = dir
        self.id = id
        if not os.path.exists(self._filename()):
            self.reset()

    def reset(self):
        self._write(ChunkIndicesComplete(complete_indices=[]))

    def add_completed(self, indices: Sequence[int]):
        existing = set(self._read().complete_indices)
        union = existing.union(set(indices))
        self._write(ChunkIndicesComplete(complete_indices=list(union)))

    def completed(self):
        return self._read().complete_indices

    def remaining(self, n_indices: int):
        existing = self._read().complete_indices
        return np.setdiff1d(np.arange(0, n_indices, dtype=int), existing)

    def _filename(self):
        return os.path.join(self.dir, self.id + ".json")

    def _read(self):
        with open(self._filename(), "r") as f:
            indices = TypeAdapter(ChunkIndicesComplete).validate_json(f.read())
            return indices

    def _write(self, indices: ChunkIndicesComplete):
        with open(self._filename(), "w") as f:
            f.write(json.dumps(indices.model_dump()))


class DroughtIndicator(IndicatorModel[BatchItem]):
    def __init__(
        self,
        working_zarr_store: MutableMapping,
        progress_store_path: str,
        window_years: int = MultiYearAverageIndicatorBase._default_window_years,
        gcms: Iterable[str] = MultiYearAverageIndicatorBase._europe_gcms,
        scenarios: Iterable[str] = MultiYearAverageIndicatorBase._default_scenarios,
        central_years: Sequence[int] = [2005, 2030, 2040, 2050, 2080],
    ):
        self.calib_start, self.calib_end = datetime(1985, 1, 1), datetime(2015, 1, 1)
        self.calc_start, self.calc_end = datetime(1985, 1, 1), datetime(2100, 12, 31)
        self.freq, self.window, self.dist, self.method = "MS", 12, "gamma", "APP"
        self.lat_min, self.lat_max = -60.0, 90.0
        self.lon_min, self.lon_max = 0, 360.0
        self.central_years = central_years
        self.histo_central_year = 2005
        self.progress_store_path = progress_store_path
        self.spei_threshold = [0, -1, -1.5, -2, -2.5, -3, -3.6]
        self.gcms = gcms
        self.scenarios = scenarios
        self.working_zarr_store = working_zarr_store
        self.window_years = window_years
        self.resource = self._resource()

    def pre_chunk(
        self,
        item: BatchItem,
        years: Union[Sequence[int], np.ndarray] = DEFAULT_YEARS,
        quantities: Sequence[str] = ["tas", "pr"],
        lat_chunk_size: int = 40,
        lon_chunk_size: int = 40,
        datasource: NexGddpCmip6 = DEFAULT_DATASOURCE,
    ):
        """Create a chunked data set for the given quantities and data source. This is for when the data
        source is either unchunked or unsuitably chunked for the calculation in hand. An SPEI index is an
        example since the calculation requires a long time series but for a limited spatial extent. The calculation
        therefore runs

        Args:
            years (_type_, optional): Years included in chunked data. Defaults to np.arange(1950, 2101).
            variables (list, optional): Quantities included in chunked data. Defaults to ['tas', 'pr'].
            lat_chunk_size (int, optional): Latitude chunks. Defaults to 40.
            lon_chunk_size (int, optional): Longitude chunks. Defaults to 40.
            datasource (_type_, optional): Source for building chunked data. Defaults to NexGddpCmip6().
        """
        prechunker = Prechunker(
            Path(self.progress_store_path) / "temp_download",
            self.working_zarr_store,
            gcms=[item.gcm],
            year_min=self.calib_start.year,
            year_max=self.calc_end.year,
            scenarios=[item.scenario],
            quantities=quantities,
        )
        prechunker.prechunk()

    def read_quantity_from_s3_store(
        self, gcm, scenario, quantity, lat_min, lat_max, lon_min, lon_max
    ) -> xr.Dataset:
        ds = self.chunked_dataset(gcm, scenario, quantity).sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        return ds

    def chunked_dataset(self, gcm, scenario, quantity) -> xr.Dataset:
        ds = xr.open_zarr(
            store=self.working_zarr_store,
            group=quantity + "_" + gcm + "_" + scenario,
            consolidated=False,
        )
        return ds

    def get_datachunks(self):
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
                        for x in zip(lat_bins[:-1], lat_bins[1:], strict=False)
                    ],
                    [
                        {"lon_min": x[0], "lon_max": x[1]}
                        for x in zip(lon_bins[:-1], lon_bins[1:], strict=False)
                    ],
                )
            )
        }
        return data_chunks

    def calculate_spei(
        self, gcm, scenario, progress_store: Optional[ProgressStore] = None
    ):
        """Calculate SPEI for the given GCM and scenario, storing"""
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
        path = "spei" + "_" + gcm + "_" + scenario
        store = self.working_zarr_store
        # consider refactoring data_chunks to give both slice and values?
        try:
            # if dataset does not already exist then create
            zarr.hierarchy.open_group(store=store, path=path, mode="r")
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
            ds_spei.to_zarr(
                store=store, group=path, mode="w", compute=False, consolidated=False
            )
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
            group=path,
            mode="r+",
            region={
                "lat": slice(lat_indexes[0], lat_indexes[-1] + 1),
                "lon": slice(lon_indexes[0], lon_indexes[-1] + 1),
                "time": slice(time_indexes[0], time_indexes[-1] + 1),
            },
            consolidated=False,
        )
        logger.info(f"written chunk {chunk_name} to zarr array.")
        return chunk_name

    def _calculate_spei_for_slice(
        self, lat_min, lat_max, lon_min, lon_max, *, gcm, scenario, num_workers=4
    ):
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
            xclim.indices.potential_evapotranspiration(tas=ds_tas["tas"], method="MB05")
            .astype("float32")
            .to_dataset(name="pet")
        )
        da_wb = xclim.indices.water_budget(pr=ds_pr["pr"], evspsblpot=ds_pet["pet"])
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
            xclim.indices.standardized_precipitation_evapotranspiration_index(
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
            f"calculated SPEI for gcm={gcm}, scenario={scenario}, lats=[{lat_min, lat_max}], lons=[{lon_min, lon_max}]"
        )
        return ds_spei

    def calculate_annual_average_spei(
        self, gcm: str, scenario: str, central_year: int, target: OscZarr
    ):
        """Calculate average number of months where 12-month SPEI index is below thresholds [0, -1, -1.5, -2, -2.5, -3.6]
        for 20 years period.

        Args:
            gcm (str): Global Circulation Model ID.
            scenario (str): Scenario ID.
            year (int): Year.
            target (OscZarr): Target to write result to.
        """

        def get_spei_full_results(gcm, scenario):
            path = "spei" + "_" + gcm + "_" + scenario
            ds_spei = xr.open_zarr(
                self.working_zarr_store, group=path, consolidated=False
            )
            return ds_spei

        period = [
            datetime(central_year - self.window_years // 2, 1, 1),
            datetime(central_year + self.window_years // 2 - 1, 12, 31),
        ]
        logger.info(
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
        if isinstance(ds_spei.time.values[0], cftime.DatetimeNoLeap):
            spei_temp = ds_spei.sel(
                time=slice(
                    cftime.DatetimeNoLeap(
                        period[0].year, period[0].month, period[0].day
                    ),
                    cftime.DatetimeNoLeap(
                        period[1].year, period[1].month, period[1].day
                    ),
                )
            )
        else:
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
                "threshold": self.spei_threshold,
                "lat": lats_all,
                "lon": lons_all,
            },
            dims=["threshold", "lat", "lon"],
        )
        path = self.resource.path.format(gcm=gcm, scenario=scenario, year=central_year)
        chunks = [len(self.spei_threshold), 256, 256]
        target.write(path, spei_annual_all, chunks=chunks)
        if central_year == self.histo_central_year:
            if scenario == "ssp126" or ("ssp126" not in self.scenarios):
                path = self.resource.path.format(
                    gcm=gcm, scenario="historical", year=central_year
                )
                target.write(path, spei_annual_all, chunks=chunks)
        return spei_annual_all

    def run_single(
        self,
        item: BatchItem,
        source,
        target: ReadWriteDataArray,
        client,
    ):
        progress_store = ProgressStore(
            str(self.progress_store_path), id=item.gcm + "_" + item.scenario
        )
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

    def calculate_multi_model(
        self, source: ReadWriteDataArray, target: ReadWriteDataArray
    ):
        for scenario in self.scenarios:
            for central_year in self.central_years:
                for i, gcm in enumerate(self.gcms):
                    path = self.resource.path.format(
                        gcm=gcm, scenario=scenario, year=central_year
                    )
                    if i == 0:
                        combined = source.read(path).copy()
                    else:
                        combined = combined + source.read(path)
                combined = combined / len(list(self.gcms))
                target_path = self.resource.path.format(
                    gcm="multi_model_0", scenario=scenario, year=central_year
                )
                target.write(target_path, combined)

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        # Do not include historical scenario. For each GCM this is taken from SSP126
        # (if present in the batch, otherwise written for each SSP).
        return [
            BatchItem(gcm, scenario, list(self.central_years))
            for gcm in self.gcms
            for scenario in [s for s in self.scenarios if s != "historical"]
            if gcm != MULTI_MODEL_ID
        ]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        create_tiles_for_resource(source, target, self.resource)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._resource()]

    def _resource(self) -> HazardResource:
        with open(
            os.path.join(os.path.dirname(__file__), "drought_index.md"), "r"
        ) as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="Drought",
            indicator_id="months/spei12m/below/threshold",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": [MULTI_MODEL_ID] + list(self.gcms)},
            path="drought/osc/v2/months_spei12m_below_threshold_{gcm}_{scenario}_{year}",
            display_name="Months 12m SPEI below threshold/{gcm}",
            description=description,
            display_groups=[
                "Months 12m SPEI below threshold"
            ],  # display names of groupings
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
                bounds=[],
                bbox=[],
                index_values=self.spei_threshold,
                path="maps/drought/osc/v2/months_spei12m_below_threshold_{gcm}_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="months/year",
            store_netcdf_coords=True,
            scenarios=[
                Scenario(
                    id=scen,
                    years=[self.central_years[0]]
                    if scen == "historical"
                    else list(self.central_years),
                )
                for scen in self.scenarios
            ],
        )
        return resource
