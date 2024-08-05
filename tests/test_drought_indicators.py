import json
import os
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr  # type: ignore

from hazard.docs_store import DocStore
from hazard.models.drought_index import DroughtIndicator, LocalZarrWorkingStore, ProgressStore, S3ZarrWorkingStore


@pytest.mark.skip(reason="incomplete")
def test_spei_indicator(test_output_dir, s3_credentials):
    # to test
    gcm = "MIROC6"
    scenario = "ssp585"
    working_store = S3ZarrWorkingStore()
    working_store = LocalZarrWorkingStore(test_output_dir)
    model = DroughtIndicator(working_store)
    model.calculate_spei(gcm, scenario, progress_store=ProgressStore(test_output_dir, "spei_prog_store"))
    # target = TestTarget()
    data_chunks = model.get_datachunks()
    test_chunk = data_chunks["Chunk_0255"]
    lat_min, lat_max = test_chunk["lat_min"], test_chunk["lat_max"]
    lon_min, lon_max = test_chunk["lon_min"], test_chunk["lon_max"]
    assert lat_min == 10.0 and lat_max == 20.0
    assert lon_min == 30.0 and lon_max == 40.0

    # model.calculate_spei("MIROC6", "ssp585")
    # model.calculate_annual_average_spei("MIROC6", "ssp585", 2080, target)

    ds_tas = model.read_quantity_from_s3_store(gcm, scenario, "tas", lat_min, lat_max, lon_min, lon_max).chunk(
        {"time": 100000}
    )
    ds_tas_local = ds_tas.compute()
    series_tas = ds_tas_local["tas"][0, 0, :].values
    # ds_pr = model.read_quantity_from_s3_store(gcm, scenario, "pr", lat_min, lat_max, lon_min, lon_max).chunk(
    #    {"time": 100000}
    # )
    series_pr = ds_tas_local["tas"][0, 0, :].values

    with open(os.path.join(test_output_dir, "drought", "data.json")) as f:
        f.write(json.dumps({"tas": list(series_tas), "pr": list(series_pr)}))

    # model.calculate_annual_average_spei("MIROC6", "ssp585", 2080, target)


def test_partial_write_zarr(test_output_dir):
    zarr_store = zarr.DirectoryStore(os.path.join(test_output_dir, "drought", "hazard.zarr"))

    lat = np.arange(-60 + 0.25 / 2, 90 + 0.25 / 2, 0.25)
    lon = np.arange(0.25 / 2, 360 + 0.25 / 2, 0.25)
    time = np.arange(
        datetime(1950, 1, 1, hour=12),
        datetime(2100, 12, 31, hour=12),
        timedelta(days=1),
    ).astype(np.datetime64)
    # data = da.zeros([len(time), len(lat), len(lon)])
    data = da.empty([len(time), len(lat), len(lon)])
    da_spei = xr.DataArray(
        data=data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
    ).chunk(chunks={"lat": 40, "lon": 40, "time": 100000})
    ds_spei = da_spei.to_dataset(name="spei")
    ds_spei.to_zarr(store=zarr_store, mode="w", compute=False)
    # see https://docs.xarray.dev/en/stable/user-guide/io.html?appending-to-existing-zarr-stores=#appending-to-existing-zarr-stores # noqa: E501
    sliced = ds_spei.sel(lat=slice(10, 20), lon=slice(30, 40))
    lat_indexes = np.where(np.logical_and(ds_spei["lat"].values >= 10, ds_spei["lat"].values <= 20))[0]
    lon_indexes = np.where(np.logical_and(ds_spei["lon"].values >= 30, ds_spei["lon"].values <= 40))[0]
    ds_spei_slice = (
        xr.DataArray(
            coords={
                "time": time,
                "lat": sliced["lat"].values,
                "lon": sliced["lon"].values,
            },
            dims=["time", "lat", "lon"],
        )
        .chunk(chunks={"lat": 40, "lon": 40, "time": 100000})
        .to_dataset(name="spei")
    )

    ds_spei_slice.to_zarr(
        store=zarr_store,
        mode="r+",
        region={
            "time": slice(None),
            "lat": slice(lat_indexes[0], lat_indexes[-1] + 1),
            "lon": slice(lon_indexes[0], lon_indexes[-1] + 1),
        },
    )


def test_progress_store(test_output_dir):
    store = ProgressStore(test_output_dir, "test_progress_store")
    store.reset()
    store.add_completed([4, 5, 8])
    store.add_completed([6, 7, 8])
    remaining = store.remaining(10)  # 0 to 9
    np.testing.assert_array_equal(remaining, [0, 1, 2, 3, 9])


@pytest.mark.skip(reason="example")
def test_doc_store(test_output_dir, s3_credentials):
    docs_store = DocStore()
    docs_store.write_new_empty_inventory()
    zarr_store = zarr.DirectoryStore(os.path.join(test_output_dir, "drought", "hazard.zarr"))
    resource = DroughtIndicator(zarr_store).resource
    docs_store.update_inventory([resource])
