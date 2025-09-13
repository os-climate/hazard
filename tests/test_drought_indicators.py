import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr  # type: ignore
from .conftest import test_dir

from hazard.docs_store import DocStore
from hazard.models.drought_index import (
    DroughtIndicator,
    ProgressStore,
)
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.s3_utilities import get_s3_fs, get_store


def create_test_data_sets(model: DroughtIndicator, test_inputs: Path):
    gcm = "MIROC6"
    scenario = "ssp585"
    lat_min = lat_max = 52.125
    lon_min = lon_max = 0.125
    ds_tas = (
        model.read_quantity_from_s3_store(
            gcm, scenario, "tas", lat_min, lat_max, lon_min, lon_max
        )
        .chunk({"time": 100000})
        .compute()
    )
    ds_pr = (
        model.read_quantity_from_s3_store(
            gcm, scenario, "pr", lat_min, lat_max, lon_min, lon_max
        )
        .chunk({"time": 100000})
        .compute()
    )
    time = [np.datetime_as_string(t, unit="D") for t in ds_tas.time.values]
    data = {
        "tas": list(ds_tas.tas.values.astype("float64").reshape((len(time)))),
        "pr": list(ds_pr.pr.values.astype("float64").reshape((len(time)))),
        "time": time,
    }
    (test_inputs / "drought").mkdir(exist_ok=True, parents=True)
    with open(test_inputs / "drought" / "spei_test.json", "w") as f:
        f.write(json.dumps(data, indent=2))


def load_test_data_sets(test_inputs: Path):
    lat_min = 52.125
    lon_min = 0.125

    with open(test_inputs / "drought" / "spei_test.json") as f:
        data = json.load(f)

    ds_tas = xr.DataArray(
        data=np.array(data["tas"]).reshape([len(data["time"]), 1, 1]),
        coords={
            "time": np.array(data["time"], dtype=np.datetime64),
            "lat": np.array([lat_min]),
            "lon": np.array([lon_min]),
        },
        dims=["time", "lat", "lon"],
    ).to_dataset(name="tas")
    ds_pr = xr.DataArray(
        data=np.array(data["pr"]).reshape([len(data["time"]), 1, 1]),
        coords={
            "time": np.array(data["time"], dtype=np.datetime64),
            "lat": np.array([lat_min]),
            "lon": np.array([lon_min]),
        },
        dims=["time", "lat", "lon"],
    ).to_dataset(name="pr")
    ds_pr.pr.attrs["standard_name"] = "precipitation_flux"
    ds_pr.pr.attrs["units"] = "kg m-2 s-1"
    ds_tas.tas.attrs["standard_name"] = "air_temperature"
    ds_tas.tas.attrs["units"] = "K"
    ds_tas.lat.attrs.update(
        {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )
    ds_tas.lon.attrs.update(
        {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )
    ds_pr.lat.attrs.update(
        {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )
    ds_pr.lon.attrs.update(
        {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )
    return ds_tas, ds_pr


def test_batches(test_dir):
    # Creamos el filesystem sobre S3 (entorno dev)
    s3 = get_s3_fs(use_dev=True)
    # Montamos el store en el grupo base 'hazard/hazard.zarr'
    working_store = get_store(
        s3=s3, use_dev=True, group_path_suffix="hazard/hazard.zarr"
    )

    # Instanciamos el modelo usando el working store real de S3
    model = DroughtIndicator(
        working_zarr_store=working_store, progress_store_path=test_dir
    )

    batches = model.batch_items()

    # Comprobamos que se generan los 18 batch items esperados
    assert len(batches) == 18


@pytest.mark.skip(reason="incomplete")
def test_spei_indicator(test_dir, test_output_dir, s3_credentials):
    gcm = "MIROC6"
    scenario = "ssp585"

    # s3 = get_s3_fs(use_dev=True)
    # working_store = get_store(
    #     s3=s3, use_dev=True, group_path_suffix="hazard/hazard.zarr"
    # )

    working_store = local_zarr_working_store(test_output_dir)
    working_store = in_memory_zarr_working_store()

    model = DroughtIndicator(working_zarr_store=working_store)

    test_input_path = Path(test_dir)

    ds_tas, ds_pr = load_test_data_sets(test_input_path)
    ds_tas.to_zarr(
        store=working_store,
        group="tas" + "_" + gcm + "_" + scenario,
        mode="w",
    )
    ds_pr.to_zarr(
        store=working_store,
        group="pr" + "_" + gcm + "_" + scenario,
        mode="w",
    )
    # we test for a cut-down set of just one lat/lon
    lat_min = lat_max = 52.125
    lon_min = lon_max = 0.125

    ds_slice = model._calculate_spei_for_slice(
        lat_min, lat_max, lon_min, lon_max, gcm=gcm, scenario=scenario
    )
    path = os.path.join("spei", gcm + "_" + scenario)
    ds_slice.to_zarr(store=working_store, group=path, mode="w")

    zarr_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "drought", "hazard.zarr")
    )
    target = OscZarr(store=zarr_store)

    da = model.calculate_annual_average_spei(gcm, scenario, 2005, target=target)
    np.testing.assert_almost_equal(
        da.values.reshape([7]), [6.35, 2.6, 1.1, 0.35, 0.0, 0.0, 0.0]
    )

    data_chunks = model.get_datachunks()
    test_chunk = data_chunks["Chunk_0255"]
    lat_min, lat_max = test_chunk["lat_min"], test_chunk["lat_max"]
    lon_min, lon_max = test_chunk["lon_min"], test_chunk["lon_max"]
    assert lat_min == 10.0 and lat_max == 20.0
    assert lon_min == 30.0 and lon_max == 40.0


def test_partial_write_zarr(test_output_dir):
    zarr_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "drought", "hazard.zarr")
    )

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
    lat_indexes = np.where(
        np.logical_and(ds_spei["lat"].values >= 10, ds_spei["lat"].values <= 20)
    )[0]
    lon_indexes = np.where(
        np.logical_and(ds_spei["lon"].values >= 30, ds_spei["lon"].values <= 40)
    )[0]
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
    docs_store = DocStore(local_path=test_output_dir)
    docs_store.write_new_empty_inventory()
    zarr_store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "drought", "hazard.zarr")
    )
    resource = DroughtIndicator(zarr_store).resource
    docs_store.update_inventory([resource])
