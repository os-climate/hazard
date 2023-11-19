from datetime import datetime, timedelta
import json
from os import path
import os
import dask.array as da
import fsspec.implementations.local as local # type: ignore
import numpy as np
import s3fs
import xarray as xr
import zarr
from hazard.docs_store import DocStore


from hazard.models.drought_index import DroughtIndicator
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource
from .utilities import TestTarget, s3_credentials, test_output_dir

def test_spei_indicator(test_output_dir, s3_credentials):
    # to test 
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY_DEV"], secret=os.environ["OSC_S3_SECRET_KEY_DEV"])
    working_path = os.environ["OSC_S3_BUCKET_DEV"] + "/drought/osc/v01"
    zarr_store = s3fs.S3Map(root=working_path, s3=s3, check=False)
    model = DroughtIndicator(zarr_store)
    target = TestTarget()
    data_chunks = model.get_datachunks()
    test_chunk = data_chunks["Chunk_0255"]
    gcm = "MIROC6"
    scenario = "ssp585"
    lat_min, lat_max = test_chunk["lat_min"], test_chunk["lat_max"]
    lon_min, lon_max = test_chunk["lon_min"], test_chunk["lon_max"]
    assert lat_min == 10.0 and lat_max == 20.0
    assert lon_min == 30.0 and lon_max == 40.0

    #model.calculate_spei("MIROC6", "ssp585")
    #model.calculate_annual_average_spei("MIROC6", "ssp585", 2080, target)

    ds_tas = model.read_quantity_from_s3_store(gcm, scenario, "tas", lat_min, lat_max, lon_min, lon_max).chunk({'time': 100000})
    ds_tas_local = ds_tas.compute()
    series_tas = ds_tas_local["tas"][0, 0, :].values
    ds_pr = model.read_quantity_from_s3_store(gcm, scenario, "pr", lat_min, lat_max, lon_min, lon_max).chunk({'time': 100000})
    series_pr = ds_tas_local["tas"][0, 0, :].values

    with open(os.path.join(test_output_dir, "drought", "data.json")) as f:
        f.write(json.dumps({"tas": list(series_tas), "pr": list(series_pr)}))

    
    #model.calculate_annual_average_spei("MIROC6", "ssp585", 2080, target)

def test_partial_write_zarr(test_output_dir):
    zarr_store = zarr.DirectoryStore(os.path.join(test_output_dir, 'drought', 'hazard.zarr'))

    lat = np.arange(-60 + 0.25 / 2, 90 + 0.25 / 2, 0.25)
    lon = np.arange(0.25 / 2, 360 + 0.25 / 2, 0.25)
    time = np.arange(datetime(1950, 1, 1, hour=12), datetime(2100, 12, 31, hour=12), timedelta(days=1)).astype(np.datetime64)
    #data = da.zeros([len(time), len(lat), len(lon)])
    data = da.empty([len(time), len(lat), len(lon)])
    da_spei = xr.DataArray(data=data, coords={'time': time, 'lat': lat,'lon': lon}, 
                       dims=["time", "lat", "lon"]).chunk(chunks={'lat': 40,'lon': 40,'time': 100000})
    ds_spei = da_spei.to_dataset(name='spei')
    ds_spei.to_zarr(store=zarr_store, mode='w', compute=False)
    # see https://docs.xarray.dev/en/stable/user-guide/io.html?appending-to-existing-zarr-stores=#appending-to-existing-zarr-stores
    sliced = ds_spei.sel(lat=slice(10, 20), lon=slice(30, 40))
    lat_indexes = np.where(np.logical_and(ds_spei['lat'].values >= 10, ds_spei['lat'].values <= 20))[0]
    lon_indexes = np.where(np.logical_and(ds_spei['lon'].values >= 30, ds_spei['lon'].values <= 40))[0]
    ds_spei_slice = xr.DataArray(coords={'time': time, 'lat': sliced["lat"].values,'lon': sliced["lon"].values}, 
                       dims=["time", "lat", "lon"]).chunk(chunks={'lat': 40,'lon': 40,'time': 100000}).to_dataset(name='spei')
    
    ds_spei_slice.to_zarr(store=zarr_store, mode='r+', region={"time": slice(None),
                                                               "lat": slice(lat_indexes[0], lat_indexes[-1] + 1), "lon": slice(lon_indexes[0], lon_indexes[-1] + 1)})


def test_check_result(test_output_dir, s3_credentials):
    source = OscZarr()
    path = "drought/osc/v1/months_spei12m_below_MIROC6_ssp585_2050"
    za = source.read_zarr(path)
    attrs = za.attrs
    print(attrs)


def test_doc_store(test_output_dir, s3_credentials):
    docs_store = DocStore()
    docs_store.write_new_empty_inventory()
    zarr_store = zarr.DirectoryStore(os.path.join(test_output_dir, 'drought', 'hazard.zarr'))
    resource = DroughtIndicator(zarr_store).resource
    docs_store.update_inventory([resource])
    #source, target = OscZarr(), OscZarr()
    #create_tiles_for_resource(source, target, resource)