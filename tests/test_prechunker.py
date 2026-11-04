import os
import pickle
import threading
import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import cloudpickle
import numpy as np
import pytest
import xarray as xr

from hazard.models.prechunker import NexGddpCmip6CachedSource, download_paths


class _FakeS3Client:
    """Fake boto3 S3 client whose objects are all the same small NetCDF file."""

    def __init__(self):
        self.downloaded_keys = []
        self._lock = threading.Lock()
        self._data = xr.Dataset(
            {"tas": (["time", "lat", "lon"], np.full((1, 2, 2), 300.0))},
            coords={"time": [0], "lat": [1.0, 0.0], "lon": [0.0, 1.0]},
        ).to_netcdf()

    def download_fileobj(self, bucket, key, fileobj):
        time.sleep(0.1)  # give other threads the chance to request the same file
        with self._lock:
            self.downloaded_keys.append(key)
        fileobj.write(self._data)


def test_cached_source_downloads_missing_files(tmp_path: Path):
    """A file not in the cache is downloaded when first opened, then read from the cache."""
    client = _FakeS3Client()
    source = NexGddpCmip6CachedSource(tmp_path, client=client)
    with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030) as ds:
        assert float(ds.tas.max()) == 300.0
    with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030) as ds:
        assert float(ds.tas.max()) == 300.0
    assert len(client.downloaded_keys) == 1
    assert client.downloaded_keys[0].endswith(
        "tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2030_v2.0.nc"
    )
    assert [p.name for p in tmp_path.iterdir()] == [
        "tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2030_v2.0.nc"
    ]


def test_cached_source_downloads_once_for_concurrent_opens(tmp_path: Path):
    """Several threads opening the same missing file download it only once."""
    client = _FakeS3Client()
    source = NexGddpCmip6CachedSource(tmp_path, client=client)

    def open_max(_):
        with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030) as ds:
            return float(ds.tas.max())

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(open_max, range(4))) == [300.0] * 4
    assert len(client.downloaded_keys) == 1


def test_cached_source_downloads_once_across_copies(tmp_path: Path):
    """Separate instances (e.g. the copies dask gives each task) share the per-file download locks."""
    client = _FakeS3Client()
    sources = [NexGddpCmip6CachedSource(tmp_path, client=client) for _ in range(4)]

    def open_max(source):
        with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030) as ds:
            return float(ds.tas.max())

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(open_max, sources)) == [300.0] * 4
    assert len(client.downloaded_keys) == 1


def test_cached_source_can_be_pickled(tmp_path: Path):
    """The source can be pickled (as dask does with task arguments); the copy reads from the cache."""
    source = NexGddpCmip6CachedSource(tmp_path, client=_FakeS3Client())
    with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030):
        pass  # downloads to the cache
    copy = pickle.loads(cloudpickle.dumps(source))
    assert copy._client is None  # the S3 client is not pickled
    with copy.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030) as ds:
        assert float(ds.tas.max()) == 300.0


def test_cached_source_download_inputs(tmp_path: Path):
    """download_inputs downloads only the files not already in the cache, each once; dry_run downloads nothing."""
    client = _FakeS3Client()
    source = NexGddpCmip6CachedSource(tmp_path, client=client)
    inputs = [("ACCESS-CM2", "ssp585", "tas", y) for y in [2030, 2031, 2031]]
    assert len(source.download_inputs(inputs, dry_run=True)) == 2
    assert client.downloaded_keys == []
    with source.open_dataset_year("ACCESS-CM2", "ssp585", "tas", 2030):
        pass  # downloads 2030 to the cache
    missing = source.download_inputs(inputs)
    assert [Path(p).name for p in missing] == [
        "tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2031_v2.0.nc"
    ]
    assert len(client.downloaded_keys) == 2
    assert source.download_inputs(inputs) == []


def test_cached_source_remove_stale_downloads(tmp_path: Path):
    """The partial files of interrupted downloads are removed once stale; recent ones are kept, and complete files never."""
    source = NexGddpCmip6CachedSource(tmp_path, client=_FakeS3Client())
    stale, recent, complete = (
        tmp_path / "a.nc.download",
        tmp_path / "b.nc.download",
        tmp_path / "c.nc",
    )
    for f in [stale, recent, complete]:
        f.write_bytes(b"x")
    hour_ago = time.time() - 3601
    os.utime(stale, (hour_ago, hour_ago))
    os.utime(complete, (hour_ago, hour_ago))
    assert source.remove_stale_downloads() == [stale]
    assert not stale.exists() and recent.exists() and complete.exists()


def test_download_paths_interrupted(tmp_path: Path):
    """If interrupted, download_paths does not start the downloads still queued."""
    client = _FakeS3Client()
    source = NexGddpCmip6CachedSource(tmp_path, client=client)
    paths = [source._path("ACCESS-CM2", "ssp585", "tas", y) for y in range(2030, 2050)]

    def interrupt(fs, *args, **kwargs):
        raise KeyboardInterrupt

    with mock.patch.object(futures, "as_completed", interrupt):
        with pytest.raises(KeyboardInterrupt):
            download_paths(paths, tmp_path, client)
    # at most the 8 downloads started before the interrupt
    assert len(client.downloaded_keys) <= 8
