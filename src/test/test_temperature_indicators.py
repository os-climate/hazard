import os

import fsspec.implementations.local as local # type: ignore
import pytest # type: ignore
from hazard.models.days_tas_above import DaysTasAboveIndicator # type: ignore
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
import s3fs # type: ignore
import xarray as xr
import zarr # type: ignore

from hazard.sources.osc_zarr import OscZarr # type: ignore

from .utilities import TestSource, TestTarget, _create_test_datasets_tas, test_output_dir


def test_days_tas_above_mocked():
    """Test degree days calculation based on mocked data."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = TestSource(_create_test_datasets_tas("tas"))
    target = TestTarget()
    # cut down the transform
    model = DaysTasAboveIndicator(threshold_temps_c = [27], window_years=2, gcms=[gcm], scenarios=[scenario], central_years=[year])  
    model.run_all(source, target)
    with source.open_dataset_year(gcm, scenario, "tas", 2029) as y0:
        with source.open_dataset_year(gcm, scenario, "tas", 2030) as y1:
            scale = 365.0 / len(y0.time)
            ind0 = scale * xr.where(y0.tas > (27 + 273.15), 1, 0).sum(dim=["time"])
            ind1 = scale * xr.where(y1.tas > (27 + 273.15), 1, 0).sum(dim=["time"])
            expected = (ind0 + ind1) / 2 
    assert expected.values == pytest.approx(target.dataset.values)


@pytest.mark.skip(reason="inputs large and downloading slow")
def test_days_tas_above(test_output_dir):
    """Test an air temperature indicator that provides days over $x$ degrees."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    years = [2028, 2029, 2030]
    # For 
    download_test_datasets(test_output_dir, gcm, scenario, years)
    source = NexGddpCmip6(root=os.path.join(test_output_dir, NexGddpCmip6.bucket), fs=local.LocalFileSystem())
    store = zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr'))
    model = DaysTasAboveIndicator(threshold_temps_c=[10], window_years=1, gcms=[gcm], scenarios=[scenario], central_years=[years[0]])
    target = OscZarr(store=store)
    model.run_all(source, target)

def download_test_datasets(test_output_dir, gcm, scenario, years, indicators=["tas"]):
    store = NexGddpCmip6()
    s3 = s3fs.S3FileSystem(anon=True)
    for year in years:
        for indicator in indicators:
            path, _ = store.path(gcm, scenario, indicator, year)
            if not os.path.exists(os.path.join(test_output_dir, path)):
                s3.download(path, os.path.join(test_output_dir, path))
    assert True
