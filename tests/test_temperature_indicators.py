import os

import fsspec.implementations.local as local  # type: ignore
import numpy as np
import pytest  # type: ignore
import s3fs  # type: ignore
import xarray as xr
import zarr  # type: ignore

from hazard.models.days_tas_above import DaysTasAboveIndicator  # type: ignore
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr  # type: ignore

from .conftest import (
    TestSource,
    TestTarget,
    _create_test_datasets_hurs,
    _create_test_datasets_tas,
)


def test_days_tas_above_mocked():
    """Test degree days calculation based on mocked data."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = TestSource(_create_test_datasets_tas("tas"), [gcm])
    target = TestTarget()
    # cut down the transform
    model = DaysTasAboveIndicator(
        threshold_temps_c=[27],
        window_years=2,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[year],
    )
    model.run_all(source, target)
    with source.open_dataset_year(gcm, scenario, "tas", 2029) as y0:
        with source.open_dataset_year(gcm, scenario, "tas", 2030) as y1:
            scale = 365.0 / len(y0.time)
            ind0 = scale * xr.where(y0.tas > (27 + 273.15), 1, 0).sum(dim=["time"])
            ind1 = scale * xr.where(y1.tas > (27 + 273.15), 1, 0).sum(dim=["time"])
            expected = (ind0 + ind1) / 2
    assert expected.values == pytest.approx(
        target.datasets[
            "chronic_heat/osc/v2/days_tas_above_27c_NorESM2-MM_ssp585_2030"
        ].values
    )


def test_days_wbgt_above_mocked():
    """Test degree days calculation based on mocked data."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    test_sets = _create_test_datasets_tas(quantity="tas")
    test_sets.update(_create_test_datasets_hurs())
    threshold_temps_c = 27.0
    source = TestSource(test_sets, [gcm])
    target = TestTarget()
    # cut down the transform
    model = WetBulbGlobeTemperatureAboveIndicator(
        threshold_temps_c=[threshold_temps_c],
        window_years=2,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[year],
    )
    model.run_all(source, target, debug_mode=True)
    result = target.datasets[
        "chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}".format(
            gcm=gcm, scenario=scenario, year=year
        )
    ]
    with source.open_dataset_year(gcm, scenario, "tas", 2029).tas as t0:
        with source.open_dataset_year(gcm, scenario, "hurs", 2029).hurs as h0:
            tas_c = t0 - 273.15  # convert from K to C
            # vpp is water vapour partial pressure in kPa
            vpp = (h0 / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
            wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
            scale = 365 / len(wbgt.time)
            ind0 = xr.where(wbgt > threshold_temps_c, scale, 0.0).sum(dim=["time"])
    with source.open_dataset_year(gcm, scenario, "tas", 2030).tas as t1:
        with source.open_dataset_year(gcm, scenario, "hurs", 2030).hurs as h1:
            tas_c = t1 - 273.15
            vpp = (h1 / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
            wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
            scale = 365 / len(wbgt.time)
            ind1 = xr.where(wbgt > threshold_temps_c, scale, 0.0).sum(dim=["time"])
    expected = (ind0 + ind1) / 2.0
    assert np.allclose((expected.values - result.values).reshape(9), 0.0)


@pytest.mark.skip(reason="inputs large and downloading slow")
def test_days_tas_above(test_output_dir):  # noqa: F811
    """Test an air temperature indicator that provides days over $x$ degrees."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    years = [2028, 2029, 2030]
    # For
    download_test_datasets(test_output_dir, gcm, scenario, years)
    source = NexGddpCmip6(
        root=os.path.join(test_output_dir, NexGddpCmip6.bucket),
        fs=local.LocalFileSystem(),
    )
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    model = DaysTasAboveIndicator(
        threshold_temps_c=[10],
        window_years=1,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[years[0]],
    )
    target = OscZarr(store=store)
    model.run_all(source, target)


def download_test_datasets(test_output_dir, gcm, scenario, years, indicators=None):  # noqa: F811
    if indicators is None:
        indicators = ["tas"]
    store = NexGddpCmip6()
    s3 = s3fs.S3FileSystem(anon=True)
    for year in years:
        for indicator in indicators:
            path, _ = store.path(gcm, scenario, indicator, year)
            if not os.path.exists(os.path.join(test_output_dir, path)):
                s3.download(path, os.path.join(test_output_dir, path))
    assert True
