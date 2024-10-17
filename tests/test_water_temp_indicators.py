import numpy as np
import xarray as xr

from hazard.models.water_temp import FutureStreamsSource, WaterTemperatureAboveIndicator

from .conftest import TestSource, TestTarget, _create_test_datasets_tas


def test_future_streams_source():
    source = FutureStreamsSource("")
    assert 1976 == source.from_year("MIROC", 1985) and 1979 == source.from_year(
        "E2O", 1985
    )
    _, url = source.water_temp_download_path("E2O", "historical", 1995)
    assert (
        url
        == "https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/"  # noqa:W503
        + "waterTemp/hist/E2O/waterTemp_weekAvg_output_E2O_hist_1986-01-07_to_1995-12-30.nc"  # noqa:W503
    )
    _, url = source.water_temp_download_path("NorESM", "rcp8p5", 2019)
    assert (
        url
        == "https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/"  # noqa:W503
        + "waterTemp/rcp8p5/noresm/waterTemp_weekAvg_output_noresm_rcp8p5_2006-01-07_to_2019-12-30.nc"  # noqa:W503
    )


def test_water_temp_above_mocked():
    year = 2030
    gcm = "NorESM"
    scenario = "rcp2p6"
    quantity = "waterTemperature"
    threshold_temps_c = 27.0
    dataset = _create_test_datasets_tas(quantity)
    dataset = dict(
        zip(
            dataset.keys(),
            [
                dataset[key].rename({"lat": "latitude", "lon": "longitude"})
                for key in dataset
            ],
            strict=False,
        )
    )
    source = TestSource(dataset, [gcm])
    target = TestTarget()
    # cut down the transform
    model = WaterTemperatureAboveIndicator(
        threshold_temps_c=[threshold_temps_c],
        window_years=2,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[year],
    )
    model.run_all(source, target)

    result = target.datasets[
        "chronic_heat/nluu/v2/weeks_water_temp_above_{gcm}_{scenario}_{year}".format(
            gcm=gcm, scenario=scenario, year=year
        )
    ]
    threshold_temps_k = threshold_temps_c + 273.15
    with source.open_dataset_year(gcm, scenario, quantity, 2029) as y0:
        scale0 = 52.0 / len(y0.time)
        ind0 = xr.where(y0.waterTemperature > threshold_temps_k, scale0, 0.0).sum(
            dim=["time"]
        )
        with source.open_dataset_year(gcm, scenario, quantity, 2030) as y1:
            scale1 = 52.0 / len(y1.time)
            ind1 = xr.where(y1.waterTemperature > threshold_temps_k, scale1, 0.0).sum(
                dim=["time"]
            )
            expected = (ind0 + ind1) / 2.0
    assert np.allclose((expected.values - result.values).reshape(9), 0.0)
