import json
import os

import numpy as np
import pytest  # type: ignore
import s3fs  # type: ignore
import typer
import xarray as xr
import zarr  # type: ignore
from fsspec.implementations import local  # type: ignore

from hazard.cli import app
from hazard.models.days_tas_above import DaysTasAboveIndicator  # type: ignore
from hazard.models.degree_days import HeatingCoolingDegreeDays
from hazard.models.deprecated_v2 import (
    DaysTasAboveIndicatorDeprecated,
    WetBulbGlobeTemperatureAboveIndicator,
)
from hazard.models.multi_year_average import MULTI_MODEL_ID, RunConfig
from hazard.models.wet_bulb_globe_temp import DaysWBGTAboveIndicator
from hazard.onboard import indicator_model_onboarder as onboarders
from hazard.onboard.indicator_model_onboarder import ThresholdIndicatorOnboarder
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr  # type: ignore
from tests.mocks import (
    TestSource,
    TestTarget,
    create_test_datasets_hurs,
    create_test_datasets_tas,
    create_test_source_tas_tasmax_hurs,
)


def test_days_tas_above_mocked():
    """Test degree days calculation based on mocked data."""
    gcm = "NorESM2-MM"
    scenario = "ssp585"
    year = 2030
    source = TestSource(create_test_datasets_tas("tas"), [gcm])
    target = TestTarget()
    # cut down the transform
    with pytest.deprecated_call():
        model = DaysTasAboveIndicatorDeprecated(
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
    test_sets = create_test_datasets_tas(quantity="tas")
    test_sets.update(create_test_datasets_hurs())
    threshold_temps_c = 27.0
    source = TestSource(test_sets, [gcm])
    target = TestTarget()
    # cut down the transform
    with pytest.deprecated_call():
        model = WetBulbGlobeTemperatureAboveIndicator(
            threshold_temps_c=[threshold_temps_c],
            window_years=2,
            gcms=[gcm],
            scenarios=[scenario],
            central_years=[year],
        )
    model.run_all(source, target, debug_mode=True)
    result = target.datasets[
        f"chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}"
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


def test_days_wbgt_v3_mocked():
    """Test the daily-mean and daily-maximum WBGT indicators, including the multi-model mean."""
    gcms = ["ACCESS-CM2", "EC-Earth3"]
    scenario = "ssp585"
    year = 2030
    thresholds = [26.0, 30.0, 34.0]
    source = create_test_source_tas_tasmax_hurs(gcms)
    target = TestTarget()
    model = DaysWBGTAboveIndicator(
        thresholds=thresholds,
        run=RunConfig(
            window_years=2,
            gcms=gcms,
            scenarios=[scenario],
            central_years=[year],
        ),
    )
    model.run_all(source, target, debug_mode=True)

    def expected_days_above(y: int):
        tas_c = source.open_dataset_year(gcms[0], scenario, "tas", y).tas - 273.15
        tasmax_c = (
            source.open_dataset_year(gcms[0], scenario, "tasmax", y).tasmax - 273.15
        )
        hurs = source.open_dataset_year(gcms[0], scenario, "hurs", y).hurs
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        scale = 365.0 / len(tas_c.time)
        days = {
            key: np.stack(
                [
                    scale * ((0.567 * t + 0.393 * vpp + 3.94) > th).sum(dim="time")
                    for th in thresholds
                ]
            )
            for key, t in [
                ("mean", tas_c),
                ("max", tasmax_c),
                ("halfway", (tas_c + tasmax_c) / 2.0),
            ]
        }
        days["444"] = (days["mean"] + days["halfway"] + days["max"]) / 3.0
        return days

    y0, y1 = expected_days_above(2029), expected_days_above(2030)
    for key in ["mean", "max", "444"]:
        expected = (y0[key] + y1[key]) / 2.0
        for gcm in gcms + [MULTI_MODEL_ID]:
            result = target.datasets[
                f"chronic_heat/osc/v3/days_wbgt_{key}_above_{gcm}_{scenario}_{year}"
            ]
            assert result.dims[0] == "threshold"
            assert list(result["threshold"].values) == thresholds
            assert result["threshold"].attrs["units"] == "°C"
            assert result.attrs["units"] == "days/year"
            np.testing.assert_allclose(result.values, expected, rtol=1e-6)
    # the test data should exercise the thresholds: not all zero or all 365
    assert 0 < y0["max"].mean() < 365
    assert (y0["max"] >= y0["mean"]).all() and (y0["max"] > y0["mean"]).any()
    # the working-day indicator lies between the two, and differs from their mean
    assert (y0["mean"] <= y0["444"]).all() and (y0["444"] <= y0["max"]).all()
    assert not np.allclose(y0["444"], (y0["mean"] + y0["max"]) / 2.0)


def test_days_wbgt_v3_multi_model_missing():
    """Multi-model mean raises an error, before writing anything, if any GCM's array is missing."""
    gcms = ["ACCESS-CM2", "EC-Earth3"]
    model = DaysWBGTAboveIndicator(
        thresholds=[30.0],
        run=RunConfig(
            window_years=2,
            gcms=gcms,
            scenarios=["ssp585"],
            central_years=[2030],
        ),
    )
    target = TestTarget()
    model.run_all(
        create_test_source_tas_tasmax_hurs(gcms), target, debug_mode=True
    )  # writes all, including multi-model
    target.datasets.pop("chronic_heat/osc/v3/days_wbgt_max_above_EC-Earth3_ssp585_2030")
    n_written = len(target.datasets)
    with pytest.raises(ValueError, match="days_wbgt_max_above_EC-Earth3_ssp585_2030"):
        model.calculate_multi_model(target)
    assert len(target.datasets) == n_written


@pytest.mark.parametrize(
    "model_class,indicator_ids,thresholds",
    [
        (
            DaysWBGTAboveIndicator,
            ["days_wbgt_mean_above", "days_wbgt_max_above", "days_wbgt_444_above"],
            [[20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44]] * 3,
        ),
        (
            DaysTasAboveIndicator,
            ["days_tas_above", "days_tasmax_above"],
            [[25, 30, 35, 40, 45, 50, 55]] * 2,
        ),
        (
            HeatingCoolingDegreeDays,
            ["cooling_degree_days", "heating_degree_days"],
            [[18, 22, 24], [10, 12, 14, 15.5, 18, 22]],
        ),
    ],
)
def test_v3_inventory(model_class, indicator_ids, thresholds):
    """Resources follow the v3 conventions; thresholds are given for each resource."""
    model = model_class()
    resources = list(model.inventory())
    assert [r.indicator_id for r in resources] == indicator_ids
    for r, resource_thresholds in zip(resources, thresholds, strict=True):
        assert r.path.startswith("chronic_heat/osc/v3/")
        assert r.map is not None and r.map.path == "maps/" + r.path + "_map"
        assert r.map.source == "map_array_pyramid"
        assert r.map.index_values == resource_thresholds
        assert r.resolution == "0.25°"
        assert r.params["gcm"] == [
            MULTI_MODEL_ID,
            "ACCESS-CM2",
            "CNRM-CM6-1",
            "EC-Earth3",
            "GFDL-ESM4",
            "MRI-ESM2-0",
            "TaiESM1",
        ]
        assert {s.id: s.years for s in r.scenarios} == {
            "historical": [2005],
            "ssp126": [2030, 2040, 2050, 2080],
            "ssp245": [2030, 2040, 2050, 2080],
            "ssp585": [2030, 2040, 2050, 2080],
        }
        assert "multi_model_0" in r.description
        assert "'historical' (averaged over 1995-2014)" in r.description
        assert "2080 (2070-2089)" in r.description
    # one batch item per GCM, scenario and central year; the multi-model mean is not a batch item
    assert len(list(model.batch_items())) == 6 * (1 + 3 * 4)


def test_days_tas_v3_mocked():
    """Test the daily-mean and daily-maximum temperature indicators, including the multi-model mean."""
    gcms = ["ACCESS-CM2", "EC-Earth3"]
    scenario = "ssp585"
    year = 2030
    thresholds = [25.0, 30.0, 35.0]
    source = create_test_source_tas_tasmax_hurs(gcms)
    target = TestTarget()
    model = DaysTasAboveIndicator(
        thresholds=thresholds,
        run=RunConfig(
            window_years=2,
            gcms=gcms,
            scenarios=[scenario],
            central_years=[year],
        ),
    )
    model.run_all(source, target, debug_mode=True)

    def expected_days_above(quantity: str):
        days = []
        for y in [2029, 2030]:
            t_c = (
                source.open_dataset_year(gcms[0], scenario, quantity, y)[quantity]
                - 273.15
            )
            scale = 365.0 / len(t_c.time)
            days.append(
                np.stack([scale * (t_c > th).sum(dim="time") for th in thresholds])
            )
        return (days[0] + days[1]) / 2.0

    for quantity in ["tas", "tasmax"]:
        expected = expected_days_above(quantity)
        assert 0 < expected.mean() < 365
        for gcm in gcms + [MULTI_MODEL_ID]:
            result = target.datasets[
                f"chronic_heat/osc/v3/days_{quantity}_above_{gcm}_{scenario}_{year}"
            ]
            assert result.dims[0] == "threshold"
            assert list(result["threshold"].values) == thresholds
            assert result["threshold"].attrs["units"] == "°C"
            assert result.attrs["units"] == "days/year"
            np.testing.assert_allclose(result.values, expected, rtol=1e-6)


def test_v3_zarr_threshold_dimension_and_units():
    """The threshold dimension and units are written to, and read back from, the Zarr arrays."""
    gcms = ["ACCESS-CM2", "EC-Earth3"]
    target = OscZarr(store=zarr.MemoryStore())
    model = DaysTasAboveIndicator(
        thresholds=[25.0, 30.0],
        run=RunConfig(
            window_years=2, gcms=gcms, scenarios=["ssp585"], central_years=[2030]
        ),
    )
    model.run_all(create_test_source_tas_tasmax_hurs(gcms), target, debug_mode=True)
    for gcm in gcms + [MULTI_MODEL_ID]:
        path = f"chronic_heat/osc/v3/days_tasmax_above_{gcm}_ssp585_2030"
        attrs = target.root[path].attrs
        assert attrs["dimensions"] == ["threshold", "latitude", "longitude"]
        assert attrs["threshold_values"] == [25.0, 30.0]
        assert attrs["threshold_units"] == "°C"
        assert attrs["units"] == "days/year"
        assert target.root[path].chunks == (2, 500, 500)
        da = target.read(path)
        assert da.dims == ("threshold", "latitude", "longitude")
        assert da["threshold"].attrs["units"] == "°C"
        assert da.attrs["units"] == "days/year"


def test_v3_zarr_write_lazy_array_whole_chunks():
    """An array read (lazily) from a Zarr store can be written back, including chunks it covers whole.

    As for the multi-model mean, calculated from the arrays of each GCM read from the target.
    """
    target = OscZarr(store=zarr.MemoryStore())
    model = DaysTasAboveIndicator(
        thresholds=[25.0, 30.0],
        run=RunConfig(
            window_years=2,
            gcms=["ACCESS-CM2"],
            scenarios=["ssp585"],
            central_years=[2030],
        ),
    )
    model.run_all(
        create_test_source_tas_tasmax_hurs(["ACCESS-CM2"]), target, debug_mode=True
    )
    da = target.read("chronic_heat/osc/v3/days_tasmax_above_ACCESS-CM2_ssp585_2030")
    # chunks smaller than the array, so that some are covered whole
    chunks = [da.shape[0], da.shape[1] // 2, da.shape[2] // 2]
    target.write("test/lazy", da, chunks=chunks)
    np.testing.assert_array_equal(target.read("test/lazy").values, da.values)


def test_v3_source_resolution_mismatch():
    """run_all raises an error, before calculating anything, if the source has a different resolution."""
    gcms = ["ACCESS-CM2"]
    source = create_test_source_tas_tasmax_hurs(gcms)
    source.resolution = "2.2 km"  # type: ignore[attr-defined]
    target = TestTarget()
    model = DaysTasAboveIndicator(
        run=RunConfig(gcms=gcms, scenarios=["ssp585"], central_years=[2030])
    )
    with pytest.raises(ValueError, match="2.2 km"):
        model.run_all(source, target, debug_mode=True)
    assert target.datasets == {}


def test_v3_inputs():
    """The inputs of a run: each quantity of each year averaged, for each GCM and scenario, once."""
    run = RunConfig(
        gcms=["ACCESS-CM2"],
        scenarios=["historical", "ssp585"],
        central_years=[2030, 2040],
        window_years=20,
    )
    inputs = DaysTasAboveIndicator(run=run).inputs(None)  # type: ignore[arg-type]
    # historical: 1995-2014; ssp585: 2020-2039 and 2030-2049, overlapping
    years = list(range(1995, 2015)) + list(range(2020, 2050))
    assert len(inputs) == len(years) * 2
    assert {(s, y) for _, s, _, y in inputs} == {
        ("historical" if y < 2015 else "ssp585", y) for y in years
    }
    assert {q for _, _, q, _ in inputs} == {"tas", "tasmax"}


def test_v3_onboarder():
    """A v3 indicator is onboarded like any other hazard, through its onboarder."""
    gcms = ["ACCESS-CM2"]
    run = RunConfig(
        window_years=2, gcms=gcms, scenarios=["ssp585"], central_years=[2030]
    )
    onboarder = ThresholdIndicatorOnboarder(DaysTasAboveIndicator(run=run))
    onboarder.source = create_test_source_tas_tasmax_hurs(gcms)
    target = TestTarget()
    onboarder.onboard(target)
    assert set(target.datasets) == {
        f"chronic_heat/osc/v3/{name}_{gcm}_ssp585_2030"
        for name in ["days_tas_above", "days_tasmax_above"]
        for gcm in gcms + [MULTI_MODEL_ID]
    }


def test_v3_onboard_cli(tmp_path, monkeypatch):
    """A v3 indicator is onboarded through the generic CLI, with the settings of its run given as JSON."""
    gcms = ["ACCESS-CM2"]
    monkeypatch.setattr(
        onboarders,
        "get_source_dataset_instance",
        lambda *args: create_test_source_tas_tasmax_hurs(gcms),
    )
    # the mocked 3x3 grid is too small for map tiles (maximum zoom level 0)
    monkeypatch.setattr(
        onboarders.ThresholdIndicatorOnboarder,
        "create_maps",
        lambda *args, **kwargs: None,
    )
    hazard_kwargs = {
        "run": {
            "window_years": 2,
            "gcms": gcms,
            "scenarios": ["ssp585"],
            "central_years": [2030],
        },
        "dask_cluster_kwargs": {"n_workers": 1, "threads_per_worker": 1},
    }
    store = tmp_path / "store"
    # run the command directly: CliRunner's capture of output conflicts with pytest's live logging
    cli = typer.main.get_command(app)
    cli.main(
        [
            "onboard",
            "DaysTasAboveIndicator",
            "--store",
            str(store),
            "--source-dir-base",
            str(tmp_path),
            "--hazard-kwargs",
            json.dumps(hazard_kwargs),
        ],
        standalone_mode=False,
    )
    root = zarr.open_group(zarr.DirectoryStore(str(store)), mode="r")
    for gcm in gcms + [MULTI_MODEL_ID]:
        assert f"chronic_heat/osc/v3/days_tas_above_{gcm}_ssp585_2030" in root
    inventory = json.loads((store / "inventory.json").read_text())
    assert {r["indicator_id"] for r in inventory["resources"]} == {
        "days_tas_above",
        "days_tasmax_above",
    }
    # invalid JSON, or JSON that is not an object, is rejected before onboarding
    for bad in ["{run:", "[1, 2]"]:
        with pytest.raises(typer.BadParameter):
            cli.main(
                ["onboard", "DaysTasAboveIndicator", "--store", str(store)]
                + ["--hazard-kwargs", bad],
                standalone_mode=False,
            )


def test_run_config():
    """RunConfig cannot be modified, and the multi-model mean cannot be given as a GCM."""
    run = RunConfig(gcms=["ACCESS-CM2"], central_years=[2030, 2050])
    assert run.gcms == ("ACCESS-CM2",)
    assert run.scenario_years() == [
        ("historical", 2005),
        ("ssp126", 2030),
        ("ssp126", 2050),
        ("ssp245", 2030),
        ("ssp245", 2050),
        ("ssp585", 2030),
        ("ssp585", 2050),
    ]
    with pytest.raises(AttributeError):
        run.window_years = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match=MULTI_MODEL_ID):
        RunConfig(gcms=[MULTI_MODEL_ID, "ACCESS-CM2"])
    # resolution defaults to that of the source dataset, unless given
    assert run.resolution == "0.25°"
    assert RunConfig(source_dataset="UKCP18").resolution == "12 km"
    assert (
        RunConfig(source_dataset="UKCP18", resolution="2.2 km").resolution == "2.2 km"
    )
    # the same RunConfig can be shared by indicators with different thresholds
    model = DaysTasAboveIndicator(thresholds=[30.0], run=run)
    assert model.thresholds == [30.0] and model.run is run
    assert DaysTasAboveIndicator(run=run).thresholds == [25, 30, 35, 40, 45, 50, 55]


@pytest.mark.skip(reason="inputs large and downloading slow")
def test_days_tas_above(test_output_dir):
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
    model = DaysTasAboveIndicatorDeprecated(
        threshold_temps_c=[10],
        window_years=1,
        gcms=[gcm],
        scenarios=[scenario],
        central_years=[years[0]],
    )
    target = OscZarr(store=store)
    model.run_all(source, target)


def download_test_datasets(test_output_dir, gcm, scenario, years, indicators=None):
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
