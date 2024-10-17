import logging
import os
import sys
from pathlib import Path, PurePath, PurePosixPath

import fsspec.implementations.local as local  # type: ignore
import pytest
import tempfile as temp
import zarr
import zarr.convenience
from hazard.onboard.flopros_flood import FLOPROSFloodStandardOfProtection
from hazard.onboard.rain_european_winter_storm import RAINEuropeanWinterStorm
from hazard.models.water_temp import WaterTemperatureAboveIndicator
from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays
from hazard.models.drought_index import DroughtIndicator

from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.ipcc_drought import IPCCDrought
from hazard.onboard.iris_wind import IRISIndicator  # type: ignore
from hazard.onboard.jrc_landslides import JRCLandslides
from hazard.onboard.jrc_subsidence import JRCSubsidence
from hazard.onboard.jupiter import Jupiter  # type: ignore
from hazard.onboard.storm_wind import STORMIndicator
from hazard.onboard.tudelft_flood import TUDelftCoastalFlood, TUDelftRiverFlood
from hazard.onboard.tudelft_wildfire import TUDelftFire
from hazard.onboard.tudelft_wind import TUDelftConvectiveWindstorm
from hazard.onboard.wisc_european_winter_storm import WISCEuropeanWinterStorm
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood  # type: ignore
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk
from hazard.onboarder import Onboarder
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import s3_utilities, zarr_utilities


@pytest.fixture
def s3_credentials():
    zarr_utilities.set_credential_env_variables()
    yield "s3_credentials"


@pytest.fixture
def log_to_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    yield "log_to_stdout"


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir


@pytest.fixture
def onboarding_output_dir():
    """Provides directory for onboarding data tests, organized in the same way for all hazard onboards.

    It checks whether you have it already in your teporary files and creates it for you.

    """
    fs = local.LocalFileSystem()
    temp_dir = os.path.join(temp.gettempdir(), "onboarding_raw_data")

    if not fs.exists(temp_dir):
        temp_dir = temp.TemporaryDirectory()
        new_temp_dir_path = os.path.join(
            os.path.dirname(temp_dir.name), "onboarding_raw_data"
        )
        os.rename(temp_dir.name, new_temp_dir_path)
        temp_dir.name = new_temp_dir_path
        os.makedirs(
            os.path.join(new_temp_dir_path, "hazard", "hazard.zarr"), exist_ok=True
        )
        os.makedirs(os.path.join(new_temp_dir_path, "downloads"), exist_ok=True)
        temp_dir._finalizer.detach()
        temp_dir = temp_dir.name

    yield temp_dir


class TestOnboarder(Onboarder):
    __test__ = False

    def __init__(self, source_dir=None):
        if source_dir:
            source_dir = source_dir / "test_onboarder"
        super().__init__(source_dir)
        self.source_dir = source_dir

    def onboard(self, target):
        pass

    def inventory(self):
        return []

    def run_single(self, item, source, target, client):
        pass

    def batch_items(self):
        pass

    def prepare(self, working_dir, force_download=False):
        pass

    def create_maps(self, source, target):
        pass

    def is_prepared(self):
        pass

    def source_dir_from_base(self, source_dir_base: PurePath):
        return source_dir_base / "test_onboarder"


def test_onboarder(test_output_dir):
    onboarder_no_path = TestOnboarder()
    assert onboarder_no_path

    # Verifica que TestOnboarder inicializa correctamente la ruta de origen
    onboarder = TestOnboarder(Path(test_output_dir))
    test_path = Path(test_output_dir) / "test_onboarder"
    assert str(onboarder.source_dir) == str(test_path)

    # Verifica que la ruta se crea correctamente
    assert onboarder.source_dir.exists() or onboarder.source_dir.parent.exists()


@pytest.mark.skip(reason="on-boarding script")
def test_wri_aqueduct(onboarding_output_dir):
    model = WRIAqueductFlood()
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_iris(onboarding_output_dir):
    model = IRISIndicator(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "iris_wind")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


def promote_iris(s3_credentials):
    for name in ["max_speed_ssp585_2050_map"]:
        prefix = "hazard/hazard.zarr/wind/iris/v1/" + name
        s3_utilities.remove_from_prod(prefix, dry_run=False)
        s3_utilities.copy_dev_to_prod(
            "hazard/hazard.zarr/wind/iris/v1/" + name, dry_run=True
        )


def copy_iris_files(s3_credentials):
    bucket = os.environ["OSC_S3_BUCKET_DEV"]  # physrisk-hazard-indicators-dev01
    s3 = s3_utilities.get_s3_fs()
    files = [
        "/wind/IRIS/return_value_maps/IRIS_return_value_map_README.txt",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP1_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP2_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP5_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_PRESENT_tenthdeg.nc",
    ]
    for file in files:
        parts = file.strip("/").split("/")
        filepath = os.path.join(test_output_dir, *parts)
        s3_path = str(PurePosixPath(bucket, "inputs", *parts))
        s3.put(filepath, s3_path, recursive=True)


@pytest.mark.skip(reason="on-boarding script")
def test_jupiter(onboarding_output_dir):
    # we need Jupiter osc-main.zip to be in downloads
    model = Jupiter()
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="example")
def test_check_result(s3_credentials, test_output_dir):
    """Example for viewing S3 directory structure."""

    s3 = s3_utilities.get_s3_fs()
    path = PurePosixPath(
        "redhat-osc-physical-landing-647521352890",
        "hazard_test",
        "hazard.zarr",
        "ChronicHeat",
        "v1",
    )
    check = s3.ls(path)
    assert check is not None


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_river_tudelft(onboarding_output_dir):
    model = TUDelftRiverFlood(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "tudelft")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_coastal_tudelft(onboarding_output_dir):
    model = TUDelftCoastalFlood(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "tudelft")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_wri_aqueduct_water_risk(onboarding_output_dir):
    model = WRIAqueductWaterRisk()
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_water_temp(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = WaterTemperatureAboveIndicator()
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_wind_onboarding(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = STORMIndicator()
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_wet_bulb_globe_temp(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = WetBulbGlobeTemperatureAboveIndicator()
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_days_tas_above_temp(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = DaysTasAboveIndicator()
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_days_tas_above_temp_single_onboard(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = DegreeDays()
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_days_tas_above_temp_temp_run_all(onboarding_output_dir):
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model = DroughtIndicator(store)
    model.run_all(None, target)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_landslides_jrc(onboarding_output_dir):
    model = JRCLandslides(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "jrc_landslides")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_subsidence_jrc(onboarding_output_dir):
    model = JRCSubsidence(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "jrc_subsidence")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_fire_tudelft(onboarding_output_dir):
    model = TUDelftFire(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "tudelft")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_conv_wind_tudelft(onboarding_output_dir):
    model = TUDelftConvectiveWindstorm(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "tudelft")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_litpop(onboarding_output_dir):
    model = ETHZurichLitPop(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "ethz_litpop")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)

    # model.create_maps(target, target)


# def test_promote(s3_credentials):
#    s3_utilities.copy_dev_to_prod("hazard/hazard.zarr/" + "maps/inundation/flopros_coastal/v1/flood_sop", dry_run=False, sync=False)
#    s3_utilities.copy_dev_to_prod("hazard/hazard.zarr/" + "inundation/flopros_coastal/v1/flood_sop", dry_run=False, sync=False)


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_flopros(test_output_dir):
    model = FLOPROSFloodStandardOfProtection()
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store, write_xarray_compatible_zarr=True)
    model.onboard_single(target, test_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_csm_subsidence(onboarding_output_dir):
    model = DavydzenkaEtAlLandSubsidence(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "csm_subsidence")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_ipcc_drought(onboarding_output_dir):
    model = IPCCDrought(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "ipcc_drought")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)
    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_rain_european_storm(onboarding_output_dir):
    model = RAINEuropeanWinterStorm(onboarding_output_dir)
    download_dir = os.path.join(onboarding_output_dir, "downloads", "rain_proj")
    if not os.path.exists(download_dir):
        raise AssertionError(f"There is no source data in {download_dir}")
    store = zarr.DirectoryStore(
        os.path.join(onboarding_output_dir, "hazard", "hazard.zarr")
    )
    target = OscZarr(store=store)

    model.onboard_single(target, onboarding_output_dir)


@pytest.mark.skip(reason="on-boarding script")
def test_wisc_european_storm(test_output_dir):
    # source_dir = os.path.join(test_output_dir, "wisc")
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    # source = WISCWinterStormEventSource(source_dir)
    model = WISCEuropeanWinterStorm()
    # model.run_all(source, target)
    model.create_maps(target, target)
