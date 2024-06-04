import logging
import os
import sys
from pathlib import PurePosixPath

import fsspec.implementations.local as local  # type: ignore
import pytest
import s3fs
import zarr
import zarr.convenience

from hazard.docs_store import DocStore
from hazard.models.water_temp import FutureStreamsSource, WaterTemperatureAboveIndicator
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.iris_wind import IRISIndicator  # type: ignore
from hazard.onboard.jupiter import Jupiter  # type: ignore
from hazard.onboard.jupiter import JupiterOscFileSource
from hazard.onboard.tudelft_flood import TUDelftRiverFlood
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood  # type: ignore
from hazard.onboard.wri_aqueduct_water_risk import (
    WRIAqueductWaterRisk,
    WRIAqueductWaterRiskSource,
    WRIAqueductWaterSupplyDemandBaselineSource,
)
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
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


@pytest.mark.skip(reason="on-boarding script")
def test_wri_aqueduct(test_output_dir, s3_credentials, log_to_stdout):
    model = WRIAqueductFlood()
    items = model.batch_items()
    print(items)
    # source = WRIAqueductSource()
    target = OscZarr()
    # target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr')))
    s3 = s3fs.S3FileSystem(
        key=os.environ.get("OSC_S3_ACCESS_KEY", None),
        secret=os.environ.get("OSC_S3_SECRET_KEY", None),
    )
    target = OscZarr(bucket=os.environ["OSC_S3_BUCKET"], s3=s3)
    for item in items:
        map_path = item.resource.map.path.format(scenario=item.scenario, year=item.year)
        if map_path != (item.path + "_map"):
            raise ValueError(f"unexpected map path {map_path}")
        # model.run_single(item, source, target, None)
        model.generate_tiles_single(item, target, target)


@pytest.mark.skip(reason="on-boarding script")
def test_iris(test_output_dir, s3_credentials):
    # upload IRIS
    # copy_iris_files(s3_credentials)
    # promote_iris(s3_credentials)
    model = IRISIndicator(test_output_dir)
    # s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY_DEV"],
    # secret=os.environ["OSC_S3_SECRET_KEY_DEV"])
    target = OscZarr(
        store=zarr.DirectoryStore(
            os.path.join(test_output_dir, "hazard", "hazard.zarr")
        )
    )  # save locally
    # target = OscZarr() # default dev bucket
    for item in model.batch_items():
        model.generate_single_map(item, target, target)
    # model.run_all(None, target, debug_mode=True)
    # create_tile_set(source, source_path, target, target_path, nodata=-9999.0, nodata_as_zero=True)


def promote_iris(s3_credentials):
    for name in ["max_speed_ssp585_2050_map"]:
        prefix = "hazard/hazard.zarr/wind/iris/v1/" + name
        s3_utilities.remove_from_prod(prefix, dry_run=False)
        s3_utilities.copy_dev_to_prod(
            "hazard/hazard.zarr/wind/iris/v1/" + name, dry_run=True
        )


def copy_iris_files(s3_credentials):
    bucket = os.environ["OSC_S3_BUCKET_DEV"]  # physrisk-hazard-indicators-dev01
    s3 = s3fs.S3FileSystem(
        key=os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        secret=os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
    )
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
def test_jupiter(test_output_dir, s3_credentials):
    # we need Jupiter OSC_Distribution to be in test_output, e.g.:
    # hazard/src/test/test_output/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv
    local_fs = local.LocalFileSystem()
    source = JupiterOscFileSource(test_output_dir, local_fs)
    # target = OscZarr(prefix='hazard') # hazard_test
    # docs_store = DocStore(prefix="hazard")
    target = OscZarr(
        store=zarr.DirectoryStore(
            os.path.join(test_output_dir, "hazard", "hazard.zarr")
        )
    )
    docs_store = DocStore(bucket=test_output_dir, fs=local_fs, prefix="hazard")

    jupiter = Jupiter()
    docs_store.update_inventory(jupiter.inventory(), remove_existing=True)
    jupiter.run_all(source, target, debug_mode=True)


@pytest.mark.skip(reason="example")
def test_check_result(test_output_dir):
    """Example for viewing S3 directory structure."""
    zarr_utilities.set_credential_env_variables()
    import s3fs  # type: ignore

    s3 = s3fs.S3FileSystem(
        key=os.environ.get("OSC_S3_ACCESS_KEY", None),
        secret=os.environ.get("OSC_S3_SECRET_KEY", None),
    )
    path = os.path.join(
        "redhat-osc-physical-landing-647521352890",
        "hazard_test",
        "hazard.zarr",
        "ChronicHeat",
        "v1",
    )
    check = s3.ls(path)
    assert check is not None


@pytest.mark.skip(reason="on-boarding script")
def test_onboard_tudelft(s3_credentials, test_output_dir):
    source_path = os.path.join(test_output_dir, "tudelft", "tudelft_river")
    model = TUDelftRiverFlood(source_path)
    model.prepare()
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    model.run_all(None, target)
    # batch_items = model.batch_items()
    # model.run_single(batch_items[4], None, target, None)
    model.create_maps(target, target)
    # path = "inundation/river_tudelft/v2/flood_depth_historical_1971"
    # map_path = "maps/inundation/river_tudelft/v2/flood_depth_historical_1971_map"
    # create_tile_set(target, path, target, map_path, max_zoom=10)

    # "flood_depth_historical_1971",
    # files = ["flood_depth_rcp8p5_2035", "flood_depth_rcp8p5_2085",
    # "flood_depth_rcp4p5_2035", "flood_depth_rcp4p5_2085"]

    # s3_utilities.copy_dev_to_prod("hazard/hazard.zarr/" + "inundation/river_tudelft/v2", False)
    # s3_utilities.copy_dev_to_prod("hazard/hazard.zarr/" + "maps/inundation/river_tudelft/v2", False)

    # for file in files:
    #     s3_utilities.copy_local_to_dev(
    #         str(pathlib.Path(test_output_dir, "hazard/hazard.zarr")),
    #         f"inundation/river_tudelft/v2/{file}",
    #     )
    #     for i in range(10, 0, -1):
    #         s3_utilities.copy_local_to_dev(
    #             str(pathlib.Path(test_output_dir, "hazard/hazard.zarr")),
    #             f"maps/inundation/river_tudelft/v2/{file}_map/{i}",
    #         )


@pytest.mark.skip(reason="on-boarding script")
def test_wri_aqueduct_water_risk(test_output_dir):
    source_dir = os.path.join(test_output_dir, "wri_aqueduct_water_risk")
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    model = WRIAqueductWaterRisk()
    source = WRIAqueductWaterRiskSource(
        source_dir=source_dir, fs=local.LocalFileSystem()
    )
    model.run_all(source, target)
    source = WRIAqueductWaterSupplyDemandBaselineSource(
        source_dir=source_dir, fs=local.LocalFileSystem()
    )
    model.run_all(source, target)
    model.create_maps(target, target)


@pytest.mark.skip(reason="on-boarding script")
def test_water_temp(test_output_dir):
    working_dir = os.path.join(test_output_dir, "water_temp")
    source = FutureStreamsSource(working_dir=working_dir)
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    model = WaterTemperatureAboveIndicator()
    model.run_all(source, target)
    model.create_maps(target, target)


@pytest.mark.skip(reason="on-boarding script")
def test_wet_bulb_globe_temp(test_output_dir):
    source = NexGddpCmip6(
        root=os.path.join(test_output_dir, NexGddpCmip6.bucket),
        fs=local.LocalFileSystem(),
    )
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    model = WetBulbGlobeTemperatureAboveIndicator()
    model.run_all(source, target)
    model.create_maps(target, target)


@pytest.mark.skip(reason="on-boarding script")
def test_litpop(test_output_dir):
    store = zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr"))
    target = OscZarr(store=store)
    model = ETHZurichLitPop(source_dir=test_output_dir)
    model.run_all(None, target)
    model.create_maps(target, target)
