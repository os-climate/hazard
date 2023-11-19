
import logging
import os
from pathlib import PurePosixPath
import sys
import fsspec.implementations.local as local # type: ignore
from hazard.docs_store import DocStore
from hazard.onboard.iris_wind import IRISIndicator # type: ignore
from hazard.onboard.jupiter import Jupiter, JupiterOscFileSource # type: ignore
import pytest
import s3fs
import zarr
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood # type: ignore
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.wri_aqueduct import WRIAqueductSource
from hazard.utilities import s3_utilities, zarr_utilities

@pytest.fixture
def s3_credentials():
    zarr_utilities.set_credential_env_variables()
    yield "s3_credentials"


@pytest.fixture
def log_to_stdout():
    logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])
    yield "log_to_stdout"


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir



@pytest.mark.skip(reason="example")
def test_wri_aqueduct(test_output_dir, s3_credentials, log_to_stdout):
    model = WRIAqueductFlood()
    items = model.batch_items()
    print(items)
    source = WRIAqueductSource()
    target = OscZarr()
    #target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr')))
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    target = OscZarr(bucket=os.environ["OSC_S3_BUCKET"], s3=s3)
    for item in items:
        map_path = item.resource.map.path.format(scenario=item.scenario, year=item.year)
        if map_path != (item.path + "_map"):
            raise ValueError(f"unexpected map path {map_path}") 
        #model.run_single(item, source, target, None)
        model.generate_tiles_single(item, target, target)


@pytest.mark.skip(reason="requires input data")
def test_iris(test_output_dir, s3_credentials):
    # upload IRIS
    #copy_iris_files(s3_credentials)
    #promote_iris(s3_credentials)
    model = IRISIndicator(test_output_dir)
    #s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY_DEV"], secret=os.environ["OSC_S3_SECRET_KEY_DEV"])
    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr'))) # save locally
    #target = OscZarr() # default dev bucket
    for item in model.batch_items():
        model.generate_single_map(item, target, target)
    #model.run_all(None, target, debug_mode=True)
    #create_tile_set(source, source_path, target, target_path, nodata=-9999.0, nodata_as_zero=True)


def promote_iris(s3_credentials):
    for name in ['max_speed_ssp585_2050_map']:
        prefix = "hazard/hazard.zarr/wind/iris/v1/" + name
        s3_utilities.remove_from_prod(prefix, dry_run=False)
        s3_utilities.copy_dev_to_prod("hazard/hazard.zarr/wind/iris/v1/" + name, dry_run=True)


def copy_iris_files(s3_credentials):
    bucket = os.environ["OSC_S3_BUCKET_DEV"] # physrisk-hazard-indicators-dev01
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY_DEV"], secret=os.environ["OSC_S3_SECRET_KEY_DEV"])
    files = [
        "/wind/IRIS/return_value_maps/IRIS_return_value_map_README.txt",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP1_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP2_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP5_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_PRESENT_tenthdeg.nc"]
    for file in files:
        parts = file.strip("/").split("/")
        filepath = os.path.join(test_output_dir, *parts)
        s3_path = str(PurePosixPath(bucket, "inputs", *parts))
        s3.put(filepath, s3_path, recursive=True)   


@pytest.mark.skip(reason="requires input data")
def test_jupiter(test_output_dir, s3_credentials):
    # we need Jupiter OSC_Distribution to be in test_output, e.g.:
    # hazard/src/test/test_output/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv
    local_fs = local.LocalFileSystem()
    source = JupiterOscFileSource(test_output_dir, local_fs)
    #target = OscZarr(prefix='hazard') # hazard_test
    #docs_store = DocStore(prefix="hazard")
    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr')))
    docs_store = DocStore(bucket=test_output_dir, fs=local_fs, prefix="hazard")

    jupiter = Jupiter()
    docs_store.update_inventory(jupiter.inventory(), remove_existing=True)
    jupiter.run_all(source, target, debug_mode=True)


@pytest.mark.skip(reason="example")
def test_check_result(test_output_dir):
    """Example for viewing S3 directory structure."""
    zarr_utilities.set_credential_env_variables()
    import s3fs # type: ignore
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    path = os.path.join("redhat-osc-physical-landing-647521352890", "hazard_test", "hazard.zarr", "ChronicHeat", "v1")
    check = s3.ls(path)
    assert True
