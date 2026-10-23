import logging
import os
import sys
from pathlib import Path, PurePosixPath

import fsspec.implementations.local as local  # type: ignore
import pytest
import s3fs
import tempfile as temp
import zarr
from hazard import get_hazards_onboarding
from hazard.docs_store import DocStore
from hazard.onboard.general_onboarding import initialize_class, onboard_hazards_to
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

    It checks whether you have it already in your temporary files and creates it for you.
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


@pytest.mark.manual
def test_onboard_named_hazard_local(test_output_dir):
    """Used to onboard into a local directory a named hazard."""
    hazard_to_onboard = "DavydzenkaEtAlLandSubsidence" # "JRCRiverFlood" # "WRIAqueductWaterRisk"
    maps_only = False  # True to skip prepare/onboard and just regenerate maps
    # from array data already in test_output_dir (e.g. while iterating on map code).

    store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")
    )
    target = OscZarr(store=store)

    if maps_only:
        hazard_map = get_hazards_onboarding()
        model = initialize_class(
            hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
        )
        model.create_maps(source=target, target=target)
        return

    doc_store = DocStore(local_path=test_output_dir)
    onboard_hazards_to(
        target=target,
        doc_store=doc_store,
        source_dir_base=test_output_dir,
        hazards=[hazard_to_onboard],
        download_dir=str(Path(test_output_dir) / "downloads"),
    )


@pytest.mark.manual
def test_sync_named_hazard(test_output_dir, s3_credentials):
    """Used to sync a named hazard to an S3 bucket.

    Run test_onboard_named_hazard first so that the data this looks for is actually
    present in test_output_dir.
    """
    hazard_to_onboard = "DavydzenkaEtAlLandSubsidence"
    dry_run = False 
    maps_only = True  # True to sync only the map arrays (e.g. after regenerating maps).

    zarr_dir = os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")

    hazard_map = get_hazards_onboarding()
    model = initialize_class(
        hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
    )

    # resource.path/resource.map.path are format strings templated with
    # scenario/year (e.g. "water_risk/wri/v2/{key}_{scenario}_{year}"); the directory
    # holding every expanded variant is the parent of that template.
    array_paths = set()
    for resource in model.inventory():
        if not maps_only:
            array_paths.add(str(PurePosixPath(resource.path).parent))
        if resource.map is not None:
            array_paths.add(str(PurePosixPath(resource.map.path).parent))

    for array_path in sorted(array_paths):
        s3_utilities.sync_local_to_prod(zarr_dir, array_path, dry_run=dry_run)


@pytest.mark.manual
def test_create_maps_from_prod_readonly(test_output_dir):
    """Generate maps locally from the array data already on the production bucket.

    Deliberately uses an anonymous (unsigned) S3FileSystem -- no access key, secret,
    or token -- so the production bucket can only ever be read here, never written
    to: any attempted write would be rejected by S3 for lack of credentials, with no
    reliance on us getting read-only permissions/IAM policy right.
    """
    hazard_to_onboard = "DavydzenkaEtAlLandSubsidence"

    anon_s3 = s3fs.S3FileSystem(anon=True)
    prod_store = s3fs.S3Map(
        root=str(
            PurePosixPath(
                "os-climate-physical-risk", "hazard-indicators", "hazard.zarr"
            )
        ),
        s3=anon_s3,
        check=False,
    )
    source = OscZarr(store=prod_store)

    local_dir = os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")
    target = OscZarr(store=zarr.DirectoryStore(local_dir))

    hazard_map = get_hazards_onboarding()
    model = initialize_class(
        hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
    )

    model.create_maps(source=source, target=target)


def copy_iris_files(s3_credentials):
    bucket = os.environ["OSC_S3_BUCKET_DEV"] 
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