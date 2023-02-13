
import os
import fsspec.implementations.local as local # type: ignore
from hazard.onboard.jupiter import Jupiter, JupiterOscFileSource # type: ignore
from hazard.onboard.osc_chronic_heat import OscChronicHeat
import pytest
from pytest import approx
import zarr # type: ignore

from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities

@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir


def test_jupiter(test_output_dir):
    # we need Jupiter OSC_Distribution to be in test_output, e.g.:
    # hazard/src/test/test_output/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv
    source = JupiterOscFileSource(test_output_dir, local.LocalFileSystem())
    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard_test', 'hazard.zarr')))
    Jupiter().run_all(source, target)


@pytest.mark.skip(reason="example")
def test_onboard_chronic_heat_work_loss(test_output_dir):
    """Example of running the work loss onboarding script."""
    fs = local.LocalFileSystem()

    # this on-boarding script works with local files:
    onboarder = OscChronicHeat(root=os.path.join(test_output_dir, "work_loss_inputs"))

    # we can specify as local file system zarr store for testing:
    store = zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard_test', 'hazard.zarr'))
    target = OscZarr(store=store)

    # or an S3 target
    # zarr_utilities.set_credential_env_variables()
    # target = OscZarr(prefix="hazard") # hazard_test

    onboarder.onboard(target)
    onboarder.onboard_maps(target, working_dir=test_output_dir)
    
    #maps_dir = os.path.join(test_output_dir, "work_loss_maps")
    #os.makedirs(maps_dir, exist_ok=True)
    #onboarder.onboard_maps(target, maps_dir)


@pytest.mark.skip(reason="example")
def test_check_result(test_output_dir):
    """Example for viewing S3 directory structure."""
    zarr_utilities.set_credential_env_variables()
    import s3fs # type: ignore
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    path = os.path.join("redhat-osc-physical-landing-647521352890", "hazard_test", "hazard.zarr", "ChronicHeat", "v1")
    check = s3.ls(path)
    assert True
