
import os
import fsspec.implementations.local as local
from hazard.onboard.osc_chronic_heat import OscChronicHeat
import pytest
from pytest import approx
import rasterio
import zarr

from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities

@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir


def test_onboard_chronic_heat_work_loss(test_output_dir):
    #fs = local.LocalFileSystem()
      
    onboarder = OscChronicHeat(root=os.path.join(test_output_dir, "work_loss_inputs"))
    store = zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard', 'hazard.zarr'))
    #target = OscZarr(store=store)
    zarr_utilities.set_credential_env_variables()
    target = OscZarr(prefix="hazard") # hazard_test
    onboarder.onboard(target)
    maps_dir = os.path.join(test_output_dir, "work_loss_maps")
    os.makedirs(maps_dir, exist_ok=True)
    onboarder.onboard_maps(target, maps_dir)

def test_check_result(test_output_dir):
    zarr_utilities.set_credential_env_variables()
    import s3fs
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    path = os.path.join("redhat-osc-physical-landing-647521352890", "hazard_test", "hazard.zarr", "ChronicHeat", "v1")
    check = s3.ls(path)
    assert True
