
import os
import fsspec.implementations.local as local # type: ignore
from hazard.docs_store import DocStore # type: ignore
from hazard.onboard.jupiter import Jupiter, JupiterOscFileSource # type: ignore
import pytest
from pytest import approx
import zarr # type: ignore

from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities

@pytest.fixture
def s3_credentials():
    zarr_utilities.set_credential_env_variables()
    yield "s3_credentials"


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir


@pytest.mark.skip(reason="example")
def test_jupiter(test_output_dir, s3_credentials):
    # we need Jupiter OSC_Distribution to be in test_output, e.g.:
    # hazard/src/test/test_output/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv
    local_fs = local.LocalFileSystem()
    source = JupiterOscFileSource(test_output_dir, local_fs)
    #target = OscZarr(prefix='hazard') # hazard_test
    #docs_store = DocStore(prefix="hazard")
    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, 'hazard_test', 'hazard.zarr')))
    docs_store = DocStore(bucket=test_output_dir, fs=local_fs, prefix="hazard_test")

    jupiter = Jupiter()
    docs_store.update_inventory(jupiter.inventory(), remove_existing=True)
    jupiter.run_all(source, target)


@pytest.mark.skip(reason="example")
def test_check_result(test_output_dir):
    """Example for viewing S3 directory structure."""
    zarr_utilities.set_credential_env_variables()
    import s3fs # type: ignore
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY"], secret=os.environ["OSC_S3_SECRET_KEY"])
    path = os.path.join("redhat-osc-physical-landing-647521352890", "hazard_test", "hazard.zarr", "ChronicHeat", "v1")
    check = s3.ls(path)
    assert True
