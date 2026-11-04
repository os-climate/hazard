import os
import shutil
import tempfile

import pytest

from hazard.utilities import zarr_utilities


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `manual` unless exactly one test was selected.

    Selecting a single test on its own -- by clicking it individually in an IDE's
    test explorer, or naming its exact node id on the command line -- runs it for
    real. Any broader selection (a whole file, a whole directory, the full suite)
    skips `manual` tests instead, since those are typically real on-boarding
    scripts: slow, and dependent on network access/credentials/prior local state.
    """
    if len(items) == 1:
        return
    skip_manual = pytest.mark.skip(reason="marked manual; select it by itself to run")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)


@pytest.fixture
def s3_credentials():
    zarr_utilities.set_credential_env_variables()
    yield "s3_credentials"


@pytest.fixture
def test_input_dir():
    """Provides directory for test inputs."""
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_input")
    yield input_dir


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir
    # if we want to clean-up (most likely not)
    # shutil.rmtree(output_dir)


@pytest.fixture
def test_dir():
    """Like test_output_dir but uses a temp directory which is cleaned-up."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)
