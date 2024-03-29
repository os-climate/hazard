import os

import pytest  # type: ignore
import zarr  # type: ignore
from affine import Affine

from hazard.onboard.storm_wind import BatchItem, STORMIndicator  # type: ignore
from hazard.sources.osc_zarr import OscZarr

from .utilities import test_output_dir


@pytest.mark.skip(reason="on-boarding script")
def test_wind_onboarding(test_output_dir):
    target = OscZarr(store=zarr.DirectoryStore(os.path.join(test_output_dir, "hazard", "hazard.zarr")))
    model = STORMIndicator(os.path.join(test_output_dir, "wind"))
    model.run_single(BatchItem(gcm="HADGEM3-GC31-HM", model=""), None, target, None)
