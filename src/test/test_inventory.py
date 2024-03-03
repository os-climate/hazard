import os

import fsspec.implementations.local as local  # type: ignore
import pytest

from hazard.docs_store import DocStore, HazardResources
from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays, HeatingCoolingDegreeDays
from hazard.models.drought_index import DroughtIndicator
from hazard.models.water_temp import WaterTemperatureAboveIndicator
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.models.work_loss import WorkLossIndicator
from hazard.onboard.iris_wind import IRISIndicator
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk
from hazard.utilities import zarr_utilities  # type: ignore

from .utilities import test_output_dir


def test_create_inventory(test_output_dir):
    """Create inventory for all indicators and write into this repo."""
    zarr_utilities.set_credential_env_variables()
    local_fs = local.LocalFileSystem()

    from pathlib import Path

    path = os.path.join(Path(__file__).parents[1], "inventories")

    # path = os.path.join(test_output_dir)

    docs_store = DocStore(bucket=path, fs=local_fs, prefix="hazard")
    # docs_store = DocStore(prefix="hazard") # for writing direct to S3

    models = [
        WRIAqueductFlood(),
        DegreeDays(),
        Jupiter(),
        WorkLossIndicator(),
        DaysTasAboveIndicator(),
        IRISIndicator(None),
        HeatingCoolingDegreeDays(),
        WaterTemperatureAboveIndicator(),
        WetBulbGlobeTemperatureAboveIndicator(),
        WRIAqueductWaterRisk(),
        DroughtIndicator(None)
    ]

    docs_store.write_new_empty_inventory()
    # docs_store.write_inventory_json(json_str)
    for model in models:
        docs_store.update_inventory(model.inventory())


@pytest.mark.skip(reason="just example")
def test_check_inventory(test_output_dir):
    zarr_utilities.set_credential_env_variables()
    prefix = "hazard"
    docs_store = DocStore(prefix=prefix)
    json_str = docs_store.read_inventory_json()
    with open(os.path.join(test_output_dir, prefix, "inventory_live.json"), "w") as f:
        f.write(json_str)
