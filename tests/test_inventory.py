import os

import fsspec.implementations.local as local  # type: ignore
import pytest
import tempfile as temp

from hazard.docs_store import DocStore
from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays, HeatingCoolingDegreeDays
from hazard.models.drought_index import DroughtIndicator
from hazard.models.water_temp import WaterTemperatureAboveIndicator
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.models.work_loss import WorkLossIndicator
from hazard.onboard.flopros_flood import FLOPROSFloodStandardOfProtection
from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.iris_wind import IRISIndicator
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.probabilistic_european_wildfire import FireRiskIndicators
from hazard.onboard.tudelft_flood import TUDelftRiverFlood
from hazard.onboard.wisc_european_winter_storm import WISCEuropeanWinterStorm
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk
from hazard.utilities import zarr_utilities
from hazard.utilities.s3_utilities import get_store  # type: ignore


def test_create_inventory(test_output_dir):
    """Create inventory for all indicators and write into the test output directory."""
    # zarr_utilities.set_credential_env_variables()
    # path = os.path.join(Path(__file__).parents[1], "src", "inventories")

    docs_store = DocStore(local_path=test_output_dir)

    models = [
        WRIAqueductFlood(),
        DegreeDays(),
        Jupiter(test_output_dir),
        WorkLossIndicator(),
        DaysTasAboveIndicator(),
        IRISIndicator(test_output_dir),
        HeatingCoolingDegreeDays(),
        WaterTemperatureAboveIndicator(),
        WetBulbGlobeTemperatureAboveIndicator(),
        WRIAqueductWaterRisk(test_output_dir),
        DroughtIndicator(None, test_output_dir),
        TUDelftRiverFlood(test_output_dir),
        FLOPROSFloodStandardOfProtection(test_output_dir),
        DavydzenkaEtAlLandSubsidence(test_output_dir),
        WISCEuropeanWinterStorm(test_output_dir),
        FireRiskIndicators(test_output_dir),
    ]

    docs_store.write_new_empty_inventory()
    # docs_store.write_inventory_json(json_str)
    for model in models:
        docs_store.update_inventory(model.inventory())


@pytest.mark.skip(reason="just example")
def test_check_inventory(test_output_dir):  # noqa: F811
    zarr_utilities.set_credential_env_variables()
    temp_dir = temp.TemporaryDirectory()
    local_fs = local.LocalFileSystem(root=temp_dir)
    docs_store = DocStore(fs=local_fs)
    json_str = docs_store.read_inventory_json()
    with open(
        os.path.join(test_output_dir, "hazard_test", "inventory_live.json"), "w"
    ) as f:
        f.write(json_str)


@pytest.mark.skip(reason="Requires credentials")
def test_local_inventory(test_dir):
    root = os.getcwd()
    docs_store = DocStore(local_path=root)
    resources = docs_store.read_inventory()
    resources_list = list(resources)
    assert len(resources_list) > 0


@pytest.mark.skip(reason="Requires credentials")
def test_inventory_functionalities():
    zarr_utilities.set_credential_env_variables()
    local_inventory = "C:\\Users\\vcrespo\\dev_violeta\\hazard\\inventory.json"
    extra_s3fs_kwargs = {
        "key": os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        "secret": os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
        "token": os.environ.get("OSC_S3_TOKEN", None),
        "endpoint_url": os.environ.get("OSC_S3_ENDPOINT_DEV", None),
    }
    store = get_store(use_dev=True, extra_s3fs_kwargs=extra_s3fs_kwargs)
    doc_store = DocStore(s3_store=store)
    # find_available_s3_paths()
    paths = doc_store.find_available_s3_paths(store)
    assert len(paths[0]) > 0 and len(paths[1]) > 0
    # check_s3_data()
    check = doc_store.check_s3_data(extra_s3fs_kwargs)
    assert check is True, "Some datasets or maps are empty"
    # check_inventory_paths()
    check, missing_paths = doc_store.check_inventory_paths(
        extra_s3fs_kwargs, local_inventory
    )
    assert check is True, f"This files are missing in the s3 bucket:\n{missing_paths}"
    # check_s3_paths()
    check, missing_paths = doc_store.check_s3_paths(extra_s3fs_kwargs, local_inventory)
    assert check is True, (
        f"This files are orphaned (not referenced by the inventory):\n{missing_paths}"
    )
    print()
