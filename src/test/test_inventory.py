import os
from hazard.docs_store import DocStore
import fsspec.implementations.local as local # type: ignore
from hazard.docs_store import DocStore, HazardModels
from hazard.models.degree_days import DegreeDays
from hazard.models.work_loss import WorkLossIndicator
from hazard.onboard.jupiter import Jupiter
from hazard.utilities import zarr_utilities # type: ignore
from .utilities import test_output_dir

def test_create_inventory(test_output_dir):
    """Create inventory for all indicators."""
    zarr_utilities.set_credential_env_variables() 
    local_fs = local.LocalFileSystem()
    
    docs_store = DocStore(bucket=test_output_dir, fs=local_fs, prefix="hazard_test2")
    #docs_store = DocStore(prefix="hazard")

    models = [DegreeDays(), Jupiter(), WorkLossIndicator()]

    docs_store.write_new_empty_inventory()
    #docs_store.write_inventory_json(json_str)
    for model in models:
        docs_store.update_inventory(model.inventory())

def test_check_inventory(test_output_dir):
    zarr_utilities.set_credential_env_variables() 
    prefix = "hazard"
    docs_store = DocStore(prefix=prefix)
    json_str = docs_store.read_inventory_json()
    with open(os.path.join(test_output_dir, prefix, "inventory_live.json"), "w") as f:
        f.write(json_str)
