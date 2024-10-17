"""Data onboarding."""

import os
import tempfile as temp
from pathlib import Path
from typing_extensions import Optional, Sequence

import fsspec.implementations.local as local
import zarr

from hazard import get_hazards_onboarding
from hazard.docs_store import DocStore
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities
from hazard.utilities.s3_utilities import get_store, load_s3_parameters


def onboard_hazards(
    local: bool = False,
    credentials_path: Optional[str] = None,
    source_dir_base: Optional[str] = None,
    hazards: Sequence[str] = "",
    download_dir: Optional[str] = None,
    force_download=False,
    force_prepare=False,
):
    """Onboards data to a local or S3 target based on the provided paths and for a series of given hazards.

    It also updates the inventory each time you onboard some hazard.

    Args:
        local: Path to local directory where data should be saved.
        credentials_path: Path to TOML file with S3 parameters and credentials (optional).
        source_dir_base (Optional[str], optional): Base directory for source data. If None, a temporary directory
            will be created. Defaults to None.
        hazards: Path to the file containing the list of hazards to onboard.
        download_dir: Path to the directory where data will be downloaded.
            Required for local storage. Defaults to None.
        force_download: If True, forces the download of hazard data even if it already exists.
            Defaults to False.
        force_prepare: If True, forces the preparation of hazard data even if it is ready.
            Defaults to False.

    """
    if not source_dir_base:
        source_dir = create_temporary_directory()
    else:
        source_dir = source_dir_base

    if credentials_path:
        print("Using S3 as target")
        zarr_utilities.set_credential_env_variables()
        s3_parameters = load_s3_parameters(credentials_path)
        s3_store = get_store(**s3_parameters)
        target = OscZarr(store=s3_store)
        doc_store = DocStore(s3_store=s3_store)
    elif local:
        print("Using local storage as target")
        store = zarr.DirectoryStore(source_dir)
        target = OscZarr(store=store)
        download_dir = os.path.join(
            Path.home(), "Downloads"
        )  # assume local inventory is here
        doc_store = DocStore(local_path=download_dir)
    else:
        raise ValueError("No local_path or credentials specification")

    updt_inv_hazards = []
    hazard_map = get_hazards_onboarding()

    for hazard in hazards:
        if hazard in hazard_map:
            try:
                model = initialize_class(hazard, hazard_map, source_dir_base=source_dir)

                for resource in model.inventory():
                    updt_inv_hazards.append(resource)

                if (
                    not model.is_prepared(force_download=force_download)
                    and not force_prepare
                ):
                    model.prepare(
                        download_dir=download_dir,
                    )

                model.onboard(
                    target,
                )

                model.create_maps(source=target, target=target)

                print(f"Successfully onboarded {hazard}")

            except Exception as e:
                # Catch the exception and log the error, but continue processing other hazards
                print(f"Failed to onboard {hazard}: {e}")
        else:
            print(f"Unknown hazard: {hazard}")
    doc_store.update_inventory(updt_inv_hazards)
    print("Inventory updated succesfully")


def initialize_class(
    hazard: str, hazard_map: dict, source_dir_base: Optional[str] = None
):
    """Initialize the hazard model class based on the hazard name, passing directory if needed."""
    special_hazards = [
        "WRIAqueductFlood",
        "WaterTemperatureAboveIndicator",
        "WetBulbGlobeTemperatureAboveIndicator",
        "DaysTasAboveIndicator",
        "Degreedays",
        "HeatingCoolingDegreeDays",
    ]

    hazard_class = hazard_map.get(hazard)
    if not hazard_class:
        raise ValueError(f"Hazard '{hazard}' not recognized.")

    if hazard in special_hazards:
        return hazard_class()
    else:
        return hazard_class(source_dir_base)


def load_hazards(hazard_list_path: str):
    """Load the list of hazards from a file."""
    with open(hazard_list_path, "r") as f:
        return [eval(hazard) for hazard in f.read().splitlines()]


def create_temporary_directory():
    """Provide directory for onboarding data tests, organized in the same way for all hazard onboards.

    It checks whether you have it already in your teporary files and creates it for you.

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

    return temp_dir
