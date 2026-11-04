"""Data onboarding."""

import logging
import os
import tempfile as temp
from pathlib import Path
from typing_extensions import Any, Dict, Optional, Sequence, Type, Union

import fsspec.implementations.local as local
import zarr

from hazard import get_hazards_onboarding
from hazard.docs_store import DocStore
from hazard.onboarder import Onboarder
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import zarr_utilities
from hazard.utilities.s3_utilities import get_store, load_s3_parameters

logger = logging.getLogger(__name__)


def onboard_hazards(
    local: bool = False,
    credentials_path: Optional[str] = None,
    source_dir_base: Optional[str] = None,
    hazards: Sequence[str] = [],
    download_dir: Optional[str] = None,
    force_download=False,
    force_prepare=False,
    store: Optional[str] = None,
    hazard_kwargs: Optional[Dict[str, Any]] = None,
):
    """Onboards data to a local or S3 target based on the provided paths and for a series of given hazards.

    Each hazard is prepared and onboarded independently; a failure for one hazard is logged
    and does not stop the rest from being processed. Once all hazards have been attempted,
    the inventory is updated once with the resources for every hazard that was successfully
    onboarded -- a hazard whose prepare/onboard/create_maps step failed is excluded.

    Args:
        local: If True, onboard to a local zarr store. If False, onboard to S3 instead,
            using credentials_path (if given) to source S3 parameters and credentials;
            if not given, S3 access falls back to whatever credentials are already
            available in the environment.
        credentials_path: Path to TOML file with S3 parameters and credentials (optional).
            Only consulted when local=False; has no effect when local=True.
        source_dir_base (Optional[str], optional): Base directory for source data. If None, a temporary directory
            will be created. Defaults to None.
        hazards: Names of the hazards to onboard, e.g. as returned by
            get_hazards_onboarding().keys().
        download_dir: Path to the directory used by each hazard's prepare() step. Most
            hazards download their source data into this directory; a few instead
            require the source file(s) to already be present there and raise if they
            are not. Defaults to None.
        force_download: If True, forces the download of hazard data even if it already exists.
            Defaults to False.
        force_prepare: If True, forces the preparation of hazard data even if it is ready.
            Defaults to False.
        store: Path of a local directory to onboard to: the zarr store and inventory are written
            there. If given, this is the target regardless of local and credentials_path.
            Defaults to None.
        hazard_kwargs: Keyword arguments passed to the constructor of each hazard's onboarder
            (in addition to source_dir_base), e.g. {"run": {"gcms": ["ACCESS-CM2"]}} for the
            threshold-based indicators. Defaults to None.

    """
    if not source_dir_base:
        source_dir = create_temporary_directory()
    else:
        source_dir = source_dir_base

    if store is not None:
        logger.info(f"Using local store {store} as target")
        target = OscZarr(store=zarr.DirectoryStore(store))
        doc_store = DocStore(local_path=store)
    elif local:
        logger.info("Using local storage as target")
        store = zarr.DirectoryStore(source_dir)
        target = OscZarr(store=store)
        inventory_dir = source_dir
        doc_store = DocStore(local_path=inventory_dir)
    else:
        logger.info("Using S3 as target")
        zarr_utilities.set_credential_env_variables()
        s3_parameters = load_s3_parameters(credentials_path)
        s3_store = get_store(**s3_parameters)
        target = OscZarr(store=s3_store)
        doc_store = DocStore(s3_store=s3_store)

    onboard_hazards_to(
        target,
        doc_store,
        source_dir,
        hazards,
        download_dir,
        force_download,
        force_prepare,
        hazard_kwargs,
    )


def onboard_hazards_to(
    target: OscZarr,
    doc_store: DocStore,
    source_dir_base: str,
    hazards: Sequence[Union[str, Onboarder]] = [],
    download_dir: Optional[str] = None,
    force_download=False,
    force_prepare=False,
    hazard_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Onboards a series of given hazards to an explicitly provided target and doc_store.

    This is the storage-agnostic core of onboard_hazards: it has no opinion on whether
    target/doc_store are backed by local storage or S3, and does no path resolution --
    the caller is expected to have already built these building blocks. Use this directly
    when you already have a target/doc_store (e.g. in tests or notebooks); use
    onboard_hazards if you just want local-vs-S3 selection handled for you.

    Each hazard is prepared and onboarded independently; a failure for one hazard is logged
    and does not stop the rest from being processed. Once all hazards have been attempted,
    the inventory is updated once with the resources for every hazard that was successfully
    onboarded -- a hazard whose prepare/onboard/create_maps step failed is excluded.

    Args:
        target: Target to onboard hazard indicators to.
        doc_store: DocStore to update with the resulting inventory.
        source_dir_base: Base directory for source data.
        hazards: Hazards to onboard: names, e.g. as returned by get_hazards_onboarding().keys(),
            or onboarders already constructed (e.g. with settings other than the defaults, such
            as the GCMs and years of an indicator model).
        download_dir: Path to the directory used by each hazard's prepare() step. Most
            hazards download their source data into this directory; a few instead
            require the source file(s) to already be present there and raise if they
            are not. Defaults to None.
        force_download: If True, forces the download of hazard data even if it already exists.
            Defaults to False.
        force_prepare: If True, forces the preparation of hazard data even if it is ready.
            Defaults to False.
        hazard_kwargs: Keyword arguments passed to the constructor of each hazard given by name
            (in addition to source_dir_base). Not used for onboarders already constructed.
            Defaults to None.

    """
    updt_inv_hazards = []
    hazard_map: Dict[str, Type[Onboarder]] = get_hazards_onboarding()
    failed_hazards = []

    for hazard in hazards:
        name = hazard if isinstance(hazard, str) else type(hazard).__name__
        if isinstance(hazard, str) and hazard not in hazard_map:
            logger.warning(f"Unknown hazard: {hazard}")
            failed_hazards.append(name)
            continue

        try:
            model = (
                initialize_class(
                    hazard,
                    hazard_map,
                    source_dir_base=source_dir_base,
                    **(hazard_kwargs or {}),
                )
                if isinstance(hazard, str)
                else hazard
            )

            if not model.is_prepared() or force_prepare or force_download:
                model.prepare(
                    download_dir=download_dir,
                    force_download=force_download,
                )

            model.onboard(
                target,
            )

            model.create_maps(source=target, target=target)

            for resource in model.inventory():
                updt_inv_hazards.append(resource)

            logger.info(f"Successfully onboarded {name}")

        except Exception:
            logger.error(f"Failed to onboard {name}", exc_info=True)
            failed_hazards.append(name)

    doc_store.update_inventory(updt_inv_hazards)
    logger.info("Inventory updated successfully")

    if failed_hazards:
        logger.error(
            f"Failed to onboard {len(failed_hazards)} of {len(hazards)} "
            f"hazard(s): {failed_hazards}"
        )


def initialize_class(
    hazard: str,
    hazard_map: Dict[str, Type[Onboarder]],
    source_dir_base: str,
    **kwargs: Any,
) -> Onboarder:
    """Initialize the hazard model class based on the hazard name, passing directory if needed, and any other keyword arguments."""
    hazard_class = hazard_map.get(hazard)
    if not hazard_class:
        raise ValueError(f"Hazard '{hazard}' not recognized.")

    return hazard_class(source_dir_base, **kwargs)


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
