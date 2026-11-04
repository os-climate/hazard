from typing import Any, Callable, Dict, Literal, Mapping

from hazard.protocols import OpenDataset
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.ukcp18 import Ukcp18

SourceDataset = Literal[
    "NEX-GDDP-CMIP6",
    "UKCP18",
]

_SOURCE_DATASETS: Mapping[str, Callable[..., OpenDataset]] = {
    "NEX-GDDP-CMIP6": NexGddpCmip6,
    "UKCP18": Ukcp18,
}


# Resolution of each source dataset with its default arguments; this allows resources to be
# created without an instance (which may need credentials). See the 'resolution' of the sources.
_SOURCE_DATASET_DEFAULT_RESOLUTIONS: Mapping[str, str] = {
    "NEX-GDDP-CMIP6": NexGddpCmip6.resolution,
    "UKCP18": "12 km",  # Ukcp18(resolution="12km")
}


def get_source_dataset_default_resolution(source_dataset: str) -> str:
    """Resolution of the source dataset with its default arguments, e.g. "0.25°"."""
    if source_dataset not in _SOURCE_DATASET_DEFAULT_RESOLUTIONS:
        raise ValueError(f"Invalid source dataset: {source_dataset}")
    return _SOURCE_DATASET_DEFAULT_RESOLUTIONS[source_dataset]


def get_source_dataset_instance(
    source_dataset: SourceDataset, source_dataset_kwargs: Dict[str, Any]
) -> OpenDataset:
    if source_dataset not in _SOURCE_DATASETS:
        raise ValueError(f"Invalid source dataset: {source_dataset}")
    return _SOURCE_DATASETS[source_dataset](**source_dataset_kwargs)
