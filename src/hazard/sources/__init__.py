from collections.abc import Callable, Mapping
from typing import Any, Dict, Literal

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


def get_source_dataset_instance(
    source_dataset: SourceDataset, source_dataset_kwargs: dict[str, Any]
) -> OpenDataset:
    if source_dataset not in _SOURCE_DATASETS:
        raise ValueError(f"Invalid source dataset: {source_dataset}")
    return _SOURCE_DATASETS[source_dataset](**source_dataset_kwargs)
