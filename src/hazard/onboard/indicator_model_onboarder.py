"""Adapters that let IndicatorModels be driven through the Onboarder pipeline.

An IndicatorModel is designed to work with a choice of data source (passed into
run_all), rather than owning a single source_dir_base the way an Onboarder does.
These adapters compose a concrete IndicatorModel with the source it should read
from, and expose that composition as an Onboarder so it can be onboarded through
the same onboard_hazards pipeline as any other hazard.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dask.distributed import Client, LocalCluster
from typing_extensions import Iterable

from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource
from hazard.models.days_tas_above import DaysTasAboveIndicator as _DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays as _DegreeDays
from hazard.models.degree_days import (
    HeatingCoolingDegreeDays as _HeatingCoolingDegreeDays,
)
from hazard.models.deprecated_v2 import (
    DaysTasAboveIndicatorDeprecated as _DaysTasAboveIndicatorDeprecated,
)
from hazard.models.deprecated_v2 import (
    HeatingCoolingDegreeDaysDeprecated as _HeatingCoolingDegreeDaysDeprecated,
)
from hazard.models.deprecated_v2 import (
    WetBulbGlobeTemperatureAboveIndicator as _WetBulbGlobeTemperatureAboveIndicator,
)
from hazard.models.drought_index import DroughtIndicator as _DroughtIndicator
from hazard.models.drought_index import local_zarr_working_store
from hazard.models.multi_year_average import RunConfig, ThresholdBasedAverageIndicator
from hazard.models.prechunker import NexGddpCmip6CachedSource
from hazard.models.water_temp import FutureStreamsSource
from hazard.models.water_temp import (
    WaterTemperatureAboveIndicator as _WaterTemperatureAboveIndicator,
)
from hazard.models.wet_bulb_globe_temp import (
    DaysWBGTAboveIndicator as _DaysWBGTAboveIndicator,
)
from hazard.models.work_loss import WorkLossIndicator as _WorkLossIndicator
from hazard.onboarder import Onboarder
from hazard.protocols import OpenDataset
from hazard.sources import get_source_dataset_instance
from hazard.sources.osc_zarr import OscZarr

logger = logging.getLogger(__name__)


class IndicatorModelOnboarder(Onboarder):
    """Adapt an IndicatorModel and its data source to the Onboarder interface.

    Concrete subclasses set self.model and self.source in their own __init__;
    this base class is not meant to be instantiated directly.
    """

    model: IndicatorModel
    source: OpenDataset
    # arguments of the dask LocalCluster used by onboard, e.g. to limit the threads (so memory used)
    dask_cluster_kwargs: Dict[str, Any] = {}

    def inventory(self) -> Iterable[HazardResource]:
        return self.model.inventory()

    def onboard(self, target):
        with (
            LocalCluster(processes=False, **self.dask_cluster_kwargs) as cluster,
            Client(cluster) as client,
        ):
            logger.info(
                f"{type(self.model).__name__}: dask dashboard {client.dashboard_link}"
            )
            self.model.run_all(self.source, target, client=client)

    def create_maps(self, source: OscZarr, target: OscZarr):
        self.model.create_maps(source, target)

    def prepare(self, download_dir=None, force_download=False) -> None:
        """No local preparation needed; the model reads directly from its source."""

    def is_prepared(self) -> bool:
        """No local preparation needed; the model reads directly from its source."""
        return True


class ThresholdIndicatorOnboarder(IndicatorModelOnboarder):
    """Onboarder of a ThresholdBasedAverageIndicator, reading NEX-GDDP-CMIP6 directly from S3 or from a local cache.

    If prepare is given a download_dir, the inputs of the run are downloaded (in parallel) to a
    cache there and then read from it. Files already in the cache are not downloaded again, so
    prepare is always run (is_prepared is False) and can be re-run after a failure.
    """

    model: ThresholdBasedAverageIndicator
    # limit the years calculated at once, so the memory used: each year of WBGT holds about 6-8 GB
    dask_cluster_kwargs = {"n_workers": 1, "threads_per_worker": 2}

    def __init__(
        self,
        model: ThresholdBasedAverageIndicator,
        dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct the onboarder.

        Args:
            model (ThresholdBasedAverageIndicator): The model, with the settings of its run.
            dask_cluster_kwargs (Optional[Dict[str, Any]], optional): Arguments of the dask LocalCluster
                used by onboard. Defaults to the dask_cluster_kwargs of the class.
        """
        self.model = model
        self.source = get_source_dataset_instance(model.run.source_dataset, {})  # type: ignore[arg-type]
        if dask_cluster_kwargs is not None:
            self.dask_cluster_kwargs = dask_cluster_kwargs

    def prepare(self, download_dir=None, force_download=False) -> None:
        """Download the inputs to a cache in download_dir, if given, to be read from there."""
        name = type(self.model).__name__
        if download_dir is None:
            logger.info(
                f"{name}: no download_dir, so inputs are read directly from the source (no cache)"
            )
            return
        if self.model.run.source_dataset != "NEX-GDDP-CMIP6":
            raise ValueError("a download_dir is only supported for NEX-GDDP-CMIP6")
        logger.info(f"{name}: preparing cache of inputs in {download_dir}")
        start = time.perf_counter()
        source = NexGddpCmip6CachedSource(Path(download_dir))
        missing = source.download_inputs(self.model.inputs(source))
        self.source = source
        logger.info(
            f"{name}: cache prepared in {time.perf_counter() - start:.0f} s ({len(missing)} files "
            "downloaded); inputs will be read from the cache"
        )

    def is_prepared(self) -> bool:
        return False


def _run_config(run: Optional[Union[RunConfig, Dict[str, Any]]]) -> Optional[RunConfig]:
    """The RunConfig of run, which can also be given as a dict of its fields (e.g. parsed from the JSON of the CLI)."""
    return RunConfig(**run) if isinstance(run, dict) else run


class DaysTasAboveIndicator(ThresholdIndicatorOnboarder):
    def __init__(
        self,
        source_dir_base: Optional[str] = None,
        *,
        run: Optional[Union[RunConfig, Dict[str, Any]]] = None,
        dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            _DaysTasAboveIndicator(run=_run_config(run)), dask_cluster_kwargs
        )


class DaysTasAboveIndicatorDeprecated(IndicatorModelOnboarder):
    """Deprecated: superseded by DaysTasAboveIndicator (v3). Kept so that the existing v2 arrays remain in the inventory."""

    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _DaysTasAboveIndicatorDeprecated()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class DegreeDays(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _DegreeDays()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class HeatingCoolingDegreeDays(ThresholdIndicatorOnboarder):
    def __init__(
        self,
        source_dir_base: Optional[str] = None,
        *,
        run: Optional[Union[RunConfig, Dict[str, Any]]] = None,
        dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            _HeatingCoolingDegreeDays(run=_run_config(run)), dask_cluster_kwargs
        )


class HeatingCoolingDegreeDaysDeprecated(IndicatorModelOnboarder):
    """Deprecated: superseded by HeatingCoolingDegreeDays (v3). Kept so that the existing v2 arrays remain in the inventory."""

    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _HeatingCoolingDegreeDaysDeprecated()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class WetBulbGlobeTemperatureAboveIndicator(IndicatorModelOnboarder):
    """Deprecated: superseded by DaysWBGTAboveIndicator (v3). Kept so that the existing v2 arrays remain in the inventory."""

    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _WetBulbGlobeTemperatureAboveIndicator()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class DaysWBGTAboveIndicator(ThresholdIndicatorOnboarder):
    def __init__(
        self,
        source_dir_base: Optional[str] = None,
        *,
        run: Optional[Union[RunConfig, Dict[str, Any]]] = None,
        dask_cluster_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            _DaysWBGTAboveIndicator(run=_run_config(run)), dask_cluster_kwargs
        )


class WorkLossIndicator(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _WorkLossIndicator()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class WaterTemperatureAboveIndicator(IndicatorModelOnboarder):
    """Unlike the other adapters above, this one needs source_dir_base: its source
    (FutureStreamsSource) downloads data to a local working directory rather than
    streaming from S3, and has no registry entry in hazard.sources.
    """

    def __init__(self, source_dir_base: str):
        self.model = _WaterTemperatureAboveIndicator()
        self.source = FutureStreamsSource(working_dir=source_dir_base)


class DroughtIndicator(IndicatorModelOnboarder):
    """DroughtIndicator reads from a pre-chunked working zarr store rather than a
    source passed at run time, and needs each batch item pre-chunked (downloaded and
    rechunked from NexGddpCmip6 via Prechunker) before it can be onboarded.
    """

    def __init__(self, source_dir_base: str):
        working_dir = os.path.join(source_dir_base, "drought_indicator")
        os.makedirs(working_dir, exist_ok=True)
        self.model: _DroughtIndicator = _DroughtIndicator(
            working_zarr_store=local_zarr_working_store(working_dir),
            progress_store_path=working_dir,
        )

    def onboard(self, target):
        # DroughtIndicator reads from its pre-chunked working store, not a run-time
        # source, so there is nothing to pass as source here.
        self.model.run_all(None, target)  # type: ignore[arg-type]

    def prepare(self, download_dir=None, force_download=False) -> None:
        for item in self.model.batch_items():
            self.model.pre_chunk(item)

    def is_prepared(self) -> bool:
        # pre_chunk/Prechunker are themselves idempotent (they skip years already
        # chunked), so it is safe to always attempt rather than track completion here.
        return False
