"""Adapters that let IndicatorModels be driven through the Onboarder pipeline.

An IndicatorModel is designed to work with a choice of data source (passed into
run_all), rather than owning a single source_dir_base the way an Onboarder does.
These adapters compose a concrete IndicatorModel with the source it should read
from, and expose that composition as an Onboarder so it can be onboarded through
the same onboard_hazards pipeline as any other hazard.
"""

import os
from typing import Optional

from typing_extensions import Iterable

from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource
from hazard.models.days_tas_above import DaysTasAboveIndicator as _DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays as _DegreeDays
from hazard.models.degree_days import (
    HeatingCoolingDegreeDays as _HeatingCoolingDegreeDays,
)
from hazard.models.drought_index import DroughtIndicator as _DroughtIndicator
from hazard.models.drought_index import local_zarr_working_store
from hazard.models.water_temp import FutureStreamsSource
from hazard.models.water_temp import (
    WaterTemperatureAboveIndicator as _WaterTemperatureAboveIndicator,
)
from hazard.models.wet_bulb_globe_temp import (
    WetBulbGlobeTemperatureAboveIndicator as _WetBulbGlobeTemperatureAboveIndicator,
)
from hazard.models.work_loss import WorkLossIndicator as _WorkLossIndicator
from hazard.onboarder import Onboarder
from hazard.protocols import OpenDataset
from hazard.sources import get_source_dataset_instance
from hazard.sources.osc_zarr import OscZarr


class IndicatorModelOnboarder(Onboarder):
    """Adapt an IndicatorModel and its data source to the Onboarder interface.

    Concrete subclasses set self.model and self.source in their own __init__;
    this base class is not meant to be instantiated directly.
    """

    model: IndicatorModel
    source: OpenDataset

    def inventory(self) -> Iterable[HazardResource]:
        return self.model.inventory()

    def onboard(self, target):
        self.model.run_all(self.source, target)

    def create_maps(self, source: OscZarr, target: OscZarr):
        self.model.create_maps(source, target)

    def prepare(self, download_dir=None, force_download=False) -> None:
        """No local preparation needed; the model reads directly from its source."""

    def is_prepared(self) -> bool:
        """No local preparation needed; the model reads directly from its source."""
        return True


class DaysTasAboveIndicator(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _DaysTasAboveIndicator()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class DegreeDays(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _DegreeDays()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class HeatingCoolingDegreeDays(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _HeatingCoolingDegreeDays()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


class WetBulbGlobeTemperatureAboveIndicator(IndicatorModelOnboarder):
    def __init__(self, source_dir_base: Optional[str] = None):
        self.model = _WetBulbGlobeTemperatureAboveIndicator()
        self.source = get_source_dataset_instance("NEX-GDDP-CMIP6", {})


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
