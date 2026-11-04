"""Module for calculating multi-year average indicators for climate hazard models."""

import logging
import time
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing_extensions import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import xarray as xr
from dask.distributed import Client
from rasterio.crs import CRS  # type: ignore


from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource
from hazard.protocols import (
    Averageable,
    OpenDataset,
    ReadWriteDataArray,
    WriteDataArray,
)
from hazard.sources import get_source_dataset_default_resolution
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Averageable)

# 'GCM' identifier of the mean over the GCMs of a multi-model set of indicators
MULTI_MODEL_ID = "multi_model_0"


@dataclass
class Indicator:
    """Representation of a climate hazard indicator with a DataArray and file path."""

    array: xr.DataArray  # Union[xr.DataArray, Sequence[xr.DataArray]]
    path: PurePosixPath


class MultiYearAverageIndicatorBase(IndicatorModel[T]):
    """Indicator which is the average of indicators produced for a number of individual years.

    Such calculations can be split by year and run in parallel.
    """

    _default_window_years: int = 20
    _default_gcms: Iterable[str] = [
        "ACCESS-CM2",
        "CMCC-ESM2",
        "CNRM-CM6-1",
        "MPI-ESM1-2-LR",
        "MIROC6",
        "NorESM2-MM",
    ]
    # see Palmer et al. 'Performance-based sub-selection of CMIP6 models for impact assessments in Europe'
    # https://esd.copernicus.org/articles/14/457/2023/

    _europe_gcms: Iterable[str] = [
        "ACCESS-CM2",
        "CNRM-CM6-1",
        "EC-Earth3",
        "GFDL-ESM4",
        "MRI-ESM2-0",
        "TaiESM1",
    ]
    _default_scenarios: Iterable[str] = ["historical", "ssp126", "ssp245", "ssp585"]
    _default_central_year_historical: int = 2005
    _default_central_years: Iterable[int] = [2030, 2040, 2050]
    _default_source_dataset: str = "NEX-GDDP-CMIP6"
    # if set, arrays are written in chunks of this size in latitude and longitude, the whole of the
    # index dimension being in each chunk; otherwise the target's default chunking is used
    chunk_size_lat_lon: Optional[int] = None

    def __init__(
        self,
        window_years: int = 20,
        gcms: Iterable[str] = _default_gcms,
        scenarios: Iterable[str] = _default_scenarios,
        central_year_historical: int = _default_central_year_historical,
        central_years: Iterable[int] = _default_central_years,
        source_dataset: str = _default_source_dataset,
    ):
        """Construct model to calculate degree days from temperature data sets.

        Args:
            window_years (int, optional): Number of years for average. Defaults to 20.
            gcms (Iterable[str], optional): Global Circulation Models to include in calculation.
                                            Defaults to ["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1",
                                            "MPI-ESM1-2-LR", "MIROC6", "NorESM2-MM"].
            scenarios (Iterable[str], optional): Scenarios to include in calculation.
                                                 Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
            central_year_historical (int): Central year to include in calculation for historical scenario.
                                           Defaults to 2005.
            central_years (Iterable[int], optional): Central years to include in calculation.
                                                     Defaults to [2010, 2030, 2040, 2050].
            source_dataset: Source dataset. Defaults to "NEX-GDDP-CMIP6"

        """
        # 1995 to 2014 (2010), 2021 tp 2040 (2030), 2031 to 2050 (2040), 2041 to 2060 (2050)
        self.window_years = window_years
        self.gcms = gcms
        self.scenarios = scenarios
        self.central_years = central_years
        self.central_year_historical = central_year_historical
        self.source_dataset = source_dataset

    def run_single(
        self, item: T, source: OpenDataset, target: ReadWriteDataArray, client: Client
    ):
        """Execute the averaging process for a single climate indicator."""
        averaged_indicators = self._averaged_indicators(client, source, target, item)
        for indicator in averaged_indicators:
            indicator.array.attrs["crs"] = CRS.from_epsg(4326)
            logger.info(
                f"{type(self).__name__}: writing array to {str(indicator.path)}"
            )
            target.write(
                str(indicator.path),
                indicator.array,
                chunks=self._write_chunks(indicator.array),
            )

    def _write_chunks(self, array: xr.DataArray) -> Optional[List[int]]:
        if self.chunk_size_lat_lon is None:
            return None
        return [array.shape[0], self.chunk_size_lat_lon, self.chunk_size_lat_lon]

    def _years(self, _: OpenDataset, item: Averageable):
        return range(
            item.central_year - self.window_years // 2,
            item.central_year + self.window_years // 2 + (self.window_years % 2),
        )

    def _averaged_indicators(
        self,
        client: Client,
        source: OpenDataset,
        target: WriteDataArray,
        item: Averageable,
    ) -> List[Indicator]:
        """Calculate average annual degree days for given window for the GCM and scenario specified."""
        years = self._years(source, item)
        logger.info(
            f"{type(self).__name__}: calculating average indicator for batch item {str(item)}, "
            f"years={list(years)}"
        )
        futures = []
        for year in years:
            future = client.submit(
                self._calculate_single_year_indicators, source, item, year
            )
            futures.append(future)
        single_year_sets: List[List[Indicator]] = list(
            client.gather(futures)
        )  # indicators for each year
        indics_per_year = len(single_year_sets[0])  # number of indicators for each year
        res: List[Indicator] = []
        for i in range(indics_per_year):
            average = sum(set[i].array for set in single_year_sets) / float(len(years))
            assert isinstance(average, xr.DataArray)  # must be non-zero
            # arithmetic drops the attributes (e.g. units) of the single-year arrays
            average.attrs = dict(single_year_sets[0][i].array.attrs)
            res.append(
                Indicator(
                    array=enforce_conventions_lat_lon(average),
                    path=single_year_sets[0][i].path,
                )
            )
        return res

    @abstractmethod
    def batch_items(self) -> Iterable[T]:
        """Get batch items (batch items can be calculated independently from one another)."""
        ...

    @abstractmethod
    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory."""
        ...

    @abstractmethod
    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: T, year: int
    ) -> List[Indicator]:
        """Calculate indicators for a single year for a single batch item.

        If just a single indicator per batch, a list of length one is expected.
        """
        ...


@dataclass
class BatchItem:
    """Represents a batch item for processing climate data with specific model, scenario, and year parameters."""

    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int

    def __str__(self):
        """Return a descriptive string representation of the batch item."""
        return f"gcm={self.gcm}, scenario={self.scenario}, central_year={self.central_year}"


@dataclass(frozen=True)
class RunConfig:
    """Settings for a run of a ThresholdBasedAverageIndicator: GCMs, scenarios, periods and source dataset.

    The defaults are those of the v3 indicators. A RunConfig cannot be modified once created, so
    the same one can be shared by several indicators.

    Attributes:
        gcms (Sequence[str]): Global Circulation Models to include in calculation.
            Defaults to ["ACCESS-CM2", "CNRM-CM6-1", "EC-Earth3", "GFDL-ESM4", "MRI-ESM2-0", "TaiESM1"].
        scenarios (Sequence[str]): Scenarios to include in calculation.
            Defaults to ["historical", "ssp126", "ssp245", "ssp585"].
        central_year_historical (int): Central year for historical scenario. Defaults to 2005.
        central_years (Sequence[int]): Central years for other scenarios. Defaults to [2030, 2040, 2050, 2080].
        window_years (int): Number of years for average. Defaults to 20.
        source_dataset (str): Source dataset. Defaults to "NEX-GDDP-CMIP6".
        multi_model (bool): If True, calculate the multi-model mean over the GCMs at the end of run_all.
            Defaults to True.
        resolution (Optional[str]): Resolution of the indicators, e.g. "0.25°". Defaults to that of the
            source dataset with its default arguments; set this if the source has other arguments (e.g.
            UKCP18 at 2.2 km). run_all checks that it matches the source, if the source gives one.
    """

    gcms: Sequence[str] = tuple(MultiYearAverageIndicatorBase._europe_gcms)
    scenarios: Sequence[str] = ("historical", "ssp126", "ssp245", "ssp585")
    central_year_historical: int = (
        MultiYearAverageIndicatorBase._default_central_year_historical
    )
    central_years: Sequence[int] = (2030, 2040, 2050, 2080)
    window_years: int = MultiYearAverageIndicatorBase._default_window_years
    source_dataset: str = MultiYearAverageIndicatorBase._default_source_dataset
    multi_model: bool = True
    resolution: Optional[str] = None

    def __post_init__(self):
        # store sequences as tuples, so that they cannot be modified either
        for name in ["gcms", "scenarios", "central_years"]:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.resolution is None:
            object.__setattr__(
                self,
                "resolution",
                get_source_dataset_default_resolution(self.source_dataset),
            )
        if MULTI_MODEL_ID in self.gcms:
            raise ValueError(
                f"'{MULTI_MODEL_ID}' is the mean over the GCMs, so cannot be one of them"
            )

    def scenario_years(self) -> List[Tuple[str, int]]:
        """(Scenario, central year) pairs of the run: central_year_historical for the historical scenario."""
        return [
            (scenario, central_year)
            for scenario in self.scenarios
            for central_year in (
                [self.central_year_historical]
                if scenario == "historical"
                else self.central_years
            )
        ]


@dataclass
class ThresholdBatchItem:
    """Batch item of a ThresholdBasedAverageIndicator: a GCM, scenario and central year."""

    gcm: str
    scenario: str
    central_year: int

    def __str__(self):
        """Return a descriptive string representation of the batch item."""
        return f"gcm={self.gcm}, scenario={self.scenario}, central_year={self.central_year}"


class ThresholdBasedAverageIndicator(MultiYearAverageIndicatorBase[ThresholdBatchItem]):
    """Model for calculating threshold-based multi-year average indicators, the thresholds being the index dimension.

    For each GCM, scenario and central year, one array per resource (see _resources) is
    written, of shape (index, lat, lon): one slice per threshold. A model can have more
    than one resource where these are calculated from the same inputs (e.g. degree days
    above and below thresholds); all are calculated in a single pass over the inputs.

    Unless run.multi_model is False, run_all then calculates the multi-model mean of each
    array over the GCMs, written with gcm=MULTI_MODEL_ID.

    Subclasses provide the input quantities, the calculation of the single-year arrays
    and the resources.
    """

    # thresholds used if none are given to the constructor
    default_thresholds: Sequence[float]
    # units of the thresholds, i.e. of the values of the "threshold" dimension
    threshold_units: str = "°C"
    # small chunks, for fast look-up of the indicators at single locations
    chunk_size_lat_lon = 500

    def __init__(
        self,
        thresholds: Optional[Sequence[float]] = None,
        run: Optional[RunConfig] = None,
    ):
        """Construct the model.

        Args:
            thresholds (Optional[Sequence[float]], optional): Thresholds, i.e. the values of the index
                dimension. Defaults to the default_thresholds of the class.
            run (Optional[RunConfig], optional): GCMs, scenarios, periods and source dataset of the
                run. Defaults to RunConfig(), i.e. the v3 settings.

        """
        run = RunConfig() if run is None else run
        super().__init__(
            window_years=run.window_years,
            gcms=list(run.gcms),
            scenarios=list(run.scenarios),
            central_year_historical=run.central_year_historical,
            central_years=list(run.central_years),
            source_dataset=run.source_dataset,
        )
        self.run = run
        self.thresholds = list(
            self.default_thresholds if thresholds is None else thresholds
        )
        self.resources = self._resources()

    def resource_thresholds(self, key: str) -> List[float]:
        """Thresholds of the resource with key (see _resources): by default, the thresholds of the model."""
        return self.thresholds

    @property
    @abstractmethod
    def quantities(self) -> Sequence[str]:
        """Input quantities (e.g. "tas") opened for each year."""
        ...

    @abstractmethod
    def _single_year_arrays(
        self, inputs: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        """Calculate the arrays for a single year.

        Args:
            inputs (Dict[str, xr.DataArray]): Daily data for the year, keyed by quantity.

        Returns:
            Dict[str, xr.DataArray]: Arrays of shape (index, lat, lon), with the same keys
            as _resources. See _stack_thresholds.
        """
        ...

    @abstractmethod
    def _resources(self) -> Dict[str, HazardResource]:
        """Resources of the model; see hazard.models.indicator_conventions.threshold_indicator_resource."""
        ...

    def batch_items(self) -> Iterable[ThresholdBatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
        return [
            ThresholdBatchItem(gcm, scenario, central_year)
            for gcm in self.gcms
            for scenario, central_year in self.run.scenario_years()
        ]

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return list(self.resources.values())

    def inputs(self, source: OpenDataset) -> List[Tuple[str, str, str, int]]:
        """(GCM, scenario, quantity, year) of each input of the run, as opened from source; each once."""
        return sorted(
            {
                (item.gcm, item.scenario, quantity, year)
                for item in self.batch_items()
                for year in self._years(source, item)
                for quantity in self.quantities
            }
        )

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        for resource in self.inventory():
            create_tiles_for_resource(source, target, resource)

    def run_all(
        self,
        source: OpenDataset,
        target: ReadWriteDataArray,
        client: Optional[Client] = None,
        debug_mode=False,
    ):
        """Run all items in the batch, then calculate the multi-model mean (if multi_model)."""
        source_resolution = getattr(source, "resolution", None)
        if source_resolution is not None and source_resolution != self.run.resolution:
            raise ValueError(
                f"resolution of source ({source_resolution}) does not match that of the run "
                f"({self.run.resolution}); set RunConfig.resolution"
            )
        name = type(self).__name__
        logger.info(
            f"{name}: starting run of {len(list(self.batch_items()))} batch items, indicators "
            + ", ".join(
                f"{r.indicator_id} (thresholds {self.resource_thresholds(key)})"
                for key, r in self.resources.items()
            )
        )
        start = time.perf_counter()
        super().run_all(source, target, client=client, debug_mode=debug_mode)
        if self.run.multi_model:
            logger.info(f"{name}: calculating multi-model mean")
            self.calculate_multi_model(target)
        logger.info(f"{name}: run complete in {time.perf_counter() - start:.0f} s")

    def calculate_multi_model(self, target: ReadWriteDataArray):
        """Calculate the mean over the GCMs of each array, written with gcm=MULTI_MODEL_ID.

        Raises an error, before anything is written, if the array of any GCM is missing
        (e.g. because its batch item failed).
        """
        paths = {
            (key, scenario, central_year): [
                resource.path.format(gcm=gcm, scenario=scenario, year=central_year)
                for gcm in self.gcms
            ]
            for key, resource in self.resources.items()
            for scenario, central_year in self.run.scenario_years()
        }
        arrays: Dict[str, xr.DataArray] = {}
        missing: List[str] = []
        for gcm_paths in paths.values():
            for path in gcm_paths:
                try:
                    arrays[path] = target.read(path)
                except KeyError:
                    missing.append(path)
        if missing:
            raise ValueError(
                f"cannot calculate multi-model mean; missing arrays: {', '.join(missing)}"
            )
        for (key, scenario, central_year), gcm_paths in paths.items():
            # join="exact" so that arrays with mismatched coordinates raise an error
            gcm_arrays = xr.align(*(arrays[path] for path in gcm_paths), join="exact")
            mean = sum(gcm_arrays[1:], gcm_arrays[0]) / len(gcm_arrays)
            mean.attrs["units"] = self.resources[key].units
            mean.attrs["crs"] = CRS.from_epsg(4326)
            path = self.resources[key].path.format(
                gcm=MULTI_MODEL_ID, scenario=scenario, year=central_year
            )
            logger.info(
                f"{type(self).__name__}: writing multi-model mean array to {path}"
            )
            target.write(path, mean, chunks=self._write_chunks(mean))

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: ThresholdBatchItem, year: int
    ) -> List[Indicator]:
        logger.info(
            f"{type(self).__name__}: starting calculation for {item}, year {year}"
        )
        with ExitStack() as stack:
            inputs = {
                quantity: stack.enter_context(
                    source.open_dataset_year(item.gcm, item.scenario, quantity, year)
                )[quantity]
                for quantity in self.quantities
            }
            for da in inputs.values():
                self._check_input(da, year)
            # inputs must be on identical grids: raise rather than silently align
            with xr.set_options(arithmetic_join="exact"):
                arrays = self._single_year_arrays(inputs)
        logger.info(
            f"{type(self).__name__}: calculation complete for {item}, year {year}"
        )
        return [
            Indicator(
                array=arrays[key].assign_attrs(units=resource.units),
                path=PurePosixPath(
                    resource.path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
            )
            for key, resource in self.resources.items()
        ]

    def _check_input(self, da: xr.DataArray, year: int):
        if any(coord not in da.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (da["time"].dt.year != year).any():
            raise ValueError("unexpected year found")

    def _stack_thresholds(
        self,
        threshold_fn: Callable[[float], xr.DataArray],
        thresholds: Optional[Sequence[float]] = None,
    ) -> xr.DataArray:
        """Stack the (lat, lon) arrays given by threshold_fn for each threshold along a new "threshold" dimension.

        The thresholds default to those of the model.
        """
        thresholds = self.thresholds if thresholds is None else list(thresholds)
        return xr.concat(
            [threshold_fn(threshold) for threshold in thresholds],
            dim=xr.DataArray(
                thresholds,
                dims="threshold",
                name="threshold",
                attrs={"units": self.threshold_units},
            ),
        ).astype("float32")

    def _days_above(self, da: xr.DataArray) -> xr.DataArray:
        """Days per year (normalized to 365 days) for which daily values of da are above each threshold."""
        scale = 365.0 / len(da["time"])
        return self._stack_thresholds(
            lambda threshold: scale * (da > threshold).sum(dim="time")
        )
