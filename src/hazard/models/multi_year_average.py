"""Module for calculating multi-year average indicators for climate hazard models."""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing_extensions import Iterable, List, Tuple, TypeVar

import xarray as xr
from dask.distributed import Client
from rasterio.crs import CRS  # type: ignore


from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource, MapInfo
from hazard.protocols import (
    Averageable,
    OpenDataset,
    ReadWriteDataArray,
    WriteDataArray,
)
from hazard.utilities.xarray_utilities import enforce_conventions_lat_lon

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Averageable)


@dataclass
class Indicator:
    """Representation of a climate hazard indicator with a DataArray, file path, and geographic bounds."""

    array: xr.DataArray  # Union[xr.DataArray, Sequence[xr.DataArray]]
    path: PurePosixPath
    bounds: List[Tuple[float, float]]


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
        "TaiESM1"
    ]
    _default_scenarios: Iterable[str] = ["historical", "ssp126", "ssp245", "ssp585"]
    _default_central_year_historical: int = 2005
    _default_central_years: Iterable[int] = [2030, 2040, 2050]
    _default_source_dataset: str = "NEX-GDDP-CMIP6"

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
            logger.info(f"Writing array to {str(indicator.path)}")
            target.write(str(indicator.path), indicator.array)

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
            f"Calculating average indicator for batch item {str(item)}, years={list(years)}"
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
            res.append(
                Indicator(
                    array=enforce_conventions_lat_lon(average),
                    path=single_year_sets[0][i].path,
                    bounds=single_year_sets[0][i].bounds,
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


class ThresholdBasedAverageIndicator(MultiYearAverageIndicatorBase[BatchItem]):
    """Model for calculating threshold-based multi-year average indicators."""

    def batch_items(self) -> Iterable[BatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
        resource = self._resource()
        for gcm in self.gcms:
            for scenario in self.scenarios:
                central_years = (
                    [self.central_year_historical]
                    if scenario == "historical"
                    else self.central_years
                )
                for central_year in central_years:
                    yield BatchItem(
                        resource=resource,
                        gcm=gcm,
                        scenario=scenario,
                        central_year=central_year,
                    )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return [self._resource()]  # .expand()

    def _get_indicators(
        self, item: BatchItem, data_arrays: List[xr.DataArray], param: str
    ) -> List[Indicator]:
        """Generate a list of indicators for given thresholds and data arrays.

        Args:
            item (BatchItem): _description_
            data_arrays (List[xr.DataArray]): _description_
            param (str): _description_

        Returns:
            List[Indicator]: _description_

        """
        resource = item.resource
        paths = [
            item.resource.path.format(
                temp_c=threshold,
                gcm=item.gcm,
                scenario=item.scenario,
                year=item.central_year,
            )
            for threshold in resource.params[param]
        ]
        assert isinstance(resource.map, MapInfo)
        return [
            Indicator(
                array=array, path=PurePosixPath(paths[i]), bounds=resource.map.bounds
            )
            for i, array in enumerate(data_arrays)
        ]

    @abstractmethod
    def _resource(self) -> HazardResource: ...  # noqa:E704
