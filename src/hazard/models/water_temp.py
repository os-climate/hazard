# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9200746/

import logging
import os
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing import Iterable, List, Tuple

import requests  # type: ignore
import xarray as xr
from attr import dataclass

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (Indicator,
                                              ThresholdBasedAverageIndicator)
from hazard.protocols import Averageable, OpenDataset
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int


class FutureStreamsSource(OpenDataset):
    def __init__(self, working_dir: str):
        """Source requires a working_dir. The data is not hosted in S3 but is rather
        downloaded to the working directory.

        Args:
            working_dir (str): Working directory path.
        """
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            if 0 < len(self.working_dir):
                os.makedirs(self.working_dir)

        self.from_years = [
            1976,
            1986,
            1996,
            2006,
            2020,
            2030,
            2040,
            2050,
            2060,
            2070,
            2080,
            2090,
        ]
        self.to_years = [
            1985,
            1995,
            2005,
            2019,
            2029,
            2039,
            2049,
            2059,
            2069,
            2079,
            2089,
            2099,
        ]

    def from_year(self, gcm: str, year: int) -> int:
        if year not in self.to_years:
            raise ValueError(
                f"The input year {year} is not within the available from_years={list(self.to_years)}"
            )
        from_year = self.from_years[self.to_years.index(year)]
        if gcm == "E2O" and from_year == 1976:
            return 1979
        return from_year

    def open_dataset_year(
        self, gcm: str, scenario: str, _: str, year: int, chunks=None
    ) -> xr.Dataset:
        path, url = self.water_temp_download_path(gcm, scenario, year)
        self.download_file(url, path)
        return xr.open_dataset(path, chunks=chunks)

    def delete_file(self, gcm: str, scenario: str, year: int) -> None:
        path, _ = self.water_temp_download_path(gcm, scenario, year)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)

    def water_temp_download_path(
        self, gcm: str, scenario: str, year: int
    ) -> Tuple[str, str]:
        adjusted_gcm = gcm if gcm == "E2O" else gcm.lower()
        adjusted_scenario = scenario[:4] if scenario == "historical" else scenario
        filename = self.filename(gcm, scenario, year)
        url = (
            "https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/"
            + f"original/waterTemp/{adjusted_scenario}/{adjusted_gcm}/"
            + filename
        )
        return os.path.join(self.working_dir, filename), url

    def filename(self, gcm: str, scenario: str, year: int) -> str:
        from_year = self.from_year(gcm, year)
        from_date = f"{from_year}-01-07"
        to_date = f"{year}-12-30"
        adjusted_gcm = gcm if gcm == "E2O" else gcm.lower()
        adjusted_scenario = scenario[:4] if scenario == "historical" else scenario
        return f"waterTemp_weekAvg_output_{adjusted_gcm}_{adjusted_scenario}_{from_date}_to_{to_date}.nc"

    def download_all(self, gcm: str, scenario: str) -> None:
        years = self.to_years[:3] if scenario == "historical" else self.to_years[3:]
        for year in years:
            filename, url = self.water_temp_download_path(gcm, scenario, year)
            self.download_file(url, os.path.join(self.working_dir, filename))

    def download_file(self, url: str, path: str) -> None:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)


class WaterTemperatureAboveIndicator(ThresholdBasedAverageIndicator):
    def __init__(
        self,
        threshold_temps_c: List[float] = [
            5,
            7.5,
            10,
            12.5,
            15,
            17.5,
            20,
            22.5,
            25,
            27.5,
            30,
            32.5,
            35,
            37.5,
            40,
        ],
        window_years: int = 20,
        gcms: Iterable[str] = ["E2O", "GFDL", "HadGEM", "IPSL", "MIROC", "NorESM"],
        scenarios: Iterable[str] = [
            "historical",
            "rcp2p6",
            "rcp4p5",
            "rcp6p0",
            "rcp8p5",
        ],
        central_year_historical: int = 1991,
        central_years: Iterable[int] = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090],
    ):
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c
        self.resource = self._resource()

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        if "E2O" in self.gcms and "historical" in self.scenarios:
            return [self.resource, self._other_resource()]
        return [self.resource]

    def batch_items(self) -> Iterable[BatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
        items = [item for item in super().batch_items() if item.gcm != "E2O"]
        if "E2O" in self.gcms and "historical" in self.scenarios:
            items.append(
                BatchItem(
                    resource=self._other_resource(),
                    gcm="E2O",
                    scenario="historical",
                    central_year=self.central_year_historical,
                )
            )
        return items

    def _years(self, source: OpenDataset, item: Averageable) -> List[int]:
        if hasattr(source, "from_years") and hasattr(source, "to_years"):
            if item.scenario == "historical":
                return source.to_years[:3]
            return [
                to_year
                for (from_year, to_year) in zip(
                    source.from_years[3:], source.to_years[3:]
                )
                if from_year == item.central_year or to_year + 1 == item.central_year
            ]
        return super()._years(source, item)

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        from_year: int = (
            source.from_year(item.gcm, year)
            if hasattr(source, "from_year")
            else (year - self.window_years // 2 + 1)
        )
        correction: float = 1.0
        if item.gcm == "E2O" and item.scenario == "historical":
            correction = float(year - from_year + 1) / 9.0
        """For a single year and batch item calculate the indicators (i.e. one per threshold temperature)."""
        logger.info(f"Starting calculation for years from {from_year} to {year}")
        with ExitStack() as stack:
            dataset = stack.enter_context(
                source.open_dataset_year(
                    item.gcm,
                    item.scenario,
                    "waterTemperature",
                    year,
                    chunks={"time": -1, "latitude": 540, "longitude": 1080},
                )
            ).waterTemperature
            if from_year == 2006:
                cutoff = item.central_year - self.window_years // 2 - 1970
                from_index = min(
                    [
                        i
                        for i, time in enumerate(dataset.time.values)
                        if cutoff <= time.astype("datetime64[Y]").astype(int)
                    ]
                )
                dataset = dataset[from_index:, :, :]
            results = self._weeks_water_temp_above_indicators(dataset, correction)
        logger.info(f"Calculation complete for years from {from_year} to {year}")
        if hasattr(source, "delete_file"):
            source.delete_file(item.gcm, item.scenario, year)
        path = (
            item.resource.path.format(scenario=item.scenario, year=item.central_year)
            if item.gcm == "E2O"
            else item.resource.path.format(
                gcm=item.gcm, scenario=item.scenario, year=item.central_year
            )
        )
        return [
            Indicator(
                results,
                PurePosixPath(path),
                item.resource.map.bounds,
            )
        ]

    def _weeks_water_temp_above_indicators(
        self, input: xr.DataArray, correction: float
    ) -> List[xr.DataArray]:
        """Create DataArrays containing indicators the thresholds for a single year."""
        if any(
            coord not in input.coords.keys()
            for coord in ["latitude", "longitude", "time"]
        ):
            raise ValueError("expect coordinates: 'latitude', 'longitude' and 'time'")
        # normalize to 52 weeks
        scale = 52.0 * correction / len(input.time)
        coords = {
            "index": self.threshold_temps_c,
            "lat": input.coords["latitude"].values,
            "lon": input.coords["longitude"].values,
        }
        output = xr.DataArray(coords=coords, dims=coords.keys())
        for i, threshold_c in enumerate(self.threshold_temps_c):
            threshold_k = 273.15 + threshold_c
            output[i, :, :] = xr.where(input > threshold_k, scale, 0.0).sum(
                dim=["time"]
            )
        return output

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        for resource in self.inventory():
            create_tiles_for_resource(source, target, resource)

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(os.path.join(os.path.dirname(__file__), "water_temp.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="weeks_water_temp_above",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={
                "gcm": [gcm for gcm in self.gcms if gcm != "E2O"],
            },
            path="chronic_heat/nluu/v2/weeks_water_temp_above_{gcm}_{scenario}_{year}",
            display_name="Weeks with average water temperature above threshold in °C/{gcm}",
            description=description,
            display_groups=[
                "Weeks with average water temperature above threshold in °C"
            ],  # display names of groupings
            group_id="",
            map=MapInfo(
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=100,
                    max_index=255,
                    units="weeks/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                path="maps/chronic_heat/nluu/v2/weeks_water_temp_above_{gcm}_{scenario}_{year}_map",
                index_values=self.threshold_temps_c,
                source="map_array_pyramid",
            ),
            units="weeks/year",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="rcp2p6", years=list(self.central_years)),
                Scenario(id="rcp4p5", years=list(self.central_years)),
                Scenario(id="rcp6p0", years=list(self.central_years)),
                Scenario(id="rcp8p5", years=list(self.central_years)),
            ],
        )
        return resource

    def _other_resource(self) -> HazardResource:
        map = MapInfo(
            colormap=self.resource.map.colormap,
            bounds=self.resource.map.bounds,
            path=self.resource.map.path.format(
                gcm="E2O", scenario="{scenario}", year="{year}"
            ),
            index_values=self.resource.map.index_values,
            source=self.resource.map.source,
        )
        return HazardResource(
            hazard_type=self.resource.hazard_type,
            indicator_id=self.resource.indicator_id,
            indicator_model_id=self.resource.indicator_model_id,
            indicator_model_gcm=self.resource.indicator_model_gcm.format(gcm="E2O"),
            params={},
            path=self.resource.path.format(
                gcm="E2O", scenario="{scenario}", year="{year}"
            ),
            display_name=self.resource.display_name.format(gcm="E2O"),
            description=self.resource.description.replace("1976", "1979"),
            display_groups=self.resource.display_groups,
            group_id=self.resource.group_id,
            map=map,
            units=self.resource.units,
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
            ],
        )
