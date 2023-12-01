# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9200746/

import logging
import os
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing import Iterable, List

import requests  # type: ignore
import xarray as xr
from attr import dataclass

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import Indicator, ThresholdBasedAverageIndicator
from hazard.protocols import Averageable, OpenDataset

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
            os.makedirs(self.working_dir)

        self.from_historical_years = [1976, 1986, 1996]
        self.to_historical_years = [1985, 1995, 2005]

        self.from_years = [2006, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]
        self.to_years = [2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099]

    def open_dataset_year(
        self, gcm: str, scenario: str, _: str, year: int, chunks=None
    ) -> xr.Dataset:
        from_years = (
            self.from_historical_years if scenario == "historical" else self.from_years
        )
        to_years = (
            self.to_historical_years if scenario == "historical" else self.to_years
        )
        set_from_year, set_to_year = next(
            (from_year, to_year)
            for from_year, to_year in zip(from_years, to_years)
            if year >= from_year and year <= to_year
        )
        path, url = self.water_temp_download_path(
            gcm, scenario, set_from_year, set_to_year
        )
        self.download_file(url, path)
        return xr.open_dataset(path, chunks=chunks)

    def water_temp_download_path(
        self, gcm: str, scenario: str, from_year: int, to_year: int
    ):
        adjusted_gcm = gcm if gcm == "E2O" else gcm.lower()
        adjusted_scenario = scenario[:4] if scenario == "historical" else scenario
        filename = self.filename(gcm, scenario, from_year, to_year)
        url = (
            "https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/"
            + f"original/waterTemp/{adjusted_scenario}/{adjusted_gcm}/"
            + filename
        )
        return os.path.join(self.working_dir, filename), url

    def filename(self, gcm: str, scenario: str, from_year: int, to_year: int):
        adjusted_from_year = (
            1979
            if scenario == "historical"
            and gcm == "E2O"
            and from_year == self.from_historical_years[0]
            else from_year
        )
        from_date = f"{adjusted_from_year}-01-07"
        to_date = f"{to_year}-12-30"
        adjusted_gcm = gcm if gcm == "E2O" else gcm.lower()
        adjusted_scenario = scenario[:4] if scenario == "historical" else scenario
        return f"waterTemp_weekAvg_output_{adjusted_gcm}_{adjusted_scenario}_{from_date}_to_{to_date}.nc"

    def download_all(self, gcm: str, scenario: str):
        from_years = (
            self.from_historical_years if scenario == "historical" else self.from_years
        )
        to_years = (
            self.to_historical_years if scenario == "historical" else self.to_years
        )
        for from_year, to_year in zip(from_years, to_years):
            filename, url = self.water_temp_download_path(
                gcm, scenario, from_year, to_year
            )
            self.download_file(url, os.path.join(self.working_dir, filename))

    def download_file(self, url, path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)


class WaterTemperatureIndicator(ThresholdBasedAverageIndicator):
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

    def _years(self, source: OpenDataset, item: Averageable) -> List[int]:
        if item.scenario == "historical":
            return source.from_historical_years
        return [
            source.from_year
            for (from_year, to_year) in zip(source.from_years, source.to_years)
            if from_year == item.year or to_year + 1 == item.year
        ]

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        """For a single year and batch item calculate the indicators (i.e. one per threshold temperature)."""
        logger.info(f"Starting calculation for year {year}")
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
            results = self._weeks_water_temp_above_indicators(dataset)
        logger.info(f"Calculation complete for year {year}")
        return [
            Indicator(
                results,
                PurePosixPath(
                    self._resource.path.format(
                        gcm=item.gcm, scenario=item.scenario, year=item.central_year
                    )
                ),
                self.resources["above"].map.bounds,
            )
        ]

    def _weeks_water_temp_above_indicators(
        self, input: xr.DataArray
    ) -> List[xr.DataArray]:
        """Create DataArrays containing indicators the thresholds for a single year."""
        if any(
            coord not in input.coords.keys()
            for coord in ["latitude", "longitude", "time"]
        ):
            raise ValueError("expect coordinates: 'latitude', 'longitude' and 'time'")
        # normalize to 52 weeks
        scale = 52.0 / len(input.time)
        coords = {
            "index": self.threshold_temps_c,
            "lat": input.coords["latitude"].values,
            "lon": input.coords["longitude"].values,
        }
        output = xr.DataArray(coords=coords, dims=coords.keys())
        for i, threshold_c in enumerate(self.threshold_temps_c):
            threshold_k = 273.15 + threshold_c
            output[i, :, :] = scale * xr.where(input > threshold_k, 1.0, 0.0).sum(
                dim=["time"]
            )
        return input

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
                "temp_c": [str(t) for t in self.threshold_temps_c],
                "gcm": list(self.gcms),
            },
            path="chronic_heat/osc/v2/weeks_water_temp_above_{gcm}_{scenario}_{year}",
            display_name="Weeks with average temperature above threshold in degrees celsius/{gcm}",
            description=description,
            display_groups=[
                "Weeks with average temperature above threshold in degrees celsius"
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
                bounds=[(-180.0, 90.0), (180.0, 90.0), (180.0, -90.0), (-180.0, -90.0)],
                path="weeks_water_temp_above_{gcm}_{scenario}_{year}_map",
                source="map_array",
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
