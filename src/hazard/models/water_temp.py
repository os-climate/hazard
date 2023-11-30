# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9200746/

import logging
import os
from contextlib import ExitStack
from typing import Iterable, List

import requests  # type: ignore
import xarray as xr
from attr import dataclass

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import Indicator, ThresholdBasedAverageIndicator
from hazard.protocols import OpenDataset

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

    @staticmethod
    def adjust_from_year(gcm, scenario, from_year):
        if gcm == "historical" and scenario == "E2O":
            return 1979
        return from_year

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
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
        filename, url = self.water_temp_download_path(
            gcm, scenario, set_from_year, set_to_year
        )
        self.download_file(url, os.path.join(self.working_dir, filename))
        return xr.open_dataset(filename)

    def water_temp_download_path(
        self, gcm: str, scenario: str, from_year: int, to_year: int
    ):
        filename = self.filename(gcm, scenario, from_year, to_year)
        url = (
            "https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/"
            + f"original/waterTemp/{scenario}/{gcm}/"
            + filename
        )
        return filename, url

    def filename(self, gcm: str, scenario: str, from_year: int, to_year: int):
        adjusted_from_year = self.adjust_from_year(gcm, scenario, from_year)
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
        threshold_temps_c: List[float] = [25, 30, 35, 40, 45, 50, 55],
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
        central_years: Iterable[int] = [2030, 2040, 2050, 2060, 2070, 2080, 2090],
    ):
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        """For a single year and batch item calculate the indicators (i.e. one per threshold temperature)."""
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            results = self._weeks_tas_above_indicators(
                tas, year, self.threshold_temps_c
            )
        logger.info(f"Calculation complete for year {year}")
        # Get Indicators for reach array, looking up path using "threshold"
        return self._get_indicators(item, results, "temp_c")

    def _weeks_tas_above_indicators(
        self, tas: xr.DataArray, year: int, threshold_temps_c: List[float]
    ) -> List[xr.DataArray]:
        """Create DataArrays containing indicators the thresholds for a single year."""
        if any(coord not in tas.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        if (tas.time.dt.year != year).any():
            raise ValueError("unexpected year found")
        # normalize to 52 weeks
        scale = 52.0 / len(tas.time)
        # will raise error if tas not present
        return [
            scale * xr.where(tas > (273.15 + threshold_c), 1.0, 0.0).sum(dim=["time"])
            for threshold_c in threshold_temps_c
        ]

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(os.path.join(os.path.dirname(__file__), "water_temp.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="weeks_water_temp/above/{temp_c}c",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={
                "temp_c": [str(t) for t in self.threshold_temps_c],
                "gcm": list(self.gcms),
            },
            path="chronic_heat/osc/v2/weeks_water_temp_above_{temp_c}c_{gcm}_{scenario}_{year}",
            display_name="Weeks with average temperature above {temp_c}°C/{gcm}",
            description=description,
            display_groups=[
                "Weeks with average temperature above"
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
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                path="weeks_tas_above_{temp_c}c_{gcm}_{scenario}_{year}_map",
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
