

# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/rcp8p5/ipsl/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9200746/

# https://www.wri.org/data/aqueduct-water-stress-projections-data
# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/rcp8p5/ipsl/waterTemp_weekAvg_output_ipsl_rcp8p5_2006-01-07_to_2019-12-30.nc
# https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/rcp8p5/gfdl/waterTemp_weekAvg_output_gfdl_rcp8p5_2006-01-07_to_2019-12-30.nc

from contextlib import ExitStack
import logging
import os
from typing import Iterable, List
from attr import dataclass
from dask.distributed import Client
import requests # type: ignore
import xarray as xr

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import Indicator, MultiYearAverageIndicatorBase
from hazard.protocols import OpenDataset, ReadWriteDataArray

logger = logging.getLogger(__name__)

@dataclass
class BatchItem():
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
        self.from_years =    [2006, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]
        self.to_years =      [2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099]
    
    def open_dataset_year(self, gcm: str, scenario: str, quantity: str, year: int, chunks = None) -> xr.Dataset:
        set_from_year, set_to_year = next((from_year, to_year) for from_year, to_year in zip(self.from_years, self.to_years) 
                              if year >= from_year and year <= to_year)
        filename = self.filename(gcm, scenario, set_from_year, set_to_year)
        return xr.open_dataset(filename)

    def water_temp_download_path(self, from_year: int, to_year: int):
        scenario = "rcp8p5"
        gcm = "gfdl"
        filename = self.filename(gcm, scenario, from_year, to_year)
        url = f"https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/" + \
                f"original/waterTemp/{scenario}/{gcm}/" + filename
        return filename, url        

    def filename(self, gcm: str, scenario: str, from_year: int, to_year: int):
        from_date = f"{from_year}-01-07"
        to_date = f"{to_year}-12-30"
        return f"waterTemp_weekAvg_output_{gcm}_{scenario}_{from_date}_to_{to_date}.nc"

    def download_all(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        from_years =    [2006, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]
        to_years =      [2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099]
        for from_year, to_year in zip(from_years, to_years):
            filename, url = self.water_temp_download_path(from_year, to_year)
            self.download_file(url, os.path.join(self.working_dir, filename))

    def download_file(self, url, path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 4): 
                    f.write(chunk)


class WaterTemperatureIndicator(MultiYearAverageIndicatorBase[BatchItem]):
    def __init__(self,
                window_years: int=20,
                gcms: Iterable[str]=["ACCESS-CM2", "CMCC-ESM2", "CNRM-CM6-1",
                    "MIROC6", "MPI-ESM1-2-LR", "NorESM2-MM"],
                scenarios: Iterable[str]=["rcp8p5"],
                central_year_historical: int = 2005,
                central_years: Iterable[int]=[2030, 2040, 2050]):
        self.gcms = gcms


    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        ...

    def _calculate_single_year_indicators(self, source: OpenDataset, item: BatchItem, year: int) -> List[Indicator]:
        """Calculate indicators for a single year for a single batch item. If just a single indicator per batch, a list
        of length one is expected."""
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(source.open_dataset_year(item.gcm, item.scenario, "tas", year)).tas
            results = self._days_tas_above_indicators(tas, year, self.threshold_temps_c)
            

        logger.info(f"Calculation complete for year {year}")
        # Get Indicators for reach array, looking up path using "threshold" 
        return self._get_indicators(item, results, "temp_c")


    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        ...
    
    def run_single(self, item: BatchItem, source: OpenDataset, target: ReadWriteDataArray, client: Client):
        """Run a single item of the batch."""
        ...

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(os.path.join(os.path.dirname(__file__), "days_tas_above.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="days_water_temp/above/{temp_c}c",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"temp_c": [str(t) for t in self.threshold_temps_c], "gcm": list(self.gcms)},
            path="chronic_heat/osc/v2/days_water_temp_above_{temp_c}c_{gcm}_{scenario}_{year}",
            display_name="Days with average temperature above {temp_c}°C/{gcm}",
            description=description,
            display_groups=["Days with average temperature above"], # display names of groupings
            group_id = "",
            map = MapInfo( 
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=100,
                    max_index=255,
                    units="days/year"),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                path="days_tas_above_{temp_c}c_{gcm}_{scenario}_{year}_map",
                source="map_array"
            ),
            units="days/year",
            scenarios=[
                Scenario(
                    id="historical",
                    years=[self.central_year_historical]),
                Scenario(
                    id="ssp126",
                    years=list(self.central_years)),
                Scenario(
                    id="ssp245",
                    years=list(self.central_years)),
                Scenario(
                    id="ssp585",
                    years=list(self.central_years)),
                ]
        )
        return resource
    