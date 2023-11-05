import logging
import os, itertools
from datetime import datetime
from typing import Iterable, Sequence
import dask.array as da

import numpy as np # type: ignore
import s3fs # type: ignore
import xarray as xr
import xclim.indices # type: ignore
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import MultiYearAverageIndicatorBase # type: ignore

from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr

logger = logging.getLogger(__name__)

class BatchItem():
    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int

class WorkingStore:
    def get_zarr_store(self):
        pass

class DroughtIndicator:
    def __init__(self,
                s3_working,
                working_source_path,
                window_years: int=MultiYearAverageIndicatorBase._default_window_years,
                gcms: Iterable[str]=MultiYearAverageIndicatorBase._default_gcms,
                scenarios: Iterable[str]=MultiYearAverageIndicatorBase._default_scenarios,
                central_years: Sequence[int]=[2005, 2030, 2040, 2050, 2080]):
        
        self.calib_start, self.calib_end = datetime(1985, 1, 1), datetime(2015, 1, 1)
        self.calc_start, self.calc_end = datetime(1985,1,1), datetime(2100,12,31)
        self.freq, self.window, self.dist, self.method = "MS", 12, "gamma", "APP"
        self.lat_min, self.lat_max = -60.0, 90.0
        self.lon_min, self.lon_max = 0, 360.0
        self.central_years = central_years
        self.spei_threshold = [0, -1, -1.5, -2, -2.5, -3, -3.6]
        self.gcms = gcms
        self.scenarios = scenarios
        self.s3_working=s3_working
        self.working_source_path=working_source_path
        self.window_years = window_years
        self.resource = self._resource()
    
    def pre_chunk(self,
        years = np.arange(1950, 2101),
        quantities = ['tas', 'pr'],
        lat_chunk_size = 40,
        lon_chunk_size = 40,
        datasource = NexGddpCmip6()):
        """Create a chunked data set for the given quantities and data source. This is for when the data
        source is either unchunked or unsuitably chunked for the calculation in hand. An SPEI index is an
        example since the calculation requires a long time series but for a limited spatial extent. The calculation
        therefore runs 

        Args:
            years (_type_, optional): Years included in chunked data. Defaults to np.arange(1950, 2101).
            variables (list, optional): Quantities included in chunked data. Defaults to ['tas', 'pr'].
            lat_chunk_size (int, optional): Latitude chunks. Defaults to 40.
            lon_chunk_size (int, optional): Longitude chunks. Defaults to 40.
            datasource (_type_, optional): Source for building chunked data. Defaults to NexGddpCmip6().
        """
        def download_dataset(variable, year, gcm, scenario, datasource = datasource):
            scenario_ = "historical" if year < 2015 else scenario
            with datasource.open_dataset_year(gcm, scenario_, variable, year) as ds_temp:
                ds = ds_temp.astype('float32').compute()
                return ds

        for variable in quantities:
            zarr_root = os.path.join(self.working_source_path, variable + "_" + self.gcm + "_" + self.scenario)
            zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3, check=False)
            for year in years:
                ds = download_dataset(variable, year, self.gcm, self.scenario).chunk({'time': 365,'lat': lat_chunk_size,'lon': lon_chunk_size})
                if year == years[0]:
                    ds.to_zarr(store=zarr_store, mode='w')
                else:
                    ds.to_zarr(store=zarr_store, append_dim='time')
                logger.info(f"completed processing: variable={variable}, year={year}.") 

    def read_quantity_from_s3_store(self, gcm, scenario, quantity, lat_min, lat_max, lon_min, lon_max) -> xr.Dataset:
        ds = self.chunked_dataset(gcm, scenario, quantity).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        return ds

    def chunked_dataset(self, gcm, scenario, quantity) -> xr.Dataset:
        zarr_root = os.path.join(self.working_source_path, quantity + "_" + gcm + "_" + scenario)
        zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3_working, check=False)
        ds = xr.open_zarr(store=zarr_store)
        return ds

    def get_datachunks(self):
        lat_delta = 10.0
        lon_delta = 10.0
        lat_bins = np.arange(self.lat_min, self.lat_max + 0.1 * lat_delta, lat_delta)
        lon_bins = np.arange(self.lon_min, self.lon_max + 0.1 * lon_delta, lon_delta)
        data_chunks = {"Chunk_" + str(i).zfill(4) : dict(list(d[0].items()) + list(d[1].items())) \
                    for i, d in enumerate(itertools.product([{'lat_min': x[0],'lat_max': x[1]} \
                    for x in zip(lat_bins[:-1], lat_bins[1:])],[{'lon_min': x[0],'lon_max': x[1]} \
                    for x in zip(lon_bins[:-1], lon_bins[1:])]))}
        return data_chunks

    def calculate_spei(self, gcm, scenario):
        """Calculate SPEI for the given GCM and scenario, storing """
        # we infer the lats and lons from the source dataset:
        ds_chunked = self.chunked_dataset(gcm, scenario, "tas")
        data_chunks = self.get_datachunks()
        ds_spei = None
        zarr_root = os.path.join(self.working_source_path, "spei", gcm + "_" + scenario)
        zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3_working, check=False)
        chunk_names = list(data_chunks.keys())
        for chunk_name in chunk_names:             
            data_chunk = data_chunks[chunk_name]
            lat_min, lat_max = data_chunk['lat_min'], data_chunk['lat_max']
            lon_min, lon_max = data_chunk['lon_min'], data_chunk['lon_max']
            ds_spei_slice = self.calculate_spei_for_slice(lat_min, lat_max, lon_min, lon_max, gcm=gcm, scenario=scenario)
            lats_all, lons_all = ds_chunked['lat'].values, ds_chunked['lon'].values
            # consider refactoring data_chunks to give both slice and values?
            lat_indexes = np.where(np.logical_and(ds_spei['lat'].values >= lat_min, ds_spei['lat'].values <= lat_max))[0]
            lon_indexes = np.where(np.logical_and(ds_spei['lon'].values >= lon_min, ds_spei['lon'].values <= lon_max))[0]
            if ds_spei is None:
                # must use deferred dask array to avoid allocating memory for whole array
                data = da.empty([len(ds_spei_slice['time'].values), len(lats_all), len(lons_all)])
                ds_spei = xr.DataArray(data=data, coords={'time': ds_spei_slice['time'].values, 'lat': lats_all,'lon': lons_all}, 
                                       dims=["time", "lat", "lon"]).chunk(chunks={'lat': 40,'lon': 40,'time': 100000}).to_dataset(name='spei')
                # compute=False to avoid calculating array
                ds_spei.to_zarr(store=zarr_store, mode='w', compute=False)
                logger.info(f"created new zarr array.") 
            else:
                # see https://docs.xarray.dev/en/stable/user-guide/io.html?appending-to-existing-zarr-stores=#appending-to-existing-zarr-stores
                ds_spei_slice.to_zarr(store=zarr_store, mode='r+', 
                                      region={"lat": slice(lat_indexes[0], lat_indexes[-1] + 1), "lon": slice(lon_indexes[0], lon_indexes[-1] + 1)})
                logger.info(f"written chunk {chunk_name} to zarr array.") 
            
    def calculate_spei_for_slice(
            self,
            lat_min, lat_max,
            lon_min, lon_max,
            *,
            gcm,
            scenario,
            num_workers = 4):
        ds_tas = self.read_quantity_from_s3_store(gcm, scenario, "tas", lat_min, lat_max, lon_min, lon_max).chunk({'time': 100000})
        ds_pr = self.read_quantity_from_s3_store(gcm, scenario, "pr", lat_min, lat_max, lon_min, lon_max).chunk({'time': 100000}) 
        ds_tas = ds_tas.drop_duplicates(dim=..., keep='last').sortby('time')
        ds_pr = ds_pr.drop_duplicates(dim=..., keep='last').sortby('time')
        ds_pet = xclim.indices.potential_evapotranspiration(tas=ds_tas['tas'], method='MB05').astype('float32').to_dataset(name='pet')
        da_wb = xclim.indices.water_budget(pr=ds_pr['pr'], evspsblpot=ds_pet['pet'])
        with xr.set_options(keep_attrs=True):
            da_wb = da_wb - 1.01 * da_wb.min()
        da_wb_calib = da_wb.sel(time=slice(str(self.calib_start)[:10], str(self.calib_end)[:10]))
        da_wb_calc = da_wb.sel(time=slice(str(self.calc_start)[:10], str(self.calc_end)[:10]))
        ds_spei = (
                xclim.indices.standardized_precipitation_evapotranspiration_index(
                                                                        da_wb_calc,
                                                                        da_wb_calib,
                                                                        freq=self.freq,
                                                                        window=self.window,
                                                                        dist=self.dist,
                                                                        method=self.method
                )
                .astype('float32')
                .to_dataset(name='spei')
                .compute(scheduler='processes', num_workers=num_workers)
        )
        logger.info(f"calculated SPEI for gcm={gcm}, lats=[{lat_min, lat_max}], lons=[{lon_min, lon_max}]")
        return ds_spei
    
    def calculate_annual_average_spei(self, gcm: str, scenario: str, central_year: int, target: OscZarr):
        """Calculate average number of months where 12-month SPEI index is below thresholds [0, -1, -1.5, -2, -2.5, -3.6]
        for 20 years period. 

        Args:
            gcm (str): Global Circulation Model ID.
            scenario (str): Scenario ID.
            year (int): Year.
            target (OscZarr): Target to write result to.
        """
        def get_spei_full_results(gcm, scenario, chunked_source_path = self.working_source_path):
            zarr_root = os.path.join(chunked_source_path, "SPEI", "Aggregated", gcm + "_" + scenario)
            zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3_working, check=False)
            ds_spei = xr.open_zarr(store=zarr_store)
            return ds_spei    

        period = [datetime(central_year - self.window_years // 2, 1, 1), datetime(central_year + self.window_years // 2-1, 12, 31)]
        print(gcm + " " + scenario + " " + str(central_year) + " period:   " + str(period[0]) + "---" + str(period[1]))
        ds_spei = get_spei_full_results(gcm, scenario)
        lats_all = ds_spei["lat"].values
        lons_all = ds_spei["lon"].values
        spei_annual = np.nan * np.zeros([len(self.spei_threshold), len(lats_all), len(lons_all)])
        spei_temp = ds_spei.sel(time=slice(period[0], period[1]))
        spei_temp = spei_temp.compute()
        spei_temp = spei_temp['spei']
        for i in range(len(self.spei_threshold)):
            spei_ext=xr.where((spei_temp <= self.spei_threshold[i]), 1, 0)
            spei_ext_sum=spei_ext.mean("time")
            spei_annual[i, :, :] = spei_ext_sum
        spei_annual_all = xr.DataArray(spei_annual, coords={"spei_idx": self.spei_threshold, "lat": lats_all, "lon": lons_all,}, dims=["spei_idx", "lat", "lon"])
        path = self.resource.path.format(gcm=gcm, scenario=scenario, year=central_year)
        target.write(path, spei_annual_all)
        return spei_annual_all   

    def run_single(self, item: BatchItem, target: OscZarr):
        self.calculate_spei(item.gcm, item.scenario)
        self.calculate_annual_average_spei(item.gcm, item.scenario, item.central_year, target)

    def _resource(self) -> HazardResource:
        #with open(os.path.join(os.path.dirname(__file__), "days_tas_above.md"), "r") as f:
        #    description = f.read()
        resource = HazardResource(
            hazard_type="AcuteDrought",
            indicator_id="months/spei12m/below",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": list(self.gcms)},
            path="acute_drought/osc/v1/months_spei12m_below_{gcm}_{scenario}_{year}",
            display_name="Drought SPEI index",
            description="",
            display_groups=["Drought SPEI index"], # display names of groupings
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
                path="acute_drought/osc/v1/months_spei12m_below_{gcm}_{scenario}_{year}_map",
                source="map_array"
            ),
            units="days/year",
            scenarios=[
                Scenario(
                    id="historical",
                    years=[self.central_years[0]]),
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

               
        
         
