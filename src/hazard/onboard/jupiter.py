from dataclasses import dataclass
import os
from pathlib import PosixPath
from dask.distributed import Client
from fsspec.spec import AbstractFileSystem # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import rasterio # type: ignore
from rasterio.crs import CRS # type: ignore
import rasterio.enums # type: ignore
import rioxarray
import xarray as xr
from typing import Dict, Iterable
from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, WriteDataArray, WriteDataset
from hazard.utilities import xarray_utilities
from hazard.utilities.map_utilities import check_map_bounds, transform_epsg4326_to_epsg3857

@dataclass
class BatchItem:
    model: HazardResource # type of hazard
    csv_filename: str
    jupiter_array_name: str

class JupiterOscFileSource():
    def __init__(self, dir: str, fs: AbstractFileSystem):   
        """Source to load data set provided by Jupiter Intelligence for use by OS-Climate
        to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
        The data is provided as a set of csv files.

        Args:
            dir (str): Directory containing OSC_Distribution; path to files are e.g. {dir}/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv.
            fs (AbstractFileSystem): File system.
        """
        self._dir = dir
        self._fs = fs

    def read(self, csv_filename: str) -> Dict[str, xr.DataArray]:
        """Read Jupiter csv data and convert into a set of DataArrays. 

        Args:
            csv_filename (str): Name of csv file, e.g. etlfire.csv.

        Returns:
            Dict[str, xr.DataArray]: Data arrays, keyed by Jupiter name.
        """
        df = pd.read_csv(os.path.join(self._dir, "OSC_Distribution", "OS-C-DATA", "OS-C Tables", csv_filename))
        ids = [c for c in df.columns if c not in ["key", "latitude", "longitude"]]
        df_pv = df.pivot(index="latitude", columns="longitude", values=ids)
        arrays: Dict[str, xr.DataArray] = {}
        for id in ids:
            da = xr.DataArray(data=df_pv[id], attrs={"crs" : CRS.from_epsg(4326)})
            da = da.where(da.data > -9999) # Jupiter set no-data
            arrays[id] = da
        return arrays

class Jupiter(IndicatorModel):
    """On-board data set provided by Jupiter Intelligence for use by OS-Climate
    to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
    """
    
    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        csv_info = { 
            "fire_probability": ( "etlfire.csv", "fire{scenario}{year}metric_mean" ),
            "months/spei3m/below/-2": ( "etldrought.csv", "drought{scenario}{year}monthsextreme3mospeimetric_mean" ),
            "flooded_fraction": ( "etlcombinedflood.csv", "combinedflood{scenario}{year}floodedfraction200yrmetric_mean" ),
            "days/above/5cm": ( "etlhail.csv", "hail{scenario}{year}dayslargehailpossiblemetric_mean" ),
            "days/above/35c": ( "etlheat.csv", "heat{scenario}{year}daysexceeding35cmetric_mean" ),
            "max/daily/water_equivalent": ( "etlprecip.csv", "precip{scenario}{year}onedayprecip100yrmetric_mean" ),
            "max/1min": ( "etlwind.csv", "wind{scenario}{year}windspeed100yrmetric_mean" )
            }
        for model in self.inventory():
            if model.indicator_id not in csv_info:
                continue
            (csv_filename, jupiter_array_name) = csv_info[model.indicator_id]
            yield BatchItem(model=model, csv_filename=csv_filename, jupiter_array_name=jupiter_array_name)

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="fire_probability",
                indicator_model_gcm="unknown",
                path="fire/jupiter/v1/fire_probability_{scenario}_{year}",
                params={},
                display_name="Fire probability",
                description="""
The maximum value, found across all months, of the probability of a wildfire occurring
at some point in an individual month within 100km of the location. For example, if the probability
of occurrence of a wildfire is 5% for July, 20% in August, 10% in September and 0% for
other months, the hazard indicator value is 20%.
                """,
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=0.7,
                        units="none"),
                    array_name="fire_probability_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="Drought",
                indicator_id="months/spei3m/below/-2",
                indicator_model_gcm="unknown",
                path="drought/jupiter/v1/months_spei3m_below_-2_{scenario}_{year}",
                params={},
                display_name="Drought",
                description="""
Months per year where the rolling 3-month averaged Standardized Precipitation Evapotranspiration Index 
is below -2.
                """,
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=12.0,
                        units="months/year"),
                    array_name="months_spei3m_below_-2_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="months/year",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="Precipitation",
                indicator_id="max/daily/water_equivalent",
                indicator_model_gcm="unknown",
                path="precipitation/jupiter/v1/max_daily_water_equivalent_{scenario}_{year}",
                params={},
                display_name="Precipitation",
                description="""
Maximum daily total water equivalent precipitation experienced at a return period of 100 years.
                """, 
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",                        
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=1000.0,
                        units="mm"),
                    array_name="max_daily_water_equivalent_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="mm",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="Hail",
                indicator_id="days/above/5cm",
                indicator_model_gcm="unknown",
                path="hail/jupiter/v1/days_above_5cm_{scenario}_{year}",
                params={},
                display_name="Large hail days per year",
                description="""
Number of days per year where large hail (> 5cm diameter) is possible.
                """, 
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=10.0,
                        units="days/year"),
                    array_name="days_above_5cm_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="days/year",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="ChronicHeat",
                indicator_id="days/above/35c",
                indicator_model_gcm="unknown",
                path="chronic_heat/jupiter/v1/days_above_35c_{scenario}_{year}",
                params={},
                display_name="Days per year above 35°C",
                description="""
Maximum daily total water equivalent precipitation experienced at a return period of 200 years.
                """, 
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=365.0,
                        units="mm"),
                    array_name="days_above_35c_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="days/year",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="Wind",
                indicator_id="max/1min",
                indicator_model_gcm="unknown",
                path="wind/jupiter/v1/max_1min_{scenario}_{year}",
                params={},
                display_name="Max 1 minute sustained wind speed",
                description="""
Maximum 1-minute sustained wind speed in km/hour experienced at different return periods.
                """, 
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=120.0,
                        units="km/hour"),
                    array_name="max_1min_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="km/hour",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ]),
            HazardResource(
                hazard_type="CombinedInundation",
                indicator_id="flooded_fraction",
                indicator_model_gcm="unknown",
                path="combined_flood/jupiter/v1/fraction_{scenario}_{year}",
                params={},
                display_name="Flooded fraction",
                description="""
The fraction of land within a 30-km grid cell that experiences flooding at different return periods.
                """, 
                group_id = "jupiter_osc",
                display_groups=[],
                map = MapInfo(
                    bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=1.0,
                        units="none"),
                    array_name="fraction_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="ssp126",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(
                        id="ssp585",
                        years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    ])]
    
    def run_single(self, item: BatchItem, source: JupiterOscFileSource, target: WriteDataArray, client: Client):
        """Run a single item of the batch."""
        arrays = source.read(item.csv_filename)
        (min, max) = (float("inf"), float("-inf"))
        for scenario in item.model.scenarios:
            for year in scenario.years:
                da = arrays[item.jupiter_array_name.format(scenario=scenario.id, year=year)]
                da = da.reindex(latitude=da.latitude[::-1]) # by convention latitude reversed
                (min, max) = np.minimum(min, da.min()), np.maximum(max, da.max()) # type: ignore
                pp = PosixPath(item.model.path, item.model.path.format(scenario=scenario.id, year=year)) # type: ignore
                target.write(str(pp), da)
                reprojected = transform_epsg4326_to_epsg3857(da.sel(latitude=slice(85, -85)))
                reprojected = da.sel(latitude=slice(85, -85)).rio.reproject("EPSG:3857", resampling=rasterio.enums.Resampling.max) #, shape=da.data.shape, nodata=0) # from EPSG:4326 to EPSG:3857 (Web Mercator)
                bounds = check_map_bounds(reprojected)
                pp_map = pp.with_name(pp.name + "_map")
                target.write(str(pp_map), reprojected)
        print(min, max)
