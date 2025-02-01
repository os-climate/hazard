
import glob
import logging
import os
import math
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings
import zipfile

import cdsapi
import geopandas as gpd
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
import numpy as np
from rasterio import features
from rasterio.enums import MergeAlg
import urllib3
import xarray as xr

#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
import zarr

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_and_unzip
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import (
    empty_data_array,
    global_crs_transform,
)

logger = logging.getLogger(__name__)


class WISCWinterStormEventSource(OpenDataset):
    def __init__(self, source_dir: str, fs: Optional[AbstractFileSystem] = None):
        """Source that can combined WISC 
        https://cds.climate.copernicus.eu/datasets/sis-european-wind-storm-synthetic-events?tab=overview

        https://cds.climate.copernicus.eu/how-to-api

        https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1017/S1350482799001103
        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
            If None, a LocalFileSystem is used.
        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.years = [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
                 2006, 2007, 2008, 2009, 2010, 2011]
        
        #synth_sets = [1.2, 2.0, 3.0]
        self.synth_set = 1.2

    def prepare_source_files(self, working_dir: Path):
        self._download_all(working_dir)
        self._extract_all(working_dir)

    def _download_all(self, working_dir: Path):
        try:
            client = cdsapi.Client(sleep_max=5, timeout=240, retry_max=50)
            synth_set = 1.2
            for i, year in enumerate(self.years):
                logger.info(f"downloading year {i + 1} / len(years) for synth set {synth_set}")
                target = working_dir / f"{year}_data_{synth_set}.zip"
                client.retrieve(
                    "sis-european-wind-storm-synthetic-events",
                    {
                        "format": "zip",
                        "version_id": "synthetic_set_2_0",
                        "year": f"{year}",
                        "month": ["01", "02", "03", "04", "05", "09", "10", "11", "12"],
                        "variable": "wind_speed_of_gusts",
                    },
                    target)
        except:
            logger.exception("Failed to download. Check https://cds.climate.copernicus.eu/how-to-api for setting up account and credentials.")
        
    def _extract_all(self, working_dir: Path):
        for year in self.years:
            set_name = f"{year}_data_{self.synth_set}"
            with zipfile.ZipFile(str(working_dir / (set_name + ".zip")), "r") as zip_ref:
                zip_ref.extractall(str(working_dir/ str(self.synth_set).replace(".", "_") / str(year)))

    def _upload_to_s3(self, working_dir: Path, s3_path: PurePosixPath):
        ...

    def occurrence_exceedance_count(self):
        wind_speeds = np.concatenate([np.arange(5, 29, 1), np.arange(28.5, 40, 0.5), np.arange(40, 50), np.arange(50, 105, 5)])
        
        first_file = self._file_list(self.years[0])[0]
        first = xr.open_dataset(first_file)
        data = np.zeros((len(wind_speeds), len(first.lat), len(first.lon)))
        # exceedance count is the number of events where maximum wind speed exceeds the threshold
        # over all years in the sets
        exceedance_count = xr.DataArray(data=data,
                        coords={"wind_speed": wind_speeds, "lat": first.lat, "lon": first.lon},
                        dims = ['wind_speed', 'lat', 'lon'])
        for year in self.years:
            files = self._file_list(year)
            logger.info(f"Processing {len(files)} events for year {year}")

            for i, f in enumerate(files):
                if i % 50 == 0:
                    logger.info(f"Event {i} of year {year}")
                with xr.open_dataset(f) as ds:
                    if not np.array_equal(ds.lat, first.lat) or not np.array_equal(ds.lon, first.lon): 
                        raise ValueError("spatial dimensions not aligned")
                    for i, speed in enumerate(wind_speeds):
                        count = xr.where(ds.wind_speed_of_gust > speed, 1.0, 0.0)
                        exceedance_count[i, :, :] += count
        return exceedance_count    
    
    def peak_annual_gust_speed(self, working_dir: Path):
        first_file = self._file_list(self.years[0])[0]
        all_files = [f for y in self.years for f in self._file_list(y)]
        first = xr.open_dataset(first_file)
        transform = first.rio.transform()

        n_years = 5 * 26
        target = empty_data_array(first.lon.size, first.lat.size,
                                  transform=transform,
                                  crs=str(first.attrs.get("crs", "EPSG:4326")),
                                  index_name="year",
                                  index_values=list(np.arange(n_years)),
                                  chunks=[n_years, 100, 100])

        events_per_ensemble = np.zeros(5)
        for f in all_files:
            ensemble_index = int(Path(all_files[0]).name.split("_")[6][-1]) - 1
            events_per_ensemble[ensemble_index] = events_per_ensemble[ensemble_index] + 1

        for year_set_index, y in enumerate(self.years):
            for f in self._file_list(y):
                ensemble_index = int(Path(all_files[0]).name.split("_")[6][-1]) - 1
                year_index = year_set_index * 5 + ensemble_index
                with xr.open_dataset(f) as eds:
                    target[year_index, :, :] = np.maximum(target[year_index, :, :], eds[:, :])

        # store = zarr.DirectoryStore(Path(working_dir) / "temp")

        # #https://docs.xarray.dev/en/stable/user-guide/io.html
        # # distributed writes
        # ds = target.to_dataset(name="event_wind_speed")
        # ds.to_zarr(store=store,
        #            group="event_wind_speed",
        #            compute=False,
        #            mode="w",
        #            encoding={"event_wind_speed" : {
        #                 "chunks" : (1, 100, 100), # n_events
        #                 "write_empty_chunks": False,
        #             }
        #         })
        # i = 0
        # for file in all_files:
        #     if i % 50 == 0:
        #         logger.info(f"Event {i} of {len(all_files)}")

                #partial.to_zarr(store=store, 
                #    group="event_wind_speed",
                #    region = { "event_index": slice(i, i + 1), 
                #            "latitude": slice(0, partial.latitude.size),
                #            "longitude": slice(0, partial.longitude.size)  }
                #)

    def _gev_cdf(self, x, mu, xi, sigma):
        exponent = -(1 + xi * ((x - mu) / sigma))**(-1 / xi)
        return np.exp(exponent)

    def _gev_icdf(self, p, mu, xi, sigma):
        x_p = mu - (sigma / xi) * (1 - (-np.log(1 - p))**(-xi))
        return x_p

    def fit(self, exceedance_count: xr.DataArray, p_tail: float = 1/5):
        return_periods = np.array([5, 10, 20, 50, 100, 200, 500])     
        data = np.zeros((len(return_periods), exceedance_count.lat.size, exceedance_count.lon.size))
        fitted_speeds = xr.DataArray(data=data, coords={"return_period": return_periods, 
                                                        "lat": exceedance_count.lat, "lon": exceedance_count.lon},
                                     dims = ['return_period', 'lat', 'lon'])
        # all values set as nan so they display well if cannot be calculated later
        fitted_speeds[:, :, :] = float('nan')
        
        n_events = 7660
        n_years = 26 * 5
        wind_speed = exceedance_count.wind_speed.data
        probs = (1 / return_periods)
        # 1-in-10 year events should happen on average 0.1 * 26 * 5 = 13 times with 5 ensembles of 26 years each.
        # 13 events represents the top 13/7660 = 0.17% of the event set; this gives an idea where the 'tail' starts.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(exceedance_count.lat.size):
                if j % 1 ==0:
                    percent = str(round(j * 100 / exceedance_count.lat.size, 2))
                    logger.info(f"{percent}% complete")
            
                for i in range(exceedance_count.lon.size):
                    #set_exceed_prob = exceedance_count[:, j, i].data / n_events

                    exceed_prob = exceedance_count[:, j, i].data / n_years

                    condition = (exceed_prob > 0) & (exceed_prob <= p_tail)
                    # we calculate parameters using the occurrence exceedance probability and associated wind speed
                    # Note it is 1-prob as our probs range from 0 -> 0.05 but in a CDF probs increase always therefore we need 0.95 -> 100 
                    try:
                        cdf_params = curve_fit(self._gev_icdf, exceed_prob[condition], wind_speed[condition])
                        #cdf_params = curve_fit(self._gev_cdf, wind_speed[condition], 1 - set_exceed_prob[condition], 
                        #                       maxfev=800) # p0 = [40, 2, 0.0001]
                        [mu, xi, sigma] = cdf_params[0]
                        fitted_speeds[:, j, i] = self._gev_icdf(probs, mu, xi, sigma)
                    except Exception as e:
                        fitted_speeds[:, j, i] = float("nan")
        return fitted_speeds

    def _file_list(self, year: int):
        return glob.glob(str(Path(self.source_dir) / str(self.synth_set).replace(".", "_") / str(year) / "*.nc"))

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> xr.Dataset:
        raise NotImplementedError()