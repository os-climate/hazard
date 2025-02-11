import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import warnings
import zipfile

import cdsapi
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
import numpy as np
import xarray as xr

# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import data_array

logger = logging.getLogger(__name__)


class WISCWinterStormEventSource(OpenDataset):
    def __init__(self, source_dir: str, fs: Optional[AbstractFileSystem] = None):
        """Source that can create WISC return period wind speed maps from the event set:
        https://cds.climate.copernicus.eu/datasets/sis-european-wind-storm-synthetic-events?tab=overview
        https://cds.climate.copernicus.eu/how-to-api

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
            If None, a LocalFileSystem is used.
        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.years = [
            1986,
            1987,
            1988,
            1989,
            1990,
            1991,
            1992,
            1993,
            1994,
            1995,
            1996,
            1997,
            1998,
            1999,
            2000,
            2001,
            2002,
            2003,
            2004,
            2005,
            2006,
            2007,
            2008,
            2009,
            2010,
            2011,
        ]
        # synth_sets = [1.2, 2.0, 3.0]
        self.synth_set = 1.2

    def prepare_source_files(self, working_dir: Path):
        self._download_all(working_dir)
        self._extract_all(working_dir)

    def _download_all(self, working_dir: Path):
        try:
            client = cdsapi.Client(sleep_max=5, timeout=240, retry_max=50)
            synth_set = 1.2
            for i, year in enumerate(self.years):
                logger.info(
                    f"downloading year {i + 1} / len(years) for synth set {synth_set}"
                )
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
                    target,
                )
        except Exception:
            logger.exception(
                "Failed to download. Check https://cds.climate.copernicus.eu/how-to-api for setting up account and credentials."
            )

    def _extract_all(self, working_dir: Path):
        for year in self.years:
            set_name = f"{year}_data_{self.synth_set}"
            with zipfile.ZipFile(
                str(working_dir / (set_name + ".zip")), "r"
            ) as zip_ref:
                zip_ref.extractall(
                    str(working_dir / str(self.synth_set).replace(".", "_") / str(year))
                )

    def occurrence_exceedance_count(self):
        """Calculate occurrence exceedance count (i.e. number of events). Only used for diagnostic purposes.

        Returns:
            DataArray: Occurrence exceedance count.
        """
        wind_speeds = np.concatenate(
            [
                np.arange(5, 29, 1),
                np.arange(28.5, 40, 0.5),
                np.arange(40, 50),
                np.arange(50, 105, 5),
            ]
        )

        first_file = self._file_list(self.years[0])[0]
        first = xr.open_dataset(first_file)
        data = np.zeros((len(wind_speeds), len(first.lat), len(first.lon)))
        # exceedance count is the number of events where maximum wind speed exceeds the threshold
        # over all years in the sets
        exceedance_count = xr.DataArray(
            data=data,
            coords={"wind_speed": wind_speeds, "lat": first.lat, "lon": first.lon},
            dims=["wind_speed", "lat", "lon"],
        )
        for year in self.years:
            files = self._file_list(year)
            logger.info(f"Processing {len(files)} events for year {year}")

            for i, f in enumerate(files):
                if i % 50 == 0:
                    logger.info(f"Event {i} of year {year}")
                with xr.open_dataset(f) as ds:
                    if not np.array_equal(ds.lat, first.lat) or not np.array_equal(
                        ds.lon, first.lon
                    ):
                        raise ValueError("spatial dimensions not aligned")
                    for i, speed in enumerate(wind_speeds):
                        count = xr.where(ds.wind_speed_of_gust > speed, 1.0, 0.0)
                        exceedance_count[i, :, :] += count
        return exceedance_count

    def peak_annual_gust_speed(self):
        first_file = self._file_list(self.years[0])[0]
        all_files = [f for y in self.years for f in self._file_list(y)]
        first = xr.open_dataset(first_file)
        transform = first.rio.transform()

        n_years = 5 * 26
        target = data_array(
            np.zeros((n_years, first.lat.size, first.lon.size)),
            transform=transform,
            crs=str(first.attrs.get("crs", "EPSG:4326")),
            index_name="year",
            index_values=list(np.arange(n_years)),
        )

        events_per_ensemble = np.zeros(5)
        for f in all_files:
            ensemble_index = int(Path(f).name.split("_")[6][-1]) - 1
            events_per_ensemble[ensemble_index] = (
                events_per_ensemble[ensemble_index] + 1
            )

        for year_set_index, y in enumerate(self.years):
            i = 0
            file_list = self._file_list(y)
            logger.info(f"Processing {len(file_list)} events for year {y}")
            for f in file_list:
                if i % 50 == 0:
                    logger.info(f"Event {i} of {len(file_list)}")
                ensemble_index = int(Path(f).name.split("_")[6][-1]) - 1
                year_index = year_set_index * 5 + ensemble_index
                assert int(Path(f).name.split("_")[4][0:4]) == y
                with xr.open_dataset(f) as eds:
                    eds = eds.rename({"lon": "longitude", "lat": "latitude"}).compute()
                    target.data[year_index, :, :] = np.maximum(
                        target.data[year_index, :, :], eds.wind_speed_of_gust.data[:, :]
                    )
                i = i + 1
        return target

    def _gev_cdf(self, x, mu, xi, sigma):
        exponent = -((1 + xi * ((x - mu) / sigma)) ** (-1 / xi))
        return np.exp(exponent)

    def _gev_icdf(self, p, mu, xi, sigma):
        x_p = mu - (sigma / xi) * (1 - (-np.log(1 - p)) ** (-xi))
        return x_p

    def fit_gumbel(self, annual_max_gust_speed: xr.DataArray):
        """See for example "A review of methods to calculate extreme wind speeds", Palutikof et al. (1999).
        https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1017/S1350482799001103
        we fit to a Type I extreme distribution, where $X_T$ is the 1-in-T year 3s peak gust wind speed:
        $ X_T = \beta - \alpha \ln \left[ -\ln(1 - \frac{1}{T}) \right] $
        We fit wind speed $x$ to Gumbel reduced yariate, $y$.
        $ x = \alpha y + \beta $
        Here the fit is a simple unweighted least squares fit.
        Args:
            annual_max_gust_speed (xr.DataArray): Maximum annual peak gust wind speed across all storms.

        Returns:
            DataArray: 1-in-T year 3s peak gust wind speed.
        """
        return_periods = np.array([5, 10, 20, 50, 100, 200, 500])
        data = np.zeros(
            (
                len(return_periods),
                annual_max_gust_speed.latitude.size,
                annual_max_gust_speed.longitude.size,
            )
        )
        fitted_speeds = xr.DataArray(
            data=data,
            coords={
                "return_period": return_periods,
                "latitude": annual_max_gust_speed.latitude,
                "longitude": annual_max_gust_speed.longitude,
            },
            dims=["return_period", "latitude", "longitude"],
        )
        # all values set as nan so they display well if cannot be calculated later
        fitted_speeds[:, :, :] = float("nan")
        for j in range(annual_max_gust_speed.latitude.size):
            if j % 1 == 0:
                percent = str(round(j * 100 / annual_max_gust_speed.latitude.size, 2))
                logger.info(f"{percent}% complete")
            for i in range(annual_max_gust_speed.longitude.size):
                samples = annual_max_gust_speed[:, j, i].data
                sorted_samples = np.sort(samples)
                rank = np.arange(len(samples)) + 1
                a = -0.44
                b = 0.12
                cum_prob = (rank + a) / (len(rank) + b)  # unbiased plotting positions
                reduced_variate = -np.log(-np.log(cum_prob))
                cond = reduced_variate > 1
                y = reduced_variate[cond]
                x = sorted_samples[cond]
                a = np.vstack([y, np.ones(len(y))]).T
                # w = np.ones(len(x)) # diagonal of weight matrix
                # Aw = A * np.sqrt(w[:,np.newaxis])
                # wx = x * np.sqrt(w)
                alpha, beta = np.linalg.lstsq(a, x, rcond=None)[0]
                cum_prob = 1 - 1 / return_periods
                fitted_speeds[:, j, i] = alpha * -np.log(-np.log(cum_prob)) + beta
        return fitted_speeds

    def fit_gev(self, exceedance_count: xr.DataArray, p_tail: float = 1 / 5):
        """For reference, but fit_gumbel is preferred."""
        return_periods = np.array([5, 10, 20, 50, 100, 200, 500])
        data = np.zeros(
            (len(return_periods), exceedance_count.lat.size, exceedance_count.lon.size)
        )
        fitted_speeds = xr.DataArray(
            data=data,
            coords={
                "return_period": return_periods,
                "lat": exceedance_count.lat,
                "lon": exceedance_count.lon,
            },
            dims=["return_period", "lat", "lon"],
        )
        # all values set as nan so they display well if cannot be calculated later
        fitted_speeds[:, :, :] = float("nan")
        n_years = 26 * 5
        wind_speed = exceedance_count.wind_speed.data
        probs = 1 / return_periods
        # 1-in-10 year events should happen on average 0.1 * 26 * 5 = 13 times with 5 ensembles of 26 years each.
        # 13 events represents the top 13/7660 = 0.17% of the event set; this gives an idea where the 'tail' starts.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(exceedance_count.lat.size):
                if j % 1 == 0:
                    percent = str(round(j * 100 / exceedance_count.lat.size, 2))
                    logger.info(f"{percent}% complete")

                for i in range(exceedance_count.lon.size):
                    exceed_prob = exceedance_count[:, j, i].data / n_years
                    condition = (exceed_prob > 0) & (exceed_prob <= p_tail)
                    # we calculate parameters using the occurrence exceedance probability and associated wind speed
                    # Note it is 1-prob as our probs range from 0 -> 0.05 but in a CDF probs increase always therefore we need 0.95 -> 100
                    try:
                        cdf_params = curve_fit(
                            self._gev_icdf,
                            exceed_prob[condition],
                            wind_speed[condition],
                        )
                        [mu, xi, sigma] = cdf_params[0]
                        fitted_speeds[:, j, i] = self._gev_icdf(probs, mu, xi, sigma)
                    except Exception:
                        fitted_speeds[:, j, i] = float("nan")
        return fitted_speeds

    def _file_list(self, year: int):
        return glob.glob(
            str(
                Path(self.source_dir)
                / str(self.synth_set).replace(".", "_")
                / str(year)
                / "*.nc"
            )
        )

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> xr.Dataset:
        logger.info("Computing peak annual 3s gust wind speeds")
        peak_annual_gust_speed = self.peak_annual_gust_speed()
        # any deferred behaviour undesirable here
        peak_annual_gust_speed = peak_annual_gust_speed.compute()
        logger.info("Fitting peak wind speeds")
        return self.fit_gumbel(peak_annual_gust_speed).to_dataset(name="wind_speed")


class WISCEuropeanWinterStorm(IndicatorModel[str]):
    def __init__(self):
        """
        Peak 3s gust wind speed for different return periods inferred from the WISC event set.

        METADATA:
        Link: https://cds.climate.copernicus.eu/datasets/sis-european-wind-storm-synthetic-events?tab=overview
        Data type: Synthetic European winter storm events.
        Hazard indicator: Wind
        Region: Europe
        Resolution: 2.4 arcmin
        Scenarios: Historical
        Time range: Not applicable
        File type: NetCDF

        DATA DESCRIPTION:
        The WISC dataset contains a set of synthetic windstorm events consisting of 22,980 individual
        storm footprints over Europe. These are a physically realistic set of plausible windstorm
        events based on the modelled climatic conditions, calculated using the Met Office HadGEM3 model
        (Global Atmosphere 3 and Global Land 3 configurations).
        Return period maps of peak gust wind speed are inferred from this data.

        Special thanks to Annabel Hall; her work on investigating the fitting of the WISC data set is adapted hereunder.
        """

    def batch_items(self):
        """Get a list of all batch items."""
        return ["historical"]

    def run_single(
        self, item: str, source: Any, target: ReadWriteDataArray, client: Client
    ):
        assert isinstance(source, WISCWinterStormEventSource)
        logger.info("Creating data set from events")
        resource = self._resource()
        for scenario in resource.scenarios:
            for year in scenario.years:
                ds = source.open_dataset_year("", scenario.id, "", year)
                # note that the co-ordinates will be written into the parent of resource.path
                target.write(
                    resource.path.format(scenario=scenario.id, year=year),
                    ds["wind_speed"].compute(),
                    spatial_coords=resource.store_netcdf_coords,
                )

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        for resource in self.inventory():
            create_tiles_for_resource(
                source,
                target,
                resource,
                nodata_as_zero=True,
                nodata_as_zero_coarsening=True,
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return [self._resource()]

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "wisc_european_winter_storm.md"),
            "r",
        ) as f:
            description = f.read()

        return HazardResource(
            hazard_type="Wind",
            indicator_id="wind_speed/3s",
            indicator_model_id="wisc",
            indicator_model_gcm="",
            path=(
                "wind/wisc/v1/max_speed_{scenario}_{year}/max_speed"
            ),  # the double path allows an XArray-readable data array to be written
            params={},
            display_name="Max 3 second gust wind speed (WISC)",
            description=description,
            group_id="",
            display_groups=[],
            map=MapInfo(
                bbox=[],
                bounds=[],
                colormap=Colormap(
                    max_index=255,
                    min_index=1,
                    nodata_index=0,
                    name="flare",
                    min_value=0.0,
                    max_value=50.0,
                    units="years",
                ),
                index_values=None,
                path="maps/wind/wisc/v1/max_speed_{scenario}_{year}/max_speed_map",
                source="map_array_pyramid",
            ),
            units="years",
            store_netcdf_coords=True,
            scenarios=[Scenario(id="historical", years=[1999])],
        )
