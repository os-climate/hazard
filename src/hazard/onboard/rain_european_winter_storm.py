# Final version

import logging
import os
from pathlib import Path, PurePath
from typing import List, Optional, Sequence, Tuple

from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import curve_fit
import xarray as xr

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.utilities.download_utilities import download_file


logger = logging.getLogger(__name__)


class RAINEuropeanWinterStorm:
    def __init__(
        self,
        source_dir: str,
        fs: Optional[AbstractFileSystem] = None,
        scenarios: Sequence[str] = ["historical", "rcp4p5", "rcp8p5"],
        years: Sequence[int] = [2035, 2085],
        return_periods: Sequence[float] = [5, 10, 20, 50, 100, 200, 500],
    ):
        """_summary_

        METADATA:
        Link: https://data.4tu.nl/datasets/6bd24d17-6873-4d43-9355-fe2867a2e0d0 # noqa: E501
        Data type: historical and scenario return period
        Hazard indicator: sustained 10 metre wind speed
        Region: Pan-Europe
        Resolution:
        Return periods: 5, 10, 20, 50
        Scenarios: RCP 4.5, 8.5
        Time range: 1970-2000, 2020-2050, 2070-2100
        File type: Map (.nc)

        Args:
            source_dir (str): Source directory path.
            fs (Optional[AbstractFileSystem], optional): File system. Defaults to None in which case the source
                is taken to be a local file system (as opposed to S3 file system).
            scenarios (Sequence[str], optional): _description_. Defaults to ["historical", "rcp4p5", "rcp8p5"].
            years (Sequence[int], optional): _description_. Defaults to [2035, 2085].
            return_periods (Sequence[float], optional): _description_. Defaults to [5, 10, 20, 50, 100, 200, 500].

        Returns:
            _type_: _description_
        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = PurePath(source_dir)
        self.return_periods = np.array(return_periods)
        self.base_periods = np.array([5, 10, 20, 50])
        self.p = 1 / self.base_periods
        self.scenarios = scenarios
        self.years = years
        self.year_lookup = {-1: "1970-2000", 2035: "2020-2050", 2085: "2070-2100"}
        self.scenario_lookup = {
            "rcp4p5": "RCP45",
            "rcp8p5": "RCP85",
            "historical": "historical",
        }
        self.returns_lookup = {5: "5yr", 10: "10yr", 20: "20yr", 50: "50yr"}
        self.file_types = ["return_level", "probability_change"]
        self.resource = self._hazard_resource()

    def prepare(self, working_dir: str):
        """Download required files to working_dir.
        If using LocalFileSystem (default), this can simply be self.source_dir.

        Args:
            working_dir (str): working directory.
        """
        url = "https://opendap.4tu.nl/thredds/fileServer/data2/uuid/fbf3f7ba-d7e7-4f67-b4b4-19838251db10/{filename}"

        Path(working_dir).mkdir(parents=True, exist_ok=True)
        for scenario in self.scenario_lookup:
            file_type = (
                "return_level" if scenario == "historical" else "probability_change"
            )
            for year in [-1] if scenario == "historical" else self.years:
                for ret in self.returns_lookup:
                    filename = f"wind_speed_{file_type}_{self.returns_lookup[ret]}_{self.scenario_lookup[scenario]}_{self.year_lookup[year]}.nc"
                    logger.info(
                        f"Downloading {filename} from url {url.format(filename=filename)}"
                    )
                    filename_ret = download_file(
                        url.format(filename=filename),
                        directory=working_dir,
                        filename=filename,
                    )
                    assert filename_ret == filename
                    logger.info(f"Downloaded {filename}")

    def run_all(self, target: ReadWriteDataArray):
        """This method runs through all processes for both scenario and historical data sets.
        It will run through 2035/2085 for scenario, but only 1970-2000 for historical.
        Outputs a data array for each combination (5 in total).
        """
        # use EPSG 3035
        dataarrays = self.dataarrays_for_returns("historical", -1)
        params = self.gev_fit_params(dataarrays)
        for scenario in self.scenarios:
            if scenario == "historical":
                self.run_historical(target, params)
            else:
                for year in self.years:
                    self.run_scenario(target, scenario, year, params)

    def run_historical(
        self,
        target: ReadWriteDataArray,
        gev_params: List[List[Tuple[float, float, float]]],
    ):
        """This section completes all processes (extrapolation, interpolation, data array production) on historical data.
        Historical data exists as absolute wind speed return levels for a given return period.
        run_historical() is different to run_scenario() as it does not require conversion from relative data.

        It goes through the following steps:
        - historical datasets are loaded;
        - data for return period 5 to 50 are used to extrapolate up to 500 year (extrapolated), using a curve fit function and GEV_fit();
        - the newly extrapolated data is joined to the 5 to 50 year data to create a list of data sets (combined);
        - this data is interpolated to produce finer resolution points for each return period (interpolated);
        - this function also arranges the list of data sets into a single data array.

        Writes data array with dimensions lat, lon, index. Index describes the return period for each data set within the data array.
        Holds windspeed data.
        """
        scenario = "historical"
        year = -1
        logger.info("Processing historical scenario")
        dataarrays = self.dataarrays_for_returns(scenario, year)
        logger.info("Extrapolating returns")
        extrapolated = self.extrapolate(dataarrays, gev_params)
        combined = dataarrays + extrapolated
        logger.info("Interpolating (reprojection)")
        result = self.interpolate(combined)
        logger.info("Saving result")
        target.write(
            self.resource.path.format(scenario="historical", year=1985), result
        )

    def run_scenario(
        self,
        target: ReadWriteDataArray,
        scenario,
        year,
        gev_params: List[List[Tuple[float, float, float]]],
    ):
        """
        This section completes all processes (conversion, extrapolation, interpolation) on scenario data.
        Scenario data exist as increase/decrease in exceedence probability relative to the hisotrical set.
        There is RCP 4.5 and RCP 8.5 for both 2035 and 2085.
        run_scenario() is different to run_historical() as it requires conversion from relative data.

        It goes through the following steps:
        - historical data is loaded
        - scenario data is loaded
        - scenario data is converted from relative probabilities to absolute wind speeds through comparison with historical data (historical, exceed_datasets, converted).
        - scenario data now has the same format at historical data so it is treated the same.
        - data for return period 5 to 50 are used to extrapolate up to 500 year (extrapolated), using a curve fit function and GEV_fit().
        - the newly extrapolated data is joined to the 5 to 50 year data to create a list of data sets (combined).
        - this data is interpolated to produce finer resolution points for each return period (interpolated).
        - this function also arranges the list of data sets into a single data array.

        Args:
            scenario (string): This defines the scenario we want to calculate for ie Historical, RCP 4.5, RCP 8.5. This will only run through RCP4.5, RCP8.5.
            year (integer): This defines the year we want to calculate for ie 2035, 2085.
        """
        # historical = self.get_historical()
        logger.info(f"Processing scenario {scenario} and year {year}")
        logger.info("Calculating returns from probability change.")
        exceed_datasets = self.dataarrays_for_returns(scenario, year)
        converted = self.convert_datasets(exceed_datasets, gev_params)
        logger.info("Extrapolating returns")
        extrapolated = self.extrapolate(converted, gev_params)
        combined = converted + extrapolated
        logger.info("Interpolating (reprojection)")
        result = self.interpolate(combined)
        logger.info("Saving result")
        target.write(self.resource.path.format(scenario=scenario, year=year), result)

    def dataarrays_for_returns(self, scenario: str, year: int):
        """Get list of DataArrays for scenario and year, one for each return period."""
        file_type = "return_level" if scenario == "historical" else "probability_change"
        result = [
            xr.open_dataset(
                self.source_dir
                / f"wind_speed_{file_type}_{r}_{self.scenario_lookup[scenario]}_{self.year_lookup[year]}.nc"
            )[f"sfcWindmax_{r}_return_level"]
            for r in ["5yr", "10yr", "20yr", "50yr"]
        ]
        return result

    def get_historical(self):
        result = [
            xr.open_dataset(
                self.source_dir / f"wind_speed_return_level_{r}_historical_1970-2000.nc"
            )[f"sfcWindmax_{r}_return_level"]
            for r in ["5yr", "10yr", "20yr", "50yr"]
        ]
        return result

    def convert_datasets(
        self, exceed_datasets, gev_params: List[List[Tuple[float, float, float]]]
    ):
        """
        Two types of datasets are used in this document - historical and scenario.
        The historical data has wind speed return level data stored as real values.
        The scenario data exists as changes in probability relative to the historical values.
        Therefore, we need to convert scenario data from relative to absolute values so we can carry on further processes.

        In this, for every pixel (103x106 here) we produce a GEV curve (defined in GEV_fit()) fitted to historical data, then interpolate along the curve to give the new scenario point.
        This is done by using the new value of 'p', given by the change in p recorded in the original relative scenario data.

        Args:
            exceed_datasets (dataset): Datasets holding wind speed return level data as 'mean change of exceedance probability' relative to historical values.
            historical (dataset): Datasets holding wind speed return level data (ie speed of a 1 in 50 year storm, for p=1/50)

        Returns:
            converted_datasets: The scenario datasets with newly calculated absolute wind speed return values, as opposed to values relative to historical probabilities.
        """
        # Create empty datasets for newly calculated absolute return value data
        ds5_abs = xr.zeros_like(exceed_datasets[0])
        ds10_abs = xr.zeros_like(exceed_datasets[1])
        ds20_abs = xr.zeros_like(exceed_datasets[2])
        ds50_abs = xr.zeros_like(exceed_datasets[3])

        # With this dataset, we expect dimensions y-103 and x-106
        for i in range(exceed_datasets[0].y.size):
            for j in range(exceed_datasets[0].x.size):
                [mu, xi, sigma] = gev_params[i][j]

                ds5_p_change_value = exceed_datasets[0][0][i, j].data
                ds10_p_change_value = exceed_datasets[1][0][i, j].data
                ds20_p_change_value = exceed_datasets[2][0][i, j].data
                ds50_p_change_value = exceed_datasets[3][0][i, j].data

                x_5 = self.gev_fit(0.2 + ds5_p_change_value, mu, xi, sigma)
                x_10 = self.gev_fit(0.1 + ds10_p_change_value, mu, xi, sigma)
                x_20 = self.gev_fit(0.05 + ds20_p_change_value, mu, xi, sigma)
                x_50 = self.gev_fit(0.02 + ds50_p_change_value, mu, xi, sigma)

                ds5_abs[0][i, j] = x_5
                ds10_abs[0][i, j] = x_10
                ds20_abs[0][i, j] = x_20
                ds50_abs[0][i, j] = x_50

        converted_datasets = [ds5_abs, ds10_abs, ds20_abs, ds50_abs]
        return converted_datasets

    def gev_fit(self, p, mu, xi, sigma):
        x_p = mu - (sigma / xi) * (1 - (-np.log(1 - p)) ** (-xi))
        return x_p

    def extrapolate(self, datasets, gev_params: List[List[Tuple[float, float, float]]]):
        """
        This method takes DataArrays which hold wind speed return values (absolute values, not relative), and extend return periods from 5-50 to 100-500.
        This is done by fitting each pixel to a GEV curve (defined in GEV_fit()) and then using this curve to calculate return values for p=1/100, 1/200, 1/500.
        This results in artificially high return period datasets.

        Args:
            datasets (dataset): These original datasets contain wind speed return values for 5-year to 50-year return periods. They form the basis to extrapolate to 500 years.

        Returns:
            new_datasets: The freshly produced artificial datasets which hold wind speed return values extrapolated for 100, 200, and 500 year return periods.
        """
        # Create empty datasets for newly calculated return value data
        ds100 = xr.zeros_like(datasets[0])
        ds200 = xr.zeros_like(datasets[0])
        # ds500 = xr.zeros_like(datasets[0])

        # Produce fitted and predicted values for 100/200/500 return period datasets
        # With this dataset, we expect dimensions y-103 and x-106
        for i in range(datasets[0].y.size):
            for j in range(datasets[0].x.size):
                [mu, xi, sigma] = gev_params[i][j]

                x_100 = self.gev_fit(1 / 100, mu, xi, sigma)
                x_200 = self.gev_fit(1 / 200, mu, xi, sigma)
                # x_500 = self.GEV_fit(1/500, mu, xi, sigma)

                ds100[0][i, j] = x_100
                ds200[0][i, j] = x_200
                # ds500[0][i,j] = x_500

        new_dataarrays = [ds100, ds200]  # , ds500]
        return new_dataarrays

    def gev_fit_params(self, datasets):
        logger.info("Calculating GEV fit")
        gev_params = [[None] * datasets[0].x.size for _ in range(datasets[0].y.size)]
        for i in range(datasets[0].y.size):
            logger.info(f"Row {i}/{datasets[0].y.size}")
            for j in range(datasets[0].x.size):
                ds5_value = datasets[0][0][i, j].data
                ds10_value = datasets[1][0][i, j].data
                ds20_value = datasets[2][0][i, j].data
                ds50_value = datasets[3][0][i, j].data
                return_levels = [ds5_value, ds10_value, ds20_value, ds50_value]
                params = curve_fit(self.gev_fit, self.p, return_levels)
                gev_params[i][j] = params[0]
        return gev_params

    def interpolate(self, da_returns_list):
        """
        This method interpolates data arrays onto a uniform latitude/longitude grid.
        Usually a geospatial specific algorithm (e.g. rasterio reproject) would be used, but in this case the original
        data set contains 2 dimensional latitude and longitude grids. That is, the problem is one of interpolation
        using irregular grid for which SciPy routines are used.

        Args:
            da_returns_list (dataset): This makes up a full dataset per scenario with the original 5-50 year return periods
            with newer extrapolated 100-500 year periods.

        Returns:
            da_comb: All 'slices' of return values for different return periods are put together in one data array.
                    The data array has dimension lat, lon, and index. Index is the return period (ie 5, 10, 20 etc.)
        """
        interpolated_das = []
        returns = [5, 10, 20, 50, 100, 200]  # 500
        assert len(returns) == len(da_returns_list)
        # check original resolution
        # min_delta_lat = (combined[0].lat[1:, :] - combined[0].lat.min().data).min().data
        # min_delta_lon = (combined[0].lon[:, 1:] - combined[0].lon.min().data).min().data

        for i in range(len(returns)):
            min_lat = da_returns_list[i].lat.min().data
            min_lon = da_returns_list[i].lon.min().data
            max_lat = da_returns_list[i].lat.max().data
            max_lon = da_returns_list[i].lon.max().data

            lon_array = da_returns_list[i].lon.to_numpy()
            assert lon_array.size == 10918
            np.reshape(lon_array, lon_array.size)

            lat_array = da_returns_list[i].lat.to_numpy()
            np.reshape(lat_array, lon_array.size)

            windspeed_array = da_returns_list[i][0].to_numpy()
            np.reshape(windspeed_array, lon_array.size)

            x = np.linspace(min_lon, max_lon, 500)
            y = np.linspace(min_lat, max_lat, 500)
            x, y = np.meshgrid(x, y)  # 2D grid for interpolation

            # we use the Delaunay triangulation for linear interpolation, but check interpolated points at the edges:
            # these are excluded using NearestNDInterpolator if distance to nearest neighbour > 1.
            interp_near = NearestNDInterpolator(
                list(
                    zip(
                        np.reshape(lon_array, lon_array.size),
                        np.reshape(lat_array, lon_array.size),
                    )
                ),
                np.reshape(windspeed_array, lon_array.size),
            )
            interp = LinearNDInterpolator(
                list(
                    zip(
                        np.reshape(lon_array, lon_array.size),
                        np.reshape(lat_array, lon_array.size),
                    )
                ),
                np.reshape(windspeed_array, lon_array.size),
            )
            mask = np.isnan(interp_near(x, y, distance_upper_bound=1.0))
            z = interp(x, y)
            z[mask] = np.nan

            da = xr.DataArray(
                data=z, coords={"lat": y[:, 0], "lon": x[0, :]}, dims=["lat", "lon"]
            )
            interpolated_das.append(da)

        # Creating a combined data array with all data
        # We currently have slices but we want a cube
        combined = np.zeros((len(returns), 500, 500))

        for i in range(len(returns)):
            combined[i, :, :] = interpolated_das[i]
        da_comb = xr.DataArray(
            data=combined,
            coords={"index": returns, "lat": y[:, 0], "lon": x[0, :]},
            dims=["index", "lat", "lon"],
        )

        return da_comb

    def _hazard_resource(self) -> HazardResource:
        with open(
            os.path.join(os.path.dirname(__file__), "rain_european_winter_storm.md"),
            "r",
        ) as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="Wind",
            indicator_id="max_speed",
            indicator_model_id=None,
            indicator_model_gcm="combined",
            path="wind/rain_proj/v1/max_speed_{scenario}_{year}",
            params={},
            display_name="Max wind speed (RAIN)",
            description=description,
            group_id="iris_osc",
            display_groups=[],
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                bounds=[],
                bbox=[],
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_index=255,
                    max_value=120.0,
                    units="m/s",
                ),
                index_values=None,
                path="maps/wind/rain_proj/v1/max_speed_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="m/s",
            scenarios=[
                Scenario(id="historical", years=[1985]),
                Scenario(id="rcp4p5", years=[2035, 2085]),
                Scenario(id="ssp8p5", years=[2035, 2085]),
            ],
        )
        return resource
