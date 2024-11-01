import os
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Dict, Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import rasterio  # type: ignore
import rasterio.enums  # type: ignore
import xarray as xr
from dask.distributed import Client
from fsspec.spec import AbstractFileSystem  # type: ignore
from rasterio.crs import CRS  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import WriteDataArray
from hazard.utilities.map_utilities import transform_epsg4326_to_epsg3857


@dataclass
class BatchItem:
    model: HazardResource  # type of hazard
    csv_filename: str
    jupiter_array_name: str


class JupiterOscFileSource:
    def __init__(self, dir: str, fs: AbstractFileSystem):
        """Source to load data set provided by Jupiter Intelligence for use by OS-Climate
        to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
        The data is provided as a set of csv files.

        Args:
            dir (str): Directory containing OSC_Distribution; path to files are
            e.g. {dir}/OSC_Distribution/OS-C-DATA/OS-C Tables/etlfire.csv.
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
        df = pd.read_csv(
            os.path.join(
                self._dir, "OSC_Distribution", "OS-C-DATA", "OS-C Tables", csv_filename
            )
        )
        ids = [c for c in df.columns if c not in ["key", "latitude", "longitude"]]
        df_pv = df.pivot(index="latitude", columns="longitude", values=ids)
        arrays: Dict[str, xr.DataArray] = {}
        for id in ids:
            da = xr.DataArray(data=df_pv[id], attrs={"crs": CRS.from_epsg(4326)})
            da = da.where(da.data > -9999)  # Jupiter set no-data
            arrays[id] = da
        return arrays


class Jupiter(IndicatorModel):
    """On-board data set provided by Jupiter Intelligence for use by OS-Climate
    to set up a OS-C ClimateScore API Service (“ClimateScore Service”).
    """

    _jupiter_description = """
These data should not be used in any manner relating to emergency management or planning, public safety,
physical safety or property endangerment. For higher-resolution data based on up-to-date methods,
subject to greater validation, and suitable for bottom-up risk analysis please contact
[Jupiter Intelligence](https://www.jupiterintel.com).
"""

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        csv_info = {
            "fire_probability": ("etlfire.csv", "fire{scenario}{year}metric_mean"),
            "months/spei3m/below/-2": (
                "etldrought.csv",
                "drought{scenario}{year}monthsextreme3mospeimetric_mean",
            ),
            "flooded_fraction": (
                "etlcombinedflood.csv",
                "combinedflood{scenario}{year}floodedfraction200yrmetric_mean",
            ),
            "days/above/5cm": (
                "etlhail.csv",
                "hail{scenario}{year}dayslargehailpossiblemetric_mean",
            ),
            "days/above/35c": (
                "etlheat.csv",
                "heat{scenario}{year}daysexceeding35cmetric_mean",
            ),
            "max/daily/water_equivalent": (
                "etlprecip.csv",
                "precip{scenario}{year}onedayprecip100yrmetric_mean",
            ),
            "max_speed": (
                "etlwind.csv",
                "wind{scenario}{year}windspeed100yrmetric_mean",
            ),
        }
        for model in self.inventory():
            if model.indicator_id not in csv_info:
                continue
            (csv_filename, jupiter_array_name) = csv_info[model.indicator_id]
            yield BatchItem(
                model=model,
                csv_filename=csv_filename,
                jupiter_array_name=jupiter_array_name,
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="fire_probability",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="fire/jupiter/v1/fire_probability_{scenario}_{year}",
                params={},
                display_name="Fire probability (Jupiter)",
                description=self._jupiter_description
                + """
This fire model computes the maximum monthly probability per annum of a wildfire within 100 km of
a given location based on several parameters from multiple bias corrected and downscaled Global Climate Models (GCMs).
For example, if the probability of occurrence of a wildfire is 5% in July, 20% in August, 10% in September
and 0% for other months, the hazard indicator value is 20%.
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=0.7,
                        units="",
                    ),
                    index_values=None,
                    path="fire_probability_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="Drought",
                indicator_id="months/spei3m/below/-2",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="drought/jupiter/v1/months_spei3m_below_-2_{scenario}_{year}",
                params={},
                display_name="Drought (Jupiter)",
                description=self._jupiter_description
                + """
This drought model is based on the Standardized Precipitation-Evapotranspiration Index (SPEI).
The SPEl is an extension of the Standardized Precipitation Index which also considers Potential Evapotranspiration (PET)
in determining drought events.
The SPEl is calculated from a log-logistic probability distribution function of climatic water balance
(precipitation minus evapotranspiration) over a given time scale.
The SPEI itself is a standardized variable with a mean value 0 and standard deviation 1.
This drought model computes the number of months per annum where the 3-month rolling average
of SPEI is below -2 based on the mean values of several parameters from
bias-corrected and downscaled multiple Global Climate Models (GCMs).
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=12.0,
                        units="months/year",
                    ),
                    index_values=None,
                    path="months_spei3m_below_-2_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="months/year",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="Precipitation",
                indicator_id="max/daily/water_equivalent",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="precipitation/jupiter/v1/max_daily_water_equivalent_{scenario}_{year}",
                params={},
                display_name="Precipitation (Jupiter)",
                description=self._jupiter_description
                + """
This model computes the maximum daily water equivalent precipitation (in mm) measured at the 100 year
return period based on the mean of the precipitation distribution from multiple bias corrected and
downscaled Global Climate Models (GCMs).
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=1000.0,
                        units="mm",
                    ),
                    index_values=None,
                    path="max_daily_water_equivalent_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="mm",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="Hail",
                indicator_id="days/above/5cm",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="hail/jupiter/v1/days_above_5cm_{scenario}_{year}",
                params={},
                display_name="Large hail days per year (Jupiter)",
                description=self._jupiter_description
                + """
This hail model computes the number of days per annum where hail exceeding 5 cm diameter is possible
based on the mean distribution of several parameters
across multiple bias-corrected and downscaled Global Climate Models (GCMs).
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=10.0,
                        units="days/year",
                    ),
                    index_values=None,
                    path="days_above_5cm_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="days/year",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="ChronicHeat",
                indicator_id="days/above/35c",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="chronic_heat/jupiter/v1/days_above_35c_{scenario}_{year}",
                params={},
                display_name="Days per year above 35°C (Jupiter)",
                description=self._jupiter_description
                + """
This heat model computes the number of days exceeding 35°C per annum based on the mean of distribution fits
to the bias-corrected and downscaled high temperature distribution
across multiple Global Climate Models (GCMs).
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=365.0,
                        units="days/year",
                    ),
                    index_values=None,
                    path="days_above_35c_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="days/year",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="Wind",
                indicator_id="max_speed",
                indicator_model_id="1min",
                indicator_model_gcm="unknown",
                path="wind/jupiter/v1/max_1min_{scenario}_{year}",
                params={},
                display_name="Max 1 minute sustained wind speed (Jupiter)",
                description=self._jupiter_description
                + """
This wind speed model computes the maximum 1-minute sustained wind speed (in m/s) experienced over a
100 year return period based on mean wind speed distributions
from multiple Global Climate Models (GCMs).
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
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
                    path="max_1min_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="m/s",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
            HazardResource(
                hazard_type="CombinedInundation",
                indicator_id="flooded_fraction",
                indicator_model_id=None,
                indicator_model_gcm="unknown",
                path="combined_flood/jupiter/v1/fraction_{scenario}_{year}",
                params={},
                display_name="Flooded fraction (Jupiter)",
                description=self._jupiter_description
                + """
Flooded fraction provides the spatial fraction of land flooded in a defined grid.
It is derived from higher-resolution flood hazards, and computed directly as the fraction of
cells within the 30-km cell that have non-zero flooding at that return period.
This model uses a 30-km grid that experiences flooding at the 200-year return period.
Open oceans are excluded.
                """,  # noqa:W503
                group_id="jupiter_osc",
                display_groups=[],
                map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                    bounds=[
                        (-180.0, 85.0),
                        (180.0, 85.0),
                        (180.0, -85.0),
                        (-180.0, -85.0),
                    ],
                    bbox=[-180.0, -85.0, 180.0, 85.0],
                    colormap=Colormap(
                        name="heating",
                        nodata_index=0,
                        min_index=1,
                        min_value=0.0,
                        max_index=255,
                        max_value=1.0,
                        units="",
                    ),
                    index_values=None,
                    path="fraction_{scenario}_{year}_map",
                    source="map_array",
                ),
                units="none",
                scenarios=[
                    Scenario(id="ssp126", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                    Scenario(id="ssp585", years=[2020, 2030, 2040, 2050, 2075, 2100]),
                ],
            ),
        ]

    def run_single(
        self,
        item: BatchItem,
        source: JupiterOscFileSource,
        target: WriteDataArray,
        client: Client,
    ):
        """Run a single item of the batch."""
        arrays = source.read(item.csv_filename)
        (min, max) = (float("inf"), float("-inf"))
        for scenario in item.model.scenarios:
            for year in scenario.years:
                da = arrays[
                    item.jupiter_array_name.format(scenario=scenario.id, year=year)
                ]
                da = da.reindex(
                    latitude=da.latitude[::-1]
                )  # by convention latitude reversed
                (min, max) = np.minimum(min, da.min()), np.maximum(max, da.max())  # type: ignore
                pp = PosixPath(item.model.path.format(scenario=scenario.id, year=year))  # type: ignore
                target.write(str(pp), da)
                reprojected = transform_epsg4326_to_epsg3857(
                    da.sel(latitude=slice(85, -85))
                )
                reprojected = da.sel(
                    latitude=slice(85, -85)
                ).rio.reproject(
                    "EPSG:3857", resampling=rasterio.enums.Resampling.max
                )  # , shape=da.data.shape, nodata=0) # from EPSG:4326 to EPSG:3857 (Web Mercator)
                # bounds = check_map_bounds(reprojected)
                pp_map = pp.with_name(pp.name + "_map")
                target.write(str(pp_map), reprojected)
        print(min, max)
