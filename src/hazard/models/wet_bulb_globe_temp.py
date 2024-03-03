import logging
import os
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing import Iterable, List

import numpy as np
import xarray as xr

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (BatchItem, Indicator,
                                              ThresholdBasedAverageIndicator)
from hazard.protocols import OpenDataset
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class WetBulbGlobeTemperatureAboveIndicator(ThresholdBasedAverageIndicator):
    def __init__(
        self,
        threshold_temps_c: List[float] = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
        ],
        window_years: int = 20,
        gcms: Iterable[str] = [
            "ACCESS-CM2",
            "CMCC-ESM2",
            "CNRM-CM6-1",
            "MPI-ESM1-2-LR",
            "MIROC6",
            "NorESM2-MM",
        ],
        scenarios: Iterable[str] = [
            "historical",
            "ssp126",
            "ssp245",
            "ssp370",
            "ssp585",
        ],
        central_year_historical: int = 2005,
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
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            hurs = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "hurs", year)
            ).hurs
            results = self._days_wbgt_above_indicators(tas, hurs)
        resource = item.resource
        path = item.resource.path.format(
            gcm=item.gcm, scenario=item.scenario, year=item.central_year
        )
        assert isinstance(resource.map, MapInfo)
        result = [
            Indicator(
                results,
                PurePosixPath(path),
                item.resource.map.bounds,
            )
        ]
        logger.info(f"Calculation complete for year {year}")
        return result

    def _days_wbgt_above_indicators(
        self, tas: xr.DataArray, hurs: xr.DataArray
    ) -> List[xr.DataArray]:
        """Create DataArrays containing indicators the thresholds for a single year."""
        tas_c = tas - 273.15  # convert from K to C
        # vpp is water vapour partial pressure in kPa
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
        scale = 365.0 / len(wbgt.time)
        if any(coord not in wbgt.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        coords = {
            "index": self.threshold_temps_c,
            "lat": wbgt.coords["lat"].values,
            "lon": wbgt.coords["lon"].values,
        }
        output = xr.DataArray(coords=coords, dims=coords.keys())
        for i, threshold_c in enumerate(self.threshold_temps_c):
            output[i, :, :] = xr.where(wbgt > threshold_c, scale, 0.0).sum(dim=["time"])
        return output

    def create_maps(self, source: OscZarr, target: OscZarr):
        """
        Create map images.
        """
        create_tiles_for_resource(source, target, self._resource())

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "wet_bulb_globe_temp.md"), "r"
        ) as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="days_wbgt_above",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": self.gcms},
            path="chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}",
            display_name="Days with wet-bulb globe temperature above threshold in degrees celsius/{gcm}",
            description=description,
            display_groups=[
                "Days with wet-bulb globe temperature above threshold in degrees celsius"
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
                    units="days/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                path="maps/chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}_map",
                index_values=self.threshold_temps_c,
                source="map_array_pyramid",
            ),
            units="days/year",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp370", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource
