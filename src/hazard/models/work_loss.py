"""Calculate work loss due to chronic heat exposure based on Wet-Bulb Globe Temperature (WBGT) and climate data from multiple Global Circulation Models (GCMs) and climate scenarios."""

import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing_extensions import Iterable, List

import numpy as np
import xarray as xr

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import Indicator, MultiYearAverageIndicatorBase
from hazard.protocols import OpenDataset
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource


logger = logging.getLogger(__name__)


@dataclass
class WorkLossBatchItem:
    """Represents a batch item for work loss calculations.

    Attributes
        resource (HazardResource): The associated hazard resource.
        gcm (str): The global climate model identifier.
        scenario (str): The climate scenario.
        central_year (int): The central year for the batch item.

    """

    resource: HazardResource
    gcm: str
    scenario: str
    central_year: int

    def __str__(self):
        """Return a string representation of the WorkLossBatchItem."""
        return f"gcm={self.gcm}, scenario={self.scenario}, central_year={self.central_year}"


class WorkLossIndicator(MultiYearAverageIndicatorBase[WorkLossBatchItem]):
    """Represents an indicator for work loss due to chronic heat.

    This class extends tocompute `MultiYearAverageIndicatorBase` work loss based on wet-bulb globe temperature (WBGT) and climate data.
    """

    def __init__(
        self,
        window_years=MultiYearAverageIndicatorBase._default_window_years,
        gcms=MultiYearAverageIndicatorBase._default_gcms,
        scenarios=MultiYearAverageIndicatorBase._default_scenarios,
        central_year_historical=MultiYearAverageIndicatorBase._default_central_year_historical,
        central_years=MultiYearAverageIndicatorBase._default_central_years,
    ):
        """Initialize the work loss indicator.

        Args:
            window_years (Iterable[int], optional): Window years for averaging.
            gcms (Iterable[str], optional): List of general circulation models (GCMs).
            scenarios (Iterable[str], optional): List of climate scenarios.
            central_year_historical (int, optional): Historical reference year.
            central_years (Iterable[int], optional): Future central years.

        """
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.alpha_light = (32.98, 17.81)
        self.alpha_medium = (30.94, 16.64)
        self.alpha_heavy = (24.64, 22.72)
        self.resource = self._resource()

    def batch_items(self) -> Iterable[WorkLossBatchItem]:
        """Get batch items (batch items can be calculated independently from one another)."""
        resource = self._resource()
        for gcm in self.gcms:
            for scenario in self.scenarios:
                central_years = (
                    [self.central_year_historical]
                    if scenario == "historical"
                    else self.central_years
                )
                for central_year in central_years:
                    yield WorkLossBatchItem(
                        resource=resource,
                        gcm=gcm,
                        scenario=scenario,
                        central_year=central_year,
                    )

    def inventory(self) -> Iterable[HazardResource]:
        """Retrieve the inventory of hazard resources.

        Returns
            Iterable[HazardResource]: A list containing the hazard resource.

        """
        return [self._resource()]

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        ...
        create_tiles_for_resource(source, target, self.resource)

    def _resource(self) -> HazardResource:
        with open(os.path.join(os.path.dirname(__file__), "work_loss.md"), "r") as f:
            description = f.read().replace("\u00c2\u00b0", "\u00b0")
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="mean_work_loss/{intensity}",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"intensity": ["low", "medium", "high"], "gcm": list(self.gcms)},
            path="chronic_heat/osc/v2/mean_work_loss_{intensity}_{gcm}_{scenario}_{year}",
            display_name="Mean work loss, {intensity} intensity/{gcm}",
            description=description,
            display_groups=["Mean work loss"],  # display names of groupings
            # we want "Mean work loss" -> "Low intensity", "Medium intensity", "High intensity" -> "GCM1", "GCM2", ...
            license="Creative Commons",
            source="",
            version="",
            group_id="",
            resolution="1800 m",
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=0.8,
                    max_index=255,
                    units="fractional loss",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 85.0],
                index_values=None,
                path="maps/chronic_heat/osc/v2/mean_work_loss_{intensity}_{gcm}_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="fractional loss",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: WorkLossBatchItem, year: int
    ) -> List[Indicator]:
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            hurs = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "hurs", year)
            ).hurs
            results = self._work_loss_indicators(tas, hurs)
        resource = item.resource
        paths = [
            item.resource.path.format(
                intensity=intensity,
                gcm=item.gcm,
                scenario=item.scenario,
                year=item.central_year,
            )
            for intensity in resource.params["intensity"]
        ]
        assert isinstance(resource.map, MapInfo)
        result = [
            Indicator(
                array=array, path=PurePosixPath(paths[i]), bounds=resource.map.bounds
            )
            for i, array in enumerate(results)
        ]
        logger.info(f"Calculation complete for year {year}")
        return result

    def _work_loss_indicators(
        self, tas: xr.DataArray, hurs: xr.DataArray
    ) -> List[xr.DataArray]:
        tas_c = tas - 273.15  # convert from K to C
        # vpp is water vapour partial pressure in kPa
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
        result: List[xr.DataArray] = []
        for alpha1, alpha2 in [self.alpha_light, self.alpha_medium, self.alpha_heavy]:
            wa = 0.1 + 0.9 / (1.0 + (wbgt / alpha1) ** alpha2)  # work-ability
            wlm = 1.0 - wa.mean(dim=["time"])  # work-loss
            result.append(wlm)
        return result
