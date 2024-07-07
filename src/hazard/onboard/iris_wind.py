import os
from dataclasses import dataclass
from typing import Iterable

import xarray as xr
from dask.distributed import Client

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import tiles


@dataclass
class BatchItem:
    resource: HazardResource  # type of hazard
    year: int
    scenario: str


class IRISIndicator(IndicatorModel[BatchItem]):
    """On-board returns data set from IRIS - Imperial College Storm Model."""

    def __init__(self, input_dir: str):
        """

        Args:
            input_dir (str): Directory containing IRIS inputs.
        """
        self.input_dir = input_dir

    def run_single(self, item: BatchItem, source, target: ReadWriteDataArray, client: Client):
        file_name = self._file_name(item.scenario, item.year)
        ds = xr.open_dataset(file_name.format(year=item.year, scenario=item.scenario))
        # dimensions: (rp: 19, latitude: 1200, longitude: 3600)
        da = OscZarr.normalize_dims(ds.vmax)
        # if the coordinates give the left, bottom of each pixel:
        da = da.assign_coords(latitude=da.latitude.data + 0.05, longitude=da.longitude.data + 0.05)
        target.write(
            item.resource.path.format(scenario=item.scenario, year=item.year),
            da,
            chunks=[len(da.index.data), 250, 250],
        )
        self.generate_single_map(item, target, target)

    def generate_single_map(self, item: BatchItem, source: ReadWriteDataArray, target: ReadWriteDataArray):
        source_path = item.resource.path.format(scenario=item.scenario, year=item.year)
        assert item.resource.map is not None
        assert isinstance(source, OscZarr) and isinstance(target, OscZarr)
        target_path = item.resource.map.path.format(scenario=item.scenario, year=item.year)
        tiles.create_tile_set(source, source_path, target, target_path, check_fill=True)
        # tiles.create_image_set(source, source_path, target, target_path)

    def batch_items(self) -> Iterable[BatchItem]:
        """Get a list of all batch items."""
        resource = self._hazard_resource()
        for scenario in resource.scenarios:
            for year in scenario.years:
                yield BatchItem(resource, year, scenario.id)

    def _file_name(self, scenario: str, year: int):
        # file name for 0.1 degree resolution sets
        # including radiative forcing (1.9 to 8.5 W/m^2): SSP1 = SSP1-1.9, SSP2 = SSP2-4.5, SSP5 = SSP5-8.5
        if scenario == "historical":
            return os.path.join(
                self.input_dir,
                "wind",
                "IRIS",
                "return_value_maps",
                "IRIS_vmax_maps_PRESENT_tenthdeg.nc",
            )
        else:
            scen_lookup = {"ssp119": "SSP1", "ssp245": "SSP2", "ssp585": "SSP5"}
            return os.path.join(
                self.input_dir,
                "wind",
                "IRIS",
                "return_value_maps",
                f"IRIS_vmax_maps_{year}-{scen_lookup[scenario]}_tenthdeg.nc",
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._hazard_resource()]

    def _hazard_resource(self) -> HazardResource:
        with open(os.path.join(os.path.dirname(__file__), "iris_wind.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="Wind",
            indicator_id="max_speed",
            indicator_model_id=None,
            indicator_model_gcm="combined",
            path="wind/iris/v1/max_speed_{scenario}_{year}",
            params={},
            display_name="Max wind speed (IRIS)",
            description=description,
            group_id="iris_osc",
            display_groups=[],
            map=MapInfo(
                bounds=[(-180.0, 60.0), (180.0, 60.0), (180.0, -60.0), (-180.0, -60.0)],
                bbox=[-180.0, -60.0, 180.0, 60.0],
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
                path="wind/iris/v1/max_speed_{scenario}_{year}_map",
                source="map_array_pyramid",
            ),
            units="m/s",
            scenarios=[
                Scenario(id="historical", years=[2010]),
                Scenario(id="ssp119", years=[2050]),
                Scenario(id="ssp245", years=[2050]),
                Scenario(id="ssp585", years=[2050]),
            ],
        )
        return resource
