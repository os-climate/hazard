import os
from typing import Iterable, Optional

from fsspec import AbstractFileSystem # type: ignore
import numpy as np
import xarray as xr
from xarray import DataArray

from hazard.protocols import WriteDataArray, WriteDataset
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import map_utilities, xarray_utilities
from hazard.inventory import HazardModel, MapInfo, Scenario

class OscChronicHeat:
    """On-boarding of LSEG-generated chronic heat sets.
    """
    def __init__(self, root: str, fs: Optional[AbstractFileSystem]=None):
        """Class to onboard chronic heat indicators from raw files.

        Args:
            fs (AbstractFileSystem, optional): if present, use to open inputs, otherwise assume local file system. Defaults to None.
            root (str): root path of input files. Defaults to None.
        """
        self.fs = fs
        self.historical_year = 2010
        self.max_valid = 1.0
        self.models = ["mean_work_loss/low", "mean_work_loss/medium", "mean_work_loss/high"]
        self.scenarios = ["historical", "ssp245", "ssp585"]
        self.scenario_years = [2030, 2040, 2050]
        self.root = root

    def check(self):
        for source_path, target_path in self._generate_source_targets():
            self._onboard_single(source_path, target, target_path)

    def inventory_entry(self) -> Iterable[HazardModel]:
        """Return the unexpanded inventory entry."""
        yield HazardModel(
            event_type="ChronicHeat",
            path="chronic_heat/osc/v1",
            id="mean_work_loss/{intensity}",
            params={"intensity" : ["high", "medium", "low"]},
            display_name="Mean work loss ({intensity} intensity)",
            description="file:///onboard/work_loss.md",
            filename="mean_work_loss_{intensity}_{scenario}_{year}",
            map=MapInfo(colormap="mean_work_loss/{intensity}"),
            units="fractional loss",
            scenarios=[
                    Scenario(id="ssp585", years=[2030, 2040, 2050]),
                    Scenario(id="ssp245", year=[2030, 2040, 2050]),
                    Scenario(id="historical", years=[2010]),
                ],
            )
        yield HazardModel(
            event_type="ChronicHeat",
            path="chronic_heat/osc/v1",
            id="mean_degree_days/above/32c",
            display_name="Mean degree days above 32°C",
            description="file:///onboard/degree_days.md", 
            filename="mean_degree_days_above_32c_{scenario}_{year}",
            map=MapInfo(colormap="mean_degree_days/above/32c"),
            units="degree days",
            scenarios=[
                    Scenario(id="ssp585", years=[2030, 2040, 2050]),
                    Scenario(id="historical", years=[1980]),
                ],
            )
    
    def onboard(self, target: WriteDataArray):
        for source_path, target_path in self._generate_source_targets():
            self._onboard_single(source_path, target, target_path)

    def onboard_maps(self, source: OscZarr, working_dir: Optional[str]=None):
        max_value = float("-inf")
        for _, path in self._generate_source_targets():
            data, transform, crs = source.read_numpy(path)
            max_value = max(max_value, data.max())
        
        for _, path in self._generate_source_targets():
            da = source.read(path).sel(latitude=slice(85, -85))
            reprojected = da.rio.reproject("EPSG:3857") #, shape=da.data.shape, nodata=0) # from EPSG:4326 to EPSG:3857 (Web Mercator)
            head, tail = os.path.split(path)
            target_path_map = os.path.join(head, tail + "_map")
            source.write(target_path_map, reprojected)
            
            #data, transform, crs = xarray_utilities.get_array_components(da)
            #print(reprojected)

            #for _, path in self._generate_source_targets():
            #    data, transform = source.read_numpy(path)
            #profile = map_utilities.geotiff_profile(3857)
            #map_utilities.write_map_geotiff_data(
            #    reprojected.data, # .values # type: ignore
            #    profile,
            #    reprojected.shape[1],
            #    reprojected.shape[0],
            #    transform,
            #    os.path.basename(path) + ".tif",
            #    working_dir,
            #    nodata_threshold=0,
            #    zero_transparent=True,
            #    max_intensity=max_value,  # float("inf"),
            #    palette="heating",
            #)   

    def _fix_array(self, ds: xr.Dataset) -> xr.DataArray:
        """The convention is not quite right."""
        ds = ds.assign_coords(xarray_utilities.coords_from_extent(ds.dims['x'], ds.dims['y'], x_dim='x', y_dim='y')) 
        ds = ds.rename({"x": "lon", "y": "lat"})
        ds = ds.squeeze(dim='band')
        da: DataArray = ds.unknown # the name of the array we are interested in
        if self.max_valid is not None:
            da = xr.where(da > self.max_valid, 0, da) 
        da = xr.where(np.isnan(da), 0, da)
        return xarray_utilities.enforce_conventions(da)     

    def _generate_source_targets(self):
        for model in self.models:
            for scenario in self.scenarios:
                if scenario == "historical":
                    source_path = self._get_source_path(model=model, scenario=scenario, year=self.historical_year)
                    target_path = self._get_target_path(model=model, scenario=scenario, year=self.historical_year)
                    yield (source_path, target_path)
                else:
                    for year in self.scenario_years:
                        source_path = self._get_source_path(model=model, scenario=scenario, year=year)
                        target_path = self._get_target_path(model=model, scenario=scenario, year=year)
                        yield (source_path, target_path)

    def _get_source_path(self, *, model: str, scenario: str, year: int):
        type, *levels = model.split("/")
        params = { "high": "25-23", "medium": "31-17", "low": "33-18" } # intensity parameters
        if type == "mean_work_loss":
            assert levels[0] in ["low", "medium", "high"]  # work intensity
            if scenario=="historical":
                return os.path.join(self.root, f"mean_work_loss_hist_{params[levels[0]]}.tiff")
            else:
                return os.path.join(self.root, f"mean_work_loss_{scenario}_{year}_{params[levels[0]]}.tiff")

    def _get_target_path(self, *, model: str, scenario: str, year: int):
        type, *levels = model.split("/")
        if type == "mean_degree_days":
            assert levels[0] in ["above", "below"]  # above or below
            assert levels[1] in ["18c", "32c"]  # threshold temperature
            return self._osc_chronic_heat_prefix() + "/" + f"{type}_{levels[0]}_{levels[1]}_{scenario}_{year}"
        elif type == "mean_work_loss":
            assert levels[0] in ["low", "medium", "high"]  # work intensity
            return self._osc_chronic_heat_prefix() + "/" + f"{type}_{levels[0]}_{scenario}_{year}"
        else:
            raise ValueError("valid types are mean_degree_days and mean_work_loss")

    def _onboard_single(self, source_path: str, target: WriteDataArray, target_path: str):
        try:
            filename_or_obj = self.fs.open(source_path) if self.fs is not None else source_path
            with xr.open_dataset(filename_or_obj, engine="rasterio", parse_coordinates=True, cache=True) as ds:
                da = self._fix_array(ds)
                # write the target to zarr
                target.write(target_path, da)                
                #point1 = da.sel(lat=15.625, lon=32.625, method="nearest")
                #point2 = target.read_floored(target_path, [32.625], [15.625])
        finally: 
            if not isinstance(filename_or_obj, str):
                filename_or_obj.close()

    def _osc_chronic_heat_prefix(self):
        return "chronic_heat/osc/v1"

