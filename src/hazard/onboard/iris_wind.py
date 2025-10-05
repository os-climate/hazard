"""Module for handling the onboarding and processing of IRIS - Imperial College Storm Model data."""

import os
from pathlib import PurePath
from typing import Optional
from typing_extensions import Iterable, override
import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from hazard.onboarder import Onboarder
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import tiles


class IRISIndicator(Onboarder):
    """On-board returns data set from IRIS - Imperial College Storm Model."""

    def __init__(self, source_dir_base: str, fs: Optional[AbstractFileSystem] = None):
        """Initialize the IRISIndicator class with the input directory for IRIS data.

        Assumes iris downloaded data is of the form wind/IRIS/return_value_maps/--files and that they are in the downloads folder.

        """
        self.source_dir = PurePath(source_dir_base, "iris_wind").as_posix() + "/"

        self.fs = fs if fs else LocalFileSystem()

    @override
    def prepare(self, force=False, download_dir=None, force_download=False):
        if download_dir and not os.path.exists(download_dir):
            msg = f"{self.__class__.__name__} requires the file return_value_maps to be in the download_dir.\nThe download_dir was {download_dir}."
            raise FileNotFoundError(msg)

        if download_dir:
            # Define la ruta específica donde se deben buscar los archivos
            search_dir = PurePath(download_dir, "wind", "IRIS", "return_value_maps")

            # Verifica si el directorio existe
            if not os.path.exists(search_dir):
                msg = f"{self.__class__.__name__} requires the files to be in the {search_dir} directory.\nThe directory was not found."
                raise FileNotFoundError(msg)

        self.fs.makedirs(self.source_dir, exist_ok=True)
        if download_dir:
            for _, _, files in os.walk(search_dir):
                for file_name in files:
                    dest_file_path = PurePath(self.source_dir, file_name)
                    if force or not os.path.exists(dest_file_path):
                        self.fs.copy(
                            PurePath(search_dir, file_name),
                            self.source_dir,
                        )

    @override
    def is_prepared(self, force=False, force_download=False) -> bool:
        """Check if the data is prepared."""
        if not os.path.exists(self.source_dir) or force or force_download:
            return False
        # Listar todos los archivos en el directorio
        try:
            files = os.listdir(self.source_dir)
        except FileNotFoundError:
            return False

        # Filtrar los archivos CSV
        nc_files = [file for file in files if file.endswith(".nc")]

        # verificar que están los 4 .nc files
        return len(nc_files) == 4

    @override
    def onboard(self, target: ReadWriteDataArray):
        items_to_process = self._get_items_to_process()
        resource = self._hazard_resource()  # Llamar una vez y almacenar en una variable

        def _pad_to_global(da_src: xr.DataArray, *, nodata_val=np.nan) -> xr.DataArray:
            """Extiende la latitud hasta ±85° manteniendo la resolución del dataset.
            Evita pérdidas por desajuste de flotantes usando `nearest` + tolerancia.
            """
            lat = da_src.latitude

            # 1) Asegurarnos de que el eje esté en orden N→S
            if lat[0] < lat[-1]:
                da_src = da_src.sortby("latitude", ascending=False)
                lat = da_src.latitude

            # 2) Resolución original (redondeada a 6 decimales para estabilidad)
            lat_res = round(float(abs(lat[1] - lat[0])), 6)

            # 3) Construct nuevo eje:  85 … -85  (mismo paso)
            full_lat = np.round(np.arange(85.0, -85.0 - lat_res / 2, -lat_res), 6)

            # 4) Reindex — nearest con tolerancia = ½ resolución
            da_full = (
                da_src.reindex(
                    latitude=full_lat,
                    method="nearest",
                    tolerance=lat_res / 2 + 1e-9,  # margen de seguridad
                )
                .fillna(nodata_val)
                .rio.write_nodata(nodata_val, inplace=False)
            )

            return da_full

        for item in items_to_process:
            file_name = self._file_name(item["scenario"], item["central_year"])
            ds = xr.open_dataset(
                file_name.format(year=item["central_year"], scenario=item["scenario"])
            )

            da = OscZarr.normalize_dims(ds["vmax"])

            # opcional: ajustar la referencia de píxel si usas centros en vez de vértices
            # da = da.assign_coords(latitude=da["latitude"] + 0.05, longitude=da["longitude"] + 0.05)
            lat = da.latitude
            lat_res = round(float(abs(lat[1] - lat[0])), 6)
            lon = da.longitude
            lon_res = round(float(abs(lon[1] - lon[0])), 6)

            da = da.assign_coords(
                latitude=lat + lat_res / 2.0,
                longitude=lon + lon_res / 2.0,
            )

            da_padded = _pad_to_global(da, nodata_val=np.nan)

            target.write(
                resource.path.format(
                    scenario=item["scenario"], year=item["central_year"]
                ),
                da_padded,  # ← se escribe la versión “global”
                chunks=[da_padded.shape[0], 250, 250],
            )

            self.generate_single_map(item=item, source=target, target=target)

    # Asumiendo que la resolución es la misma para todos los archivos

    def generate_single_map(
        self, item, source: ReadWriteDataArray, target: ReadWriteDataArray
    ):
        resource = self._hazard_resource()
        source_path = resource.path.format(
            scenario=item["scenario"], year=item["central_year"]
        )
        assert resource.map is not None
        # assert isinstance(source, OscZarr) and isinstance(target, OscZarr)
        target_path = resource.map.path.format(
            scenario=item["scenario"], year=item["central_year"]
        )
        tiles.create_tile_set(source, source_path, target, target_path, check_fill=True)  # type: ignore[arg-type]
        # tiles.create_image_set(source, source_path, target, target_path)

    def _get_items_to_process(self) -> list[dict]:
        """Get a list of dictionaries with items to process."""
        resource = self._hazard_resource()
        items = []

        for scenario in resource.scenarios:
            for year in scenario.years:
                items.append(
                    {
                        "indicator": resource.indicator_id,
                        "scenario": scenario.id,
                        "central_year": year,
                        "input_dataset_filename": self._file_name(scenario.id, year),
                    }
                )

        return items

    @override
    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images for the IRIS dataset."""
        pass

    def _file_name(self, scenario: str, year: int):
        """Return the file name for a specific scenario and year."""
        # file name for 0.1 degree resolution sets
        # including radiative forcing (1.9 to 8.5 W/m^2): SSP1 = SSP1-1.9, SSP2 = SSP2-4.5, SSP5 = SSP5-8.5
        if scenario == "historical":
            return os.path.join(
                self.source_dir,
                "IRIS_vmax_maps_PRESENT_tenthdeg.nc",
            )
        else:
            scen_lookup = {"ssp119": "SSP1", "ssp245": "SSP2", "ssp585": "SSP5"}
            return os.path.join(
                self.source_dir,
                f"IRIS_vmax_maps_{year}-{scen_lookup[scenario]}_tenthdeg.nc",
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        return [self._hazard_resource()]

    def _hazard_resource(self) -> HazardResource:
        """Return the hazard resource details, including metadata and map info, for the IRIS dataset."""
        with open(os.path.join(os.path.dirname(__file__), "iris_wind.md"), "r") as f:
            description = f.read()
        resource = HazardResource(
            hazard_type="Wind",
            indicator_id="max_speed",
            indicator_model_id="iris",
            indicator_model_gcm="combined",
            path="wind/iris/v1/max_speed_{scenario}_{year}",
            params={},
            resolution="10 km",
            display_name="Max wind speed (IRIS)",
            description=description,
            license="Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/",
            attribution="Sparks, N. & Toumi, R. (2023). IRIS: The Imperial College Storm Model Dataset (v1). Figshare. https://doi.org/10.6084/m9.figshare.c.6724251.v1",
            source="",
            version="",
            group_id="iris_osc",
            display_groups=[],
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_index=255,
                    max_value=90.0,
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
