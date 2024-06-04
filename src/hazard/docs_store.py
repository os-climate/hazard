"""Module for managing hazard inventory and documentation in various file systems.

This module provides the `DocStore` class, which supports operations such as reading,
writing, and updating hazard inventory and related documentation. The class is
designed to work with different file systems, including S3 and local storage.

Classes:
    DocStore
        Manages hazard inventory and documentation in a specified file system.
"""

import itertools
import json
import os
from pathlib import Path, PurePath, PurePosixPath
from typing import Dict, Iterable, List, Optional

import numpy as np
import s3fs  # type: ignore
from fsspec import FSMap  # type: ignore
from fsspec.implementations.local import LocalFileSystem
from pydantic import TypeAdapter

from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import DegreeDays, HeatingCoolingDegreeDays
from hazard.models.drought_index import DroughtIndicator
from hazard.models.water_temp import WaterTemperatureAboveIndicator
from hazard.models.wet_bulb_globe_temp import WetBulbGlobeTemperatureAboveIndicator
from hazard.models.work_loss import WorkLossIndicator
from hazard.onboard.csm_subsidence import DavydzenkaEtAlLandSubsidence
from hazard.onboard.ethz_litpop import ETHZurichLitPop
from hazard.onboard.ipcc_drought import IPCCDrought
from hazard.onboard.iris_wind import IRISIndicator
from hazard.onboard.jrc_landslides import JRCLandslides
from hazard.onboard.jrc_subsidence import JRCSubsidence
from hazard.onboard.jupiter import Jupiter
from hazard.onboard.tudelft_flood import TUDelftCoastalFlood, TUDelftRiverFlood
from hazard.onboard.tudelft_wildfire import TUDelftFire
from hazard.onboard.tudelft_wind import TUDelftConvectiveWindstorm
from hazard.onboard.wri_aqueduct_flood import WRIAqueductFlood
from hazard.onboard.wri_aqueduct_water_risk import WRIAqueductWaterRisk
from hazard.protocols import ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.map_utilities import epsg4326_to_epsg3857
from hazard.utilities.s3_utilities import get_store

from .inventory import HazardResource, HazardResources


class DocStore:
    """Manages hazard inventory and documentation in a specified file system.

    This class supports reading and writing hazard inventory and documentation
    to various file systems, such as S3 or local storage.

    Attributes
        _fs : AbstractFileSystem
            The file system interface (e.g., S3 or local) used for reading/writing.
        _root : str
            The root path for storing inventory and documentation.

    Raises
        ValueError
            If using LocalFileSystem without providing a local path.

    """

    def __init__(
        self,
        s3_store: Optional[FSMap] = None,
        local_path: Optional[str] = None,
    ):
        r"""Initialize the DocStore with the specified parameters.

        s3_store is used if provided. If not, a LocalFileSystem will be used with local_path as _root.
        If no local_path is provided, os.getcwd() will be used.


        Args:
            s3_store (Optional[FSMap]): S3Map to use.
            local_path (Optional[str]): Local path for inventory storage to use in the local fs.

        Notes:
            If both fs and s3_store are provided, s3_store will take precedence.

        Examples:
            extra_s3fs_kwargs = {
                "key": "abcdefghijklmno",
                "secret": "abcdefghijklmno",
                "token": "abcdefghijklmno",
                "endpoint_url": "http://abcdefghijklmno"
            }

            store = get_store(extra_s3fs_kwargs=extra_s3fs_kwargs)

            DocStore(s3_store= store)


            local_path = "C:\dev\hazard"

            DocStore(local_path=local_path)

        """
        if s3_store is not None:
            self._fs = s3_store.fs
            self._root = s3_store.root.replace("/hazard.zarr", "")

        else:
            if local_path is None:
                local_path = os.getcwd()

            self._fs = LocalFileSystem()
            self._root = PurePath(local_path)

    def read_inventory(self) -> HazardResources:
        """Read inventory at path provided and return HazardResources."""
        path = self._full_path_inventory()
        if not self._fs.exists(path):
            return HazardResources(resources=[])
        json_str = self.read_inventory_json()
        return TypeAdapter(HazardResources).validate_python(json.loads(json_str))

    def read_inventory_json(self) -> str:
        """Read inventory at path provided and return JSON."""
        with self._fs.open(self._full_path_inventory(), "r") as f:
            json_str = f.read()
        return json_str

    def write_inventory_json(self, json_str: str):
        """Write inventory."""
        path = self._full_path_inventory()
        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def write_new_empty_inventory(self):
        """Write inventory."""
        path = self._full_path_inventory()
        if not isinstance(self._fs, s3fs.S3FileSystem):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        models = HazardResources(resources=[])
        json_str = json.dumps(models.dict(), indent=4)  # pretty print

        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def write_new_inventory(self, resources: Iterable[HazardResource]):
        """Write inventory with hazard resources."""
        path = self._full_path_inventory()
        if not isinstance(self._fs, s3fs.S3FileSystem):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # combined = {}
        # for resource in resources:
        #     combined[resource.key()] = resource
        # Convert the resources to the appropriate model
        models = HazardResources(resources=list(resources))
        json_str = json.dumps(models.dict(), indent=4)  # Pretty-print JSON
        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def update_inventory(
        self, resources: Iterable[HazardResource], remove_existing: bool = False
    ):
        """Update the inventory with new hazard resources.

        Args:
            resources : Iterable[HazardResource]
                Collection of HazardResource objects to add.
            remove_existing : bool, optional
                Clear existing inventory if True. Defaults to False.

        Example:
            >>> inventory.update_inventory(resources, remove_existing=True)


        """
        path = self._full_path_inventory()
        combined = (
            {}
            if remove_existing
            else dict((i.key(), i) for i in list(self.read_inventory())[0][1])
        )
        for resource in resources:
            combined[resource.key()] = resource
        models = HazardResources(resources=list(combined.values()))
        json_str = json.dumps(models.dict(), indent=4)

        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def read_description_markdown(self, paths: List[str]) -> Dict[str, str]:
        """Read description markdown at path provided."""
        md: Dict[str, str] = {}
        for path in paths:
            try:
                with self._fs.open(self._full_path_doc(path), "r") as f:
                    md[path] = f.read()
            finally:
                continue  # noqa: B012
        return md

    def create_bucket_inventory(self):  # noqa: F811
        """Create inventory for all indicators and write into s3 bucket."""
        default_root = str(os.path.join(Path.home(), "Downloads"))
        models = [
            ETHZurichLitPop(default_root),
            DavydzenkaEtAlLandSubsidence(default_root),
            WRIAqueductFlood(),
            DegreeDays(),
            Jupiter(),
            WorkLossIndicator(),
            DaysTasAboveIndicator(),
            IRISIndicator(default_root),
            HeatingCoolingDegreeDays(),
            WaterTemperatureAboveIndicator(),
            WetBulbGlobeTemperatureAboveIndicator(),
            WRIAqueductWaterRisk(),
            DroughtIndicator(default_root),
            TUDelftRiverFlood(default_root),
            TUDelftCoastalFlood(default_root),
            TUDelftFire(default_root),
            TUDelftConvectiveWindstorm(default_root),
            JRCLandslides(default_root),
            JRCSubsidence(default_root),
            IPCCDrought(default_root),
        ]

        self.write_new_empty_inventory()
        # docs_store.write_inventory_json(json_str)
        for model in models:
            self.update_inventory(model.inventory())

    def find_available_s3_paths(self, store, local_inventory: Optional[str] = None):
        """Find and return available dataset and map paths in the S3 Zarr store.

        Args:
            store: FSMap
                The S3-compatible FSMap used to check the existence of paths.
            local_inventory: Path to local inventory

        Returns:
            list[list[str], list[str]]
                A list of two sublists:
                - The first sublist contains valid dataset paths.
                - The second sublist contains valid map paths.

        Notes:
            - If resource parameters exist, it generates combinations of parameters
            and formats the paths dynamically for each combination, scenario, and year.
            - Paths are checked in the S3 store, and only those that exist are added
            to the result.

        """
        if local_inventory is not None:
            if os.path.exists(local_inventory):
                with open(local_inventory, "r") as f:
                    json_str = f.read()

                hazard_resources = TypeAdapter(HazardResources).validate_python(
                    json.loads(json_str)
                )
            else:
                raise KeyError(f"{local_inventory} not found")
        else:
            hazard_resources = self.read_inventory()

        fs = store.fs
        paths: list[list] = [[], []]
        for resource in hazard_resources.resources:
            # Iterate over all scenarios
            for scenario in resource.scenarios:
                scenario_id = scenario.id
                # Iterate over all years for the given scenario
                for year in scenario.years:
                    # If there are params, generate combinations of params and format the path accordingly
                    if resource.params:
                        param_combinations = [
                            dict(zip(resource.params.keys(), v, strict=False))
                            for v in itertools.product(*resource.params.values())
                        ]

                        # Iterate over each combination of params
                        for param_set in param_combinations:
                            # Add scenario and year to the param_set
                            param_set["scenario"] = scenario_id
                            param_set["year"] = str(year)

                            # Format the path dynamically using all params, scenario, and year
                            path = resource.path.format(**param_set)
                            map_path = resource.map.path.format(**param_set)  # type: ignore[union-attr]
                            full_s3_path = PurePosixPath(store.root, path)
                            full_s3_map_path = PurePosixPath(store.root, map_path)
                            if fs.exists(full_s3_path) and fs.exists(full_s3_map_path):
                                paths[0].append(path)
                                if "jupiter" not in map_path:
                                    paths[1].append(map_path + "/1")
                                else:
                                    paths[1].append(map_path)
                    else:
                        # If no params, just use scenario and year
                        path = resource.path.format(scenario=scenario_id, year=year)
                        map_path = resource.map.path.format(  # type: ignore[union-attr]
                            scenario=scenario_id, year=year
                        )
                        full_s3_path = PurePosixPath(store.root, path)
                        full_s3_map_path = PurePosixPath(store.root, map_path)
                        if fs.exists(full_s3_path) and fs.exists(full_s3_map_path):
                            paths[0].append(path)
                            if "jupiter" not in map_path:
                                paths[1].append(map_path + "/1")
                            else:
                                paths[1].append(map_path)
        return paths

    def check_s3_data(self, extra_s3fs_kwargs: dict):  # noqa: F811
        """Check if the datasets and maps in S3 Zarr store are empty.

        This function reads datasets and map sets from a Zarr store
        located in an S3-compatible file system. It verifies that each
        dataset and its corresponding map is non-empty by checking their
        byte size, number of partitions, and shape.

        Args:
            extra_s3fs_kwargs (Dict): Additional arguments for S3 file system.

        Raises:
            AssertionError
                If any dataset is empty or has invalid properties such as
                0 bytes, no partitions, or an invalid shape.
                If any map is empty or has invalid properties such as 0 bytes,
                or an invalid shape.

        Side Effects:
            - Sets S3 credentials using environment variables.
            - Uses s3fs to interact with the Zarr store in S3.

        """
        store = get_store(use_dev=True, extra_s3fs_kwargs=extra_s3fs_kwargs)
        reader = OscZarr()
        paths = self.find_available_s3_paths(store=store)
        empty = 0
        empty_map = 0

        for path, map_path in zip(paths[0], paths[1], strict=False):
            data_set = reader.read(path)
            map_set = reader.root[map_path]
            if (
                not data_set.data.nbytes > 0
                or not data_set.data.npartitions > 0
                or not data_set.data.shape > (0, 0, 0)
            ):
                empty += 1

            if (
                not map_set.nbytes > 0
                or not map_set.size > 0
                or not map_set.shape > (0, 0, 0)
            ):
                empty_map += 1

        if empty == 0 and empty_map == 0:
            return True
        else:
            return False

    def check_inventory_paths(
        self, extra_s3fs_kwargs: dict, local_inventory: Optional[str]
    ):  # noqa: F811
        """Check if the combinations of inventory paths exist in the S3 Zarr store.

        Args:
            extra_s3fs_kwargs (Dict): Additional arguments for S3 file system.
            local_inventory: Path to local inventory

        Raises:
            AssertionError
                Raised if one or more paths are missing in the S3 store.

        Side Effects:
            - Sets S3 credentials using environment variables.
            - Uses s3fs to check the existence of paths in the S3 store.
            - Prints the status (found/missing) of each path.

        """
        store = get_store(use_dev=True, extra_s3fs_kwargs=extra_s3fs_kwargs)
        fs = store.fs
        error = 0
        missing_paths = []

        if local_inventory is not None:
            if os.path.exists(local_inventory):
                with open(local_inventory, "r") as f:
                    json_str = f.read()
                hazard_resources = TypeAdapter(HazardResources).validate_python(
                    json.loads(json_str)
                )
            else:
                raise KeyError(f"{local_inventory} not found")
        else:
            hazard_resources = self.read_inventory()

        for resource in hazard_resources.resources:
            # Iterate over all scenarios
            for scenario in resource.scenarios:
                scenario_id = scenario.id
                # Iterate over all years for the given scenario
                for year in scenario.years:
                    # If there are params, generate combinations of params and format the path accordingly
                    if resource.params:
                        param_combinations = [
                            dict(zip(resource.params.keys(), v, strict=False))
                            for v in itertools.product(*resource.params.values())
                        ]

                        # Iterate over each combination of params
                        for param_set in param_combinations:
                            # Add scenario and year to the param_set
                            param_set["scenario"] = scenario_id
                            param_set["year"] = str(year)

                            # Format the path dynamically using all params, scenario, and year
                            path = resource.path.format(**param_set)
                            full_s3_path = PurePosixPath(store.root, path)
                            if not fs.exists(full_s3_path):
                                error += 1
                                missing_paths.append(path)
                                continue
                    else:
                        # If no params, just use scenario and year
                        path = resource.path.format(scenario=scenario_id, year=year)
                        full_s3_path = PurePosixPath(store.root, path)
                        if not fs.exists(full_s3_path):
                            error += 1
                            missing_paths.append(path)
                            continue
        if error == 0:
            return True, missing_paths
        else:
            return False, missing_paths

    def check_s3_paths(self, extra_s3fs_kwargs: dict, local_inventory: Optional[str]):
        """Check for missing S3 paths that are not listed in the inventory.

        Args:
            extra_s3fs_kwargs (Dict): Additional arguments for S3 file system.
            local_inventory: Path to local inventory

        Raises:
            AssertionError
                Raised if there are paths in S3 that are not listed in the inventory.

        Side Effects:
            - Sets S3 credentials using environment variables.
            - Uses s3fs to walk through the directories in the S3 bucket.
            - Compares last-level S3 directories to paths in the inventory.

        """
        store = get_store(use_dev=True, extra_s3fs_kwargs=extra_s3fs_kwargs)
        fs = store.fs
        bucket = os.environ.get("OSC_S3_BUCKET_DEV", None)
        s3_path = f"{bucket}/"
        last_level_dirs = []

        for dirpath, dirnames, _ in fs.walk(s3_path):
            if not dirnames:
                last_level_dirs.append(dirpath)

        s3_paths = [
            os.path.dirname(path) if "_map" in path and "jupiter" not in path else path
            for path in last_level_dirs
        ]
        definitive_s3_paths = [path.replace(f"{store.root}/", "") for path in s3_paths]
        if local_inventory is not None:
            inventory_paths = self.find_available_s3_paths(
                store=store, local_inventory=local_inventory
            )
        else:
            inventory_paths = self.find_available_s3_paths(store=store)
        map_paths = [path.replace("/1", "") for path in inventory_paths[1]]
        missing_paths = []
        for s3_path in definitive_s3_paths:
            if s3_path not in inventory_paths[0] and s3_path not in map_paths:
                missing_paths.append(s3_path)

        if len(missing_paths) == 0:
            return True, missing_paths
        else:
            return False, missing_paths

    def _full_path_doc(self, path: str):
        return str(PurePosixPath(self._root, "docs", path))

    def _full_path_inventory(self):
        return str(PurePosixPath(self._root, "inventory.json"))

    def get_resolution(self, target: ReadWriteDataArray, path) -> Optional[str]:
        """Return the resolution of the data set.

        This is typically the resolution of the
        original data set. It is not always available and may be None.

        Returns:
            Optional[str]: Resolution of the data set.

        """
        ds = target.read(path=path)
        coords = ds.coords

        if "latitude" in coords and "longitude" in coords:
            lat = ds["latitude"].values
            lon = ds["longitude"].values

            # Asegurarse de que lat y lon sean 1D (puedes ajustar esto si es 2D)
            if lat.ndim > 1:
                lat = lat[:, 0]
            if lon.ndim > 1:
                lon = lon[0, :]

            x_vals = np.array([epsg4326_to_epsg3857(la, lat[0])[0] for la in lon])
            y_vals = np.array([epsg4326_to_epsg3857(lon[0], la)[1] for la in lat])

            dx = np.abs(np.diff(x_vals))
            dy = np.abs(np.diff(y_vals))

            res_x = np.mean(dx)
            res_y = np.mean(dy)
            res = np.sqrt(res_x * res_y)

            if res < 100:
                res_rounded = int(round(res / 10) * 10)
            else:
                res_rounded = int(round(res / 100) * 100)

            print(f"Resolution: {res_rounded} m")
            return f"{res_rounded} m"

        elif "x" in coords and "y" in coords:
            x = ds["x"].values
            y = ds["y"].values

            # Calcular diferencias consecutivas absolutas
            dx = np.abs(np.diff(x))  # diferencias E–O
            dy = np.abs(np.diff(y))  # diferencias N–S

            # Calcular la media de las diferencias
            resolution_x = np.mean(dx)
            resolution_y = np.mean(dy)
            res = np.sqrt(resolution_x * resolution_y)
            if res < 100:
                res_rounded = int(round(res / 10) * 10)
            else:
                res_rounded = int(round(res / 100) * 100)

            print(f"Resolution (metros): {res_rounded} m")
            return f"{res_rounded} m"

        else:
            print("No se encontraron coordenadas de latitud/longitud ni x/y.")
            return None
