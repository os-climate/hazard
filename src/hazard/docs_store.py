import json
import os
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional

import s3fs # type: ignore
from fsspec import AbstractFileSystem # type: ignore
from pydantic import BaseModel, parse_obj_as

from hazard.sources.osc_zarr import OscZarr, default_dev_bucket

from .inventory import HazardResource


def get_env(key: str, default: Optional[str] = None) -> str:
    value = os.environ.get(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"environment variable {key} not present")
    else:
        return value


class HazardResources(BaseModel):
    resources: List[HazardResource]


class DocStore:
    # environment variable names:
    __access_key = "OSC_S3_ACCESS_KEY"
    __secret_key = "OSC_S3_SECRET_KEY"
    __S3_bucket = "OSC_S3_BUCKET"  # e.g. redhat-osc-physical-landing-647521352890

    def __init__(
        self,
        bucket = default_dev_bucket,
        prefix: str = "hazard",
        get_env: Callable[[str, Optional[str]], str] = get_env,
        fs: Optional[AbstractFileSystem] = None,
    ):
        """Class to read hazard inventory and documentation from supplied AbstractFileSystem (e.g. S3).
        In general *array* paths are of form:
        {bucket}/hazard/hazard.zarr/{path_1}/.../{path_n}/{array_name}
        or
        {bucket}/hazard_test/hazard.zarr/{path_1}/.../{path_n}/{array_name} (for test 'hazard' -> 'hazard_test')
        Document paths follow name structure:
        {bucket}/hazard/docs/{path_1}/.../{path_n}/{doc_name}
        Inventory is stored in
        {bucket}/hazard/inventory.json

        Args:
            get_env (Callable[[str, Optional[str]], str], optional): Get environment variable. Defaults to get_env.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem. Defaults to None in which case S3FileSystem will be created. # noqa: E501
        """
        if fs is None:
            access_key = get_env(self.__access_key, None)
            secret_key = get_env(self.__secret_key, None)
            fs = s3fs.S3FileSystem(anon=False, key=access_key, secret=secret_key)

        self._fs = fs
        if type(self._fs) == s3fs.S3FileSystem:
            bucket = get_env(self.__S3_bucket, bucket)
            self._root = str(PurePosixPath(bucket, prefix))
        else:
            self._root = str(PurePosixPath(bucket, prefix))

    def read_inventory(self) -> List[HazardResource]:
        """Read inventory at path provided and return HazardResources."""
        path = self._full_path_inventory()
        if not self._fs.exists(path):
            return []
        json_str = self.read_inventory_json()
        models = parse_obj_as(HazardResources, json.loads(json_str)).resources
        return models

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models = HazardResources(resources=[])
        json_str = json.dumps(models.dict(), indent=4) # pretty print
        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def update_inventory(self, resources: Iterable[HazardResource], remove_existing: bool=False):
        """Add the hazard models provided to the inventory. If a model with the same key
        (hazard type and id) exists, replace."""
        path = self._full_path_inventory()
        combined = {} if remove_existing else dict((i.key(), i) for i in self.read_inventory()) 
        for resource in resources:
            combined[resource.key()] = resource
        models = HazardResources(resources=list(combined.values()))
        json_str = json.dumps(models.dict(), indent=4) # pretty print
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
                continue
        return md

    def _full_path_doc(self, path: str):
        return str(PurePosixPath(self._root, "docs", path))

    def _full_path_inventory(self):
        return str(PurePosixPath(self._root, "inventory.json"))