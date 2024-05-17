import json
import os
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

import s3fs  # type: ignore
from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel, parse_obj_as

from hazard.sources.osc_zarr import default_dev_bucket

from .inventory import HazardResource, HazardResources


class DocStore:
    # environment variable names:
    __access_key = "OSC_S3_ACCESS_KEY_DEV"
    __secret_key = "OSC_S3_SECRET_KEY_DEV"
    __S3_bucket = "OSC_S3_BUCKET_DEV"  # e.g. redhat-osc-physical-landing-647521352890

    def __init__(
        self,
        bucket=default_dev_bucket,
        prefix: str = "hazard",
        fs: Optional[AbstractFileSystem] = None,
        local_path: Optional[str] = None,
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
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem. Defaults to None in which case S3FileSystem will be created. # noqa: E501
            local_path (Optional[str], optional): local path where to save the inventory. only used if `fs` is a LocalFileSystem.
        """
        if fs is None:
            access_key = os.environ.get(self.__access_key, None)
            secret_key = os.environ.get(self.__secret_key, None)
            fs = s3fs.S3FileSystem(key=access_key, secret=secret_key)

        self._fs = fs
        if type(self._fs) == s3fs.S3FileSystem:  # noqa: E721 # use isinstance?
            bucket = os.environ.get(self.__S3_bucket, bucket)
            self._root = f"s3://{str(PurePosixPath(bucket, prefix))}"
        elif type(self._fs) == LocalFileSystem:  # noqa: E721 # use isinstance?
            if local_path is None:
                raise ValueError("if using a local filesystem, please provide a value for `local_path`")
            self._root = str(PurePosixPath(local_path))
        else:
            self._root = str(PurePosixPath(bucket, prefix))

    def read_inventory(self) -> List[HazardResource]:
        """Read inventory at path provided and return HazardResources."""
        path = self._full_path_inventory()
        if not self._fs.exists(path):
            return []
        json_str = self.read_inventory_json()
        return parse_obj_as(HazardResources, json.loads(json_str)).resources

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
        json_str = json.dumps(models.dict(), indent=4)  # pretty print

        with self._fs.open(path, "w") as f:
            f.write(json_str)

    def write_inventory_stac(self, resources: Iterable[HazardResource]):
        """Write a hazard models inventory as STAC."""

        if self._fs == s3fs.S3FileSystem:
            path_Root = self._root
        else:
            path_root = '.'
            
        items = HazardResources(resources=resources).to_stac_items(path_root=path_root, items_as_dicts=True)
        for it in items:
            with self._fs.open(self._full_path_stac_item(id=it["id"]), "w") as f:
                f.write(json.dumps(it, indent=4))
        catalog_path = self._full_path_stac_catalog()
        catalog = self.stac_catalog(items=items)
        with self._fs.open(catalog_path, "w") as f:
            json_str = json.dumps(catalog, indent=4)
            f.write(json_str)
        collection_path = self._full_path_stac_collection()
        collection = self.stac_collection(items=items)
        with self._fs.open(collection_path, "w") as f:
            json_str = json.dumps(collection, indent=4)
            f.write(json_str)

    def stac_catalog(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:

        return {
            "stac_version": "1.0.0",
            "id": "osc-hazard-indicators-catalog",
            "type": "Catalog",
            "description": "OS-C hazard indicators catalog",
            "links": [
                {"rel": "self", "href": "./catalog.json"},
                {"rel": "root", "href": "./catalog.json"},
                {"rel": "child", "href": "./collection.json"},
            ]
            + [{"rel": "item", "href": f"./{x['id']}.json"} for x in items],
        }

    def stac_collection(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:

        return {
            "stac_version": "1.0.0",
            "type": "Collection",
            "stac_extensions": [],
            "id": "osc-hazard-indicators-collection",
            "title": "OS-C hazard indicators collection",
            "description": "OS-C hazard indicators collection",
            "license": "CC-BY-4.0",
            "extent": {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [["1950-01-01T00:00:00Z", "2100-12-31T23:59:59Z"]]},
            },
            "providers": [{"name": "UKRI", "roles": ["producer"], "url": "https://www.ukri.org/"}],
            "links": [
                {"rel": "self", "type": "application/json", "href": "./collection.json"},
                {"rel": "root", "type": "application/json", "href": "./catalog.json"},
            ]
            + [{"rel": "item", "href": f"./{x['id']}.json"} for x in items],
        }

    def update_inventory(self, resources: Iterable[HazardResource], remove_existing: bool = False):
        """Add the hazard models provided to the inventory. If a model with the same key
        (hazard type and id) exists, replace."""

        # if format == stac, we do a round trip, stac -> osc -> stac.
        path = self._full_path_inventory()
        combined = {} if remove_existing else dict((i.key(), i) for i in self.read_inventory(format=format))
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
                continue
        return md

    def _full_path_doc(self, path: str):
        return str(PurePosixPath(self._root, "docs", path))

    def _full_path_inventory(self):
        return str(PurePosixPath(self._root, "inventory.json"))

    def _full_path_stac_item(self, id: str):
        return str(PurePosixPath(self._root, f"{id}.json"))

    def _full_path_stac_catalog(self):
        return str(PurePosixPath(self._root, "catalog.json"))

    def _full_path_stac_collection(self):
        return str(PurePosixPath(self._root, "collection.json"))
