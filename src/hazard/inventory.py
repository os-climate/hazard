import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

import pystac

# region HazardModel


class Colormap(BaseModel):
    """Provides details of colormap."""

    min_index: Optional[int] = Field(
        1,
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison.",
    )
    min_value: float = Field(
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    max_index: Optional[int] = Field(
        255,
        description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison.",
    )
    max_value: float = Field(
        description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison."
    )
    name: str = Field(description="Name of colormap, e.g. 'flare', 'heating'.")
    nodata_index: Optional[int] = Field(0, description="Index used for no data.")
    units: str = Field(description="Units, e.g. 'degree days', 'metres'.")


class MapInfo(BaseModel):
    """Provides information about map layer."""

    colormap: Optional[Colormap] = Field(description="Details of colormap.")
    path: str = Field(
        description="Name of array reprojected to Web Mercator for on-the-fly display or to hash to obtain tile ID. If not supplied, convention is to add '_map' to path."  # noqa
    )
    bounds: List[Tuple[float, float]] = Field(
        [(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
        description="Bounds (top/left, top/right, bottom/right, bottom/left) as degrees. Note applied to map reprojected into Web Mercator CRS.",  # noqa
    )
    bbox: List[float] = Field(
        [-180.0, 85.0, 180, 85.0]
    )
    index_values: Optional[List[Any]] = Field(
        None,
        description="Index values to include in maps. If None, the last index value only is included.",
    )
    # note that the bounds should be consistent with the array attributes
    source: Optional[str] = Field(
        description="""Source of map image. These are
                            'map_array': single Mercator projection array at path above
                            'map_array_pyramid': pyramid of Mercator projection arrays
                            'mapbox'.
                            """
    )


class Period(BaseModel):
    """Provides information about a period, which currently corresponds to a year, belonging to a scenario."""

    year: int
    map_id: str = Field(description="If present, identifier to be used for looking up map tiles from server.")


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: List[int]
    # periods: Optional[List[Period]]


class HazardResource(BaseModel):
    """Provides information about a set of hazard indicators, including available scenarios and years."""

    hazard_type: str = Field(description="Type of hazard.")
    group_id: Optional[str] = Field(
        "public",
        description="Identifier of the resource group (used for authentication).",
    )
    path: str = Field(description="Full path to the indicator array.")
    indicator_id: str = Field(
        description="Identifier of the hazard indicator (i.e. the modelled quantity), e.g. 'flood_depth'."
    )
    indicator_model_id: Optional[str] = Field(
        None,
        description="Identifier specifying the type of model used in the derivation of the indicator \
                                    (e.g. whether flood model includes impact of sea-level rise).",
    )
    indicator_model_gcm: str = Field(
        description="Identifier of general circulation model(s) used in the derivation of the indicator."
    )
    params: Dict[str, Sequence[str]] = Field({}, description="Parameters used to expand wild-carded fields.")
    display_name: str = Field(description="Text used to display indicator.")
    display_groups: List[str] = Field([], description="Text used to group the (expanded) indicators for display.")
    description: str = Field(
        description="Brief description in mark down of the indicator and model that generated the indicator."
    )
    map: Optional[MapInfo] = Field(description="Optional information used for display of the indicator in a map.")
    scenarios: List[Scenario] = Field(description="Climate change scenarios for which the indicator is available.")
    units: str = Field(description="Units of the hazard indicator.")

    def expand(self):
        keys = list(self.params.keys())
        return expand_resource(self, keys, self.params)

    def key(self):
        """Unique key for the resource. array_path should be unique, although indicator_id is typically not.
        Vulnerability models request a hazard indicator by indicator_id from the Hazard Model. The Hazard Model
        selects based on its own logic (e.g. selects a particular General Circulation Model)."""
        return self.path

    def to_stac_item(self) -> Iterable[pystac.Item]:
        
        """
        converts hazard resource to STAC items
        """
        
        # iteration : create as many items as iterations
        
        # over parameters: if GCM put it to CMIP extension field. otherwise properties:{param key} -> param value
        # over scenarios. year goes to datetime. scenario goes to CMIP field. 
        
        # pystac.Item(
        #     id=self.indicator_id,
        #     geometry={
        #         "type": "Polygon",
        #         "coordinates": self.map.bounds
        #     },
        #     bbox=self.map.bbox,
        #     collection="osc-hazard-indicators",
        # )
        raise NotImplementedError
    
def from_stac_items(stac_items: Iterable[pystac.Item]) -> HazardResource:
    
    """
    converts STAC item to HazardResource
    """
    
    # find indicator specific param in properties:params, gcm in CMIP. 
    # find year in datetime
    # find scenario in CMIP.
    
    raise NotImplementedError
    
    
def expand(item: str, key: str, param: str):
    return item and item.replace("{" + key + "}", param)


def expand_resource(
    resource: HazardResource, keys: List[str], params: Dict[str, List[str]]
) -> Iterable[HazardResource]:
    if len(keys) == 0:
        yield resource.copy(deep=True, update={"params": {}})
    else:
        keys = keys.copy()
        key = keys.pop()
        for item in expand_resource(resource, keys, params):
            for param in params[key]:
                yield item.copy(
                    deep=True,
                    update={
                        "id": expand(item.indicator_id, key, param),
                        "display_name": expand(item.display_name, key, param),
                        "path": expand(item.path, key, param),
                        "map": (
                            None
                            if item.map is None
                            else item.map.copy(
                                deep=True,
                                update={"path": expand(item.map.path, key, param)},
                            )
                        ),
                    },
                )


# endregion


class HazardInventory(BaseModel):
    models: List[HazardResource]
    colormaps: dict


def inventory_json(models: Iterable[HazardResource]) -> str:
    response = HazardInventory(models=models)  # type: ignore
    return json.dumps(response.dict())
