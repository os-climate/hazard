import json
from typing import Iterable, List, Optional
from pydantic import BaseModel, Field
from enum import Flag, auto
from typing import Dict, List, Optional, Tuple


# region HazardModel

class Colormap(BaseModel):
    """Provides details of colormap."""

    min_index: Optional[int] = Field(
        1, description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    min_value: float = Field(
        description="Value of colormap minimum. Constant min for a group of maps can facilitate comparison."
    )
    max_index: Optional[int] = Field(
        255, description="Value of colormap maximum. Constant max for a group of maps can facilitate comparison."
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
    array_name: Optional[str] = Field(
        description="Name of array reprojected to Web Mercator for on-the-fly display or to hash to obtain tile ID. If not supplied, convention is to add '_map' to array_name."  # noqa
    )
    bounds: Optional[List[Tuple[float, float]]] = Field(
        [[-180.0, 85.0], [180.0, 85.0], [180.0, -85.0], [-180.0, -85.0]],
        description="Bounds (top/left, top/right, bottom/right, bottom/left) as degrees. Note applied to map reprojected into Web Mercator CRS.",  # noqa
    )
    # note that the bounds should be consistent with the array attributes
    source: Optional[str] = Field(description="Source of map image: 'map_array' or 'tiles'.")


class Period(BaseModel):
    """Provides information about a period, which currently corresponds to a year, belonging to a scenario."""

    year: int
    map_id: str = Field(description="If present, identifier to be used for looking up map tiles from server.")


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""

    id: str
    years: List[int]
    periods: Optional[List[Period]]


def expanded(item: str, key: str, param: str):
    return item and item.replace("{" + key + "}", param)


class HazardModel(BaseModel):
    """Provides scenarios associated with a hazard model."""

    type: Optional[str] = Field(description="Type of hazard.")
    path: str
    id: str
    params: Optional[Dict[str, List[str]]]
    display_name: str
    description: str
    array_name: str
    map: Optional[MapInfo]
    scenarios: List[Scenario]
    units: str

    def expand(self):
        # should be only one key
        if not self.params:
            yield self
            return
        key = list(self.params.keys())[0]
        params = self.params[key]
        for param in params:
            yield self.copy(
                deep=True,
                update={
                    "id": expanded(self.id, key, param),
                    "display_name": expanded(self.display_name, key, param),
                    "array_name": expanded(self.array_name, key, param),
                    "map": self.map.copy(deep=True, update={"array_name": expanded(self.map.array_name, key, param)}),
                },
            )

# endregion

class HazardInventory(BaseModel):
    models: List[HazardModel]
    colormaps: dict


def inventory_json(models: Iterable[HazardModel]) -> str:
    response = HazardInventory(models=models)  # type: ignore
    return json.dumps(response.dict())

#def update_inventory_s3(inventory: HazardInventory, s3=None, test: bool=True):
    """Update the entry in the OS-Climate S3."""



