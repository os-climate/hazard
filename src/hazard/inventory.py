"""Hazard Inventory Management Module."""

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


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
    bbox: Optional[List[float]] = Field(default=[-180.0, -85.0, 180.0, 85.0])
    index_values: Optional[Sequence[Any]] = Field(
        default=None,
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
    map_id: str = Field(
        description="If present, identifier to be used for looking up map tiles from server."
    )


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'."""

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
    path: str = Field(
        description="Full path to the indicator array if store_netcdf_coords is False, \
                    otherwise to the group containing the indicator array and coordinates."
    )
    indicator_id: str = Field(
        description="Identifier of the hazard indicator (i.e. the modelled quantity), e.g. 'flood_depth'."
    )
    indicator_model_id: Optional[str] = Field(
        default=None,
        description="Identifier specifying the type of model used in the derivation of the indicator \
                                    (e.g. whether flood model includes impact of sea-level rise).",
    )
    indicator_model_gcm: str = Field(
        description="Identifier of general circulation model(s) used in the derivation of the indicator."
    )
    params: Dict[str, Sequence[str]] = Field(
        {}, description="Parameters used to expand wild-carded fields."
    )
    display_name: str = Field(description="Text used to display indicator.")
    display_groups: List[str] = Field(
        [], description="Text used to group the (expanded) indicators for display."
    )
    description: str = Field(
        description="Brief description in mark down of the indicator and model that generated the indicator."
    )
    license: Optional[str] = Field(
        default="",
        description="The license under which the indicator or dataset is distributed. This defines how the data can be used, shared, or modified.",
    )
    source: Optional[str] = Field(
        default="",
        description="The origin or provenance of the indicator or dataset, such as the organization, research project, or publication responsible for its creation.",
    )
    attribution: Optional[str] = Field(
        default="",
        description="Identifies the source, author, or entity responsible for the content or item in the inventory, allowing for proper credit or origin tracking.",
    )
    version: Optional[str] = Field(
        default="", description="The version identifier of the indicator or dataset."
    )
    map: Optional[MapInfo] = Field(
        description="Optional information used for display of the indicator in a map."
    )

    resolution: Optional[str] = Field(
        default=None,
        description="Resolution of the hazard indicator. This is typically the resolution of the original data set. It is not always available and may be None.",
    )

    store_netcdf_coords: bool = Field(
        False,
        description="If True, NetCDF-style coordinates are also stored, which allows XArray to read the array \
            natively. By convention, the hazard indicator data array is named 'indicator' and the path to \
            the Zarr array is then path/indicator (not path).",
    )
    scenarios: List[Scenario] = Field(
        description="Climate change scenarios for which the indicator is available."
    )
    units: str = Field(description="Units of the hazard indicator.")

    def expand(self):
        """Expand the resource using its parameters.

        Returns
            List[HazardResource]: A list of expanded hazard resources.

        """
        keys = list(self.params.keys())
        return expand_resource(self, keys, self.params)

    def key(self):
        """Use a unique key for the resource. Ensure array_path is unique, although indicator_id typically is not.

        Vulnerability models request a hazard indicator by indicator_id from the Hazard Model. The Hazard Model
        selects based on its own logic (e.g. selects a particular General Circulation Model).
        """
        return self.path


class HazardResources(BaseModel):
    """Container for a list of hazard resources.

    Args:
        resources (List[HazardResource]): A list of hazard resource objects.

    """

    resources: List[HazardResource]


def expand(item: str, key: str, param: str):
    """Replace a placeholder in a string with a given parameter.

    Args:
        item (str): The string containing the placeholder.
        key (str): The placeholder key to be replaced.
        param (str): The value to replace the placeholder with.

    Returns:
        str: The updated string with the placeholder replaced, or the original string if no replacement occurs.

    """
    return item and item.replace("{" + key + "}", param)


def expand_resource(
    resource: HazardResource, keys: List[str], params: Dict[str, List[str]]
) -> Iterable[HazardResource]:
    """Recursively expand a resource by replacing placeholders with parameter values.

    Args:
        resource (HazardResource): The base resource to expand.
        keys (List[str]): The list of parameter keys to iterate over.
        params (Dict[str, List[str]]): A dictionary mapping parameter keys to lists of values.

    Returns:
        HazardResource: A new resource instance for each expanded combination of parameters.

    """
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
    """Represents an inventory of hazard models and their associated colormaps.

    Args:
        models (List[HazardResource]): A list of hazard resources.
        colormaps (dict): A dictionary mapping hazard types to their colormap configurations.

    """

    models: List[HazardResource]
    colormaps: dict


def inventory_json(models: Iterable[HazardResource]) -> str:
    """Convert a collection of hazard resources into a JSON string.

    Args:
        models (Iterable[HazardResource]): An iterable of hazard resource objects.

    Returns:
        str: A JSON string representation of the hazard inventory.

    """
    response = HazardInventory(models=models)  # type: ignore
    return json.dumps(response.dict())


def paths_for_resources(resources: List[HazardResource], include_maps: bool = True):
    """List all the paths (to arrays or DataSets) for the HazardResources listed."""
    paths = []
    for resource in resources:
        for scenario in resource.scenarios:
            for year in scenario.years:
                path = resource.path.format(scenario=scenario.id, year=year)
                paths.append(path)
                if include_maps:
                    assert resource.map is not None
                    map_path = resource.map.path.format(scenario=scenario.id, year=year)
                    paths.append(map_path)
    return paths
