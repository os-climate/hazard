from typing import Iterable, List, Optional
from pydantic import BaseModel


class IndexValue(BaseModel):
    index: int
    value: float


class MapInfo(BaseModel):
    """Provides information about map layer"""
    colormap: Optional[str]
    nodata_index: Optional[int] # usually index 0
    colormap_id: Optional[str]
    colormap_min: Optional[IndexValue] # usually index 1
    colormap_max: Optional[IndexValue] # usually index 255


class Period(BaseModel):
    """A period belonging to a scenario"""
    year: int
    map_id: str


class Scenario(BaseModel):
    """Scenario ID and the list of available years for that scenario e.g. RCP8.5 = 'rcp8.5'"""
    id: str
    years: List[int]
    periods: Optional[List[Period]]


class HazardModel(BaseModel):
    """Provides the scenarios associated with a hazard model."""
    type: str # alias of event type
    path: str
    id: str
    display_name: str 
    description: str
    array_name: str # alias of filename
    map: Optional[MapInfo]
    scenarios: List[Scenario]
    units: str


class HazardInventory(BaseModel):
    models: List[HazardModel]
    colormaps: dict

def generate_inventory(models: Iterable[HazardModel], colormaps: dict) -> str:
    response = HazardInventory(models=models, colormaps=colormaps)  # type: ignore
    ...

def update_inventory_s3(inventory: HazardInventory, s3=None, test: bool=True):
    """Update the entry in the OS-Climate S3."""



def to_hazard_models(self) -> List[HazardModel]:
    models = parse_obj_as(List[HazardModel], self.models)

    # we populate map_id hashes programmatically
    for model in models:

        for scenario in model.scenarios:
            test_periods = scenario.periods
            scenario.periods = []
            for year in scenario.years:
                name_format = (
                    model.filename if model.map is None or model.map.filename is None else model.map.filename
                )
                filename = name_format.format(scenario=scenario.id, year=year, id=model.id, return_period=1000)
                id = alphanumeric(filename)[0:6]
                scenario.periods.append(Period(year=year, map_id=id))
            # if a period was specifed explicitly, we check that hash is the same: a build-in check
            if test_periods is not None:
                for (period, test_period) in zip(scenario.periods, test_periods):
                    if period.map_id != test_period.map_id:
                        raise Exception(
                            f"validation error: hash {period.map_id} different to specified hash {test_period.map_id}"  # noqa: E501
                        )
    return models

def colormaps(self):
    """Color maps. Key can be identical to a model identifier or more descriptive (if shared by many models)."""
    return colormap_provider.colormaps()


def alphanumeric(text):
    """Return alphanumeric hash from supplied string."""
    hash_int = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest(), "big")
    return base36encode(hash_int)


def base36encode(number, alphabet="0123456789abcdefghijklmnopqrstuvwxyz"):
    """Converts an integer to a base36 string."""
    if not isinstance(number, int):
        raise TypeError("number must be an integer")

    base36 = ""

    if number < 0:
        raise TypeError("number must be positive")

    if 0 <= number < len(alphabet):
        return alphabet[number]

    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36

    return base36

