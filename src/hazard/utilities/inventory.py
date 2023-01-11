from typing import List, Optional
from pydantic import BaseModel


class MapInfo(BaseModel):
    """Provides information about map layer"""
    colormap: Optional[str]
    filename: Optional[str]
    colormap_id: Optional[str]
    colormap_min_index: Optional[int]
    colormap_min_value: Optional[float]
    colormap_max_index: Optional[int] 
    colormap_max_value: Optional[float]
    

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
    event_type: str
    path: str
    id: str
    display_name: str
    description: str
    filename: str
    map: Optional[MapInfo]
    scenarios: List[Scenario]
    units: str

