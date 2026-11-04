"""Conventions for the resources and descriptions of indicators derived from ThresholdBasedAverageIndicator."""

import os

from typing_extensions import Sequence

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import MULTI_MODEL_ID, RunConfig
from hazard.utilities.description_utilities import get_indicator_period_descriptions

PATH_PREFIX_V3 = "chronic_heat/osc/v3"


def threshold_indicator_resource(
    *,
    run: RunConfig,
    thresholds: Sequence[float],
    indicator_id: str,
    path_name: str,
    display_name: str,
    display_group: str,
    description: str,
    units: str,
    max_value: float,
    hazard_type: str = "ChronicHeat",
    path_prefix: str = PATH_PREFIX_V3,
) -> HazardResource:
    """Create a resource for a threshold-indexed indicator, with the conventions for paths, GCMs, scenarios and maps.

    Args:
        run (RunConfig): Run settings, giving the GCMs (and whether the multi-model mean is
            included), scenarios and resolution of the resource.
        thresholds (Sequence[float]): Thresholds, i.e. the values of the index dimension.
        indicator_id (str): Indicator ID.
        path_name (str): Name of the array; the path is
            "{path_prefix}/{path_name}_{gcm}_{scenario}_{year}" and the map path that prefixed by "maps/".
        display_name (str): Display name; should include "({gcm})".
        display_group (str): Display group.
        description (str): Description; see indicator_description.
        units (str): Units of the indicator.
        max_value (float): Value at the top of the map's colour scale.
        hazard_type (str, optional): Hazard type. Defaults to "ChronicHeat".
        path_prefix (str, optional): Prefix of the path, which gives the version of the data.
            Defaults to "chronic_heat/osc/v3".

    """
    path = f"{path_prefix}/{path_name}" + "_{gcm}_{scenario}_{year}"
    return HazardResource(
        hazard_type=hazard_type,
        indicator_id=indicator_id,
        indicator_model_id=None,
        indicator_model_gcm="{gcm}",
        params={"gcm": ([MULTI_MODEL_ID] if run.multi_model else []) + list(run.gcms)},
        path=path,
        display_name=display_name,
        description=description,
        display_groups=[display_group],
        group_id="",
        license="Creative Commons",
        source="",
        version="",
        resolution=run.resolution,
        attribution="",
        map=MapInfo(
            colormap=Colormap(
                name="heating",
                nodata_index=0,
                min_index=1,
                min_value=0.0,
                max_value=max_value,
                max_index=255,
                units=units,
                # the indicators can be zero, so cannot be shown on a log scale
                scaling="linear",
            ),
            bbox=[-180.0, -60.0, 180.0, 85.0],
            index_values=list(thresholds),
            path="maps/" + path + "_map",
            source="map_array_pyramid",
        ),
        units=units,
        scenarios=[
            Scenario(
                id=scenario,
                years=(
                    [run.central_year_historical]
                    if scenario == "historical"
                    else list(run.central_years)
                ),
            )
            for scenario in run.scenarios
        ],
    )


def indicator_description(description_file: str, run: RunConfig) -> str:
    """Description made up of the contents of description_file, the source dataset, the GCMs and the periods.

    Args:
        description_file (str): Name of the file describing the indicator, in the directory of this module.
        run (RunConfig): Run settings, giving the source dataset, GCMs and periods.

    """
    models_dir = os.path.dirname(__file__)
    with open(os.path.join(models_dir, description_file), "r") as f:
        description = f.read()
    source_dataset_file = run.source_dataset.lower().replace("-", "_") + ".md"
    with open(
        os.path.join(os.path.dirname(models_dir), "sources", source_dataset_file), "r"
    ) as f:
        description += "\n" + f.read()
    description += (
        "\nIndicators are generated for each of the Global Climate Models (GCMs) "
        + ", ".join(run.gcms)
        + (
            f" and for the mean of these ('{MULTI_MODEL_ID}')."
            if run.multi_model
            else "."
        )
    )
    description += "\n" + get_indicator_period_descriptions(
        run.scenarios,
        run.central_year_historical,
        run.central_years,
        run.window_years,
    )
    return description
