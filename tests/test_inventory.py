import os
import tempfile as temp

import pytest
from anyio import Path
from fsspec.implementations import local  # type: ignore

from hazard import get_inventory_resources
from hazard.docs_store import DocStore
from hazard.inventory import Colormap, MapInfo
from hazard.models.deprecated_v2 import (
    DaysTasAboveIndicatorDeprecated,
    HeatingCoolingDegreeDaysDeprecated,
)
from hazard.utilities import zarr_utilities
from hazard.utilities.s3_utilities import get_store  # type: ignore


def test_create_inventory(test_output_dir):
    """Create inventory for the published hazards and write into the test output
    directory (see hazard.get_inventory_hazards for which hazards are included).

    With include_deprecated, deprecated resources (e.g. the v2 chronic heat indicators) are
    included alongside the resources that supersede them.
    """
    include_deprecated = True
    # zarr_utilities.set_credential_env_variables()

    docs_store = DocStore(
        local_path=os.path.join(
            Path(__file__).parents[1], "src", "inventories", "hazard"
        )
    )

    resources = get_inventory_resources(
        test_output_dir, include_deprecated=include_deprecated
    )

    docs_store.write_new_empty_inventory()
    # docs_store.write_inventory_json(json_str)
    docs_store.update_inventory(resources)


def test_inventory_resources_exclude_deprecated(test_output_dir):
    """Both v2 (deprecated) and v3 chronic heat resources are included by default; with
    include_deprecated=False, only the v3 ones are."""
    all_paths = {r.path for r in get_inventory_resources(test_output_dir)}
    current = get_inventory_resources(test_output_dir, include_deprecated=False)
    v2 = "chronic_heat/osc/v2/days_tas_above_{temp_c}c_{gcm}_{scenario}_{year}"
    v3 = "chronic_heat/osc/v3/days_tas_above_{gcm}_{scenario}_{year}"
    assert {v2, v3} <= all_paths
    assert not any(r.deprecated for r in current)
    assert v3 in {r.path for r in current} and v2 not in {r.path for r in current}


def test_colormap_scaling():
    """Colormaps are linear by default; log scaling requires a positive minimum below the maximum."""
    args = dict(name="heating", max_value=100.0, units="m")
    assert Colormap(min_value=0.0, **args).scaling == "linear"
    assert Colormap(min_value=0.1, scaling="log", **args).scaling == "log"
    with pytest.raises(ValueError, match="min_value > 0"):
        Colormap(min_value=0.0, scaling="log", **args)
    with pytest.raises(ValueError, match="max_value > min_value"):
        Colormap(min_value=200.0, scaling="log", **args)


def test_map_info_bounds_omitted_if_none():
    """bounds (deprecated in favour of bbox) is in the inventory JSON only if given."""
    colormap = Colormap(name="heating", min_value=0.0, max_value=1.0, units="m")
    without = MapInfo(colormap=colormap, path="p_map", source="map_array_pyramid")
    assert "bounds" not in without.model_dump() and "bbox" in without.model_dump()
    with_bounds = MapInfo(
        colormap=colormap,
        path="p_map",
        source="map_array_pyramid",
        bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
    )
    assert with_bounds.model_dump()["bounds"] == with_bounds.bounds


def test_deprecation_expanded():
    """superseded_by is expanded along with path; deprecation is in the inventory JSON only if given."""
    resource = DaysTasAboveIndicatorDeprecated()._resource()
    expanded = list(resource.expand())
    assert all(r.deprecated for r in expanded)
    assert {r.deprecation.superseded_by for r in expanded} == {  # type: ignore[union-attr]
        "chronic_heat/osc/v3/days_tas_above_multi_model_0_{scenario}_{year}"
    }
    degree_days = HeatingCoolingDegreeDaysDeprecated()._resource()["above"]
    assert [r.deprecation.superseded_by for r in degree_days.expand()] == [  # type: ignore[union-attr]
        "chronic_heat/osc/v3/cooling_degree_days_ACCESS-CM2_{scenario}_{year}"
    ]
    assert "deprecation" in resource.model_dump()
    assert (
        "deprecation"
        not in resource.model_copy(update={"deprecation": None}).model_dump()
    )


@pytest.mark.skip(reason="just example")
def test_check_inventory(test_output_dir):
    zarr_utilities.set_credential_env_variables()
    temp_dir = temp.TemporaryDirectory()
    local_fs = local.LocalFileSystem(root=temp_dir)
    docs_store = DocStore(fs=local_fs)
    json_str = docs_store.read_inventory_json()
    with open(
        os.path.join(test_output_dir, "hazard_test", "inventory_live.json"), "w"
    ) as f:
        f.write(json_str)


@pytest.mark.skip(reason="Requires credentials")
def test_local_inventory(test_dir):
    root = os.getcwd()
    docs_store = DocStore(local_path=root)
    resources = docs_store.read_inventory()
    resources_list = list(resources)
    assert len(resources_list) > 0


@pytest.mark.skip(reason="Requires credentials")
def test_inventory_functionalities():
    zarr_utilities.set_credential_env_variables()
    local_inventory = "C:\\Users\\vcrespo\\dev_violeta\\hazard\\inventory.json"
    extra_s3fs_kwargs = {
        "key": os.environ.get("OSC_S3_ACCESS_KEY_DEV", None),
        "secret": os.environ.get("OSC_S3_SECRET_KEY_DEV", None),
        "token": os.environ.get("OSC_S3_TOKEN", None),
        "endpoint_url": os.environ.get("OSC_S3_ENDPOINT_DEV", None),
    }
    store = get_store(use_dev=True, extra_s3fs_kwargs=extra_s3fs_kwargs)
    doc_store = DocStore(s3_store=store)
    # find_available_s3_paths()
    paths = doc_store.find_available_s3_paths(store)
    assert len(paths[0]) > 0 and len(paths[1]) > 0
    # check_s3_data()
    check = doc_store.check_s3_data(extra_s3fs_kwargs)
    assert check is True, "Some datasets or maps are empty"
    # check_inventory_paths()
    check, missing_paths = doc_store.check_inventory_paths(
        extra_s3fs_kwargs, local_inventory
    )
    assert check is True, f"This files are missing in the s3 bucket:\n{missing_paths}"
    # check_s3_paths()
    check, missing_paths = doc_store.check_s3_paths(extra_s3fs_kwargs, local_inventory)
    assert check is True, (
        f"This files are orphaned (not referenced by the inventory):\n{missing_paths}"
    )
    print()
