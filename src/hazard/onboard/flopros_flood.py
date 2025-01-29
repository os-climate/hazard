import logging
import os
import math
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
from dask.distributed import Client
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from rasterio import features
from rasterio.enums import MergeAlg
import xarray as xr

from hazard.indicator_model import IndicatorModel
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.protocols import OpenDataset, ReadWriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.download_utilities import download_and_unzip
from hazard.utilities.tiles import create_tiles_for_resource
from hazard.utilities.xarray_utilities import (
    empty_data_array,
    global_crs_transform,
)


logger = logging.getLogger(__name__)


class FLOPROSFloodStandardOfProtectionSource(OpenDataset):
    def __init__(self, source_dir, fs: Optional[AbstractFileSystem] = None):
        """Source that can provide FLOPROS data as an XArray raster.

        Args:
            source_dir (str): directory containing source files. If fs is a S3FileSystem instance
            <bucket name>/<prefix> is expected.
            fs (Optional[AbstractFileSystem], optional): AbstractFileSystem instance.
            If None, a LocalFileSystem is used.
        """
        self.fs = fs if fs else LocalFileSystem()
        self.source_dir = source_dir
        self.zip_url = "https://nhess.copernicus.org/articles/16/1049/2016/nhess-16-1049-2016-supplement.zip"
        self.archive_name = self.zip_url.split("/")[-1].split(".")[0]
        self.prepare()

    def prepare(self, working_dir: Optional[str] = None):
        if not isinstance(self.fs, LocalFileSystem):
            # e.g. we are copying to S3;  download to specified working directory, but then copy to self.source_dir
            assert working_dir is not None
            download_and_unzip(self.zip_url, working_dir, self.archive_name)
            for file in os.listdir(working_dir):
                with open(file, "rb") as f:
                    self.fs.write_bytes(PurePosixPath(self.source_dir, file), f.read())
        else:
            # download and unzip directly in location
            download_and_unzip(self.zip_url, self.source_dir, self.archive_name)
        logger.info("Reading database into GeoDataFrame")
        path = (
            Path(self.source_dir)
            / self.archive_name
            / "Scussolini_etal_Suppl_info"
            / "FLOPROS_shp_V1"
            / "FLOPROS_shp_V1.shp"
        )
        self.df = gpd.read_file(path)

    def open_dataset_year(
        self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    ) -> xr.Dataset:
        """_summary_

        Args:
            gcm (str): Ignored.
            scenario (str): Ignored.
            quantity (str): 'RiverineInundation' or 'CoastalInundation'.
            year (int): Ignored.
            chunks (_type_, optional): _description_. Defaults to None.

        Returns:
            xr.Dataset: Data set named 'indicator' with 'max' and 'min' coordinate labels in the index coordinate.
        """
        hazard_type = quantity

        def get_merged_rp(row, min_max: str, flood_type: str):
            """Calculate min or max from database entry (GeoDataFrame row).

            From the paper: "In practice, if information is available in the design layer for a given sub-country unit, then
            this information is included in the merged layer. If no information is contained in the design layer, then the policy layer
            information is included in the merged layer. Finally, if information is not available even at the policy layer, then the
            model layer information is included in the merged layer."

            Args:
                row: GeoDataFrame row
                flood_type (str, optional): "Riv" or "Co". Defaults to "Riv".

            Returns:
                float: Protection level as return period in years.
            """
            layers = ["DL", "PL", "ModL"] if flood_type == "Riv" else ["DL", "PL"]
            for layer in layers:  # design layer, policy layer, modelled layer
                # note that for the modelled layer, both min and max are set to the modelled value
                layer_rp = (
                    row[f"{layer}_{flood_type}"]
                    if layer == "ModL"
                    else row[f"{layer}_{min_max}_{flood_type}"]
                )
                if layer_rp > 0:  # if 0, layer is considered missing
                    # if design layer is present, use this, otherwise use the policy layer, otherwise the modelled, otherwise missing
                    return layer_rp
            return float("Nan")  # zero is no data, represented by NaN here.

        logger.info(f"Processing hazard type {hazard_type}")
        min_shapes: List[Tuple[float, Any]] = []
        max_shapes: List[Tuple[float, Any]] = []
        logger.info("Inferring max and min protection levels per region")
        for _, row in self.df.iterrows():
            flood_type = (
                "Riv" if hazard_type == "RiverineInundation" else "Co"
            )  # riverine and coastal
            min, max = (
                get_merged_rp(row, "Min", flood_type),
                get_merged_rp(row, "Max", flood_type),
            )
            if row["name"] is None and (min is None and max is None):
                continue
            # if either the min or max is NaN, that is OK: the vulnerability model is expected to deal with that
            if not math.isnan(min) and not math.isnan(max) and min > max:
                # it can occur that for a layer there is only information about minimum
                raise ValueError("unexpected return period")

            if not math.isnan(min):
                min_shapes.append((row.geometry, min))
            if not math.isnan(max):
                max_shapes.append((row.geometry, max))

        resolution_in_arc_mins = 1
        width, height = (
            int(60 * 360 / resolution_in_arc_mins),
            int(60 * 180 / resolution_in_arc_mins),
        )
        crs, transform = global_crs_transform(width, height)
        logger.info("Creating empty array")
        da = empty_data_array(
            width,
            height,
            transform,
            str(crs),
            index_name="min_max",
            index_values=["min", "max"],
        )
        for min_max in ["min", "max"]:
            logger.info(
                f"Creating raster at {(360 * 60) / width} arcmin resolution for {min_max} protection"
            )
            rasterized = features.rasterize(
                min_shapes if min_max == "min" else max_shapes,
                out_shape=[height, width],
                transform=transform,
                all_touched=True,
                fill=float("nan"),  # background value
                merge_alg=MergeAlg.replace,
            )
            index = 0 if min_max == "min" else 1
            da[index, :, :] = rasterized[:, :]
        return da.to_dataset(name="sop")


class FLOPROSFloodStandardOfProtection(IndicatorModel[str]):
    def __init__(self):
        """
        Flood protection standards expressed as return period.

        METADATA:
        Link: https://nhess.copernicus.org/articles/16/1049/2016/
        Data type: Global database of flood protection standards.
        Hazard indicator: Riverine and coastal flood
        Region: Global
        Resolution: Country / country unit
        Scenarios: Not applicable
        Time range: Not applicable
        File type: Shape File (.shx)

        DATA DESCRIPTION:
        FLOod PROtection Standards, FLOPROS, which comprises information in the form of the flood return period
        associated with protection measures, at different spatial scales. FLOPROS comprises three layers
        of information, and combines them into one consistent database. The design layer contains empirical
        information about the actual standard of existing protection already in place; the policy layer contains
        information on protection standards from policy regulations; and the model layer uses a validated modelling
        approach to calculate protection standards.
        """

    def batch_items(self):
        """Get a list of all batch items."""
        return ["min_max"]  # just one!

    def run_single(
        self, item: str, source: Any, target: ReadWriteDataArray, client: Client
    ):
        assert isinstance(source, FLOPROSFloodStandardOfProtectionSource)
        logger.info("Writing rasters")
        for hazard_type, resource in self._resources().items():
            ds = source.open_dataset_year("", "", hazard_type, -1)
            array_name = "sop"
            # note that the co-ordinates will be written into the parent of resource.path
            target.write(
                resource.path,
                ds[array_name].compute(),
                spatial_coords=resource.save_netcdf_coords,
            )

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        for resource in self.inventory():
            create_tiles_for_resource(
                source,
                target,
                resource,
                nodata_as_zero=True,
                nodata_as_zero_coarsening=True,
            )

    def inventory(self) -> Iterable[HazardResource]:
        """Get the inventory item(s)."""
        return self._resources().values()

    def _resources(self) -> Dict[str, HazardResource]:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "flopros_flood.md"), "r"
        ) as f:
            description = f.read()

        def path_component(hazard_type: str):
            return "riverine" if hazard_type == "RiverineInundation" else "coastal"

        return {
            k: HazardResource(
                hazard_type=k,
                indicator_id="flood_sop",
                indicator_model_id="flopros",
                indicator_model_gcm="",
                path=(
                    "inundation/flopros_" + path_component(k) + "/v1/flood_sop/sop"
                ),  # the double path allows an XArray-readable data array to be written
                params={},
                display_name="Standard of protection (FLOPROS)",
                description=description,
                group_id="",
                display_groups=[],
                map=MapInfo(
                    bbox=[],
                    bounds=[],
                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=1500.0,
                        units="years",
                    ),
                    index_values=None,
                    path=f"maps/inundation/flopros_{path_component(k)}/v1/flood_sop_map",
                    source="map_array_pyramid",
                ),
                units="years",
                save_netcdf_coords=True,
                scenarios=[Scenario(id="historical", years=[1985])],
            )
            for k in ["RiverineInundation", "CoastalInundation"]
        }
