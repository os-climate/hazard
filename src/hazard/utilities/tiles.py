import logging
import math
import os
import posixpath
from typing import Optional, Sequence

import mercantile
import numpy as np  # type: ignore
import rasterio  # type: ignore
import xarray as xr
from rasterio import CRS  # type: ignore
from rasterio.warp import Resampling  # type: ignore

from hazard.inventory import HazardResource  # type: ignore
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import xarray_utilities

logger = logging.getLogger(__name__)


def create_tiles_for_resource(source: OscZarr, target: OscZarr, resource: HazardResource):
    if resource.map is None or resource.map.source != "map_array_pyramid":
        raise ValueError("resource does not specify 'map_array_pyramid' map source.")
    indices = None
    resources = resource.expand()
    for resource in resources:
        for scenario in resource.scenarios:
            for year in scenario.years:
                path = resource.path.format(scenario=scenario.id, year=year)
                assert resource.map is not None
                map_path = resource.map.path.format(scenario=scenario.id, year=year)
                if resource.map.index_values is not None:
                    da = source.read(path)
                    indexes = list(da["index"].values)
                    indices = [indexes.index(v) for v in resource.map.index_values]
                create_tile_set(source, path, target, map_path, indices=indices)


# def create_image_set_for_resource(source: OscZarr, target: OscZarr, resource: HazardResource):


def create_image_set(
    source: OscZarr,
    source_path: str,
    target: OscZarr,
    target_path: str,
    index_slice=slice(-1, None),
    reprojection_threads=8,
    nodata=None,
):
    da = xarray_utilities.normalize_array(source.read(source_path))
    _create_image_set(
        da,
        target,
        target_path,
        index_slice=index_slice,
        reprojection_threads=reprojection_threads,
        nodata=nodata,
    )


def _create_image_set(
    source: xr.DataArray,
    target: OscZarr,
    target_path: str,
    index_slice=slice(-1, None),
    reprojection_threads=8,
    nodata=None,
):
    return_periods = source.index.data
    index_slice = slice(-1, None)
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
    src_left, src_bottom, src_right, src_top = source.rio.bounds()
    # we have to truncate into range -85 to 85
    src_bottom, src_top = (
        max(src_bottom, -85.0),
        min(src_top, 85.0),
    )
    src_width, src_height = source.sizes["longitude"], source.sizes["latitude"]
    _, width, height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=src_width,
        height=src_height,
        left=src_left,
        bottom=src_bottom,
        right=src_right,
        top=src_top,
    )
    dst_width, dst_height = src_width, max(src_height, height)
    ((dst_left, dst_right), (dst_bottom, dst_top)) = rasterio.warp.transform(
        src_crs, dst_crs, xs=[src_left, src_right], ys=[src_bottom, src_top]
    )
    dst_transform = rasterio.transform.from_bounds(dst_left, dst_bottom, dst_right, dst_top, dst_width, dst_height)

    _ = target.create_empty(
        target_path,
        dst_width,
        dst_height,
        dst_transform,
        dst_crs,
        indexes=return_periods,
        chunks=(1, 4000, 4000),
    )

    indices = range(len(return_periods))[index_slice]
    for index in indices:
        da_m = source[index, :, :].rio.reproject(
            dst_crs,
            shape=(dst_height, dst_width),
            transform=dst_transform,
            nodata=nodata,
        )
        target.write_slice(target_path, slice(index, index + 1), slice(None), slice(None), da_m.data)


def create_tile_set(
    source: OscZarr,
    source_path: str,
    target: OscZarr,
    target_path: str,
    indices: Optional[Sequence[int]] = None,
    max_tile_batch_size=64,
    reprojection_threads=8,
    nodata=None,
    nodata_as_zero=False,
    check_fill=False,
):
    """Create a set of EPSG:3857 (i.e. Web Mercator) tiles according to the
    Slippy Map standard.

    Args:
        source (OscZarr): OSC Zarr array source. Array is (z, y, x) where x and y are
        spatial coordinates and z is index coordinate (often return period).
        source_path (str): OSC Zarr source path.
        target (OscZarr): OSC Zarr array target.
        target_path (str): OSC Zarr target path. Arrays are stored in {target_path}/{level},
        e.g. my_hazard_indicator/2 for level 2.
        indices (Sequence[int], optional): Indices for which map should be generated.
        max_tile_batch_size (int, optional): Maximum number of tiles in x and y direction
        that can be processed simultaneously.
        Defaults to 64 i.e. max image size is 256 * 64 x 256 * 64 pixels.
        reprojection_threads (int, optional): Number of threads to use to perform reprojection
        to EPSG:3857.
        Defaults to 8.
    """
    if not target_path.endswith("map"):
        # for safety; should end with 'map' to avoid clash
        raise ValueError("invalid target path {target_path}")

    da = source.read(source_path)  # .to_dataset()
    return_periods = da.index.data

    _, _, src_width = (
        da.sizes["index"],
        da.sizes["latitude"],
        da.sizes["longitude"],
    )
    dst_crs = CRS.from_epsg(3857)

    max_level = int(round(math.log2(src_width / 256.0)))
    pixels_per_tile = 256
    chunk_size = 512
    if indices is None:
        index_slice = slice(-1, None)
        indices = range(len(return_periods))[index_slice]
    logger.info(f"Starting map tiles generation for array {source_path}.")
    max_dimension = 2**max_level * pixels_per_tile
    logger.info(f"Source array size is {da.shape} (z, y, x).")
    logger.info(f"Indices (z) subset to be processed: {indices}.")
    logger.info(f"Maximum zoom is level is {max_level}, i.e. of size ({max_dimension}, {max_dimension}) pixels].")

    target.remove(target_path)

    os.environ["CHECK_WITH_INVERT_PROJ"] = "YES"
    for level in range(max_level, 0, -1):
        logger.info(f"Starting level {level}.")
        level_path = posixpath.join(target_path, f"{level}")

        tiles = 2**level
        dst_dim = tiles * pixels_per_tile
        ulx, uly = mercantile.xy(*mercantile.ul(mercantile.Tile(x=0, y=0, z=level)))
        lrx, lry = mercantile.xy(*mercantile.ul(mercantile.Tile(x=tiles, y=tiles, z=level)))
        whole_map_transform = rasterio.transform.from_bounds(ulx, lry, lrx, uly, dst_dim, dst_dim)
        # i.e. chunks are <path>/<z>/<index>.<y>.<x>, e.g. flood/4/4.2.3
        _ = target.create_empty(
            level_path,
            dst_dim,
            dst_dim,
            whole_map_transform,
            dst_crs,
            indexes=return_periods,
            chunks=(1, chunk_size, chunk_size),
        )

        num_batches = max(1, 2**level // max_tile_batch_size)
        tile_batch_size = min(2**level, max_tile_batch_size)
        for index in indices:
            logger.info(f"Starting index {index}.")
            da_index = da[index, :, :]  # .compute()
            if nodata_as_zero:
                da_index.data[np.isnan(da_index.data)] = 0
                if nodata:
                    da_index.data[da_index.data == nodata] = 0
            for batch_x in range(0, num_batches):
                for batch_y in range(0, num_batches):
                    x_slice = slice(batch_x * tile_batch_size, (batch_x + 1) * tile_batch_size)
                    y_slice = slice(batch_y * tile_batch_size, (batch_y + 1) * tile_batch_size)
                    ulx, uly = mercantile.xy(*mercantile.ul(mercantile.Tile(x=x_slice.start, y=y_slice.start, z=level)))
                    lrx, lry = mercantile.xy(*mercantile.ul(mercantile.Tile(x=x_slice.stop, y=y_slice.stop, z=level)))
                    logger.info(f"Processing batch ({batch_x}/{num_batches}, {batch_y}/{num_batches}).")
                    dst_transform = rasterio.transform.from_bounds(
                        ulx,
                        lry,
                        lrx,
                        uly,
                        pixels_per_tile * tile_batch_size,
                        pixels_per_tile * tile_batch_size,
                    )
                    # if necessary per batch we could operate on just a chunk of da
                    da_m = da_index.rio.reproject(
                        "EPSG:3857",
                        resampling=Resampling.bilinear,
                        shape=(
                            pixels_per_tile * tile_batch_size,
                            pixels_per_tile * tile_batch_size,
                        ),
                        transform=dst_transform,
                        num_threads=reprojection_threads,
                        nodata=nodata,
                    )
                    logger.info(f"Reprojection complete. Writing to target {level_path}.")

                    if check_fill:
                        da_m.data = xr.where(da_m.data > 3.4e38, np.nan, da_m.data)

                    target.write_slice(
                        level_path,
                        slice(index, index + 1),
                        slice(
                            y_slice.start * pixels_per_tile,
                            y_slice.stop * pixels_per_tile,
                        ),
                        slice(
                            x_slice.start * pixels_per_tile,
                            x_slice.stop * pixels_per_tile,
                        ),
                        da_m.data,
                    )

                    logger.info("Batch complete.")
