import logging
import math
import os
import posixpath
from typing import Any, Optional, Sequence, Tuple

import mercantile
import numpy as np  # type: ignore
import rasterio  # type: ignore
import rasterio.coords
import rasterio.transform
import rasterio.warp
import rioxarray  # noqa: F401
import xarray as xr
from rasterio import CRS  # type: ignore
from rasterio.warp import Resampling  # type: ignore

from hazard.inventory import HazardResource  # type: ignore
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import xarray_utilities

logger = logging.getLogger(__name__)


def create_tiles_for_resource(
    source: OscZarr,
    target: OscZarr,
    resource: HazardResource,
    max_zoom: Optional[int] = None,
    nodata=None,
    nodata_as_zero=False,
    nodata_as_zero_coarsening=False,
    check_fill=False,
):
    """Create a set of EPSG:3857 (i.e. Web Mercator) tiles according to the
    Slippy Map standard.

    Args:
        source (OscZarr): OSCZarr source or arrays. Arrays are (z, y, x) where x and y are
            spatial coordinates and z is index coordinate (often return period).
        target (OscZarr): OSCZarr array target.
        resource (HazardResource): Resource for which map tiles should be calculated.
        max_zoom (Optional[int], optional): Maximum zoom level; inferred if not specified. Defaults to None.
        nodata (Optional[float], optional): If specified, set the nodata value. Defaults to None in which case
            this is inferred from array.
        nodata_as_zero (bool, optional): If True, nodata is set to zero for purposes of reprojection to highest zoom
            level image. Defaults to False.
        nodata_as_zero_coarsening (bool, optional): If True, nodata is set to zero for generating coarser tiles from
            the highest zoom level image. Use, e.g. for flood maps if nodata is NaN and represents no flood risk.
            Defaults to False.
        check_fill (bool, optional): If True treat infinity values as nodata. Defaults to False.
    """
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
                    index_dim = da.dims[0]  # should be the index dimension
                    indexes = list(da[index_dim].values)
                    indices = [indexes.index(v) for v in resource.map.index_values]
                create_tile_set(
                    source,
                    path,
                    target,
                    map_path,
                    indices=indices,
                    max_zoom=max_zoom,
                    nodata=nodata,
                    nodata_as_zero=nodata_as_zero,
                    nodata_as_zero_coarsening=nodata_as_zero_coarsening,
                    check_fill=check_fill,
                )


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
    dst_transform = rasterio.transform.from_bounds(
        dst_left, dst_bottom, dst_right, dst_top, dst_width, dst_height
    )

    _ = target.create_empty(
        target_path,
        dst_width,
        dst_height,
        dst_transform,
        dst_crs,
        index_values=return_periods,
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
        target.write_slice(
            target_path, slice(index, index + 1), slice(None), slice(None), da_m.data
        )


def create_tile_set(
    source: OscZarr,
    source_path: str,
    target: OscZarr,
    target_path: str,
    indices: Optional[Sequence[int]] = None,
    max_tile_batch_size: int = 32,
    reprojection_threads: int = 8,
    max_zoom: Optional[int] = None,
    nodata: Optional[float] = None,
    nodata_as_zero: bool = False,
    nodata_as_zero_coarsening: bool = False,
    check_fill: bool = False,
):
    """Create a set of EPSG:3857 (i.e. Web Mercator) tiles according to the
    Slippy Map standard.

    Args:
        source (OscZarr): OSCZarr source or arrays. Arrays are (z, y, x) where x and y are
            spatial coordinates and z is index coordinate (often return period).
        source_path (str): Path to the array accessed via the OSCZarr instance.
        target (OscZarr): OSCZarr array target.
        target_path (str): OSCZarr target path. Arrays are stored in {target_path}/{level},
            e.g. my_hazard_indicator/2 for level 2.
        indices (Sequence[int], optional): Indices for which map should be generated.
        max_tile_batch_size (int, optional): Maximum number of tiles in x and y direction
        that can be processed simultaneously.
            Defaults to 64 i.e. max image size is 256 * 64 x 256 * 64 pixels.
        reprojection_threads (int, optional): Number of threads to use to perform reprojection
        to EPSG:3857.
            Defaults to 8.
        max_zoom (Optional[int], optional): Maximum zoom level; inferred if not specified. Defaults to None.
        nodata (Optional[float], optional): If specified, set the nodata value. Defaults to None in which case
            this is inferred from array.
        nodata_as_zero (bool, optional): If True, nodata is set to zero for purposes of reprojection to highest zoom
            level image. Defaults to False.
        nodata_as_zero_coarsening (bool, optional): If True, nodata is set to zero for generating coarser tiles from
            the highest zoom level image. Use, e.g. for flood maps if nodata is NaN and represents no flood risk.
            Defaults to False.
        check_fill (bool, optional): If True treat infinity values as nodata. Defaults to False.
    """
    if not target_path.endswith("map"):
        # for safety; should end with 'map' to avoid clash
        raise ValueError("invalid target path {target_path}")

    da = source.read(source_path)  # .to_dataset()
    index_values = da[da.dims[0]].data

    src_width = da.sizes["longitude"] if "longitude" in da.sizes else da.sizes["x"]

    chunk_size: int = 512
    pixels_per_tile = 256
    os.environ["CHECK_WITH_INVERT_PROJ"] = "YES"

    # to calculate the maximum zoom level we find the bounds in the map projection
    # and calculate the number of pixels across full x range
    left, bottom, right, top = da.rio.transform_bounds(
        "EPSG:4326"
    )  # left, bottom, right, top or west, south, east, north
    ulx, uly = mercantile.xy(left, top)
    lrx, lry = mercantile.xy(right, bottom)
    ulx_whole, uly_whole = mercantile.xy(*mercantile.ul(mercantile.Tile(0, 0, 0)))
    lrx_whole, lry_whole = mercantile.xy(*mercantile.ul(mercantile.Tile(1, 1, 0)))
    whole_map_pixels_x = src_width * (lrx_whole - ulx_whole) / (lrx - ulx)
    max_zoom = (
        int(round(math.log2(whole_map_pixels_x / pixels_per_tile)))
        if max_zoom is None
        else max_zoom
    )

    if indices is None:
        index_slice = slice(-1, None)
        indices = range(len(index_values))[index_slice]
    logger.info(f"Starting map tiles generation for array {source_path}.")
    logger.info(f"Source array size is {da.shape} (z, y, x).")
    logger.info(f"Indices (z) subset to be processed: {indices}.")
    logger.info(f"Maximum zoom is level is {max_zoom}.")

    # _coarsen(target, target_path, max_zoom, (left, bottom, right, top), indices)
    target.remove(target_path)
    _create_empty_tile_pyramid(
        target,
        target_path,
        max_zoom,
        chunk_size,
        index_name=str(da.dims[0]),
        index_values=index_values,
    )

    # here we reproject and write the maximum zoom level
    _write_zoom_level(
        da,
        target,
        target_path,
        max_zoom,
        indices,
        max_tile_batch_size=max_tile_batch_size,
        reprojection_threads=reprojection_threads,
        nodata=nodata,
        nodata_as_zero=nodata_as_zero,
        check_fill=check_fill,
    )

    # and then progressively coarsen each level and write until we reach level 0"
    _coarsen(
        target,
        target_path,
        max_zoom,
        (left, bottom, right, top),
        indices,
        nodata_as_zero_coarsening=nodata_as_zero_coarsening,
    )


def _create_empty_tile_pyramid(
    target: OscZarr,
    target_path: str,
    max_zoom: int,
    chunk_size: int,
    index_name: str = "index",
    index_values: Sequence[Any] = [0],
):
    pixels_per_tile: int = 256
    ulx, uly = mercantile.xy(*mercantile.ul(mercantile.Tile(x=0, y=0, z=0)))
    lrx, lry = mercantile.xy(*mercantile.ul(mercantile.Tile(x=1, y=1, z=0)))
    dst_crs = CRS.from_epsg(3857)
    for zoom in range(max_zoom, 0, -1):
        tiles = 2**zoom
        dst_dim = tiles * pixels_per_tile
        zoom_level_path = posixpath.join(target_path, f"{zoom}")
        whole_map_transform = rasterio.transform.from_bounds(
            ulx, lry, lrx, uly, dst_dim, dst_dim
        )
        # i.e. chunks are <path>/<z>/<index>.<y>.<x>, e.g. flood/4/4.2.3
        # we create an array for the whole world, but if the map covers just a fraction then all other
        # chunks will be empty
        _ = target.create_empty(
            zoom_level_path,
            dst_dim,
            dst_dim,
            whole_map_transform,
            dst_crs,
            index_name=index_name,
            index_values=index_values,
            chunks=(1, chunk_size, chunk_size),
        )


def _write_zoom_level(
    da: xr.DataArray,
    target: OscZarr,
    target_path: str,
    zoom: int,
    indices: Optional[Sequence[int]] = None,
    max_tile_batch_size: int = 32,
    reprojection_threads: int = 8,
    nodata=None,
    nodata_as_zero=False,
    check_fill=False,
):
    logger.info(f"Starting writing of level {zoom}.")
    pixels_per_tile: int = 256

    left, bottom, right, top = da.rio.transform_bounds("EPSG:4326")
    # we find just the tiles that contain the map area
    xmin, xmax, ymin, ymax = get_tile_bounds(left, bottom, right, top, zoom)
    dst_crs = CRS.from_epsg(3857)
    zoom_level_path = posixpath.join(target_path, f"{zoom}")
    ntiles_in_level = 2**zoom
    ntiles = max(xmax - xmin + 1, ymax - ymin + 1)
    num_batches = max(1, math.ceil(ntiles / max_tile_batch_size))
    tile_batch_size = min(ntiles, max_tile_batch_size)
    for index in indices if indices is not None else [0]:
        logger.info(f"Starting index {index}.")
        da_index = da[index, :, :]  # .compute()
        if nodata_as_zero:
            da_index.data[np.isnan(da_index.data)] = 0
            if nodata:
                da_index.data[da_index.data == nodata] = 0
        for batch_x in range(0, num_batches):
            for batch_y in range(0, num_batches):
                x_slice = slice(
                    xmin + batch_x * tile_batch_size,
                    min(xmin + (batch_x + 1) * tile_batch_size, ntiles_in_level),
                )
                y_slice = slice(
                    ymin + batch_y * tile_batch_size,
                    min(ymin + (batch_y + 1) * tile_batch_size, ntiles_in_level),
                )
                ulx, uly = mercantile.xy(
                    *mercantile.ul(
                        mercantile.Tile(x=x_slice.start, y=y_slice.start, z=zoom)
                    )
                )
                lrx, lry = mercantile.xy(
                    *mercantile.ul(
                        mercantile.Tile(x=x_slice.stop, y=y_slice.stop, z=zoom)
                    )
                )
                logger.info(
                    f"Processing batch ({batch_x}/{num_batches}, {batch_y}/{num_batches})."
                )
                dst_transform = rasterio.transform.from_bounds(
                    ulx,
                    lry,
                    lrx,
                    uly,
                    (y_slice.stop - y_slice.start) * pixels_per_tile,
                    (x_slice.stop - x_slice.start) * pixels_per_tile,
                )
                # at this point, we could take just the portion of the source
                # that covers the batch of tiles
                # calculate bounds of tiles in source CRS using rasterio.warp.transform_bounds
                da_trimmed = trim_array(da_index, dst_crs, ulx, lry, lrx, uly)
                if da_trimmed is None:
                    # no overlap with batch
                    continue

                da_trimmed.attrs["nodata"] = float("nan")

                da_m = da_trimmed.rio.reproject(
                    "EPSG:3857",
                    resampling=Resampling.bilinear,
                    shape=(
                        (y_slice.stop - y_slice.start) * pixels_per_tile,
                        (x_slice.stop - x_slice.start) * pixels_per_tile,
                    ),
                    transform=dst_transform,
                    num_threads=reprojection_threads,
                    nodata=nodata,
                )

                logger.info(
                    f"Reprojection complete. Writing to target {zoom_level_path}."
                )

                if check_fill:
                    da_m.data = xr.where(da_m.data > 3.4e38, np.nan, da_m.data)

                target.write_slice(
                    zoom_level_path,
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


def get_tile_bounds(left: float, bottom: float, right: float, top: float, zoom: int):
    ul = mercantile.tile(left, min(top, 89.9999), zoom)
    lr = mercantile.tile(right, max(bottom, -89.9999), zoom)
    return ul.x, lr.x, ul.y, lr.y


def _coarsen(
    target: OscZarr,
    target_path: str,
    max_zoom: int,
    bounds: Tuple[float, float, float, float],
    indices: Optional[Sequence[int]],
    max_tile_batch_size: int = 16,
    nodata_as_zero_coarsening: bool = False,
):
    pixels_per_tile = 256
    bbox = rasterio.coords.BoundingBox(*bounds)
    for index in indices if indices is not None else [0]:
        for zoom in range(max_zoom, 1, -1):
            current_zoom_level_path = posixpath.join(target_path, f"{zoom}")
            next_zoom_level_path = posixpath.join(target_path, f"{zoom - 1}")
            z = target.read_zarr(current_zoom_level_path)
            # we go down a zoom level to find the bounds
            xmin, xmax, ymin, ymax = get_tile_bounds(
                bbox.left, bbox.bottom, bbox.right, bbox.top, zoom - 1
            )
            xmin, ymin = xmin * 2, ymin * 2
            xmax, ymax = xmax * 2 + 1, ymax * 2 + 1
            ntiles_in_level = 2**zoom
            ntiles = max(xmax - xmin + 1, ymax - ymin + 1)
            num_batches = max(1, math.ceil(ntiles / max_tile_batch_size))
            tile_batch_size = min(ntiles, max_tile_batch_size)
            # in this case we can process the batches in parallel: consider multi-threading this part?
            for batch_x in range(0, num_batches):
                for batch_y in range(0, num_batches):
                    x_slice = slice(
                        xmin + batch_x * tile_batch_size,
                        min(xmin + (batch_x + 1) * tile_batch_size, ntiles_in_level),
                    )
                    y_slice = slice(
                        ymin + batch_y * tile_batch_size,
                        min(ymin + (batch_y + 1) * tile_batch_size, ntiles_in_level),
                    )
                    zslice = z[
                        index,
                        y_slice.start * pixels_per_tile : y_slice.stop
                        * pixels_per_tile,
                        x_slice.start * pixels_per_tile : x_slice.stop
                        * pixels_per_tile,
                    ]

                    # use Zarr array directly, as opposed to e.g.:
                    # da_slice = da_slice.coarsen(x=2, y=2).mean()  # type:ignore
                    if nodata_as_zero_coarsening:
                        if not np.all(np.isnan(zslice)):
                            zslice[np.isnan(zslice)] = 0

                    zslice = (
                        zslice[::2, ::2]
                        + zslice[1::2, ::2]
                        + zslice[::2, 1::2]
                        + zslice[1::2, 1::2]
                    ) / 4
                    target.write_slice(
                        next_zoom_level_path,
                        slice(index, index + 1),
                        slice(
                            int(y_slice.start * pixels_per_tile / 2),
                            int(y_slice.stop * pixels_per_tile / 2),
                        ),
                        slice(
                            int(x_slice.start * pixels_per_tile / 2),
                            int(x_slice.stop * pixels_per_tile / 2),
                        ),
                        zslice,
                    )


def trim_array(da_index: xr.DataArray, crs, left, bottom, right, top):
    y_dim, x_dim = da_index.dims
    da_left, da_bottom, da_right, da_top = rasterio.warp.transform_bounds(
        crs, da_index.rio.crs, left, bottom, right, top
    )
    # find pixels required in order to add a pixel buffer
    x = np.where((da_index[x_dim] >= da_left) & (da_index[x_dim] <= da_right))[0]
    y = np.where((da_index[y_dim] >= da_bottom) & (da_index[y_dim] <= da_top))[0]
    if len(x) == 0 or len(y) == 0:
        return None
    # add a 2 pixel buffer
    xmin, xmax = max(0, x[0] - 2), min(len(da_index[x_dim]), x[-1] + 2)
    ymin, ymax = max(0, y[0] - 2), min(len(da_index[y_dim]), y[-1] + 2)
    return da_index[ymin:ymax, xmin:xmax]
