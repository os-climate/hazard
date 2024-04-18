# Hazard

Creation of climate hazard model data sets for OS-Climate.

_Hazard_ is a Python library for creating climate hazard model data sets for OS-Climate applications. The datasets may be simply on-boarded from existing data or derived by transforming other data sources.

An important use of _hazard_ is the preparation of hazard model data sets for use in the [Physrisk](https://github.com/os-climate/physrisk) physical climate risk analysis tool. In general the preparation is in the form of a pipeline, whereby data is sourced, transformed and stored in optimized form (generally to OS-Climate Amazon S3). It is desirable to leverage cloud computing services where tasks are memory-, I/O- and/or compute-intensive.

In line with the the _'treat your data as code'_ approach and to ensure that the creation of any data set for OS-Climate is _repeatable_ and _transparent_, a data set is associated with a particular Git commit of this repository.
A particular data set creation task is a Python script. These can be run on [OS-Climate JupyterHub](https://jupyterhub-odh-jupyterhub.apps.odh-cl2.apps.os-climate.org) environment (as script, notebook or pipeline).

## Introduction to data sets for hazard models

Hazards come in two varieties:

1. _Acute hazards_: **events**, for example heat waves, inundations (floods) and hurricanes, and
2. _Chronic hazards_: long-term shifts in climate parameters such as average temperature, sea-level or water stress indices.

See [methodology document](https://github.com/os-climate/physrisk/tree/main/methodology#:~:text=PhysicalRiskMethodology.pdf) for more details.

Two important types of model used in the assessment of the vulnerability of an asset (natural or financial) to an acute hazard are:

1. _Event-based models_, where the model provides a large number of individual simulated events, actual or plausible, and
2. _Return-period-based models_, where the model rather provides the statistic properties of the ensemble of events.

### Acute hazard model data sets

Return-period-based acute hazard model data sets contain event intensity as a function of return period for different locations. For example, the model might specify that in a certain region flood events with an inundation depth of 50 cm occur with a return period of 10 years (i.e. these are one in 10 year events) and events with an inundation depth of 100 cm occur with a return period of 200 years. In practice, flood models may have a granularity of 10 return periods or more.

An inundation depth of 100 cm for events with a 200 year return period implies that there is a probability of $1/200$ that a flood event occurs in a single year with an inundation depth greater than 100 cm (see [methodology document](https://github.com/os-climate/physrisk/tree/main/methodology#:~:text=PhysicalRiskMethodology.pdf) for discussion of different return period conventions). The probability here is an _exceedance probability_.

The dataset therefore has three dimensions (or axes); two spatial and return period.

### Chronic hazard model data sets

In contrast, chronic hazard model data sets only have the two spatial dimensions, under the convention that a single climate parameter is provided in each data set.

## Dataset creation/transformation: guidelines

[Xarrays](https://docs.xarray.dev/en/stable/) are the main containers used in creating or transforming data sets and are also used as an intermediary format when on-boarding data sets. [Dask is used](https://docs.xarray.dev/en/stable/user-guide/dask.html) to parallelize the calculations in cases where these are memory- or compute-intensive. Xarrays are chosen for convenience and performance when dealing with multidimensional data sets.

## Dataset storage format

In common with other types of geospatial data, hazard model data sets may be raster or vector, regions being determined by cells in the former case and, typically, by polygon boundaries in the latter.

### Raster data

For physical risk calculations, fast look up of data based on a large number (> millions) of latitude and longitude pairs is required. In order to access large multidimensional raster data sets efficiently, the Zarr format is preferred. Zarr is a compressed chunked format in which — in contrast to NetCDF4/HDF5 data — each chunk is a separate object in cloud object stores. This facilitates parallel read and write access, e.g. from a cluster of CPUs. The Zarr format is also convenient when using [xarrays](https://docs.xarray.dev/en/stable/) especially [with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html).

It is worth noting that use of a chunked format does not preclude federation of the data within a database (see for example the approach of [tileDB](https://tiledb.com/)).

### Raster chunk sizes and dimensions

As a guide to chunk size [the Zarr team notes](https://zarr.readthedocs.io/en/stable/tutorial.html) that 'chunks of at least 1 megabyte (1M) uncompressed size seem to provide better performance, at least when using the Blosc compression library.' [The Amazon Best Practices for S3](https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf) moreover recommends making concurrent requests for byte ranges of an object at the granularity of 8-16 MB, in rough agreement to this.

For return-period-based data sets, the recommended dimensions are ('return period', 'latitude', 'longitude'). Each chunk should contain data for all return periods since this is needed for each latitude and longitude. For more efficient compression under the "C" (row-major) layout, return period is the first dimension.

```python
import zarr
import zarr.storage.MemoryStore

# create empty Zarr array containing return period data with 21600 latitudes and 43200 longitudes
shape = (10, 21600, 43200) # ('return period', 'latitude', 'longitude')
store = zarr.storage.MemoryStore(root="hazard.zarr")
root = zarr.open(store=store, mode="w")
z = root.create_dataset("example_array_path",
    shape=(shape[0], shape[1], shape[2]),
    chunks=(shape[0], 1000, 1000),
    dtype="f4"
)
```

Note that each chunk contains all return period data for a spatial region of 1000×1000 cells.

### Affine transforms

An important and common case for raster data sets is that the transform to and from geospatial coordinates (e.g. altitude and longitudes) and raster cell location (e.g. index of the array) is _affine_. [Affine](https://pypi.org/project/affine/) is a convenient library for handling such transforms. Affine transforms are common in the metadata of GeoTIFF files as handled by, for example, the [rasterio](https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html) package.

### Use with xarrays

### Conventions for storage of Zarr arrays

The root for hazard Zarr arrays in a S3 bucket is 'hazard' or 'hazard_test' (for testing).
within this, Zarr arrays are stored in Zarr group hazard/hazard.zarr

The convention for paths to Zarr arrays is:
hazard/hazard.zarr/`<`path to array`>`/`<`version`>`/`<`array name`>`

## Arrays are typically instances of models
