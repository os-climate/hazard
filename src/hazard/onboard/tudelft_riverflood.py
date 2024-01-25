import os
import s3fs
import zarr
import numpy as np
import requests
import zipfile
import xarray as xr
import logging

from pyproj.crs import CRS
from hazard.sources.osc_zarr import OscZarr

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

class TUDELFTRiverFlood_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="RiverineInundation",
                indicator_id="flood_depth",
                indicator_model_gcm = 'historical',
                path="inundation/river_tudelft/v1/flood_depth_{scenario}_{year}",
                params={},
                display_name="River Flood Depth (tudelft)",
                description="""
                GIS-compatible files containing data related to the probability of 
                river floods occurring in Europe under present and future climate. 
                Includes gridded (GeoTIFF) datasets of river flood extents (in two 
                variants, with or without flood protection) and water depths.
                Additionally includes extreme river discharge estimates in ESRI 
                Shapefile format. Based upon CLMcom-CCLM4-8-17-EC-EARTH regional 
                climate simulation (EURO-CORDEX).
                """,
                group_id = "river_tudelft",
                display_groups=[],
                map = MapInfo(
                    bounds= [
                        [
                            -180.0,
                            85.0
                        ],
                        [
                            180.0,
                            85.0
                        ],
                        [
                            180.0,
                            -85.0
                        ],
                        [
                            -180.0,
                            -85.0
                        ]
                    ],

                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=5.0,
                        units="meters"),
                    path="flood_depth_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1971]),
                    Scenario(
                        id="rcp45",
                        years=[2050, 2070]),
                    Scenario(
                        id="rcp85",
                        years=[2050, 2070]),

                    ])]

class TUDELFTRiverFlood():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Delft University of Technology
        river flood data.

        METADATA:
        Link: https://data.4tu.nl/articles/dataset/Pan-European_data_sets_of_river_flood_probability_of_occurrence_under_present_and_future_climate/12708122
        Data type: historical and scenario return period
        Hazard indicator: flood depth
        Region: Pan-Europe
        Resolution: 100m
        Return periods: 10,30,100,300,1000
        Scenarios: RCP4.5, 8.5
        Time range: 1971-2000,2050,2070,2100
        File type: Map (.tif)

        DATA DESCRIPTION:
        GIS-compatible files containing data related to the probability of 
        river floods occurring in Europe under present and future climate. 
        Includes gridded (GeoTIFF) datasets of river flood extents (in two 
        variants, with or without flood protection) and water depths.
        Additionally includes extreme river discharge estimates in ESRI 
        Shapefile format. Based upon CLMcom-CCLM4-8-17-EC-EARTH regional 
        climate simulation (EURO-CORDEX).
        """
        
        # Bucket parameters
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.zarr_storage = zarr_storage

        # create temporary folder to store data
        self.temp_dir = os.path.join(os.getcwd(), temp_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # Download source data
        self.return_periods_str = ['0010', '0030', '0100', '0300', '1000']
        self.return_periods = [10,30,100,300,1000]
        self.zip_urls = ['https://data.4tu.nl/file/df7b63b0-1114-4515-a562-117ca165dc5b/5e6e4334-15b5-4721-a88d-0c8ca34aee17']
        self.zip_filenames = []
        self.path_to_extact_zip = [os.path.join(self.temp_dir, ax.split('.')[0]) for ax in self.zip_filenames]

        # Create tif files names and paths
        roots = ['_1971_2000_hist_', '_2021_2050_RCP45_', '_2021_2050_RCP85_', '_2071_2100_RCP45_', '_2071_2100_RCP85_']
        self.tif_filenames = ['River_flood_depth' + roots[i] for i in range(len(roots))]
        self.tif_paths =  [os.path.join(self.temp_dir, ax) for ax in self.tif_filenames]

        # zarr parameters
        hazard_type = 'inundation'
        data_source_name = 'river_tudelft'
        version = 'v1'
        roots = ['_historical_1971', '_rcp45_2050', '_rcp85_2050', '_rcp45_2070', '_rcp85_2070']
        dataset_names = ['flood_depth' + ax for ax in roots]
        self.group_path_arrays = [os.path.join(hazard_type, data_source_name, version, dataset_name) for dataset_name in dataset_names]

        #
        self.crs_latlon = CRS.from_epsg(4326)

    def connect_to_bucket(self):
        """
        Create connection to s3 bucket using access and secret keys. 
        """

        # Acess key and secret key are stored as env vars OSC_S3_HI_ACCESS_KEY and OSC_S3_HI_SECRET_KEY, resp.
        self.s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], secret=os.environ["OSC_S3_HIdev01_SECRET_KEY"])
        self.group_path = os.path.join(self.bucket_name, self.prefix, self.zarr_storage).replace('\\','/')
        self.store = s3fs.S3Map(root=self.group_path, s3=self.s3, check=False)
        self.root = zarr.group(store=self.store, overwrite=False) 

    def create_OSCZarr_object(self):
        """
        Create helper OSCZarr object to create zarr objects in s3 and write into it.
        """

        # Create OscZarr object to interact with the bucket.
        oscZ = OscZarr(bucket=self.bucket_name,
                prefix=self.prefix,
                s3=self.s3,
                store=self.store)
        
        self.oscZ = oscZ

    def download_file(self, url, path):
        """
        Function to download data from source using http request.
        """

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    def download_and_unzip_datasets(self):
        """
        Function to download every tif file (one per return period) and unzip them.
        """

        for rp_i in range(len(self.return_periods)):
            zip_filename = os.path.join(self.temp_dir, self.zip_filenames[rp_i])
            zip_url = self.zip_urls[rp_i]
            if not os.path.exists(zip_filename):
                self.download_file(zip_url, zip_filename)

            zip_path_to_extact = self.path_to_extact_zip[rp_i]
            if not os.path.exists(zip_path_to_extact):
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(zip_path_to_extact)


    def create_reduced_nc(self):
        """
        Reduce dimension of tif file by 50 and save it as netcdf.
        """

        # Read one tif file to get the metadata: transform, crs, width, height and shape


        for i in range(len(self.tif_paths)):
            for rt_i in range(len(self.return_periods_str)):

                inputfile = self.tif_paths[i] + self.return_periods_str[rt_i] + 'y.tif'
                outfile = self.tif_paths[i] + self.return_periods_str[rt_i] + 'y_reduced.zarr'

                if outfile.split('\\')[-1] in os.listdir(self.temp_dir):
                    continue

                data = xr.open_dataset(inputfile)
                data = data.isel(y=slice(0, -1, 10), x=slice(0, 38370, 10))

                store = zarr.DirectoryStore(outfile)
                data.chunk(1024).to_zarr(store)

                logger.info("Tif file dimension reduced: " + inputfile)


    def merge_das(self, i):
        """
        Merge xr.DataArrays for every return period
        """
        das = []
        for rt_i in range(len(self.return_periods_str)):

            inputfile = self.tif_paths[i] + self.return_periods_str[rt_i] + 'y_reduced.zarr'
            data = xr.open_zarr(inputfile)
            data = data.fillna(0) # interpolate?

            da = data.band_data

            aux = np.array(da.data)
            aux[aux > 10] = 10
            da.data = aux

            da = da.rio.write_crs("EPSG:3035")
            da = da.rio.reproject("EPSG:4326", shape=(da.shape[1], da.shape[2]), nodata=0)

            da = da.rename({'x': 'lon','y': 'lat', 'band':'index'})
            da['lat'] = da.lat.data.round(5)
            da['lon'] = da.lon.data.round(5)

            da['index'] = [self.return_periods[rt_i]]

            das.append(da)

        da_merged = xr.concat(das, 'index')
        da_merged = da_merged.fillna(0)
        # da_merged['index'] = self.return_periods

        logger.info("xr.DataAray merged: " + inputfile)
        return da_merged
                

    def create_map_images(self):
        """
        Create map images.
        
        """

        source = self.oscZ
        source_path = self.group_path_array
        target = source
        target_path = source_path + '_map'

        create_tile_set(source, source_path, target, target_path)


    def onboard_all(self):
        if download_zip:
            logger.info("Downloading data", extra=logger_config)
            self.download_and_unzip_datasets()

        self.create_reduced_nc()

        for i in range(len(self.group_path_arrays)):

            da_merged = self.merge_das(i)

            self.group_path_array = self.group_path_arrays[i]

            logger.info("Pushing to bucket: " + self.group_path_array, extra=logger_config)
            self.oscZ.write(self.group_path_array, da_merged)

            logger.info("Creating image: " + self.group_path_array, extra=logger_config)
            self.create_map_images()         

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger_config = logging.basicConfig(filename='onboarding.log',
                                 level=logging.INFO, 
                                 filemode = 'w', 
                                 format='%(process)d-%(levelname)s-%(message)s') 
    
    # https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
    bucket_name = 'physrisk-hazard-indicators-dev01'
    prefix = 'hazard'
    zarr_storage = 'hazard.zarr'

    temp_dir = 'data'
    download_zip = False
    JRCRiverFlood_ = TUDELFTRiverFlood(bucket_name, prefix, zarr_storage, temp_dir)

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    JRCRiverFlood_.connect_to_bucket()
    JRCRiverFlood_.create_OSCZarr_object()

    JRCRiverFlood_.onboard_all()


