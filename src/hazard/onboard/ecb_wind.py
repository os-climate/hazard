
import os
import s3fs
import boto3
import zarr
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import xarray as xr
import math
from geocube.api.core import make_geocube
import io
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO  # Python 2.x
else:
    from io import StringIO  # Python 3.x

from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

class ECBWind_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Wind",
                indicator_id="gust_speed_level",
                indicator_model_gcm = 'historical',
                path="wind/ecb/v1/gust_speed_level_{scenario}_{year}",
                params={},
                display_name="Wind Gust Speed Level",
                description="""
                ECB Gumbel analysis.
                """,
                group_id = "ecb",
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
                        max_value=300.0,
                        units="meters/second"),
                    path="gust_speed_level_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1980]),
                    ])]


class ECBWind():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the European Central Bank (ECB)
        windstorm data.

        METADATA:
        Link: provided directly by the ECB (email)
        Data type: historical return period
        Hazard indicator: gust speed
        Region: Europe
        Resolution: 50km
        Return periods: 5, 10, 50, 100, 500
        Time range: NA
        File type: Table (.csv)

        DATA DESCRIPTION:
        Gumbel analysis.

        EXTRA INFORMATION
        Since NUTS ID CODE is provided instead of latitude and longitude we need and extra
        file make the translation.
        https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
        """
        
        # Bucket parameters
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.zarr_storage = zarr_storage

        # create temporary folder to store data
        self.temp_dir = os.path.join(os.getcwd(), temp_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # Source data
        self.return_periods = [5, 10, 50, 100, 500]
        self.data_filenames = ['wind_gumbel_ECB.csv', 'NUTS_LB_2021_4326.geojson']
        self.npy_filenames = ['ecb_wind.npy']

        # zarr parameters
        hazard_type = 'windstorm'
        data_source_name = 'ecb'
        version = 'v2'
        dataset_names = ["gust_speed_level_historical_1980"]
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

    def download_files_from_s3(self):
        """
        Download data in csv format and geojson for coordinate translation. 
        """

        file_ = data_filenames[0]
        ws = read_files_from_s3(self.bucket_name, self.prefix, file_)[0]

        file_ = data_filenames[1]
        nuts = read_files_from_s3(self.bucket_name, self.prefix, file_)[0]
        nuts = nuts[['NUTS_ID', 'geometry']]

        return ws, nuts

    def create_dataArrays_and_merge(self):
        """

        """

        ws, nuts = self.download_files_from_s3()

        df_merged = ws.merge(nuts, left_on='nuts_id', right_on='NUTS_ID')
        df_merged = df_merged[['rp', 'gustspeed', 'geometry']]

        all_das = []
        for rp_int in self.return_periods:

            df_ = df_merged[df_merged.rp == rp_int][['gustspeed', 'geometry']]

            ds = make_geocube(vector_data=df_, resolution=1/2**4, interpolate_na_method='linear')

            all_das.append(ds)

        ds = xr.concat(all_das, 'spatial_ref')
        ds = ds.rename({'x': 'lon','y': 'lat', 'spatial_ref':'index'})
        ds = ds.fillna(0)
        ds['index'] = self.return_periods

        da_merged = ds['gustspeed']

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
        """
        This functions populates the s3 for every file.
        """

        da_merged = self.create_dataArrays_and_merge()

        self.group_path_array = self.group_path_arrays[0]

        logger.info("Pushing to bucket: " + self.group_path_array, extra=logger_config)
        self.oscZ.write(self.group_path_array, da_merged)

        logger.info("Creating image: " + self.group_path_array, extra=logger_config)
        self.create_map_images()  

    
def upload_files_to_s3(bucket_name, data_filenames):
    """
    Uplaod raw wind data to s3 bucket inside the folder hazard/raw_data_consortium
    Upload file to translate from NUTS code to latitude-longitude:
    https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
    """

    boto_c = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])

    base_path_hazard = os.path.join(os.getenv("physical_risk_database"), 'hazard')
    hazard_type = 'Wind'
    datasource = 'ECB'
    inputfile_path = os.path.join(base_path_hazard, hazard_type, datasource)

    for data_filename in data_filenames:
        local_file = os.path.join(inputfile_path, data_filename)
        bucket_destination = os.path.join(prefix, 'raw_data_consortium', data_filename)
        boto_c.upload_file(local_file, bucket_name, bucket_destination)


def read_files_from_s3(bucket_name, prefix, file_): 

    """
    Load file from bucket and read it depending on the type. Supported types for now are csv and geojson.
    """

    def load_files_from_s3(bucket_name, prefix, file_):
        """
        Load files form s3 bucket given its name. To do that we must filter by name.
        """

        my_bucket=s3.Bucket(bucket_name)
        bucket_list = []
        for file in my_bucket.objects.filter(Prefix = prefix):
            file_name=file.key
            if file_name.split('\\')[-1] == file_:
                bucket_list.append(file.key)

        return bucket_list

    s3 = boto3.resource('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])
    bucket_list = load_files_from_s3(bucket_name, prefix, file_)

    df = []   # Initializing empty list
    for file in bucket_list:
        if file.split('.')[-1] == 'csv':
            obj = s3.Object(bucket_name, file)
            data=obj.get()['Body'].read()
            df.append(pd.read_csv(io.BytesIO(data), header=0, delimiter=",", low_memory=False))

        elif file.split('.')[-1] == 'geojson':
            obj = s3.Object(bucket_name, file)
            data=obj.get()['Body'].read()
            df.append(gpd.read_file(io.BytesIO(data)))

    return df


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger_config = logging.basicConfig(filename='onboarding.log',
                                 level=logging.INFO, 
                                 filemode = 'w', 
                                 format='%(process)d-%(levelname)s-%(message)s')  
    
    # https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
    bucket_name = 'physrisk-hazard-indicators-dev01'
    prefix = 'hazard'
    zarr_storage = 'hazard_consortium.zarr'
    upload_files = False # Set to True to upload data to S3

    # Upload raw data to s3
    data_filenames = ['wind_gumbel_ECB.csv', 'NUTS_LB_2021_4326.geojson']
    if upload_files:
        upload_files_to_s3(bucket_name, data_filenames)

    # Onboard all
    temp_dir = 'data'
    ECBWind_ = ECBWind(bucket_name, prefix, zarr_storage, temp_dir)

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    ECBWind_.connect_to_bucket()
    ECBWind_.create_OSCZarr_object()

    ECBWind_.onboard_all()