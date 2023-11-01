
import os
import s3fs
import boto3
import zarr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
import math
import pyproj
import requests
import io
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO  # Python 2.x
else:
    from io import StringIO  # Python 3.x

from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


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
        version = 'v1'
        dataset_names = ["gust_speed_level_historical_NA_map"]
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

    def create_latlonvector_and_wsmatrix(self):
        """
        Create latitude and longitude matching nuts code from data source and from geojson.
        Create wind return period matrix with the return periods.
        """

        ws, nuts = self.download_files_from_s3()

        ws_rp_matrix = np.zeros(ws.shape[0])
        longitude = []
        latitude = []
        for i in range(ws.shape[0]):
            ws_ = ws.gustspeed[i]
            rp = ws.rp[i]
            rp_index = self.return_periods.index(rp)
            nuts_id = ws.nuts_id[i]
            lon = nuts[nuts.NUTS_ID == nuts_id].geometry.x
            lat = nuts[nuts.NUTS_ID == nuts_id].geometry.y

            longitude.append(lon.values[0])
            latitude.append(lat.values[0])

            ws_rp_matrix[i] = ws_

        ws_rp_matrix = ws_rp_matrix.reshape(len(self.return_periods), (int(ws.shape[0] / len(self.return_periods))))

        self.longitude = longitude
        self.latitude = latitude
        self.ws_rp_matrix = ws_rp_matrix


    def create_latlon_grid(self):
        """
        Create a grid of latitudes and longitudes given the bounds.
        """
        
        min_lat, max_lat = min(self.latitude), max(self.latitude)
        min_lon, max_lon = min(self.longitude), max(self.longitude)

        self.total_size = self.ws_rp_matrix.shape[1]
        self.small_size = self.total_size
        grid = np.meshgrid(np.linspace(min_lon, max_lon, self.total_size), np.linspace(min_lat, max_lat, self.small_size))

        return grid

    def create_hazard_matrix(self, npy_filename):
        """
        Create a hazard matrix given a hazrd vector. The method used is very slow and can could be optimized.
        """

        grid = self.create_latlon_grid()

        # Create and empty matrix with zeros
        ws_matrix = np.zeros((self.small_size, self.total_size, len(self.return_periods)))

        # Save the data 
        ws_matrix_name = os.path.join(self.temp_dir, npy_filename)

        if npy_filename not in os.listdir(self.temp_dir):
            # Find the nearest point and and the ssl value
            for pos_i in range(self.total_size):
                lon_i = self.longitude[pos_i]
                lat_i = self.latitude[pos_i]
                ws_i = self.ws_rp_matrix[:, pos_i]
                
                aux_min = 500000
                for i in range(self.small_size):
                    for j in range(self.total_size):
                        lon_ij = grid[0][i, j]
                        lat_ij = grid[1][i, j]

                        dist = math.dist((lon_ij, lat_ij), (lon_i, lat_i))

                        if dist < aux_min:
                            aux_min = dist
                            aux_min_i = (i, j)
                
                ws_matrix[aux_min_i[0], aux_min_i[1], :] = ws_i
                print(pos_i, self.total_size)

            np.save(ws_matrix_name, ws_matrix)
        else:
            ws_matrix = np.load(ws_matrix_name)

        return ws_matrix, grid

    def onboard_all(self):
        """
        This functions populates the s3 for every file.
        """

        self.create_latlonvector_and_wsmatrix()

        npy_filename = self.npy_filenames[0]
        ws_matrix, grid = self.create_hazard_matrix(npy_filename)

        # Define zarr shape and coordinate system
        self.width = ws_matrix.shape[1]
        self.height = ws_matrix.shape[0]
        self.shape = (self.height, self.width)

        self.longitudes = grid[0]
        self.latitudes = grid[1]

        self.create_affine_transform_from_mapbounds()

        self.group_path_array = self.group_path_arrays[0]
        self.create_empty_zarr_in_s3()

        self.populate_zarr_in_s3(ws_matrix)

    def create_affine_transform_from_mapbounds(self):
        """
        Create an affine transformation from map point and shape of bounds.

        Maybe add to map utilities
        """

        # Create Affine transformation
        min_xs = self.longitudes.min()
        max_xs = self.longitudes.max()
        min_ys = self.latitudes.min()
        max_ys = self.latitudes.max()

        bounds = (min_xs, min_ys, max_xs, max_ys)

        # Compute the parameters of the georeference
        A = (bounds[2] - bounds[0]) / self.width # pixel size in the x-direction in map units/pixel
        B = 0 # rotation about y-axis
        C = 0 # rotation about x-axis
        D = -(bounds[3] - bounds[1]) / self.height # pixel size in the y-direction in map units, almost always negative
        E = bounds[0] # x-coordinate of the center of the upper left pixel
        F = bounds[3] # y-coordinate of the center of the upper left pixel

        transform = Affine(A, B, C, D, E, F)
        self.transform_latlon = transform

    def create_empty_zarr_in_s3(self):
        """
        Create and empty zarr in s3 with dimension (number of return periods x map width x map heigh)
        """
        # Create data file inside zarr group with name dataset_name

        # Name standard is: hazard_type + _ + hazard_subtype (if exists) + '_' + hist or scenario + '_' RP (return period) or event/ emulated + '_' + data_provider

        self.oscZ._zarr_create(path=self.group_path_array,
                        shape = self.shape,
                        transform = self.transform_latlon,
                        crs = str(self.crs_latlon),
                        overwrite=False,
                        return_periods=self.return_periods)    


    def populate_zarr_in_s3(self, ssl_matrix):
        """
        Populate s3 zarr file in chunks.
        """

        chunck_size = 1000
        z = self.oscZ.root[self.group_path_array]

        for rt_i in range(len(self.return_periods)):
            for height_pos in range(0, self.height, chunck_size):
                for width_pos in range(0, self.width, chunck_size):

                    z[rt_i,height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size] = ssl_matrix[height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size, rt_i]


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
# ECBWind_.connect_to_bucket()
# ECBWind_.create_OSCZarr_object()
ECBWind_.onboard_all()