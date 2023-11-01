
import os
import s3fs
import zarr
import numpy as np
import rasterio
import xarray as xr
import math
import pyproj
import requests

from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


class JRCCoastalFlood():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        coastal flood data.

        METADATA:
        Link: https://data.jrc.ec.europa.eu/dataset/0026aa70-cc6d-4f6f-8c2f-554a2f9b17f2
        Link: https://data.jrc.ec.europa.eu/dataset/deff5a62-074c-4175-bce4-f8f13e0437a3
        Link: https://data.jrc.ec.europa.eu/dataset/a25677b7-2296-4eeb-82f2-70c78690ae10
        Data type: historical and scenario return period
        Scenarios: historical, RCP4.5 and RCP8.5
        Hazard indicator: storm surge level
        Region: Europe
        Resolution: 25km
        Return periods: 5,10,20,50,100,200,500,1000
        Time range: 1969-2004 and 2009-2099
        File type: Map (.nc)

        DATA DESCRIPTION:
        The dataset contains the extreme storm surge levels (ESSL) at a European scale. The ESSL
        are estimated from an ensemble of 8 climatic models for the period from 1/12/1969 to 
        30/11/2004 and for 8 return periods (5, 10, 20, 50, 100, 200, 500, 1000) according to 
        the Peak Over Threshold approach.

        The dataset contains the extreme storm surge levels (ESSL) at a European scale. 
        The ESSL are estimated from an ensemble of 8 climatic models and the corresponding 
        RCP45 scenarios, for the period from 1/12/2009 to 30/11/2099 and for 8 return periods
        (5, 10, 20, 50, 100, 200, 500, 1000) according to the Peak Over Threshold approach.

        The dataset contains the extreme storm surge levels (ESSL) at a European scale. 
        The ESSL are estimated from an ensemble of 8 climatic models and the corresponding 
        RCP85 scenarios, for the period from 1/12/2009 to 30/11/2099 and for 8 return periods
        (5, 10, 20, 50, 100, 200, 500, 1000) according to the Peak Over Threshold approach.
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
        self.return_periods = [5, 10, 20, 50, 100, 200, 500, 1000]
        self.nc_urls = ["https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/LISCOAST/RCP/LATEST/CoastAlRisk_Europe_EESSL_Historical.nc",
                        "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/LISCOAST/RCP/LATEST/CoastAlRisk_Europe_EESSL_RCP45.nc",
                        "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/LISCOAST/RCP/LATEST/CoastAlRisk_Europe_EESSL_RCP85.nc"]
        self.nc_filenames = [url.split('/')[-1] for url in self.nc_urls]
        self.npy_filenames = [ax.replace('nc','npy') for ax in self.nc_filenames]
        
        # zarr parameters
        hazard_type = 'inundation_coastal'
        data_source_name = 'jrc'
        version = 'v1'
        dataset_names = ["strom_surge_level_historical_1969_2004_map",
                        "strom_surge_level_rcp45_2009_2099_map",
                        "strom_surge_level_rcp85_2009_2099_map"]
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

    def download_all_files(self):
        """
        Function to download every tif file (one per return period) and unzip them.
        """

        for file_i in range(len(self.nc_filenames)):
            nc_filename = os.path.join(self.temp_dir, self.nc_filenames[file_i])
            nc_url = self.nc_urls[file_i]
            if not os.path.exists(nc_filename):
                self.download_file(nc_url, nc_filename)


    def read_nc_file(self, inputfile):
        """
        Read one tif file to load map metadata.

        Maybe add to map utilities.
        """

        # Read one tif file to get the metadata: transform, crs, width, height and shape
        nc_data = xr.open_dataset(inputfile)

        self.lat_vector = nc_data.latitude.data
        self.lon_vector = nc_data.longitude.data
        self.ssl_vector = nc_data.ssl.data
        self.vector_size = self.ssl_vector.shape[0]

    def create_latlon_grid(self):
        """
        Create a grid of latitudes and longitudes given the bounds.
        """
        
        # Create latitude and longitude grid
        min_lat, max_lat = self.lat_vector.min(), self.lat_vector.max()
        min_lon, max_lon = self.lon_vector.min(), self.lon_vector.max()

        self.total_size = self.lon_vector.shape[0]
        self.small_size = self.total_size
        grid = np.meshgrid(np.linspace(min_lon, max_lon, self.total_size), np.linspace(min_lat, max_lat, self.small_size))

        return grid

    def create_hazard_matrix(self, npy_filename):
        """
        Create a hazard matrix given a hazrd vector. The method used is very slow and can could be optimized.
        """

        grid = self.create_latlon_grid()

        # Create and empty matrix with zeros
        ssl_matrix = np.zeros((self.small_size, self.total_size, len(self.return_periods)))

        # Save the data 
        ssl_matrix_name = os.path.join(self.temp_dir, npy_filename)

        if npy_filename not in os.listdir(self.temp_dir):
            # Find the nearest point and and the ssl value
            for pos_i in range(self.total_size):
                lon_i = self.lon_vector[pos_i]
                lat_i = self.lat_vector[pos_i]
                ssl_i = self.ssl_vector[pos_i, :]
                
                aux_min = 500000
                for i in range(self.small_size):
                    for j in range(self.total_size):
                        lon_ij = grid[0][i, j]
                        lat_ij = grid[1][i, j]

                        dist = math.dist((lon_ij, lat_ij), (lon_i, lat_i))

                        if dist < aux_min:
                            aux_min = dist
                            aux_min_i = (i, j)
                
                ssl_matrix[aux_min_i[0], aux_min_i[1], :] = ssl_i
                print(pos_i, self.total_size)

            np.save(ssl_matrix_name, ssl_matrix)
        else:
            ssl_matrix = np.load(ssl_matrix_name)

        return ssl_matrix, grid

    def onboard_all(self):
        """
        This functions populates the s3 for every file.
        """

        for file_i in range(len(self.nc_filenames)):

            inputfile = os.path.join(self.temp_dir, self.nc_filenames[file_i])
            self.read_nc_file(inputfile)

            npy_filename = self.npy_filenames[file_i]
            ssl_matrix, grid = self.create_hazard_matrix(npy_filename)

            # Define zarr shape and coordinate system
            self.width = ssl_matrix.shape[1]
            self.height = ssl_matrix.shape[0]
            self.shape = (self.height, self.width)

            self.longitudes = grid[0]
            self.latitudes = grid[1]

            self.create_affine_transform_from_mapbounds()

            self.group_path_array = self.group_path_arrays[file_i]
            self.create_empty_zarr_in_s3()

            self.populate_zarr_in_s3(ssl_matrix)

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


# https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
bucket_name = 'physrisk-hazard-indicators-dev01'
prefix = 'hazard'
zarr_storage = 'hazard_consortium.zarr'

temp_dir = 'data'
# JRCCoastalFlood_ = JRCCoastalFlood(bucket_name, prefix, zarr_storage, temp_dir)
# JRCCoastalFlood_.download_all_files()
# JRCCoastalFlood_.connect_to_bucket()
# JRCCoastalFlood_.create_OSCZarr_object()
# JRCCoastalFlood_.onboard_all()

