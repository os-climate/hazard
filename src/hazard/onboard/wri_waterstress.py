import os
import s3fs
import zarr
import numpy as np
import rasterio
import requests
import zipfile
import math
from itertools import product
import geopandas as gpd
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

import pyproj
from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


class WRIWaterstress():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Water Resources Institute (WRI)
        water stress data.

        After downloading the zip data file you can find an excel file for readme purposes.

        METADATA:
        Link: https://www.wri.org/data/aqueduct-water-stress-projections-data
        Data type: Susceptability Score
        Hazard indicator: Water Stress, seasonal variability, water demand and water supply
        Region: Global
        Resolution: 22km
        Scenarios: RCP2.6, 4.5, 8.5
        Time range: 2030, 2040, 2050
        File type: Shape File (.shx)

        DATA DESCRIPTION:
        The Aqueduct Water Stress Projections Data include indicators of change in water supply, 
        water demand, water stress, and seasonal variability, projected for the coming decades 
        under scenarios of climate and economic growth.

        IMPORTANT NOTE:
        The dimesnion of the data has has been reduced by 10 (see reducing_factor variable)
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
        self.return_periods = [1]
        self.zip_urls = ["https://files.wri.org/d8/s3fs-public/aqueduct_projections_20150309_shp.zip"]
        self.zip_filenames = [url.split('/')[-1] for url in self.zip_urls]
        self.path_to_extact_zip = [os.path.join(self.temp_dir, ax.split('.')[0]) for ax in self.zip_filenames]

        # Create tif files names and paths
        self.shx_filenames = ["aqueduct_projections_20150309.shx"]
        self.shx_paths = [os.path.join(self.path_to_extact_zip[0], self.shx_filenames[0])]

        # Create column names
        indicator_code = 'ws' # For water stress
        year_codes = ['20', '30', '40']
        scenario_codes = ['24' ,'28', '38']
        data_type = 't'
        suffix = 'r'
        self.vars_ = []
        for i,j in product(range(3), range(3)):
            self.vars_.append(indicator_code + year_codes[i] + scenario_codes[j] + data_type + suffix)

        # npy names
        self.npy_filenames = [ax + '.npy' for ax in self.vars_]


        # zarr parameters
        hazard_type = 'waterstress'
        data_source_name = 'wri'
        version = 'v1'
        years = ['20' + y for y in year_codes]
        sc = ['rcp26', 'rcp45', 'rcp85']
        dataset_names = []
        for i,j in product(range(3), range(3)):
            dataset_names.append('water_stress_' + sc[i] +'_'+ years[j] + '_map')
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


    def read_shx_file(self):
        """
        Open and read shx file and filter by columns.
        """

        data = gpd.read_file(self.shx_paths[0])

        cols = ['geometry']
        cols.extend(self.vars_)
        data_ = data[cols]

        return data


    def create_latlon_from_geometry(self, data_, var_):
        """
        Create latitude and longitude vectors from geometry data.
        Create water stress vector from data.
        """

        lon_vector = []
        lat_vector = []
        ws_vector = []
        
        # Data is big so we reduce it 10 times
        reducing_factor = 10
        for row_i in range(0, len(data_), reducing_factor):

            ws = data_[var_][row_i]
            geom_ = data_['geometry'][row_i]

            if isinstance(geom_, Polygon):
                for point_i in range(len(geom_.exterior.coords.xy[0])):

                    lon = geom_.exterior.coords.xy[0][point_i]
                    lat = geom_.exterior.coords.xy[1][point_i]

                    lon_vector.append(lon)
                    lat_vector.append(lat)
                    ws_vector.append(ws)

                    # More points can be added
                    if point_i > 1:
                        break

            elif isinstance(geom_, MultiPolygon) and False: # Skip multipolygon
                lon = geom_.geoms[0].exterior.coords.xy[0][0]
                lat = geom_.geoms[0].exterior.coords.xy[1][0]

        self.longitude = lon_vector
        self.latitude = lat_vector
        self.ws_vector = ws_vector




    def create_latlon_grid(self):
        """
        Create a grid of latitudes and longitudes given the bounds.
        """
        
        min_lat, max_lat = min(self.latitude), max(self.latitude)
        min_lon, max_lon = min(self.longitude), max(self.longitude)

        self.total_size = len(self.ws_vector)
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
            # Find the nearest point and and the ws value
            for pos_i in range(self.total_size):
                lon_i = self.longitude[pos_i]
                lat_i = self.latitude[pos_i]
                ws_i = self.ws_vector[pos_i]
                
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

        data = self.read_shx_file()

        for i in range(len(self.vars_)):

            var_ = self.vars_[i]

            # check if variable is in the data
            if var_ not in data.columns: continue

            data_var = data[['geometry', var_]]
            self.create_latlon_from_geometry(data_var, var_)

            npy_filename = self.npy_filenames[i]
            ws_matrix, grid = self.create_hazard_matrix(npy_filename)

            # Define zarr shape and coordinate system
            self.width = ws_matrix.shape[1]
            self.height = ws_matrix.shape[0]
            self.shape = (self.height, self.width)

            self.longitudes = grid[0]
            self.latitudes = grid[1]

            self.create_affine_transform_from_mapbounds()

            self.group_path_array = self.group_path_arrays[i]
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


    def populate_zarr_in_s3(self, ws_matrix):
        """
        Populate s3 zarr file in chunks.
        """

        chunck_size = 1000
        z = self.oscZ.root[self.group_path_array]

        for rt_i in range(len(self.return_periods)):
            for height_pos in range(0, self.height, chunck_size):
                for width_pos in range(0, self.width, chunck_size):

                    z[rt_i,height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size] = ws_matrix[height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size, rt_i]




# https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
bucket_name = 'physrisk-hazard-indicators-dev01'
prefix = 'hazard'
zarr_storage = 'hazard_consortium.zarr'

temp_dir = 'data'
WRIWaterstress_ = WRIWaterstress(bucket_name, prefix, zarr_storage, temp_dir)
WRIWaterstress_.download_and_unzip_datasets()
WRIWaterstress_.onboard_all()