import os
import s3fs
import zarr
import numpy as np
import rasterio
import requests
import zipfile
import pyproj
from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


class JRCRiverFlood():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        river flood data.

        METADATA:
        Link: https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81
        Data type: historical return period
        Hazard indicator: flood depth
        Region: Pan-Europe
        Resolution: 100m
        Return periods: 10,20,50,100,200,500
        Time range: 1990-2016
        File type: Map (.tif)

        DATA DESCRIPTION:
        The maps depict flood prone areas for river flood events for six different flood 
        frequencies (from 1-in-10-years to 1-in-500-years). The extent comprises most 
        of the geographical Europe and all the river basins entering the Mediterranean 
        and Black Seas in the Caucasus, Middle East and Northern Africa countries. 
        Cell values indicate water depth (in m).
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
        self.return_periods_str = ['010', '020', '050', '100', '200', '500']
        self.return_periods = [int(rt) for rt in self.return_periods_str]
        self.zip_urls = ["https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/EuropeanMaps/floodMap_RP{}.zip".format(rp) for rp in self.return_periods_str]
        self.zip_filenames = [url.split('/')[-1] for url in self.zip_urls]
        self.path_to_extact_zip = [os.path.join(self.temp_dir, ax.split('.')[0]) for ax in self.zip_filenames]

        # Create tif files names and paths
        self.tif_filenames = ['floodmap_EFAS_RP{}_C.tif'.format(rp) for rp in self.return_periods_str]
        self.tif_paths = [os.path.join(self.path_to_extact_zip[axi], self.tif_filenames[axi]) for axi in range(len(self.return_periods))]

        # zarr parameters
        hazard_type = 'inundation_river'
        data_source_name = 'jrc'
        version = 'v1'
        dataset_name = 'flood_depth_historical_1990_2016_map'
        self.group_path_array = os.path.join(hazard_type, data_source_name, version, dataset_name)

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



    def get_metadata(self):
        """
        Read one tif file to load map metadata.

        Maybe add to map utilities.
        """

        # Read one tif file to get the metadata: transform, crs, width, height and shape
        inputfile = self.tif_paths[0]

        src = rasterio.open(inputfile)

        self.transform_source = src.transform
        self.crs_source = CRS.from_epsg(3035)
        self.width = src.width
        self.height = src.height
        self.shape = (self.height, self.width)

        src.close()


    def transform_matrix_to_epsg(self):
        """
        Transfrom from i-j matrix index o epsg using affine transformation.

        Maybe add to map utilities.
        """

        cols, rows = np.meshgrid([0, self.width-1], [0, self.height-1])
        xs, ys = rasterio.transform.xy(self.transform_source, rows, cols)
        return xs, ys

    def transfrom_one_epsg_to_another(self):
        """
        Transfrom from one epsg coordynate system to another. In this case from 3035 to 4326 (latlon).

        Maybe add to map utilities.
        """

        xs, ys = self.transform_matrix_to_epsg()
        proj = pyproj.Transformer.from_crs(3035, 4326, always_xy=True, authority='EPSG')
        self.longitudes, self.latitudes = proj.transform(np.array(xs),  np.array(ys))

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

    def read_window(self, src, height_pos, width_pos, chunck_size):
        """
        Read tif data by windows of dimension (chunk_size x chunk_size).

        Parameters:
            path_to_file (str): full path to tif file.

        Returns:
            fld_depth (numpy array): flood depth at (x1, y1) 3035 EPSG coordinates

        """
        window = rasterio.windows.Window(width_pos, height_pos, chunck_size, chunck_size)
        band = src.read(1, window=window)

        to_impute = band == src.nodata
        band[to_impute] = 0

        return band
    

    def populate_zarr_in_s3(self):
        """
        Populate s3 zarr file in chunks.
        """

        chunck_size = 1000
        z = self.oscZ.root[self.group_path_array]

        for rt_i in range(self.return_periods_str):

            inputfile = self.tif_paths[rt_i]

            src = rasterio.open(inputfile)

            for height_pos in range(0, self.height, chunck_size):
                for width_pos in range(0, self.width, chunck_size):

                    band = self.read_window(src, height_pos, width_pos, chunck_size)

                    z[rt_i,height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size] = band


# https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
bucket_name = 'physrisk-hazard-indicators-dev01'
prefix = 'hazard'
zarr_storage = 'hazard_consortium.zarr'

temp_dir = 'data'
# JRCRiverFlood_ = JRCRiverFlood(bucket_name, prefix, zarr_storage, temp_dir)
# JRCRiverFlood_.download_and_unzip_datasets()
# JRCRiverFlood_.get_metadata()
# JRCRiverFlood_.transfrom_one_epsg_to_another()
# JRCRiverFlood_.create_affine_transform_from_mapbounds()

# JRCRiverFlood_.connect_to_bucket()
# JRCRiverFlood_.create_OSCZarr_object()
# JRCRiverFlood_.create_empty_zarr_in_s3()
# JRCRiverFlood_.populate_zarr_in_s3()