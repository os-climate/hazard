import os
import io
import s3fs
import boto3
import zarr
import numpy as np
import logging
import xarray as xr

from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

class JRCLandslides_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Drought",
                indicator_id="susceptability",
                indicator_model_gcm = 'historical',
                path="drought/landslide_jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Landslide Susceptability",
                description="""
                The spatial dataset (GIS map) shows landslide susceptibility levels at European scale, 
                derived from heuristic-statistical modelling of main landslide conditioning factors 
                using also landslide location data. It covers all EU member states except Malta, in 
                addition to Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR Macedonia, Iceland, 
                Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.
                """,
                group_id = "landslide_jrc",
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
                    path="susceptability_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1980]),
                    ])]

class JRCLandslides():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        landslide data.

        The data must be requested submitting a form in the next link:
        https://esdac.jrc.ec.europa.eu/content/european-landslide-susceptibility-map-elsus-v2 

        Then, an email with instructions to downloading the data will be recieved.
        The data will be provided in Esri ASCII Grid (.asc) file type.

        METADATA:
        Link: https://data.jrc.ec.europa.eu/dataset/jrc-esdac-78  
        Data type: Susceptability Score
        Hazard indicator: Susceptability Score
        Region: Europe
        Resolution: 200m
        Time range: NA
        File type: Esri ASCII Grid (.asc)

        DATA DESCRIPTION:
        The spatial dataset (GIS map) shows landslide susceptibility levels at European scale, 
        derived from heuristic-statistical modelling of main landslide conditioning factors 
        using also landslide location data. It covers all EU member states except Malta, in 
        addition to Albania, Andorra, Bosnia and Herzegovina, Croatia, FYR Macedonia, Iceland, 
        Kosovo, Liechtenstein, Montenegro, Norway, San Marino, Serbia, and Switzerland.

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
        self.data_filenames = ['elsus_v2.asc']

        # zarr parameters
        hazard_type = 'drought'
        data_source_name = 'landslide_jrc'
        version = 'v1'
        dataset_names = ['susceptability_historical_1980']
        self.group_path_arrays = [os.path.join(hazard_type, data_source_name, version, dataset_name) for dataset_name in dataset_names]

        # Affine matrix metadata
        self.crs_latlon = CRS.from_epsg(4326)
        self.cell_size = 200
        self.XLLCORNER = 2636073.6872550002  
        self.YLLCORNER = 1385914.3968890002  

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
        Download data in asc format. 
        """

        file_ = self.data_filenames[0]
        data = read_files_from_s3(self.temp_dir, self.bucket_name, self.prefix, file_)[0]

        return data
    
    def create_affine_transform_from_mapbounds_3035(self, width, height):
        """
        Create an affine transformation from map point and shape of bounds.

        Maybe add to map utilities
        """

        # Create Affine transformation
        min_xs = self.XLLCORNER
        max_xs = min_xs + self.cell_size*width
        min_ys = self.YLLCORNER
        max_ys = min_ys + self.cell_size*height

        bounds = (min_xs, min_ys, max_xs, max_ys)

        # Compute the parameters of the georeference
        A = (bounds[2] - bounds[0]) / width # pixel size in the x-direction in map units/pixel
        B = 0 # rotation about y-axis
        C = 0 # rotation about x-axis
        D = -(bounds[3] - bounds[1]) / height # pixel size in the y-direction in map units, almost always negative
        E = bounds[0] # x-coordinate of the center of the upper left pixel
        F = bounds[3] # y-coordinate of the center of the upper left pixel

        transform = Affine(A, B, C, D, E, F)
        self.transform_3035 = transform

    def transfrom_ij_to_epsg_3035(self, data, width_range, height_range):
        """
        Transfrom from matrix index (i-j) to ETRS LAEA coordynate system (CRS 3035)

        Maybe add to map utilities.
        """

        A = self.transform_3035[0]
        B = self.transform_3035[1]
        C = self.transform_3035[2]
        D = self.transform_3035[3]
        E = self.transform_3035[4]
        F = self.transform_3035[5]

        mat = np.array([[A, B], [C, D]]) 
        shift = np.array([E, F])

        width_ = data.shape[1]
        height_ = data.shape[0]

        coord_x = np.zeros([height_, width_])
        coord_y = np.zeros([height_, width_])
        for w_, w in enumerate(width_range):
            for h_, h in enumerate(height_range):
                x_y = mat @ np.array([w, h]) + shift
                coord_x[h_, w_] = x_y[0]
                coord_y[h_, w_] = x_y[1]

        return coord_x, coord_y


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

        data_ = self.download_files_from_s3()

        # Reduce dimension of data
        width = data_.shape[1]
        height = data_.shape[0]
        reducing_factor = 25
        data = data_[0:height:reducing_factor, 0:width:reducing_factor]

        self.create_affine_transform_from_mapbounds_3035(width, height)

        height_range = np.arange(0, height, reducing_factor)
        width_range = np.arange(0, width, reducing_factor)

        coord_x, coord_y = self.transfrom_ij_to_epsg_3035(data, width_range, height_range)

        da = xr.DataArray(
                data=np.transpose(data),
                dims=["x", "y"],
                coords=dict(
                    x=(["x"], coord_x[0,:]),
                    y=(["y"], coord_y[:,0]),
                ),
                attrs=dict(
                    description="Landslide susceptability",
                    units="index",
                ),
            )
        
        da = da.rio.write_crs("EPSG:3035")
        da = da.transpose('y','x')
        da = da.rio.reproject("EPSG:4326", shape=(da.shape[0], da.shape[1]), nodata=0)

        self.group_path_array = self.group_path_arrays[0]

        logger.info("Pushing to bucket: " + self.group_path_array, extra=logger_config)
        self.oscZ.write(self.group_path_array, da)

        logger.info("Creating image: " + self.group_path_array, extra=logger_config)
        self.create_map_images()    

     


def upload_files_to_s3(bucket_name, data_filenames):
    """
    Uplaod raw wind data to s3 bucket inside the folder hazard/raw_data_consortium
    Upload file with susceptability map for landslides and pdf with metadata.
    """

    boto_c = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])

    base_path_hazard = os.path.join(os.getenv("physical_risk_database"), 'hazard')
    hazard_type = 'Landslide'
    datasource = 'JRC'
    inputfile_path = os.path.join(base_path_hazard, hazard_type, datasource)

    for data_filename in data_filenames:
        local_file = os.path.join(inputfile_path, data_filename)
        bucket_destination = os.path.join(prefix, 'raw_data_consortium', data_filename)
        boto_c.upload_file(local_file, bucket_name, bucket_destination)


def read_files_from_s3(temp_dir, bucket_name, prefix, file_): 

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

    # s3 = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])
    df = []   # Initializing empty list
    for file in bucket_list:
        if file.split('.')[-1] == 'asc':
            
            obj = s3.Object(bucket_name, file)
            data=obj.get()['Body'].read()
            df.append( np.loadtxt(io.BytesIO(data), skiprows=6) )

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
    zarr_storage = 'hazard.zarr'
    upload_files = False # Set to True to upload data to S3

    # Upload raw data to s3
    data_filenames = ['elsus_v2.asc', 'ELSUSv2_susceptibility_metadata.pdf']
    if upload_files:
        upload_files_to_s3(bucket_name, data_filenames)

    temp_dir = 'data'
    JRCLandslides_ = JRCLandslides(bucket_name, prefix, zarr_storage, temp_dir)

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    JRCLandslides_.connect_to_bucket()
    JRCLandslides_.create_OSCZarr_object()

    JRCLandslides_.onboard_all()