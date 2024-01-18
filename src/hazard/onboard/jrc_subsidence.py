import os
import s3fs
import boto3
import zarr
import numpy as np
import logging
import xarray as xr
# from osgeo import gdal

from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

class JRCSubsidence_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Subsidence",
                indicator_id="subsidence",
                indicator_model_gcm = 'historical',
                path="subsidence/jrc/v1/susceptability_{scenario}_{year}",
                params={},
                display_name="Subsidence Susceptability",
                description="""
                A number of layers for soil properties have been created based on data from the European
                Soil Database in combination with data from the Harmonized World Soil Database (HWSD) 
                and Soil-Terrain Database (SOTER). The available layers include: Total available water 
                content, Depth available to roots, Clay content, Silt content, Sand content, Organic 
                carbon, Bulk Density, Coarse fragments.
                """,
                group_id = "jrc",
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
                        units="index"),
                    path="susceptability_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1980]),
                    ])]

class JRCSubsidence():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Define every attribute of the onboarding class for the Joint Research Center (JRC)
        subsidence data.

        The data must be requested submitting a form in the next link:
        https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data

        Then, an email with instructions to downloading the data will be recieved.
        The data will be provided in Idrisi Raster Format (.rst) file type.

        METADATA:
        Link: https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data
        Hazard subtype: from Drought
        Data type: Historical simulation
        Hazard indicator: NA
        Region: Europe
        Resolution: 1km
        Time range: NA
        File type: Restructured Text (.rst)

        DATA DESCRIPTION:
        A number of layers for soil properties have been created based on data from the European
        Soil Database in combination with data from the Harmonized World Soil Database (HWSD) 
        and Soil-Terrain Database (SOTER). The available layers include: Total available water 
        content, Depth available to roots, Clay content, Silt content, Sand content, Organic 
        carbon, Bulk Density, Coarse fragments.

        IMPORTANT NOTES:
        To buil the hazard indicator in the form of susceptability categories the next reference was used:
        https://publications.jrc.ec.europa.eu/repository/handle/JRC114120 (page 32)

        The categories depend on the percentage of soil and sand. The next categories are used:
        very high (clay > 60 %)
        high (35% < clay < 60%)
        medium (18% < clay < 35% and >= 15% sand, or 18% < clay and 15% < sand < 65%)
        low (18% < clay and > 65% sand)


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
        self.data_filenames = ['STU_EU_T_SAND.rst', 'STU_EU_T_CLAY.rst']

        # zarr parameters
        hazard_type = 'subsidence'
        data_source_name = 'jrc'
        version = 'v2'
        dataset_names = ['susceptability_historical_1980']
        self.group_path_arrays = [os.path.join(hazard_type, data_source_name, version, dataset_name) for dataset_name in dataset_names]

        # Affine matrix metadata
        self.crs_latlon = CRS.from_epsg(4326)

        # etrs laea coordinate system. BOUNDS
        self.min_xs = 1500000
        self.max_xs = 7400000
        self.min_ys = 900000
        self.max_ys = 5500000

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

        # file_ = self.data_filenames[0]
        # data_sand = read_files_from_local(self.bucket_name, self.prefix, file_)

        # file_ = self.data_filenames[1]
        # data_clay = read_files_from_local(self.bucket_name, self.prefix, file_)

        data_sand, data_clay = read_files_from_local()

        return data_sand, data_clay
    

    def create_affine_transform_from_mapbounds_3035(self, width, height):
        """
        Create an affine transformation from map point and shape of bounds.

        Maybe add to map utilities
        """

        # Create Affine transformation

        bounds = (self.min_xs, self.min_ys, self.max_xs, self.max_ys)

        # Compute the parameters of the georeference
        A = (bounds[2] - bounds[0]) / width # pixel size in the x-direction in map units/pixel
        B = 0 # rotation about y-axis
        C = 0 # rotation about x-axis
        D = -(bounds[3] - bounds[1]) / height # pixel size in the y-direction in map units, almost always negative
        E = bounds[0] # x-coordinate of the center of the upper left pixel
        F = bounds[3] # y-coordinate of the center of the upper left pixel

        transform = Affine(A, B, C, D, E, F)
        self.transform_3035 = transform


    def transfrom_ij_to_epsg_3035(self, width, height):
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

        coord_x = np.zeros([height, width])
        coord_y = np.zeros([height, width])
        for w in range(width):
            for h in range(height):
                x_y = mat @ np.array([w, h]) + shift
                coord_x[h, w] = x_y[0]
                coord_y[h, w] = x_y[1]

        return coord_x, coord_y


    def create_categories(self, data_clay, data_sand):
        """
        https://publications.jrc.ec.europa.eu/repository/handle/JRC114120

        assess the subsidence susceptibility for different classes: 
        very high (clay > 60 %)
        high (35% < clay < 60%)
        medium (18% < clay < 35% and >= 15% sand, or 18% < clay and 15% < sand < 65%)
        low (18% < clay and > 65% sand).
        """

        # From matrix index to _etrs_laea using transform
        data_cat = np.zeros([self.height, self.width])
        for w in range(self.width):
            for h in range(self.height):
                val_clay = data_clay[h, w]
                val_sand = data_sand[h, w]
                if val_clay > 18 and val_sand > 65:
                    data_cat[h, w] = 1
                elif (val_clay >= 18 and val_clay <= 35) and val_sand <= 15:
                    data_cat[h, w] = 2
                elif val_clay > 35 and val_clay <= 60:
                    data_cat[h, w] = 3
                elif val_clay > 60:
                    data_cat[h, w] = 4

        return data_cat
    

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

        # Download data from s3
        # data_sand, data_clay = self.download_files_from_s3()

        sand_filename = 'STU_EU_T_SAND.txt'
        clay_filename = 'STU_EU_T_CLAY.txt'

        data_sand = np.loadtxt(sand_filename)
        data_clay = np.loadtxt(clay_filename)

        # 
        width = data_sand.shape[1]
        height = data_sand.shape[0]

        self.create_affine_transform_from_mapbounds_3035(width, height)
        coord_x, coord_y = self.transfrom_ij_to_epsg_3035(width, height)

        self.width = width
        self.height = height

        data_cat = self.create_categories(data_sand, data_clay)

        da = xr.DataArray(
                data=np.transpose(data_cat),
                dims=["x", "y"],
                coords=dict(
                    x=(["x"], coord_x[0,:]),
                    y=(["y"], coord_y[:,0]),
                ),
                attrs=dict(
                    description="Subsidence susceptability",
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

                    z[rt_i,height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size] = ws_matrix[height_pos:height_pos+chunck_size, width_pos:width_pos+chunck_size]



def upload_files_to_s3(bucket_name, data_filenames):
    """
    Uplaod raw wind data to s3 bucket inside the folder hazard/raw_data_consortium
    Upload files with soil variables to build subsidence hazard indicator.
    """

    boto_c = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])

    base_path_hazard = os.path.join(os.getenv("physical_risk_database"), 'hazard')
    hazard_type = 'Subsidence'
    datasource = 'JRC'
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

    s3 = boto3.client('s3', aws_access_key_id=os.environ["OSC_S3_HIdev01_ACCESS_KEY"], aws_secret_access_key=os.environ["OSC_S3_HIdev01_SECRET_KEY"])
    df = []   # Initializing empty list
    for file in bucket_list:
        if file.split('.')[-1] == 'rst':
            # obj = s3.Object(bucket_name, file)
            # data=obj.get()['Body'].read()

            with open('prueba.rst', 'wb') as f:
                s3.download_fileobj(bucket_name, file, f)

            s3.download_file(bucket_name, file,  file.split('\\')[-1])
            aux_ = os.path.join(os.getcwd(), file.split('\\')[-1])
            df.append( gdal.Open(aux_).ReadAsArray() )

    return df

def read_files_from_local():
    """
    
    """

    if "physical_risk_database" in os.environ:
        base_path_hazard = os.path.join(os.getenv("physical_risk_database"), 'hazard')
        base_path_exp = os.path.join(os.getenv("physical_risk_database"), 'exposure')
    else:
        base_path = os.getcwd()
        base_path = base_path.split('PhysicalRisk')[0]
        base_path_hazard = os.path.join(base_path, 'physical_risk_database', 'hazard')
        base_path_exp = os.path.join(base_path, 'physical_risk_database', 'exposure')

    hazard_type = 'Subsidence'
    datasource = 'JRC'
    data_filename = 'STU_EU_T_SAND.rst'

    inputfile_path = os.path.join(base_path_hazard, hazard_type, datasource)
    inputfile = os.path.join(inputfile_path, data_filename)

    data_sand = gdal.Open(inputfile).ReadAsArray()


    data_filename = 'STU_EU_T_CLAY.rst'

    inputfile = os.path.join(inputfile_path, data_filename)

    data_clay = gdal.Open(inputfile).ReadAsArray()

    return data_sand, data_clay


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
    data_filenames = ['STU_EU_T_SAND.rst', 'STU_EU_T_CLAY.rst']
    if upload_files:
        upload_files_to_s3(bucket_name, data_filenames)


    temp_dir = 'data'
    JRCSubsidence_ = JRCSubsidence(bucket_name, prefix, zarr_storage, temp_dir)

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    JRCSubsidence_.connect_to_bucket()
    JRCSubsidence_.create_OSCZarr_object()

    JRCSubsidence_.onboard_all()


