import os
import s3fs
import zarr
import numpy as np
import requests
import zipfile
from itertools import product
import geopandas as gpd

from pyproj.crs import CRS

from hazard.sources.osc_zarr import OscZarr

import geopandas as gpd
from hazard.utilities.xarray_utilities import global_crs_transform
from geocube.api.core import make_geocube

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

import logging

class WRIWaterstress_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="WaterStress",
                indicator_id="water_stress",
                indicator_model_gcm = 'historical',
                path="waterstress/wri/v2/water_stress_{scenario}_{year}",
                params={},
                display_name="WRI Water Stress",
                description="""
                The Aqueduct Water Stress Projections Data include indicators of change in water supply, 
                water demand, water stress, and seasonal variability, projected for the coming decades 
                under scenarios of climate and economic growth.

                Categories:
                6: Arid and low water use
                5: Extremely high (>80%)
                4: High (40-80%)
                3: Medium-high (20-40%)'
                2: Low-medium (10-20%)
                1: Low (<10%)
                0: No data

                """,
                group_id = "wri",
                display_groups=[],
                map = MapInfo(
                    bounds= [
                        [
                            -180.0,
                            90.0
                        ],
                        [
                            180.0,
                            90.0
                        ],
                        [
                            180.0,
                            -90.0
                        ],
                        [
                            -180.0,
                            -90.0
                        ]
                    ],

                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=6.0,
                        units="index"),
                    path="water_stress_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="rcp26",
                        years=[2020, 2030, 2040]),
                    Scenario(
                        id="rcp45",
                        years=[2020, 2030, 2040]),
                    Scenario(
                        id="rcp85",
                        years=[2020, 2030, 2040]),

                    ])]


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
        suffix = 'l'
        self.vars_ = []
        for i,j in product(range(3), range(3)):
            self.vars_.append(indicator_code + year_codes[i] + scenario_codes[j] + data_type + suffix)

        # npy names
        self.npy_filenames = [ax + '.npy' for ax in self.vars_]


        # zarr parameters
        hazard_type = 'waterstress'
        data_source_name = 'wri'
        version = 'v4'
        years = ['20' + y for y in year_codes]
        sc = ['rcp26', 'rcp45', 'rcp85']
        dataset_names = []
        for i,j in product(range(3), range(3)):
            dataset_names.append('water_stress_' + sc[i] +'_'+ years[j])
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
        return data


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

        data = self.read_shx_file()

        for i in range(len(self.vars_)):

            var_ = self.vars_[i]
            self.group_path_array = self.group_path_arrays[i]

            # check if variable is in the data
            if var_ not in data.columns: continue

            data_var = data[['geometry', var_]]
            crs, transform = global_crs_transform(360 * 4, 180 * 4)
            resolution = transform[0]

            # bounds_ = [Polygon([[-180,90],[180,90], [-180,-90], [180, -90]])]
            # aux_df = pd.DataFrame({'geometry':bounds_, var_:[0]})
            # data_var = data_var.append(aux_df)

            data_var[var_].replace(to_replace=['Arid and low water use',
                                               'Extremely high (>80%)',
                                               'High (40-80%)',
                                               'Medium-high (20-40%)',
                                               'Low-medium (10-20%)',
                                               'Low (<10%)',
                                               'No data']
                                               , value=[6,5,4,3,2,1,0]
                                               , inplace=True)

            ds = make_geocube(vector_data=data_var, resolution=resolution)
            da = ds[var_]

            da.data = np.nan_to_num(da.data)
            da = da.rename({'x': 'lon','y': 'lat'})

            logger.info("Pushing to bucket: " + self.group_path_array, extra=logger_config)
            self.oscZ.write(self.group_path_array, da)

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
    zarr_storage = 'hazard_consortium.zarr'

    temp_dir = 'data'
    download_data = False
    WRIWaterstress_ = WRIWaterstress(bucket_name, prefix, zarr_storage, temp_dir)
    if download_data:
        logger.info("Downloading data", extra=logger_config)
        WRIWaterstress_.download_and_unzip_datasets()

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    WRIWaterstress_.connect_to_bucket()
    WRIWaterstress_.create_OSCZarr_object()

    WRIWaterstress_.onboard_all()

