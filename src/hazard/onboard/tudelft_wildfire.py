import os
import s3fs
import zarr
import numpy as np
import xarray as xr
import geopandas as gpd
from geocube.api.core import make_geocube

import logging
from pyproj.crs import CRS

from hazard.sources.osc_zarr import OscZarr

from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_tile_set

class TUDELFTWildfire_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="fwi",
                indicator_model_gcm = 'historical',
                path="fire/fire_tudelft/v1/fwi20_{scenario}_{year}",
                params={},
                display_name="FWI under 20",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in 
                Europe under present and projected future climates. Includes gridded (NetCDF) 
                datasets of high forest fire danger probabilities for the present climate 
                (1981-2010) based on the ERA-Interim reanalysis and for the projected 
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
                and 2071-2100). 
                """,
                group_id = "fire_tudelft",
                display_groups=[],
                map = MapInfo(
                    bounds= [
                        [
                            -45.0,
                            73.0
                        ],
                        [
                            65.5,
                            73.0
                        ],
                        [
                            65.5,
                            21.5                    
                        ],
                        [
                            -45.0,
                            -21.5
                        ]
                    ],

                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=100.0,
                        units="prob"),
                    path="fwi20_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1971]),
                    Scenario(
                        id="rcp45",
                        years=[2050, 2100]),
                    Scenario(
                        id="rcp85",
                        years=[2050, 2100]),

                    ]),
            HazardResource(
                hazard_type="Fire",
                indicator_id="fwi",
                indicator_model_gcm = 'historical',
                path="fire/fire_tudelft/v1/fwi45_{scenario}_{year}",
                params={},
                display_name="FWI under 45",
                description="""
                NetCDF files containing daily probabilities of high forest fire danger in 
                Europe under present and projected future climates. Includes gridded (NetCDF) 
                datasets of high forest fire danger probabilities for the present climate 
                (1981-2010) based on the ERA-Interim reanalysis and for the projected 
                climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
                and 2071-2100). 
                """,
                group_id = "fire_tudelft",
                display_groups=[],
                map = MapInfo(
                    bounds= [
                        [
                            -45.0,
                            73.0
                        ],
                        [
                            65.5,
                            73.0
                        ],
                        [
                            65.5,
                            21.5                    
                        ],
                        [
                            -45.0,
                            -21.5
                        ]
                    ],

                    colormap=Colormap(
                        max_index=255,
                        min_index=1,
                        nodata_index=0,
                        name="flare",
                        min_value=0.0,
                        max_value=100.0,
                        units="prob"),
                    path="fwi45_{scenario}_{year}_map",
                    source="map_array_pyramid"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[1971]),
                    Scenario(
                        id="rcp45",
                        years=[2050, 2100]),
                    Scenario(
                        id="rcp85",
                        years=[2050, 2100]),

                    ])
                    ]

class TUDELFTWildfire():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Pan-European data sets of forest fire probability of occurrence under 
        present and future climate.

        METADATA:
        Link: https://data.4tu.nl/datasets/f9a134ad-fff9-44d5-ad4e-a0e9112b551e
        Data type: Probability 
        Hazard indicator: Probability
        Region: Pan-Europe
        Resolution: 49km
        Time range: 1971-2000,2021-2050,2071-2100
        Scenarios: RCP4.5, 8.5
        File type: Map (.nc)

        DATA DESCRIPTION:
        NetCDF files containing daily probabilities of high forest fire danger in 
        Europe under present and projected future climates. Includes gridded (NetCDF) 
        datasets of high forest fire danger probabilities for the present climate 
        (1981-2010) based on the ERA-Interim reanalysis and for the projected 
        climates under the RCP4.5 and RCP8.5 scenarios (periods 1971-2000, 2021-2050 
        and 2071-2100). 

        IMPORTANT NOTE:

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
        self.return_periods = [1]
        rcps = ['rcp45_', 'rcp85_']
        fwis = ['fwi20_', 'fwi45_']
        dates = ['2021-2050','2071-2100']
        dates_osc = [ '2050', '2100']
        self.data_filenames = []
        dataset_names = []
        for fwi in fwis:
            for rcp in rcps:
                for j in range(len(dates)):
                    data_filename = 'multi-model_' + fwi + rcp + dates[j] + '.nc'
                    self.data_filenames.append(data_filename)

                    dataset_name = fwi + rcp + dates_osc[j]
                    dataset_names.append(dataset_name)

        rcps = ['historical_']
        dates = ['1971-2000']
        dates_osc = ['1971']
        for fwi in fwis:
            for i in range(len(rcps)):
                data_filename = 'multi-model_' + fwi + rcps[i] + dates[i] + '.nc'
                self.data_filenames.append(data_filename)

                dataset_name = fwi + rcps[i] + dates_osc[i]
                dataset_names.append(dataset_name)

        # zarr parameters
        hazard_type = 'fire'
        data_source_name = 'fire_tudelft'
        version = 'v1'
        self.group_path_arrays = [os.path.join(hazard_type, data_source_name, version, dataset_name) for dataset_name in dataset_names]
        self.path_to_nc = [os.path.join(self.temp_dir, data_filename) for data_filename in self.data_filenames]

        # Affine matrix metadata
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

        for i in range(len(self.path_to_nc)):

            inputfile = self.path_to_nc[i]
            self.group_path_array = self.group_path_arrays[i]

            data = xr.open_dataset(inputfile)
            data = data.risk
            data = data.rename({'time':'index'})
            data['index'] = self.return_periods

            df = data.to_dataframe().reset_index()

            gdf = gpd.GeoDataFrame(
                df.risk, geometry=gpd.points_from_xy(df.lon,df.lat))
            
            resolution = 0.5 / 4
            ds = make_geocube(vector_data=gdf, resolution=resolution, interpolate_na_method='linear')

            da = ds.risk
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
    zarr_storage = 'hazard.zarr'

    temp_dir = 'data'
    download_data = False
    JRCLandslides_ = TUDELFTWildfire(bucket_name, prefix, zarr_storage, temp_dir)

    if download_data:
        logger.info("Downloading data", extra=logger_config)
        # WRIWaterstress_.download_and_unzip_datasets()

    logger.info("Conecting to bucket: " + bucket_name, extra=logger_config)
    logger.info("Conecting to zarr: " + zarr_storage, extra=logger_config)
    JRCLandslides_.connect_to_bucket()
    JRCLandslides_.create_OSCZarr_object()

    JRCLandslides_.onboard_all()