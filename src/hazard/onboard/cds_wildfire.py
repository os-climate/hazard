import os
import io
import s3fs
import boto3
import zarr
import numpy as np
import xarray as xr

import pyproj
from pyproj.crs import CRS
from affine import Affine

from hazard.sources.osc_zarr import OscZarr


from typing import Iterable
from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.utilities.tiles import create_image_set

class CDSWildfire_inventory():
    def __init__(self):
        pass

    def inventory(self) -> Iterable[HazardResource]:
        """Get the (unexpanded) HazardModel(s) that comprise the inventory."""
        
        return [
            HazardResource(
                hazard_type="Fire",
                indicator_id="Seasonal FWI",
                indicator_model_gcm = 'CNRM-CERFACS-CNRM-CM5',
                path="fire_cds/ecb/v1/sfwi_{scenario}_{year}",
                params={},
                display_name="Seasonal Fire Weather Index (FWI) Categories",
                description="""
                The dataset presents projections of fire danger indicators for Europe based upon the Canadian 
                Fire Weather Index System (FWI) under future climate conditions. The FWI is a meteorologically 
                based index used worldwide to estimate the fire danger and is implemented in the Global ECMWF 
                Fire Forecasting model (GEFF). In this dataset, daily FWI values, seasonal FWI values, and other
                FWI derived, threshold-specific, indicators were modelled using the GEFF model to estimate the
                fire danger in future climate scenarios. These indicators include the number of days with moderate, 
                high, or very high fire danger conditions as classified by the European Forest Fire Information System 
                (EFFIS) during the northern hemisphere's fire season (June-September):
                    very low: <5.2
                    low: 5.2 - 11.2
                    moderate: 11.2 - 21.3
                    high: 21.3 - 38.0
                    very high: 38.0 - 50
                    extreme: >=50.0
                """,
                group_id = "cds",
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
                        max_value=6.0,
                        units="prob"),
                    path="sfwi_{scenario}_{year}_map",
                    source="map_array"
                ),
                units="none",
                scenarios=[
                    Scenario(
                        id="historical",
                        years=[2004]),
                    Scenario(
                        id="rcp45",
                        years=[2075, 2080, 2098]),
                    Scenario(
                        id="rcp85",
                        years=[2075, 2080, 2098]),

                    ])]

class CDSWildfire():

    def __init__(self, bucket_name, prefix, zarr_storage, temp_dir) -> None:
        """
        Fire danger indicators for Europe from 1970 to 2098 derived from climate projections.

        METADATA:
        Link: https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-tourism-fire-danger-indicators?tab=overview
        Data type: Index
        Hazard indicator: Seasonal FWI
        Region: Europe
        Resolution: 10km
        Time range: 2000, 2075, 2080, 2098
        Scenarios: RCP4.5, 8.5
        File type: Map (.nc)

        DATA DESCRIPTION:
        The dataset presents projections of fire danger indicators for Europe based upon the Canadian 
        Fire Weather Index System (FWI) under future climate conditions. The FWI is a meteorologically 
        based index used worldwide to estimate the fire danger and is implemented in the Global ECMWF 
        Fire Forecasting model (GEFF). In this dataset, daily FWI values, seasonal FWI values, and other
        FWI derived, threshold-specific, indicators were modelled using the GEFF model to estimate the
        fire danger in future climate scenarios. These indicators include the number of days with moderate, 
        high, or very high fire danger conditions as classified by the European Forest Fire Information System 
        (EFFIS) during the northern hemisphere's fire season (June-September):
            very low: <5.2
            low: 5.2 - 11.2
            moderate: 11.2 - 21.3
            high: 21.3 - 38.0
            very high: 38.0 - 50
            extreme: >=50.0

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

        names_dict = {}
        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_historical_fwi-mean-jjas_20010101_20051231_v2.nc'] = 'sfwi_historical_2004'

        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp45_fwi-mean-jjas_20710101_20751231_v2.nc'] = 'sfwi_rcp45_2075'
        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp45_fwi-mean-jjas_20760101_20801231_v2.nc'] = 'sfwi_rcp45_2080'
        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp45_fwi-mean-jjas_20960101_20981231_v2.nc'] = 'sfwi_rcp45_2098'

        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp85_fwi-mean-jjas_20710101_20751231_v2.nc'] = 'sfwi_rcp85_2075'
        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp85_fwi-mean-jjas_20760101_20801231_v2.nc'] = 'sfwi_rcp85_2080'
        names_dict['eur11_rca4_CNRM-CERFACS-CNRM-CM5_rcp85_fwi-mean-jjas_20960101_20981231_v2.nc'] = 'sfwi_rcp85_2098'

        # zarr parameters
        hazard_type = 'fire_cds'
        data_source_name = 'ecb'
        version = 'v1'
        dataset_names = list(names_dict.values())
        self.group_path_arrays = [os.path.join(hazard_type, data_source_name, version, dataset_name) for dataset_name in dataset_names]
        
        self.data_filenames = list(names_dict.keys())
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

        create_image_set(source, source_path, target, target_path)

    def create_categories(self, data):
        """
        Create categories:

        very low: <5.2
        low: 5.2 - 11.2
        moderate: 11.2 - 21.3
        high: 21.3 - 38.0
        very high: 38.0 - 50
        extreme: >=50.0
        """

        data_cat = np.nan_to_num(data)
        data_cat = np.where((data_cat > 0.001) & (data_cat <= 5.2), 1, data_cat) # very-low
        data_cat = np.where((data_cat > 5.2) & (data_cat <= 11.2), 2, data_cat) # low
        data_cat = np.where((data_cat > 11.2) & (data_cat <= 21.3), 3, data_cat) # moderate
        data_cat = np.where((data_cat > 21.3) & (data_cat <= 38), 4, data_cat) # high
        data_cat = np.where((data_cat > 38) & (data_cat < 50), 5, data_cat) # very high
        data_cat = np.where((data_cat >= 50) , 6, data_cat) # extreme

        return data_cat
    
    def onboard_all(self):
        """
        This functions populates the s3 for every file.
        """

        for i in range(len(self.path_to_nc)):

            inputfile = self.path_to_nc[i]
            self.group_path_array = self.group_path_arrays[i]

            data = xr.open_dataset(inputfile)
            data = data['fwi-mean-jjas'][-1,:,:]
            data = data.rename({'rlat':'latitude', 'rlon':'longitude'})
            data['time'] = self.return_periods[0]

            cat_data = self.create_categories(data.data)
            data.data = cat_data

            self.oscZ.write(self.group_path_array, data)

            self.create_map_images()


# https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-dev-s3-keys
bucket_name = 'physrisk-hazard-indicators-dev01'
prefix = 'hazard'
zarr_storage = 'hazard_consortium.zarr'

temp_dir = 'data'
# JRCLandslides_ = CDSWildfire(bucket_name, prefix, zarr_storage, temp_dir)
# JRCLandslides_.connect_to_bucket()
# JRCLandslides_.create_OSCZarr_object()
# JRCLandslides_.onboard_all()