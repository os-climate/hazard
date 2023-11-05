from os import path
import os
import s3fs

from hazard.models.drought_index import DroughtIndicator
from .utilities import TestTarget, s3_credentials, test_output_dir

def test_spei_indicator(test_output_dir, s3_credentials):
    # to test 
    s3 = s3fs.S3FileSystem(anon=False, key=os.environ["OSC_S3_ACCESS_KEY_DEV"], secret=os.environ["OSC_S3_SECRET_KEY_DEV"])
    working_path = os.environ["OSC_S3_BUCKET_DEV"]+"/drought/osc/v01"
    model = DroughtIndicator(s3, working_path)
    target = TestTarget()
    model.calculate_spei("MIROC6", "ssp585")
    #model.calculate_annual_average_spei("MIROC6", "ssp585", 2080, target)