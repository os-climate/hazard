from os import path
from hazard.models.water_temp import WaterTemperatureIndicator
from .utilities import test_output_dir

def test_water_temp_indicator(test_output_dir):
    model = WaterTemperatureIndicator()
    model.download_all(path.join(test_output_dir, "future_streams"))