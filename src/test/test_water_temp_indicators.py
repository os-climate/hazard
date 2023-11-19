from os import path

import pytest
from hazard.models.water_temp import FutureStreamsSource, WaterTemperatureIndicator
from .utilities import test_output_dir

@pytest.mark.skip(reason="in development")
def test_water_temp_indicator(test_output_dir):
    source = FutureStreamsSource(path.join(test_output_dir, "future_streams"))
    source.download_all()
    model = WaterTemperatureIndicator()