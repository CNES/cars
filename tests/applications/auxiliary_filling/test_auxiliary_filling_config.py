#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Test module for
cars/applications/auxiliary_filling/auxiliary_filling_from_sensors_app.py
"""

# Third party imports
import pytest

# CARS imports
from cars.applications.auxiliary_filling import (
    auxiliary_filling_from_sensors_app as auxiliary_app,
)


@pytest.mark.unit_tests
def test_check_full_conf():
    """
    Test configuration check for AuxiliaryFillingFromSensors application
    """
    conf = {
        "save_intermediate_data": False,
        "mode": "full",
        "activated": True,
        "use_mask": True,
        "color_interpolator": "linear",
    }
    _ = auxiliary_app.AuxiliaryFillingFromSensors(conf)
