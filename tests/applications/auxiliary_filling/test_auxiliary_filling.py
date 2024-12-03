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
Test module for config of
cars/applications/auxiliary_filling/auxiliary_filling_from_sensors.py
"""

# Third party imports
import pytest

# CARS imports
from cars.applications.auxiliary_filling.auxiliary_filling_from_sensors import (
    AuxiliaryFillingFromSensors,
)


@pytest.mark.unit_tests
def test_auxiliary_filling():
    """
    Test for AuxiliaryFillingFromSensors application
    """

    conf = {
        "save_intermediate_data": False,
    }

    auxiliary_filling_application = AuxiliaryFillingFromSensors(conf)

    res = auxiliary_filling_application.run()

    # Dummy assert for now
    assert res is True
