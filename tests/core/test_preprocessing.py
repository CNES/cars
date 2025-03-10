#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
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
Test module for cars/core/preprocessing.py
"""

# Standard imports

import numpy as np

# Third party imports
import pytest

# CARS import
from cars.core import preprocessing


@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    vals = [
        (1.442299, 43.600764, 32631),
        (None, 43.600764, 32632),
        (1.442299, None, 32632),
        (None, None, 32632),
        (np.nan, 43.600764, 32632),
        (1.442299, np.nan, 32632),
        (np.nan, np.nan, 32632),
    ]

    for lon, lat, gt in vals:
        epsg = preprocessing.get_utm_zone_as_epsg_code(lon, lat)
        assert epsg == gt
