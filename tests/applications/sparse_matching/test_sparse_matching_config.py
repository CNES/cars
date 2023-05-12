#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
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
Test module for cars/applications/sparse_matching/sift.py
"""

# Third party imports
import pytest

# CARS imports
from cars.applications.sparse_matching.sift import Sift


@pytest.mark.unit_tests
def test_check_conf_with_error():
    """
    Test configuration check with forbidden values
    """
    conf = {
        "method": "sift",
        "elevation_delta_lower_bound": 1000,
        "elevation_delta_upper_bound": -1000,  # should be > lower bound
    }
    with pytest.raises(ValueError):
        _ = Sift(conf)
