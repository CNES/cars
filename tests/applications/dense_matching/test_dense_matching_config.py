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
Test module for config of cars/applications/dense_matching/census_mccnn_sgm.py
"""

# Third party imports
import pytest

# CARS imports
from cars.applications.dense_matching.census_mccnn_sgm import CensusMccnnSgm


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "max_offset,confidence_intervals,expected_error",
    [(10, False, ValueError), (30, True, KeyError)],
)
def test_check_conf_with_error(
    max_offset, confidence_intervals, expected_error
):
    """
    Test configuration check for dense matching application
    First, with forbidden value for elevation offset (min > max)
    Then, with incoherent confidence intervals conf
    not present in the loader conf
    """
    conf = {
        "method": "census_sgm",
        "min_elevation_offset": 20,
        "max_elevation_offset": max_offset,  # should be > min
        "compute_intervals": confidence_intervals,
    }
    with pytest.raises(expected_error):
        _ = CensusMccnnSgm(conf)
