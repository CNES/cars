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
Test module for cars/parameters/advanced_parameters.py
"""

import pytest

from cars.pipelines.parameters import advanced_parameters


@pytest.mark.unit_tests
def test_advanced_parameters_full_config():
    """
    Test configuration check for advanced parameters
    """

    config = {
        "debug_with_roi": True,
        "use_epipolar_a_priori": True,
        "epipolar_a_priori": {
            "left_right": {
                "grid_correction": [
                    4.288258198116454,
                    -5.126994797702842e-05,
                    -0.0001868504592756687,
                    0.7633338209640762,
                    0.00017551039816977979,
                    7.416961236000771e-05,
                ],
                "disparity_range": [-26.157764557028997, 26.277429517242638],
            }
        },
        "terrain_a_priori": {
            "dem_median": "/tmp/cars/output/dem_median.tif",
            "dem_min": "/tmp/cars/output/dem_min.tif",
            "dem_max": "/tmp/cars/output/dem_max.tif",
        },
    }

    advanced_parameters.check_advanced_parameters(config)


@pytest.mark.unit_tests
def test_advanced_parameters_minimal():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    advanced_parameters.check_advanced_parameters(config)


@pytest.mark.unit_tests
def test_advanced_parameters_no_epipolar():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    advanced_parameters.check_advanced_parameters(
        config, check_epipolar_a_priori=False
    )
