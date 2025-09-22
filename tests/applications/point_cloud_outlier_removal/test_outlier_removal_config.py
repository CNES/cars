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
Test module for config of
cars/applications/point_cloud_outliers_removing/statistical_app.py
"""

import json_checker

# Third party imports
import pytest

from cars.applications.point_cloud_outlier_removal.small_components_app import (
    SmallComponents,
)

# CARS imports
from cars.applications.point_cloud_outlier_removal.statistical_app import (
    Statistical,
)


@pytest.mark.unit_tests
def test_check_full_conf_small_components():
    """
    Test configuration check for outliers removing application
    """
    conf = {
        "method": "small_components",
        "save_intermediate_data": False,
        "save_by_pair": False,
        "on_ground_margin": 11,
        "connection_distance": 3.0,
        "nb_points_threshold": 50,
        "clusters_distance_threshold": None,
        "half_epipolar_size": 5,
    }
    _ = SmallComponents(conf)


@pytest.mark.unit_tests
def test_check_full_conf_statistical():
    """
    Test configuration check for outliers removing application
    """
    conf = {
        "method": "statistical",
        "save_intermediate_data": False,
        "save_by_pair": False,
        "k": 50,
        "filtering_constant": 0.0,
        "mean_factor": 1.0,
        "std_dev_factor": 5.0,
        "use_median": False,
        "half_epipolar_size": 5,
    }
    _ = Statistical(1, conf)


@pytest.mark.unit_tests
def test_check_conf_with_error():
    """
    Test configuration check for outliers removing application
    with forbidden value for parameter k
    """
    conf = {
        "method": "statistical",
        "k": 0,  # should be > 0
    }
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        _ = Statistical(1, conf)
