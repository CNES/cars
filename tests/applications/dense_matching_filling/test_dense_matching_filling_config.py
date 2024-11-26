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
Test module for config of cars/applications/dense_match_filling/plane.py
"""

import json_checker

# Third party imports
import pytest

# CARS imports
from cars.applications.dense_match_filling.plane import PlaneFill
from cars.applications.dense_match_filling.zero_padding import ZerosPadding


@pytest.mark.unit_tests
def test_check_full_conf_zero_padding():
    """
    Test configuration check for ZerosPadding application
    """
    conf = {
        "method": "zero_padding",
        "classification": ["water", "building"],
        "save_intermediate_data": False,
    }
    _ = ZerosPadding(conf)


@pytest.mark.unit_tests
def test_check_full_conf_plane():
    """
    Test configuration check for PlaneFill application
    """
    conf = {
        "method": "plane",
        "interpolation_type": "pandora",
        "interpolation_method": "mc_cnn",
        "max_search_distance": 100,
        "smoothing_iterations": 1,
        "ignore_nodata_at_disp_mask_borders": False,
        "ignore_zero_fill_disp_mask_values": True,
        "ignore_extrema_disp_values": True,
        "nb_pix": 20,
        "percent_to_erode": 0.2,
        "classification": ["water", "building"],
        "save_intermediate_data": False,
    }
    _ = PlaneFill(conf)


@pytest.mark.unit_tests
def test_check_conf_with_error():
    """
    Test configuration check for PlaneFill application
    with forbidden values for classification parameter (list of int
    instead of list of str)
    """
    conf = {
        "method": "plane",
        "classification": [1, 2, 3],
    }
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        _ = PlaneFill(conf)
