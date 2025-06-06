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
cars/applications/dense_matching/census_mccnn_sgm_app.py
"""

import os

# Third party imports
import pytest

# CARS imports
from cars.applications.dense_matching.census_mccnn_sgm_app import CensusMccnnSgm
from cars.applications.dense_matching.loaders import (
    __file__ as dense_matching_loaders_init_file,
)


@pytest.mark.unit_tests
def test_check_full_conf_pandora_conf_as_dict():
    """
    Test configuration check for dense matching application
    """
    conf = {
        "method": "census_sgm_default",
        "min_epi_tile_size": 300,
        "max_epi_tile_size": 1500,
        "epipolar_tile_margin_in_percent": 60,
        "min_elevation_offset": -100,
        "max_elevation_offset": 100,
        "disp_min_threshold": -40,
        "disp_max_threshold": 40,
        "performance_map_method": ["risk", "intervals"],
        "perf_eta_max_ambiguity": 0.99,
        "perf_eta_max_risk": 0.25,
        "perf_eta_step": 0.04,
        "perf_ambiguity_threshold": 0.6,
        "use_global_disp_range": False,
        "local_disp_grid_step": 30,
        "disp_range_propagation_filter_size": 300,
        "save_intermediate_data": False,
        "loader": "pandora",
        "loader_conf": {
            "input": {},
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "census",
                    "window_size": 5,
                    "subpix": 1,
                },
                "optimization": {
                    "optimization_method": "sgm",
                    "overcounting": False,
                    "penalty": {
                        "P1": 8,
                        "P2": 32,
                        "p2_method": "constant",
                        "penalty_method": "sgm_penalty",
                    },
                },
                "disparity": {
                    "disparity_method": "wta",
                    "invalid_disparity": "NaN",
                },
                "refinement": {"refinement_method": "vfit"},
                "filter": {"filter_method": "median", "filter_size": 3},
                "validation": {
                    "validation_method": "cross_checking_accurate",
                    "cross_checking_threshold": 1.0,
                },
            },
        },
    }
    _ = CensusMccnnSgm(conf)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "method_name",
    [
        "census_sgm_default",
        "census_sgm_urban",
        "census_sgm_shadow",
        "census_sgm_mountain_and_vegetation",
        "census_sgm_homogeneous",
    ],
)
def test_check_each_pandora_conf_as_dict(method_name):
    """
    Test configuration check for dense matching application
    """
    conf = {"method": method_name}
    _ = CensusMccnnSgm(conf)


@pytest.mark.unit_tests
def test_denoise_disparity_map():
    """ "
    Test denoise disparity map
    """

    conf = {"denoise_disparity_map": True}
    _ = CensusMccnnSgm(conf)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "cross_value",
    [
        True,
        "fast",
        "accurate",
    ],
)
def test_cross_validation_value(cross_value):
    """
    Test configuration check for dense matching application
    """
    conf = {"use_cross_validation": cross_value}
    _ = CensusMccnnSgm(conf)


@pytest.mark.unit_tests
def test_check_full_conf_pandora_conf_as_file():
    """
    Test configuration check for dense matching application
    """
    loader_conf_path = os.path.dirname(dense_matching_loaders_init_file)
    census_loader_conf_path = os.path.join(
        loader_conf_path, "config_census_sgm_default.json"
    )
    conf = {
        "method": "census_sgm_default",
        "min_epi_tile_size": 300,
        "max_epi_tile_size": 1500,
        "epipolar_tile_margin_in_percent": 60,
        "min_elevation_offset": -100,
        "max_elevation_offset": 100,
        "disp_min_threshold": -40,
        "disp_max_threshold": 40,
        "performance_map_method": ["risk", "intervals"],
        "perf_eta_max_ambiguity": 0.99,
        "perf_eta_max_risk": 0.25,
        "perf_eta_step": 0.04,
        "perf_ambiguity_threshold": 0.6,
        "use_global_disp_range": False,
        "local_disp_grid_step": 30,
        "disp_range_propagation_filter_size": 300,
        "save_intermediate_data": False,
        "loader": "pandora",
        "loader_conf": census_loader_conf_path,
    }
    _ = CensusMccnnSgm(conf)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "max_offset,confidence_intervals,expected_error",
    [(10, False, ValueError)],
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

    performance_map_method = []
    if confidence_intervals:
        performance_map_method = ["intervals"]

    conf = {
        "method": "census_sgm_default",
        "min_elevation_offset": 20,
        "max_elevation_offset": max_offset,  # should be > min
        "performance_map_method": performance_map_method,
        "loader_conf": {
            "input": {},
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "census",
                    "window_size": 5,
                    "subpix": 1,
                },
                "disparity": {"disparity_method": "wta"},
            },
        },
    }
    with pytest.raises(expected_error):
        _ = CensusMccnnSgm(conf)
