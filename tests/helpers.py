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
Helpers shared testing generic module:
contains global shared generic functions for tests/*.py
TODO: Try to put the most part in cars source code (if pertinent) and
organized functionnally.
TODO: add conftest.py general tests conf with tests refactor.
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pandora
import rasterio as rio
from pandora.check_json import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
)
from pandora.state_machine import PandoraMachine

# CARS imports
from cars.externals.matching.correlator_configuration.corr_conf import (
    check_input_section_custom_cars,
    get_config_input_custom_cars,
)


def cars_path():
    """
    Return root of cars source directory
    One level down from tests
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def absolute_data_path(data_path):
    """
    Return a full absolute path to test data
    environment variable.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_folder, data_path)


def get_geoid_path():
    return os.path.join(cars_path(), "cars/conf/geoid/egm96.grd")


def temporary_dir():
    """
    Returns path to temporary dir from CARS_TEST_TEMPORARY_DIR environment
    variable. Defaults to /tmp
    """
    if "CARS_TEST_TEMPORARY_DIR" not in os.environ:
        # return default tmp dir
        return "/tmp"
    # return env defined tmp dir
    return os.environ["CARS_TEST_TEMPORARY_DIR"]


def assert_same_images(actual, expected, rtol=0, atol=0):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            assert rio_actual.transform == rio_expected.transform
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            np.testing.assert_allclose(
                rio_actual.read(), rio_expected.read(), rtol=rtol, atol=atol
            )


def assert_same_datasets(actual, expected, rtol=0, atol=0):
    """
    Compare two datasets:
    """
    assert (
        list(actual.attrs.keys()).sort() == list(expected.attrs.keys()).sort()
    )
    for key in expected.attrs.keys():
        if isinstance(expected.attrs[key], np.ndarray):
            np.testing.assert_allclose(actual.attrs[key], expected.attrs[key])
        else:
            assert actual.attrs[key] == expected.attrs[key]
    assert actual.dims == expected.dims
    assert (
        list(actual.coords.keys()).sort() == list(expected.coords.keys()).sort()
    )
    for key in expected.coords.keys():
        np.testing.assert_allclose(
            actual.coords[key].values, expected.coords[key].values
        )
    assert (
        list(actual.data_vars.keys()).sort()
        == list(expected.data_vars.keys()).sort()
    )
    for key in expected.data_vars.keys():
        np.testing.assert_allclose(
            actual[key].values, expected[key].values, rtol=rtol, atol=atol
        )


def create_corr_conf():
    """
    Create correlator configuration for stereo testing
    TODO: put in CARS source code ? (external?)
    """
    user_cfg = {}
    user_cfg["input"] = {}
    user_cfg["pipeline"] = {}
    user_cfg["pipeline"]["right_disp_map"] = {}
    user_cfg["pipeline"]["right_disp_map"]["method"] = "accurate"
    user_cfg["pipeline"]["matching_cost"] = {}
    user_cfg["pipeline"]["matching_cost"]["matching_cost_method"] = "census"
    user_cfg["pipeline"]["matching_cost"]["window_size"] = 5
    user_cfg["pipeline"]["matching_cost"]["subpix"] = 1
    user_cfg["pipeline"]["optimization"] = {}
    user_cfg["pipeline"]["optimization"]["optimization_method"] = "sgm"
    user_cfg["pipeline"]["optimization"]["P1"] = 8
    user_cfg["pipeline"]["optimization"]["P2"] = 32
    user_cfg["pipeline"]["optimization"]["p2_method"] = "constant"
    user_cfg["pipeline"]["optimization"]["penalty_method"] = "sgm_penalty"
    user_cfg["pipeline"]["optimization"]["overcounting"] = False
    user_cfg["pipeline"]["optimization"]["min_cost_paths"] = False
    user_cfg["pipeline"]["disparity"] = {}
    user_cfg["pipeline"]["disparity"]["disparity_method"] = "wta"
    user_cfg["pipeline"]["disparity"]["invalid_disparity"] = 0
    user_cfg["pipeline"]["refinement"] = {}
    user_cfg["pipeline"]["refinement"]["refinement_method"] = "vfit"
    user_cfg["pipeline"]["filter"] = {}
    user_cfg["pipeline"]["filter"]["filter_method"] = "median"
    user_cfg["pipeline"]["filter"]["filter_size"] = 3
    user_cfg["pipeline"]["validation"] = {}
    user_cfg["pipeline"]["validation"]["validation_method"] = "cross_checking"
    user_cfg["pipeline"]["validation"]["cross_checking_threshold"] = 1.0
    # Import plugins before checking configuration
    pandora.import_plugin()
    # Check configuration and update the configuration with default values
    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()
    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, pandora_machine)
    # check a part of input section
    user_cfg_input = get_config_input_custom_cars(user_cfg)
    cfg_input = check_input_section_custom_cars(user_cfg_input)
    # concatenate updated config
    cfg = concat_conf([cfg_input, cfg_pipeline])
    return cfg
