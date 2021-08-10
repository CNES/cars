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
Configure correlator module:
contains functions to retrieve the correlator configuration
"""

from typing import Dict

import numpy as np
import pandora
from json_checker import Checker, Or
from pandora.check_json import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
    update_conf,
)
from pandora.state_machine import PandoraMachine


def configure_correlator(corr_file_path=None):
    """
    Provide correlator configuration through dictionary format.
    If a correlator file path is provided, read correlator
    configuration file to upgrade correlator configuration.
    Otherwise, use default configuration depending of the
    correlator type used.
    Relative paths will be made absolute.

    :param corr_file_path: Path to correlation confuiguration json file
    :type corr_file_path: str

    :returns: The dictionary with correlator configuration
    :rtype: dict
    """
    # Configure correlator
    # Configure pandora
    # Read the user configuration file
    user_cfg = None
    if corr_file_path is None:
        # Default Pandora configuration:
        #  * Census with 5 X 5 window
        #  * SGM
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
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = "NaN"
        user_cfg["pipeline"]["refinement"] = {}
        user_cfg["pipeline"]["refinement"]["refinement_method"] = "vfit"
        user_cfg["pipeline"]["filter"] = {}
        user_cfg["pipeline"]["filter"]["filter_method"] = "median"
        user_cfg["pipeline"]["filter"]["filter_size"] = 3
        user_cfg["pipeline"]["validation"] = {}
        user_cfg["pipeline"]["validation"][
            "validation_method"
        ] = "cross_checking"
        user_cfg["pipeline"]["validation"]["cross_checking_threshold"] = 1.0
    else:
        user_cfg = pandora.read_config_file(corr_file_path)

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


input_configuration_schema_custom_cars = {
    "nodata_left": Or(
        int, lambda x: np.isnan(x)  # pylint: disable=unnecessary-lambda
    ),
    "nodata_right": Or(
        int, lambda x: np.isnan(x)  # pylint: disable=unnecessary-lambda
    ),
}

default_short_configuration_input_custom_cars = {
    "input": {
        "nodata_left": -9999,
        "nodata_right": -9999,
    }
}


def get_config_input_custom_cars(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the input configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "input" in user_cfg:
        cfg["input"] = {}

        if "nodata_left" in user_cfg["input"]:
            cfg["input"]["nodata_left"] = user_cfg["input"]["nodata_left"]

        if "nodata_right" in user_cfg["input"]:
            cfg["input"]["nodata_right"] = user_cfg["input"]["nodata_right"]

    return cfg


def check_input_section_custom_cars(
    user_cfg: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input_custom_cars, user_cfg)

    # check schema
    configuration_schema = {"input": input_configuration_schema_custom_cars}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg
