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
================================
  Module "configure_correlator"
================================
This module contains functions to retrieve the correlator configuration
"""

import pandora
from pandora.JSON_checker import get_config_pipeline, check_pipeline_section,\
                                 get_config_image, check_image_section,\
                                 concat_conf


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
        # Defaut Pandora configuration:
        #  * Census with 5 X 5 window
        #  * SGM
        user_cfg = {}
        user_cfg['image'] = {}
        user_cfg['image']['valid_pixels'] = 0
        user_cfg['image']['no_data'] = 255
        user_cfg["stereo"] = {}
        user_cfg["stereo"]["stereo_method"] = "census"
        user_cfg["stereo"]["window_size"] = 5
        user_cfg["stereo"]["subpix"] = 1
        user_cfg["aggregation"] = {}
        user_cfg["aggregation"]["aggregation_method"] = "none"
        user_cfg["optimization"] = {}
        user_cfg["optimization"]["optimization_method"] = "sgm"
        user_cfg["optimization"]["P1"] = 8
        user_cfg["optimization"]["P2"] = 32
        user_cfg["optimization"]["p2_method"] = "constant"
        user_cfg["optimization"]["penalty_method"] = "sgm_penalty"
        user_cfg["optimization"]["overcounting"] = False
        user_cfg["optimization"]["min_cost_paths"] = False
        user_cfg["refinement"] = {}
        user_cfg["refinement"]["refinement_method"] = "vfit"
        user_cfg["filter"] = {}
        user_cfg["filter"]["filter_method"] = "median"
        user_cfg["filter"]["filter_size"] = 3
        user_cfg["validation"] = {}
        user_cfg["validation"]["validation_method"] = "cross_checking"
        user_cfg["validation"]["cross_checking_threshold"] = 1.0
        user_cfg["validation"]["right_left_mode"] = "accurate"
        user_cfg["validation"]["interpolated_disparity"] = "none"
    else:
        user_cfg = pandora.read_config_file(corr_file_path)

    # Import plugins before checking confifuration
    pandora.import_plugin()
    # Check configuration and update the configuration with default values
    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline)
    # check image
    user_cfg_image = get_config_image(user_cfg)
    cfg_image = check_image_section(user_cfg_image)
    # concatenate updated config
    cfg = concat_conf([cfg_image, cfg_pipeline])
    return cfg
