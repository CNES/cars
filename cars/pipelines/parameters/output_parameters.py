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
This module contains the output definition
"""

import logging
import os

from json_checker import Checker, Or
from pyproj import CRS

import cars.core.constants as cst
from cars.core.utils import safe_makedirs
from cars.pipelines.parameters import output_constants


def check_output_parameters(conf):
    """
    Check the output json configuration and fill in default values

    :param conf: configuration of output
    :type conf: dict
    :param pipeline_name: name of corresponding pipeline
    :type pipeline_name: str
    """
    overloaded_conf = conf.copy()
    out_dir = conf[output_constants.OUT_DIRECTORY]
    out_dir = os.path.abspath(out_dir)
    # Ensure that output directory and its subdirectories exist
    safe_makedirs(out_dir)

    # Overload some parameters
    overloaded_conf[output_constants.OUT_DIRECTORY] = out_dir

    overloaded_conf[output_constants.PRODUCT_LEVEL] = overloaded_conf.get(
        output_constants.PRODUCT_LEVEL, "dsm"
    )
    if isinstance(overloaded_conf[output_constants.PRODUCT_LEVEL], str):
        overloaded_conf[output_constants.PRODUCT_LEVEL] = [
            overloaded_conf[output_constants.PRODUCT_LEVEL]
        ]
    for level in overloaded_conf[output_constants.PRODUCT_LEVEL]:
        if level not in ["dsm", "depth_map", "point_cloud"]:
            raise RuntimeError("Unknown product level {}".format(level))

    overloaded_conf[output_constants.OUT_GEOID] = overloaded_conf.get(
        output_constants.OUT_GEOID, False
    )
    overloaded_conf[output_constants.EPSG] = overloaded_conf.get(
        output_constants.EPSG, None
    )

    overloaded_conf[output_constants.RESOLUTION] = overloaded_conf.get(
        output_constants.RESOLUTION, 0.5
    )

    overloaded_conf[output_constants.SAVE_BY_PAIR] = overloaded_conf.get(
        output_constants.SAVE_BY_PAIR, False
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY] = overloaded_conf.get(
        output_constants.AUXILIARY, {}
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY][output_constants.AUX_COLOR] = (
        overloaded_conf[output_constants.AUXILIARY].get(
            output_constants.AUX_COLOR, True
        )
    )
    overloaded_conf[output_constants.AUXILIARY][output_constants.AUX_MASK] = (
        overloaded_conf[output_constants.AUXILIARY].get(
            output_constants.AUX_MASK, False
        )
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CLASSIFICATION
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CLASSIFICATION, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_PERFORMANCE_MAP
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_PERFORMANCE_MAP, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CONTRIBUTING_PAIR
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CONTRIBUTING_PAIR, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_FILLING
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_FILLING, False
    )

    # Check schema
    output_schema = {
        output_constants.OUT_DIRECTORY: str,
        output_constants.PRODUCT_LEVEL: list,
        output_constants.OUT_GEOID: Or(bool, str),
        output_constants.EPSG: Or(int, None),
        output_constants.RESOLUTION: Or(int, float),
        output_constants.SAVE_BY_PAIR: bool,
        output_constants.AUXILIARY: dict,
    }
    checker_output = Checker(output_schema)
    checker_output.validate(overloaded_conf)

    # check auxiliary keys
    auxiliary_schema = {
        output_constants.AUX_COLOR: bool,
        output_constants.AUX_MASK: bool,
        output_constants.AUX_CLASSIFICATION: bool,
        output_constants.AUX_PERFORMANCE_MAP: bool,
        output_constants.AUX_CONTRIBUTING_PAIR: bool,
        output_constants.AUX_FILLING: bool,
    }

    checker_auxiliary = Checker(auxiliary_schema)
    checker_auxiliary.validate(overloaded_conf[output_constants.AUXILIARY])

    if "epsg" in overloaded_conf and overloaded_conf["epsg"]:
        spatial_ref = CRS.from_epsg(overloaded_conf["epsg"])
        if spatial_ref.is_geographic:
            if overloaded_conf[output_constants.RESOLUTION] > 10e-3:
                logging.warning(
                    "The resolution of the "
                    + "point_cloud_rasterization should be "
                    + "fixed according to the epsg"
                )

    return overloaded_conf


def intialize_product_index(orchestrator, product_levels, input_pairs):
    """
    Initialize the index dictionary according to requested levels with None
    values for all paths.

    :param orchestrator: cars orchestrator
    :type orchestrator: Orchestrator
    :param product_levels: name of corresponding pipeline
    :type product_levels: list
    :param input_pairs: list containing the pair names
    :type input_pairs: list
    """

    index = {}

    if "dsm" in product_levels:
        index["dsm"] = {
            cst.INDEX_DSM_ALT: None,
            cst.INDEX_DSM_COLOR: None,
            cst.INDEX_DSM_MASK: None,
            cst.INDEX_DSM_CLASSIFICATION: None,
            cst.INDEX_DSM_PERFORMANCE_MAP: None,
            cst.INDEX_DSM_CONTRIBUTING_PAIR: None,
            cst.INDEX_DSM_FILLING: None,
        }

    if "point_cloud" in product_levels:
        # Initialize an empty index for point cloud because its content is
        # unknown at this point (tile name, save by pair or not)
        index["point_cloud"] = {}

    if "depth_map" in product_levels:
        index["depth_map"] = {}
        for pair in input_pairs:
            index["depth_map"][pair] = {
                cst.INDEX_DEPTH_MAP_X: None,
                cst.INDEX_DEPTH_MAP_Y: None,
                cst.INDEX_DEPTH_MAP_Z: None,
                cst.INDEX_DEPTH_MAP_COLOR: None,
                cst.INDEX_DEPTH_MAP_MASK: None,
                cst.INDEX_DEPTH_MAP_CLASSIFICATION: None,
                cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP: None,
                cst.INDEX_DEPTH_MAP_FILLING: None,
                cst.INDEX_DEPTH_MAP_EPSG: None,
            }

    orchestrator.update_index(index)
