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
This module handles the geometry loader specific configuration
"""

from copy import copy
from typing import Dict

# cars import
from cars.conf import input_parameters, static_conf
from cars.core.geometry import AbstractGeometry


def geo_loader_can_open(conf) -> bool:
    """
    Check if the pair given in the configuration have valid geometric models

    :param conf: CARS input configuration
    :return:  True if the pair has valid geometrical models, False otherwise
    """
    geo_conf = extract_geo_conf(conf)
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            static_conf.get_geometry_plugin()
        )
    )
    return geo_plugin.check_products_consistency(geo_conf)


def geo_conf_schema() -> Dict[str, str]:
    """
    Retrieve the geometry loader configuration schema

    :return: a json checker schema
    """
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            static_conf.get_geometry_plugin()
        )
    )
    return geo_plugin.geo_conf_schema()


def extract_geo_conf(conf) -> Dict[str, str]:
    """
    Extract from the input configuration the fields required by the geometry
    loader

    :param conf: CARS input configuration dictionary
    :return: the geometry loader configuration
    """
    geo_model = {}

    required_geo_conf = geo_conf_schema()

    for key in required_geo_conf.keys():
        geo_model[key] = conf[key]
    return geo_model


def get_input_schema_with_geo_info() -> input_parameters.InputConfigurationType:
    """
    Retrieve input configuration with the geometry loader specific features

    :return: the input configuration dictionary
    """
    # copy cars minimal configuration dictionary
    schema = copy(input_parameters.INPUT_CONFIGURATION_SCHEMA)

    # update it with the configuration required by the geometry loader
    schema.update(geo_conf_schema())

    return schema
