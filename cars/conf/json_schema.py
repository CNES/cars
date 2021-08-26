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
This module handles the cars input schema as well as loaders ones to construct
the user final configuration json schema
"""

from copy import copy

from cars.conf import input_parameters, static_conf


def input_conf_schema():
    """
    Retrieve user input configuration.
    This function merges the loaders schema with the basic configuration
    required by cars (INPUT_CONFIGURATION_SCHEMA
    defined in the input_parameters module)

    :return: the user input configuration schema
    """
    # copy cars minimal configuration schema
    schema = copy(input_parameters.INPUT_CONFIGURATION_SCHEMA)

    # update it with the configuration required by the geometry loader
    geometry_loader = static_conf.get_geometry_loader()
    geo_schema = geometry_loader.conf_schema
    schema.update(geo_schema)

    return schema
