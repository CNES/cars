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
This module contains the output  constants
"""

# Pipeline output keys
OUT_DIRECTORY = "directory"
PRODUCT_LEVEL = "product_level"
DSM_BASENAME = "dsm_basename"
MASK_BASENAME = "mask_basename"
CLASSIFICATION_BASENAME = "classification_basename"
COLOR_BASENAME = "color_basename"
FILLING_BASENAME = "filling_basename"
INFO_BASENAME = "info_basename"
PERFORMANCE_MAP_BASENAME = "performance_map_basename"
CONTRIBUTING_PAIR_BASENAME = "contributing_pair_basename"
OUT_GEOID = "geoid"
EPSG = "epsg"
RESOLUTION = "resolution"
SAVE_BY_PAIR = "save_by_pair"
AUXILIARY = "auxiliary"

# Auxiliary keys
AUX_COLOR = "color"
AUX_MASK = "mask"
AUX_CLASSIFICATION = "classification"
AUX_PERFORMANCE_MAP = "performance_map"
AUX_FILLING = "filling"
AUX_CONTRIBUTING_PAIR = "contributing_pair"


# Output tree constants
DSM_DIRECTORY = "dsm"
