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
this module contains the constants used in sensor_to_full_resolution
 pipeline.
"""

# Sensor input

SENSORS = "sensors"
PAIRING = "pairing"

INITIAL_ELEVATION = "initial_elevation"

CHECK_INPUTS = "check_inputs"
ROI = "roi"
GEOID = "geoid"
DEM_PATH = "dem"
ALTITUDE_DELTA_MIN = "altitude_delta_min"
ALTITUDE_DELTA_MAX = "altitude_delta_max"

INPUT_IMG = "image"
INPUT_MSK = "mask"
INPUT_CLASSIFICATION = "classification"
INPUT_GEO_MODEL = "geomodel"
INPUT_GEO_MODEL_TYPE = "geomodel_type"
INPUT_GEO_MODEL_FILTER = "geomodel_filters"
INPUT_NODATA = "no_data"
INPUT_COLOR = "color"

CARS_DEFAULT_ALT = 0  # Default altitude used in cars pipelines
