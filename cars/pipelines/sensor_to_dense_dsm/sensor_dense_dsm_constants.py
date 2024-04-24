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

EPSG = "epsg"
INITIAL_ELEVATION = "initial_elevation"
USE_ENDOGENOUS_ELEVATION = "use_endogenous_elevation"

CHECK_INPUTS = "check_inputs"
DEFAULT_ALT = "default_alt"
ROI = "roi"
GEOID = "geoid"
DEBUG_WITH_ROI = "debug_with_roi"

INPUT_IMG = "image"
INPUT_MSK = "mask"
INPUT_CLASSIFICATION = "classification"
INPUT_GEO_MODEL = "geomodel"
INPUT_GEO_MODEL_TYPE = "geomodel_type"
INPUT_GEO_MODEL_FILTER = "geomodel_filters"
INPUT_NODATA = "no_data"
INPUT_COLOR = "color"
USE_EPIPOLAR_A_PRIORI = "use_epipolar_a_priori"
EPIPOLAR_A_PRIORI = "epipolar_a_priori"
TERRAIN_A_PRIORI = "terrain_a_priori"

# inner epipolar a priori constants
GRID_CORRECTION = "grid_correction"
DISPARITY_RANGE = "disparity_range"
DEM_MEDIAN = "dem_median"
DEM_MIN = "dem_min"
DEM_MAX = "dem_max"

# Pipeline output
OUT_DIR = "out_dir"
DSM_BASENAME = "dsm_basename"
CLR_BASENAME = "clr_basename"
INFO_BASENAME = "info_basename"
