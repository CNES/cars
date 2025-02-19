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
This module contains the advanced parameter definitions
"""

SAVE_INTERMEDIATE_DATA = "save_intermediate_data"
PHASING = "phasing"
DEBUG_WITH_ROI = "debug_with_roi"

USE_EPIPOLAR_A_PRIORI = "use_epipolar_a_priori"
EPIPOLAR_A_PRIORI = "epipolar_a_priori"
GROUND_TRUTH_DSM = "ground_truth_dsm"


MERGING = "merging"

# inner epipolar a priori constants
GRID_CORRECTION = "grid_correction"
DISPARITY_RANGE = "disparity_range"

TERRAIN_A_PRIORI = "terrain_a_priori"

DEM_MEDIAN = "dem_median"
DEM_MIN = "dem_min"
DEM_MAX = "dem_max"
ALTITUDE_DELTA_MAX = "altitude_delta_max"
ALTITUDE_DELTA_MIN = "altitude_delta_min"

# ground truth dsm
INPUT_GROUND_TRUTH_DSM = "dsm"
INPUT_CLASSIFICATION = "classification"
INPUT_AUX_PATH = "auxiliary_data"
INPUT_AUX_INTERP = "auxiliary_data_interpolation"
INPUT_GEOID = "geoid"
INPUT_EPSG = "epsg"

PERFORMANCE_MAP_CLASSES = "performance_map_classes"
