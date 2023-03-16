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
file contains all the constants used in rasterization module
"""

RASTERIZATION_RUN_TAG = "rasterization_run"

# Params
METHOD = "method"
DSM_RADIUS = "dsm_radius"
SIGMA = "sigma"
GRID_POINTS_DIVISION_FACTOR = "grid_points_division_factor"
RESOLUTION = "resolution"


# Run infos
EPSG_TAG = "epsg"
DSM_TAG = "dsm"
COLOR_TAG = "color"
MSK_TAG = "msk"
CONFIDENCE_TAG = "disparity_confidence"
DSM_MEAN_TAG = "dsm_mean"
DSM_STD_TAG = "dsm_std"
DSM_N_PTS_TAG = "dsm_n_pts"
DSM_POINTS_IN_CELL_TAG = "dsm_points_in_cell"
DSM_NO_DATA_TAG = "dsm_no_data"
COLOR_NO_DATA_TAG = "color_no_data"
