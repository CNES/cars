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
this module contains the constants of sparse matching.
"""


# USED VARIABLES


SPARSE_MATCHING_RUN_TAG = "sparse_matching_run"


# INFOS

# Sparse matching PARAMS
DISPARITY_MARGIN_TAG = "disparity_margin"
ELEVATION_DELTA_LOWER_BOUND = "elevation_delta_lower_bound"
ELEVATION_DELTA_UPPER_BOUND = "elevation_delta_upper_bound"
EPIPOLAR_ERROR_UPPER_BOUND = "epipolar_error_upper_bound"
EPIPOLAR_ERROR_MAXIMUM_BIAS = "epipolar_error_maximum_bias"

SIFT_THRESH_HOLD = "sift_matching_threshold"
SIFT_N_OCTAVE = "sift_n_octave"
SIFT_N_SCALE_PER_OCTAVE = "sift_n_scale_per_octave"
SIFT_PEAK_THRESHOLD = "sift_peak_threshold"
SIFT_EDGE_THRESHOLD = "sift_edge_threshold"
SIFT_MAGNIFICATION = "sift_magnification"
SIFT_BACK_MATCHING = "sift_back_matching"

# Sparse matching RUN
DISP_LOWER_BOUND = "disp_lower_bound"
DISP_UPPER_BOUND = "disp_upper_bound"


# disparity range computation
DISPARITY_RANGE_COMPUTATION_TAG = "disparity_range_computation_run"
MINIMUM_DISPARITY_TAG = "minimum_disparity"
MAXIMUM_DISPARITY_TAG = "maximum_disparity"
MATCHES_TAG = "matches"
DISPARITY_MARGIN_PARAM_TAG = "disparity_margin_param"

# Matches filtering
METHOD = "method"
MATCHES_FILTERING_TAG = "matches_filtering"
RAW_MATCHES_TAG = "raw_matches"
FILTERED_MATCHES_TAG = "filtered_matches"
NUMBER_MATCHES_TAG = "number_matches"
RAW_NUMBER_MATCHES_TAG = "raw_number_matches"
BEFORE_CORRECTION_EPI_ERROR_MEAN = "before_correction_epi_error_mean"
BEFORE_CORRECTION_EPI_ERROR_STD = "before_correction_epi_error_std"
BEFORE_CORRECTION_EPI_ERROR_MAX = "before_correction_epi_error_max"
