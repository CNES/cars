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
this module contains the constants of fill_disp.
"""


# USED VARIABLES


FILL_DISP_PARAMS_TAG = "fill_dip_params"
FILL_DISP_RUN_TAG = "fill_disp_run"


# PARAMS
METHOD = "method"  # default : 'plane'
INTERP_TYPE = "interpolation_type"
INTERP_METHOD = "interpolation_method"
MAX_DIST = "max_search_distance"
SMOOTH_IT = "smoothing_iterations"
IGNORE_NODATA = "ignore_nodata_at_disp_mask_borders"
IGNORE_ZERO = "ignore_zero_fill_disp_mask_values"
IGNORE_EXTREMA = "ignore_extrema_disp_values"
