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
CARS Disparity Constants module
"""

# disparity map
MAP = "disp"
CONFIDENCE = "confidence"
CONFIDENCE_KEY = "cost_volume_confidence"
INTERVAL = "interval_bounds"
INTERVAL_INF = "confidence_from_interval_bounds_inf"
INTERVAL_SUP = "confidence_from_interval_bounds_sup"
EPI_DISP_MIN_GRID = "disp_min_grid"
EPI_DISP_MAX_GRID = "disp_max_grid"

# disparity mask
VALID = "disp_msk"
FILLING = "filling"
INVALID_REF = "msk_invalid_ref"
INVALID_SEC = "msk_invalid_sec"
MASKED_REF = "msk_masked_ref"
MASKED_SEC = "msk_masked_sec"
OCCLUSION = "msk_occlusion"
FALSE_MATCH = "msk_false_match"
INCOMPLETE_DISP = "msk_incomplete_disp"
STOPPED_INTERP = "msk_stopped_interp"
FILLED_OCCLUSION = "msk_filled_occlusion"
FILLED_FALSE_MATCH = "msk_filled_false_match"
INSIDE_SEC_ROI = "msk_inside_sec_roi"
DISP_TO_0 = "msk_disp_to_0"
