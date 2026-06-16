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
this module contains the constants of dense_matching.
"""
from pandora.constants import Criteria

from cars.core import constants_disparity as disp_cst

# USED VARIABLES


DENSE_MATCHING_RUN_TAG = "dense_matching"

# grids disp
DISP_MIN_GRID = "disp_min_grid"
DISP_MAX_GRID = "disp_max_grid"


# PARAMS
METHOD = "method"
MIN_EPI_TILE_SIZE = "min_epi_tile_size"
MAX_EPI_TILE_SIZE = "max_epi_tile_size"
EPI_TILE_MARGIN_IN_PERCENT = "epipolar_tile_margin_in_percent"
MIN_ELEVATION_OFFSET = "min_elevation_offset"
MAX_ELEVATION_OFFSET = "max_elevation_offset"

# ABRIDGED PANDORA CONSTANTS
IN_VALIDITY_MASK_LEFT = "IN_VALIDITY_MASK_LEFT"
IN_VALIDITY_MASK_RIGHT = "IN_VALIDITY_MASK_RIGHT"
RIGHT_INCOMPLETE_DISPARITY_RANGE = "RIGHT_INCOMPLETE_DISPARITY_RANGE"
STOPPED_INTERPOLATION = "STOPPED_INTERPOLATION"
FILLED_OCCLUSION = "FILLED_OCCLUSION"
FILLED_MISMATCH = "FILLED_MISMATCH"
LEFT_NODATA_OR_BORDER = "LEFT_NODATA_OR_BORDER"
RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING = (
    "RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING"
)
OCCLUSION = "OCCLUSION"
MISMATCH = "MISMATCH"
INCOMPLETE_VARIABLE_DISPARITY_RANGE = "INCOMPLETE_VARIABLE_DISPARITY_RANGE"


def get_criteria(key):
    """
    get pandora criteria name from abridged key

    :param key: abridged key of pandora criteria

    Returns:
        _type_: pandora criteria
    """
    return Criteria.__dict__.get("PANDORA_MSK_PIXEL_" + key)


# CORRESPONDING MSK TABLE PANDORA CARS
MASK_HASH_TABLE = {
    disp_cst.MASKED_REF: get_criteria(IN_VALIDITY_MASK_LEFT),
    disp_cst.MASKED_SEC: get_criteria(IN_VALIDITY_MASK_RIGHT),
    disp_cst.INCOMPLETE_DISP: get_criteria(RIGHT_INCOMPLETE_DISPARITY_RANGE),
    disp_cst.STOPPED_INTERP: get_criteria(STOPPED_INTERPOLATION),
    disp_cst.FILLED_OCCLUSION: get_criteria(FILLED_OCCLUSION),
    disp_cst.FILLED_FALSE_MATCH: get_criteria(FILLED_MISMATCH),
    disp_cst.INVALID_REF: get_criteria(LEFT_NODATA_OR_BORDER),
    disp_cst.INVALID_SEC: get_criteria(RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING),
    disp_cst.OCCLUSION: get_criteria(OCCLUSION),
    disp_cst.FALSE_MATCH: get_criteria(MISMATCH),
    disp_cst.INCOMPLETE_VARIABLE_DISP: get_criteria(
        INCOMPLETE_VARIABLE_DISPARITY_RANGE
    ),
}
