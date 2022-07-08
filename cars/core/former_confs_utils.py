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
former_conds_utils contains function used to get data in artificially
 created configuration, to imitate former prepare and compute_dsm
  outputs.

TODO will be removed
"""


# Former output tags of prepare.py (conf/output_prepare.py)

PREPROCESSING_SECTION_TAG = "preprocessing"
PREPROCESSING_OUTPUT_SECTION_TAG = "output"

MINIMUM_DISPARITY_TAG = "minimum_disparity"
MAXIMUM_DISPARITY_TAG = "maximum_disparity"
DISP_TO_ALT_RATIO_TAG = "disp_to_alt_ratio"

EPIPOLAR_SIZE_X_TAG = "epipolar_size_x"
EPIPOLAR_SIZE_Y_TAG = "epipolar_size_y"

LEFT_EPIPOLAR_GRID_TAG = "left_epipolar_grid"
RIGHT_EPIPOLAR_GRID_TAG = "right_epipolar_grid"
RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG = "right_epipolar_uncorrected_grid"


# Former output tags of compute_dsm.py (conf/output_compute_dsm.py)


def get_disp_min_max(conf):
    """
    Get disp min and disp max from former cars conf

    :param conf: former cars conf
    :type conf: dict

    :return: disp min, disp max
    :rtype: tuple(float, float)

    """

    preprocessing_output_conf = conf[PREPROCESSING_SECTION_TAG][
        PREPROCESSING_OUTPUT_SECTION_TAG
    ]
    minimum_disparity = preprocessing_output_conf[MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_conf[MAXIMUM_DISPARITY_TAG]

    return minimum_disparity, maximum_disparity


def get_epi_sizes_from_cars_post_prepare_configuration(conf):
    """
    Get epipolar sizes x and y from former cars conf

    :param conf: former cars conf
    :type conf: dict

    :return: epi_size_x, epi_size_y
    :rtype: tuple(int, int)

    """

    preprocessing_output_conf = conf[PREPROCESSING_SECTION_TAG][
        PREPROCESSING_OUTPUT_SECTION_TAG
    ]

    epi_size_x = preprocessing_output_conf[EPIPOLAR_SIZE_X_TAG]

    epi_size_y = preprocessing_output_conf[EPIPOLAR_SIZE_Y_TAG]

    return epi_size_x, epi_size_y


def get_grid_from_cars_post_prepare_configurations(conf):
    """
    Get grids path from former cars conf

    :param conf: former cars conf
    :type conf: dict

    :return: grid1, grid2, uncorrected_grid_2
    :rtype: tuple(str, str, str)

    """

    preprocessing_output_conf = conf[PREPROCESSING_SECTION_TAG][
        PREPROCESSING_OUTPUT_SECTION_TAG
    ]

    grid1 = preprocessing_output_conf[LEFT_EPIPOLAR_GRID_TAG]
    grid2 = preprocessing_output_conf[RIGHT_EPIPOLAR_GRID_TAG]

    uncorrected_grid_2 = preprocessing_output_conf.get(
        RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG, None
    )

    return grid1, grid2, uncorrected_grid_2
