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
# pylint: disable=too-many-lines
"""
This module is responsible for the filling disparity algorithms:
thus it fills the disparity map with values estimated according to
their neighbourhood.
"""

# Standard imports


import numpy as np

# Third party imports
import xarray as xr

from cars.applications.dense_match_filling import (
    fill_disp_wrappers as fill_wrap,
)
from cars.conf import mask_cst

# Cars import
from cars.core import constants as cst


def classif_to_stacked_array(disp_map, class_index):
    """
    Convert disparity dataset to mask correspoding to all classes

    :param disp_map: disparity dataset
    :type disp_map: xarray Dataset
    :param class_index: classification tags
    :type class_index: list of str

    """

    index_class = np.where(
        np.isin(
            np.array(disp_map.coords[cst.BAND_CLASSIF].values),
            np.array(class_index),
        )
    )[0].tolist()
    # get index for each band classification of the non zero values
    stack_index = np.any(
        disp_map[cst.EPI_CLASSIFICATION].values[index_class, :, :] > 0,
        axis=0,
    )

    return stack_index


def fill_disp_using_zero_padding(
    disp_map: xr.Dataset,
    class_index,
    fill_valid_pixels,
) -> xr.Dataset:
    """
    Fill disparity map holes

    :param disp_map: disparity map
    :type disp_map: xr.Dataset
    :param class_index: class index according to the classification tag
    :type class_index: int
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool
    """
    # get index of the application class config
    # according the coords classif band
    if cst.BAND_CLASSIF in disp_map.coords:
        # get index for each band classification
        stack_index = classif_to_stacked_array(disp_map, class_index)
        # Exclude pixels outside of epipolar footprint
        mask = (
            disp_map[cst.EPI_MSK].values
            != mask_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
        )
        if not fill_valid_pixels:
            # Exclude valid pixels
            mask = np.logical_and(mask, disp_map["disp_msk"].values == 0)
        stack_index = np.logical_and(stack_index, mask)
        # set disparity value to zero where the class is
        # non zero value and masked region
        disp_map["disp"].values[stack_index] = 0
        disp_map["disp_msk"].values[stack_index] = 255
        disp_map[cst.EPI_MSK].values[stack_index] = 0
        # Add a band to disparity dataset to memorize which pixels are filled
        fill_wrap.update_filling(disp_map, stack_index, "zeros_padding")
