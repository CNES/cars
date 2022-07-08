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
CARS mask classes module
"""

# Standard imports
import logging
import warnings
from typing import List

# Third party imports
import numpy as np

# CARS imports

# Specific values
# 0 = valid pixels
# 255 = value used as no data during the rectification in the epipolar geometry
VALID_VALUE = 0
NO_DATA_IN_EPIPOLAR_RECTIFICATION = 255
PROTECTED_VALUES = [NO_DATA_IN_EPIPOLAR_RECTIFICATION]

# tags for mask classes json parameters
ignored_by_corr_tag = "ignored_by_correlation"  # pylint: disable=invalid-name
set_to_ref_alt_tag = "set_to_ref_alt"  # pylint: disable=invalid-name
ignored_by_sift_matching_tag = (  # pylint: disable=invalid-name
    "ignored_by_sift_matching"
)


def check_mask_classes(mask_classes_dict: dict):
    """
    Check if mask classes use protected key,
    and logs a warning

    :param mask_classes_dict: dictionary of the mask classes to use in CARS
    """

    # check that required values are not protected for CARS internal usage
    used_values = []
    for key in mask_classes_dict.keys():
        if mask_classes_dict[key] is not None:
            used_values.extend(mask_classes_dict[key])

    for i in PROTECTED_VALUES:
        if i in used_values:
            logging.warning(
                "{} value cannot be used as a mask class, "
                "it is reserved for CARS internal use".format(i)
            )


def is_multiclasses_mask(msk: np.ndarray) -> bool:
    """
    Check if the mask has several classes.
    The VALID_VALUE and all protected values defined
    in the PROTECTED_VALUES mask module global variable
    are not taken into account.

    :param msk: mask to test
    :return: True if the mask has several classes, False otherwise
    """
    # search the locations of valid values in the mask
    msk_classes = np.where(msk == VALID_VALUE, True, False)

    # update with the locations of the protected values in the mask
    for i in PROTECTED_VALUES:
        msk_classes = np.logical_or(
            msk_classes, np.where(msk == i, True, False)
        )

    # set these location to nan in order to discard them
    msk_only_classes = msk.astype(np.float64)
    msk_only_classes[msk_classes] = np.nan

    # check if mask has several classes
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message="All-NaN slice encountered"
        )
        res = bool(np.nanmin(msk_only_classes) != np.nanmax(msk_only_classes))

    return res


def create_msk_from_classes(
    mc_msk: np.ndarray,
    classes_to_use: List[int],
    out_msk_pix_value: int = 255,
    out_msk_dtype: np.dtype = np.uint16,
    mask_intern_no_data_val: bool = False,
) -> np.ndarray:
    """
    Create a mask of type out_msk_dtype set to the out_msk_pix_value for pixels
    belonging to the required classes (defined by the classes_to_use parameter)
    in the multi-classes mask in input.

    The NO_DATA_IN_EPIPOLAR_RECTIFICATION value is added to the classes to mask
    if the add_intern_no_data_val parameter is set to True.

    :param mc_msk: multi-classes mask
    :param classes_to_use: List of values to use to create the output mask
    :param out_msk_pix_value: pixel value to assign to the output mask
           in the locations of the required classes'
           pixels. If the out_msk_dtype parameter is set to bool,
           this parameter will be automatically set to True.
    :param out_msk_dtype: numpy dtype of the output mask
    :return: the output mask
    """
    # initiate the required classes final mask
    if out_msk_dtype == bool:
        not_msk_pix_value = False
    else:
        not_msk_pix_value = 0
    out_msk = np.full(
        mc_msk.shape, fill_value=not_msk_pix_value, dtype=out_msk_dtype
    )

    # create boolean mask with the pixels of the required classes as True
    msk_with_selected_classes = np.zeros(mc_msk.shape, dtype=bool)

    for i in classes_to_use:
        msk_with_selected_classes = np.logical_or(
            msk_with_selected_classes, np.where(mc_msk == i, True, False)
        )

    # if intern nodata must be masked
    if mask_intern_no_data_val:
        msk_with_selected_classes = np.logical_or(
            msk_with_selected_classes,
            np.where(mc_msk == NO_DATA_IN_EPIPOLAR_RECTIFICATION, True, False),
        )

    out_msk[msk_with_selected_classes] = out_msk_pix_value

    return out_msk
