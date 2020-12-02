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

from typing import Dict, List
import logging
import json
from json_checker import OptionalKey

import numpy as np

from cars import utils

# Specific values
# 0 = valid pixels
# 255 = value used as no data during the resampling in the epipolar geometry
VALID_VALUE = 0
NO_DATA_IN_EPIPOLAR_RECTIFICATION = 255
PROTECTED_VALUES = [NO_DATA_IN_EPIPOLAR_RECTIFICATION]

# tags for mask classes json parameters
ignored_by_corr_tag = "ignored_by_correlation"
set_to_ref_alt_tag = "set_to_ref_alt"
ignored_by_sift_matching_tag = "ignored_by_sift_matching"

# Schema for mask json
msk_classes_json_schema = {
    OptionalKey(ignored_by_corr_tag): [int],
    OptionalKey(set_to_ref_alt_tag): [int],
    OptionalKey(ignored_by_sift_matching_tag): [int]
}


def mask_classes_can_open(mask_classes_path: str) -> bool:
    """
    Check if the json file describing the mask classes usage in the CARS API has the right format.

    :param mask_classes_path: path to the json file
    :return: True if the json file validates the msk_classes_json_schema, False otherwise
    """
    with open(mask_classes_path, 'r') as f:
        classes_usage_dict = json.load(f)
        try:
            utils.check_json(classes_usage_dict, msk_classes_json_schema)
            return True
        except Exception as e:
            logging.error("Exception caught while trying to read file %s: %s"
                           % (mask_classes_path, e))
            return False


def read_mask_classes(mask_classes_path: str) -> Dict[str, List[int]]:
    """
    Read the json file describing the mask classes usage in the CARS API and return it as a dictionary.

    :param mask_classes_path: path to the json file
    :return: dictionary of the mask classes to use in CARS
    """
    classes_usage_dict = dict()

    with open(mask_classes_path, 'r') as f:
        classes_usage_dict = json.load(f)

    # check that required values are not protected for CARS internal usage
    used_values = []
    for key in classes_usage_dict.keys():
        used_values.extend(classes_usage_dict[key])

    for i in PROTECTED_VALUES:
        if i in used_values:
            logging.warning('%s value cannot be used as a mask class, '
                            'it is reserved for CARS internal use' % i)

    return classes_usage_dict


def is_multiclasses_mask(msk: np.ndarray) -> bool:
    """
    Check if the mask has several classes.
    The VALID_VALUE and all protected values defined in the PROTECTED_VALUES mask module global variable
    are not taken into account.

    :param msk: mask to test
    :return: True if the mask has several classes, False otherwise
    """
    # search the locations of valid values in the mask
    msk_classes = np.where(msk == VALID_VALUE, True, False)

    # update with the locations of the protected values in the mask
    for i in PROTECTED_VALUES:
        msk_classes = np.logical_or(msk_classes, np.where(msk == i, True, False))

    # set these location to nan in order to discard them
    msk_only_classes = msk.astype(np.float)
    msk_only_classes[msk_classes] = np.nan

    # check if mask has several classes
    if np.nanmin(msk_only_classes) != np.nanmax(msk_only_classes):
        return True
    else:
        return False


def create_msk_from_classes(mc_msk: np.ndarray, classes_to_use: List[int], out_msk_pix_value: int=255,
                            out_msk_dtype: np.dtype=np.uint16) -> np.ndarray:
    """
    Create a mask of type out_msk_dtype set to the out_msk_pix_value for pixels belonging to the required classes
    (defined by the classes_to_use parameter) in the multi-classes mask in input.

    :param mc_msk: multi-classes mask
    :param classes_to_use: List of values to use to create the output mask
    :param out_msk_pix_value: pixel value to assign to the output mask in the locations of the required classes'
    pixels. If the out_msk_dtype parameter is set to np.bool, this parameter will be automatically set to True.
    :param out_msk_dtype: numpy dtype of the output mask
    :return: the output mask
    """
    # initiate the required classes final mask
    if out_msk_dtype == np.bool:
        not_msk_pix_value = False
    else:
        not_msk_pix_value = 0
    out_msk = np.full(mc_msk.shape, fill_value=not_msk_pix_value, dtype=out_msk_dtype)

    # create boolean mask with the pixels of the required classes as True
    msk_with_selected_classes = np.zeros(mc_msk.shape, dtype=np.bool)

    for i in classes_to_use:
        msk_with_selected_classes = np.logical_or(msk_with_selected_classes,
                                                  np.where(mc_msk == i, True, False))
    out_msk[msk_with_selected_classes] = out_msk_pix_value

    return out_msk


def create_msk_from_tag(mc_msk: np.ndarray, msk_classes_path: str, classes_to_use_tag: str, out_msk_pix_value: int=255,
                        out_msk_dtype: np.dtype=np.uint16, mask_intern_no_data_val: bool=False) -> np.ndarray:
    """
    Create a mask of type out_msk_dtype set to the out_msk_pix_value for pixels belonging to the required classes  in
    the multi-classes mask in input. The classes are defined by the classes_to_use_tag field in the msk_classes_path
    json.
    The NO_DATA_IN_EPIPOLAR_RECTIFICATION value is added to the classes to mask if the add_intern_no_data_val parameter
    is set to True.
    If the classes_to_use_tag is not set in the input json, no class will be mask except if the add_intern_no_data_val
    is set to True.

    :param mc_msk: multi-classes mask
    :param msk_classes_path: mask classes json path
    :param classes_to_use_tag: tag of the field to use in the msk_classes_path json
    :param out_msk_pix_value: pixel value to assign to the output mask in the locations of the required classes'
    pixels. If the out_msk_dtype parameter is set to np.bool, this parameter will be automatically set to True.
    :param out_msk_dtype: numpy dtype of the output mask
    :param mask_intern_no_data_val: boolean activating the masking of all values equal to the
    NO_DATA_IN_EPIPOLAR_RECTIFICATION variable in the output mask
    :return: the output mask
    """
    classes_dict = read_mask_classes(msk_classes_path)
    classes_to_mask = []

    # retrieve the required classes to use
    if classes_to_use_tag in classes_dict.keys():
        classes_to_mask.extend(classes_dict[classes_to_use_tag])
    else:
        logging.warning('No class specified by the %s tag in %s. No class will be used to mask the image data in '
                        'the corresponding step in cars' % (classes_to_use_tag, msk_classes_path))

    # add specific values
    if mask_intern_no_data_val:
        classes_to_mask.append(NO_DATA_IN_EPIPOLAR_RECTIFICATION)

    # create final mask
    out_msk = create_msk_from_classes(mc_msk, classes_to_mask, out_msk_pix_value, out_msk_dtype)

    return out_msk
