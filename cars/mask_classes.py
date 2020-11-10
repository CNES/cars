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
set_to_input_dem_tag = "set_to_input_dem"
ignored_by_sift_matching_tag = "ignored_by_sift_matching"

# Schema for mask json
msk_classes_json_schema = {
    OptionalKey(ignored_by_corr_tag): [int],
    OptionalKey(set_to_input_dem_tag): [int],
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
            logging.error("Exception caught while trying to read file {}: {}"
                          .format(mask_classes_path, e))
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
            logging.warning('{} value cannot be used as a mask class, '
                            'it is reserved for CARS internal use'.format(i))

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
