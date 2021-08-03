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
This input module refers to parameters and configuration inputs
"""
# Standard imports
import json
import os
from typing import Dict, Union

# Third party imports
from json_checker import And, OptionalKey

# CARS imports
from cars.conf import mask_classes
from cars.core.inputs import rasterio_can_open
from cars.core.utils import make_relative_path_absolute

# tag for static conf file
STATIC_PARAMS_TAG = "static_parameters"


def read_input_parameters(filename):
    """
    Read an input parameters json file.
    Relative paths will be made absolute.

    :param filename: Path to json file
    :type filename: str

    :returns: The dictionary read from file
    :rtype: dict
    """
    config = {}
    with open(filename, "r") as fstream:
        # Load json file
        config = json.load(fstream)
        json_dir = os.path.abspath(os.path.dirname(filename))
        # make potential relative paths absolute
        for tag in [
            IMG1_TAG,
            IMG2_TAG,
            GEOMETRIC_MODEL1_TAG,
            GEOMETRIC_MODEL2_TAG,
            MASK1_TAG,
            MASK2_TAG,
            MASK1_CLASSES_TAG,
            MASK2_CLASSES_TAG,
            COLOR1_TAG,
            SRTM_DIR_TAG,
        ]:
            if tag in config:
                config[tag] = make_relative_path_absolute(config[tag], json_dir)
    return config


# tags for input parameters
INPUT_SECTION_TAG = "input"
IMG1_TAG = "img1"
IMG2_TAG = "img2"
GEOMETRIC_MODEL1_TAG = "geometric_model1"
GEOMETRIC_MODEL2_TAG = "geometric_model2"
GEOMETRIC_MODEL_TYPE_TAG = "geometric_model_type"
SRTM_DIR_TAG = "srtm_dir"
COLOR1_TAG = "color1"
MASK1_TAG = "mask1"
MASK2_TAG = "mask2"
MASK1_CLASSES_TAG = "mask1_classes"
MASK2_CLASSES_TAG = "mask2_classes"
NODATA1_TAG = "nodata1"
NODATA2_TAG = "nodata2"
DEFAULT_ALT_TAG = "default_alt"

# Schema for input configuration json
INPUT_CONFIGURATION_SCHEMA = {
    IMG1_TAG: And(str, rasterio_can_open),
    IMG2_TAG: And(str, rasterio_can_open),
    OptionalKey(SRTM_DIR_TAG): And(str, os.path.isdir),
    OptionalKey(COLOR1_TAG): And(str, rasterio_can_open),
    OptionalKey(MASK1_TAG): And(str, rasterio_can_open),
    OptionalKey(MASK2_TAG): And(str, rasterio_can_open),
    OptionalKey(MASK1_CLASSES_TAG): And(
        str, mask_classes.mask_classes_can_open
    ),
    OptionalKey(MASK2_CLASSES_TAG): And(
        str, mask_classes.mask_classes_can_open
    ),
    OptionalKey(DEFAULT_ALT_TAG): float,
    NODATA1_TAG: int,
    NODATA2_TAG: int,
}

# Type for input configuration json
InputConfigurationType = Dict[str, Union[int, str]]
