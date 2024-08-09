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
This module contains the output definition
"""

import os

from json_checker import Checker, Or

from cars.core.utils import safe_makedirs
from cars.pipelines.parameters import output_constants


def check_output_parameters(conf):
    """
    Check the output json configuration and fill in default values

    :param conf: configuration of output
    :type conf: dict
    """
    overloaded_conf = conf.copy()
    out_dir = conf[output_constants.OUT_DIRECTORY]
    out_dir = os.path.abspath(out_dir)
    # Ensure that output directory and its subdirectories exist
    safe_makedirs(out_dir)
    safe_makedirs(os.path.join(out_dir, output_constants.DSM_DIRECTORY))

    # Overload some parameters
    overloaded_conf[output_constants.OUT_DIRECTORY] = out_dir

    overloaded_conf[output_constants.PRODUCT_LEVEL] = overloaded_conf.get(
        output_constants.PRODUCT_LEVEL, "dsm"
    )

    overloaded_conf[output_constants.DSM_BASENAME] = overloaded_conf.get(
        output_constants.DSM_BASENAME, "dsm.tif"
    )
    overloaded_conf[output_constants.CLR_BASENAME] = overloaded_conf.get(
        output_constants.CLR_BASENAME, "color.tif"
    )
    overloaded_conf[output_constants.INFO_BASENAME] = overloaded_conf.get(
        output_constants.INFO_BASENAME,
        "metadata.json",
    )
    overloaded_conf[output_constants.OUT_GEOID] = overloaded_conf.get(
        output_constants.OUT_GEOID, False
    )
    overloaded_conf[output_constants.EPSG] = overloaded_conf.get(
        output_constants.EPSG, None
    )

    overloaded_conf[output_constants.RESOLUTION] = overloaded_conf.get(
        output_constants.RESOLUTION, "auto"
    )

    overloaded_conf[output_constants.SAVE_BY_PAIR] = overloaded_conf.get(
        output_constants.SAVE_BY_PAIR, False
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY] = overloaded_conf.get(
        output_constants.AUXILIARY, {}
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY][output_constants.AUX_COLOR] = (
        overloaded_conf[output_constants.AUXILIARY].get(
            output_constants.AUX_COLOR, True
        )
    )
    overloaded_conf[output_constants.AUXILIARY][output_constants.AUX_MASK] = (
        overloaded_conf[output_constants.AUXILIARY].get(
            output_constants.AUX_MASK, False
        )
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CLASSIFICATION
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CLASSIFICATION, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CONFIDENCE
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CONFIDENCE, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CONTRIBUTING_PAIR
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CONTRIBUTING_PAIR, False
    )

    # Check schema
    output_schema = {
        output_constants.OUT_DIRECTORY: str,
        output_constants.PRODUCT_LEVEL: str,
        output_constants.DSM_BASENAME: str,
        output_constants.CLR_BASENAME: str,
        output_constants.INFO_BASENAME: str,
        output_constants.OUT_GEOID: Or(bool, str),
        output_constants.EPSG: Or(int, None),
        output_constants.RESOLUTION: str,
        output_constants.SAVE_BY_PAIR: bool,
        output_constants.AUXILIARY: dict,
    }
    checker_output = Checker(output_schema)
    checker_output.validate(overloaded_conf)

    # check auxiliary keys
    auxiliary_schema = {
        output_constants.AUX_COLOR: bool,
        output_constants.AUX_MASK: bool,
        output_constants.AUX_CLASSIFICATION: bool,
        output_constants.AUX_CONFIDENCE: bool,
        output_constants.AUX_CONTRIBUTING_PAIR: bool,
    }

    checker_auxiliary = Checker(auxiliary_schema)
    checker_auxiliary.validate(overloaded_conf[output_constants.AUXILIARY])

    return overloaded_conf
