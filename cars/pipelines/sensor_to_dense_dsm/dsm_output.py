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
CARS containing inputs checking for raster full res dsm
Used for full_res and low_res pipelines
"""

import errno
import os

from json_checker import Checker

# CARS imports
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sensors_cst,
)


def dense_dsm_check_output(conf):
    """
    Check the output given

    :param conf: configuration of output
    :type conf: dict
    """
    overloaded_conf = conf.copy()
    out_dir = conf[sensors_cst.OUT_DIR]
    out_dir = os.path.abspath(out_dir)
    # Ensure that outdir exists
    try:
        os.makedirs(out_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise

    # Overload some parameters
    overloaded_conf[sensors_cst.OUT_DIR] = out_dir
    overloaded_conf[sensors_cst.DSM_BASENAME] = overloaded_conf.get(
        sensors_cst.DSM_BASENAME, "dsm.tif"
    )
    overloaded_conf[sensors_cst.CLR_BASENAME] = overloaded_conf.get(
        sensors_cst.CLR_BASENAME, "clr.tif"
    )
    overloaded_conf[sensors_cst.INFO_BASENAME] = overloaded_conf.get(
        sensors_cst.INFO_BASENAME, "content.json"
    )

    # Check schema
    output_schema = {
        sensors_cst.OUT_DIR: str,
        sensors_cst.DSM_BASENAME: str,
        sensors_cst.CLR_BASENAME: str,
        sensors_cst.INFO_BASENAME: str,
    }
    checker_output = Checker(output_schema)
    checker_output.validate(overloaded_conf)

    return overloaded_conf
