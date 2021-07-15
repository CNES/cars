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
TBB module:
contains some tbb global shared general purpose functions
"""
# Standard imports
import subprocess


def check_tbb_installed() -> bool:
    """
    Check if numba finds tbb correctly installed
    :return: tbb found
    """

    tbb_installed = False
    # Get output of 'numba -s'
    numba_output = (
        subprocess.check_output(["numba", "-s"]).decode("utf8").strip()
    )
    for item in numba_output.split("\n"):
        if "TBB Threading Layer Available" in item and "True" in item:
            tbb_installed = True
            return tbb_installed
    return tbb_installed
