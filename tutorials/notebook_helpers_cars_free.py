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
this module contains functions helpers used in notebooks without importing cars.
"""

# Standard imports
import os


def set_dask_config():
    """
    Set dask config path
    """

    # Force the use of CARS dask configuration
    dask_config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cars",
        "orchestrator",
        "cluster",
        "dask_config",
    )
    if not os.path.isdir(dask_config_path):
        raise NotADirectoryError("Wrong dask config path")
    os.environ["DASK_CONFIG"] = str(dask_config_path)
