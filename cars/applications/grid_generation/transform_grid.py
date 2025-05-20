#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
Grids module:
contains functions used for grid transformation
"""

import numpy as np

from cars.applications.grid_generation import grids


def transform_grid(grid_left, grid_right, resolution):
    grid_left.attributes["epipolar_size_x"] = np.floor(
        grid_left.attributes["epipolar_size_x"] / resolution
    )
    grid_left.attributes["epipolar_size_y"] = np.floor(
        grid_left.attributes["epipolar_size_y"] / resolution
    )
    grid_left.attributes["grid_spacing"][0] = np.floor(
        grid_left.attributes["grid_spacing"][0] / resolution
    )
    grid_left.attributes["grid_spacing"][1] = np.floor(
        grid_left.attributes["grid_spacing"][1] / resolution
    )
    grid_left.attributes["grid_origin"][0] = np.floor(
        grid_left.attributes["grid_origin"][0] / resolution
    )
    grid_left.attributes["grid_origin"][1] = np.floor(
        grid_left.attributes["grid_origin"][1] / resolution
    )

    grids.write_grid(
        grid_left[0, 0],
        grid_left.attributes["path"],
        grid_left.attributes["grid_origin"],
        grid_left.attributes["grid_spacing"],
    )
    grids.write_grid(
        grid_left[0, 0],
        grid_right.attributes["path"],
        grid_left.attributes["grid_origin"],
        grid_left.attributes["grid_spacing"],
    )

    return grid_left
