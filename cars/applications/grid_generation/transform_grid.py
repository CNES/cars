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
import rasterio

from cars.applications.grid_generation import grid_generation_algo


def transform_grid_func(grid_left, grid_right, resolution):
    """
    Transform the grid for low res resampling

    :param grid_left: the left grid
    :type grid_left: cars_dataset
    :param grid_right: the right grid
    :type grid_right: cars_dataset
    :param resolution: the resolution for the resampling
    :type resolution: int
    """
    for key, value in grid_left.items():
        if isinstance(value, (int, float, np.floating)):
            grid_left[key] = np.floor(value / resolution)
        elif isinstance(value, list):
            for i, _ in enumerate(value):
                grid_left[key][i] = np.floor(value[i] / resolution)

    # we need to charge the data to override it
    with rasterio.open(grid_left["path"]) as src:
        data_left = src.read()

    with rasterio.open(grid_right["path"]) as src:
        data_right = src.read()

    grid_generation_algo.write_grid(
        np.transpose(data_left, (1, 2, 0)),
        grid_left["path"],
        grid_left["grid_origin"],
        grid_left["grid_spacing"],
    )
    grid_generation_algo.write_grid(
        np.transpose(data_right, (1, 2, 0)),
        grid_right["path"],
        grid_left["grid_origin"],
        grid_left["grid_spacing"],
    )

    return grid_left
