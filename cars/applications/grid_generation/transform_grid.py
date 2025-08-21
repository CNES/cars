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


def transform_grid_func(grid, resolution, right=False):
    """
    Transform the grid for low res resampling

    :param grid: the grid
    :type grid: cars_dataset
    :param resolution: the resolution for the resampling
    :type resolution: int
    """
    for key, value in grid.items():
        if right:
            if key not in ("grid_origin", "grid_spacing"):
                scale(key, value, grid, resolution)
        else:
            scale(key, value, grid, resolution)

    # we need to charge the data to override it
    with rasterio.open(grid["path"]) as src:
        data_left = src.read()

    grid_generation_algo.write_grid(
        np.transpose(data_left, (1, 2, 0)),
        grid["path"],
        grid["grid_origin"],
        grid["grid_spacing"],
    )

    return grid


def scale(key, value, grid, resolution):
    """
    Scale attributes by the resolution
    """

    if key == "grid_origin":
        for i, _ in enumerate(value):
            grid[key][i] = np.floor(value[i] / resolution)
    elif key == "grid_spacing":
        for i, _ in enumerate(value):
            grid[key][i] = np.floor(value[i] / resolution)
    elif key == "disp_to_alt_ratio":
        grid[key] = value * resolution
    elif key == "epipolar_size_x":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_size_y":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_origin_x":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_origin_y":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_spacing_x":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_spacing":
        grid[key] = np.floor(value / resolution)
    elif key == "epipolar_step":
        grid[key] = np.floor(value / resolution)
