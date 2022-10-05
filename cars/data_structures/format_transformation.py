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
Contains function for format transformation between region and windows:
"""

# Third party imports
import copy

# Standard imports
import math

import numpy as np

# CARS imports


def tiling_grid_2_cars_dataset_grid(
    regions_grid, resolution=1, from_terrain=False
):
    """
    Convert Region grid to Grid format used in CarsDatasets, in pixels

    :param regions_grid: region grid
    :type regions_grid: np.ndarray
    :param resolution: resolution of image
    :type resolution: float
    :param from_terrain: is a terrain raster
    :type from_terrain: bool

    :return: Grid with CarsDataset format
    :rtype: np.ndarray

    """

    nb_rows, nb_cols = regions_grid.shape[0] - 1, regions_grid.shape[1] - 1

    cars_ds_grid = np.zeros((nb_rows, nb_cols, 4))
    # rows min : ymin
    cars_ds_grid[:, :, 0] = np.squeeze(regions_grid[:-1, :-1, 1])

    # rows max : ymax
    cars_ds_grid[:, :, 1] = np.squeeze(regions_grid[1:, :-1, 1])

    # cols min : xmin
    cars_ds_grid[:, :, 2] = np.squeeze(regions_grid[:-1, :-1, 0])

    # cols max : xmax
    cars_ds_grid[:, :, 3] = np.squeeze(regions_grid[:-1, 1:, 0])

    # convert position to pixels

    x_min = np.min(regions_grid[:, :, 0])
    y_min = np.min(regions_grid[:, :, 1])
    y_max = np.max(regions_grid[:, :, 1])

    if from_terrain:
        arr0 = np.copy(cars_ds_grid[:, :, 0])
        arr1 = np.copy(cars_ds_grid[:, :, 1])

        cars_ds_grid[:, :, 0] = y_max - arr1
        cars_ds_grid[:, :, 1] = y_max - arr0
    else:
        cars_ds_grid[:, :, 0] -= y_min
        cars_ds_grid[:, :, 1] -= y_min

    cars_ds_grid[:, :, 2] -= x_min
    cars_ds_grid[:, :, 3] -= x_min

    cars_ds_grid = np.round(cars_ds_grid / resolution)

    return cars_ds_grid


def grid_margins_2_overlaps(grid, margins):
    """
    Convert margins to overlap grid format used in CarsDatasets

    :param grid: region grid
    :type grid: np.ndarray
    :param margins: margin
    :type margins: List


    :return: overlap grid
    :rtype: np.ndarray

    """

    nb_rows, nb_cols = grid.shape[0], grid.shape[1]

    cars_ds_overlaps = np.zeros((nb_rows, nb_cols, 4))

    # margins : pandora convention : ['left','up', 'right', 'down']
    overlap_row_up = abs(math.floor(margins[1]))
    overlap_row_down = abs(math.ceil(margins[3]))
    overlap_col_left = abs(math.floor(margins[0]))
    overlap_col_right = abs(math.ceil(margins[2]))

    row_max = grid[-1, 0, 1]
    col_max = grid[0, -1, 3]

    for j in range(0, nb_cols):
        for i in range(0, nb_rows):

            row_up = grid[i, j, 0]
            row_down = grid[i, j, 1]
            col_left = grid[i, j, 2]
            col_right = grid[i, j, 3]

            # fill overlap [OL_row_up, OL_row_down, OL_col_left,
            #  OL_col_right]
            cars_ds_overlaps[i, j, 0] = row_up - max(0, row_up - overlap_row_up)
            cars_ds_overlaps[i, j, 1] = (
                min(row_max, row_down + overlap_row_down) - row_down
            )

            cars_ds_overlaps[i, j, 2] = col_left - max(
                0, col_left - overlap_col_left
            )
            cars_ds_overlaps[i, j, 3] = (
                min(col_max, col_right + overlap_col_right) - col_right
            )

    return cars_ds_overlaps


def region_margins_from_window(
    initial_margin, window, left_overlaps, right_overlaps
):
    """
    Convert window to margins.

    :param initial_margin: Initial Margin computed
    :type initial_margin: List
    :param window: window to use
    :type window: Dict
    :param left_overlaps: left overlap to use
    :type left_overlaps: Dict
    :param right_overlaps: right overlap to use
    :type right_overlaps: Dict


    :return: Region, Margin
    :rtype: Tuple(List, List)

    """

    # margin : 'left', 'up', 'right', 'down'

    left_margin = [
        abs(left_overlaps["left"]),
        abs(left_overlaps["up"]),
        abs(left_overlaps["right"]),
        abs(left_overlaps["down"]),
    ]

    right_margin = [
        abs(right_overlaps["left"]),
        abs(right_overlaps["up"]),
        abs(right_overlaps["right"]),
        abs(right_overlaps["down"]),
    ]

    margin = copy.copy(initial_margin)
    margin["left_margin"].data = np.array(left_margin)
    margin["right_margin"].data = np.array(right_margin)

    # region :  xmin, ymin, xmax, ymax
    # window : row_min, row_max, col_min, col_max

    region = [
        window["col_min"],
        window["row_min"],
        window["col_max"],
        window["row_max"],
    ]

    return region, margin
