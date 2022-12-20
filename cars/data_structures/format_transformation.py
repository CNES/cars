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

    cars_ds_overlaps = np.ndarray(shape=(nb_rows, nb_cols, 4), dtype=float)

    # margins : pandora convention : ['left','up', 'right', 'down']
    overlap_row_up = abs(math.floor(margins[1]))
    overlap_row_down = abs(math.ceil(margins[3]))
    overlap_col_left = abs(math.floor(margins[0]))
    overlap_col_right = abs(math.ceil(margins[2]))

    row_max = np.max(grid[:, :, 1])
    col_max = np.max(grid[:, :, 3])

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


def get_corresponding_indexes(row, col):
    """
    Get point cloud tiling grid indexes, corresponding to
    given raster indexes.
    In rasterio convention.

    :param row: row
    :type row: int
    :param col: col
    :type col: int

    :return: corresponding indexes (row, col)
    :rtype: tuple(int, int)

    """

    pc_row = col
    pc_col = row

    return pc_row, pc_col


def terrain_coords_to_pix(point_cloud_cars_ds, resolution):
    """
    Compute the tiling grid in pixels, from a tiling grid in
    geocoordinates

    :param point_cloud_cars_ds: point clouds
    :type point_cloud_cars_ds: CarsDataset
    :param resolution: resolution
    :type resolution: float

    :return: new tiling grid
    :rtype: np.ndarray

    """

    raster_tiling_grid = np.empty(
        point_cloud_cars_ds.tiling_grid.shape
    ).transpose(1, 0, 2)

    for row in range(raster_tiling_grid.shape[0]):
        for col in range(raster_tiling_grid.shape[1]):
            # get corresponding tile in point cloud
            pc_row, pc_col = get_corresponding_indexes(row, col)

            # Get window
            window_dict = point_cloud_cars_ds.get_window_as_dict(
                pc_row,
                pc_col,
                from_terrain=True,
                resolution=resolution,
            )

            # apply window
            raster_tiling_grid[row, col, :] = [
                window_dict["row_min"],
                window_dict["row_max"],
                window_dict["col_min"],
                window_dict["col_max"],
            ]

    return raster_tiling_grid


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
