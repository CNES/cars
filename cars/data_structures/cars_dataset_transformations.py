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
# pylint: disable=too-many-lines
"""
cars_dataset module:

"""

from shapely.geometry import Polygon

from cars.data_structures.cars_dataset import CarsDataset


def region_to_polygon(region):
    """
    Convert region to polygon

    :param region: region [row_min, row_max, col_min, col_max]
    :type region: list

    :return: polygon of the epipolar region
    :rtype: shapely.geometry.Polygon

    """

    row_min, row_max, col_min, col_max = region
    polygon_coords = [
        (col_min, row_min),
        (col_max, row_min),
        (col_max, row_max),
        (col_min, row_max),
    ]

    # convert to shapely polygon
    polygon = Polygon(polygon_coords)
    return polygon


def extract_cars_dataset(in_cars_ds, region):
    """
    Extract CarsDataset from disparity map and epipolar region

    :param in_cars_ds:  cars datasert to crop
    :type in_cars_ds: xr.Dataset
    :param region: epipolar region
    :type region: xr.Dataset

    :return: CarsDataset
    :rtype: CarsDataset

    """

    # compute tiles to use
    row_min = in_cars_ds.shape[0] - 1
    row_max = 0
    col_min = in_cars_ds.shape[1] - 1
    col_max = 0

    for tile_row in range(in_cars_ds.shape[0]):
        for tile_col in range(in_cars_ds.shape[1]):
            tile_region = in_cars_ds.tiling_grid[tile_row, tile_col]
            if region_to_polygon(region).intersects(
                region_to_polygon(tile_region)
            ):
                row_min = min(row_min, tile_row)
                row_max = max(row_max, tile_row)
                col_min = min(col_min, tile_col)
                col_max = max(col_max, tile_col)

    # Generate new CarsDataset
    new_cars_dataset = CarsDataset(
        in_cars_ds.dataset_type, name=in_cars_ds.name
    )
    new_cars_dataset.attributes = in_cars_ds.attributes

    new_cars_dataset.tiling_grid = in_cars_ds.tiling_grid[
        row_min : row_max + 1, col_min : col_max + 1
    ]

    # fill with content
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            new_cars_dataset[row - row_min, col - col_min] = in_cars_ds[
                row, col
            ]

    return new_cars_dataset
