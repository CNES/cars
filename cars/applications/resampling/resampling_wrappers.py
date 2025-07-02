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
Resampling module:
contains functions used for epipolar resampling
"""

import copy

# Standard imports
import logging

# Third party imports
import numpy as np

# CARS imports
from cars.core import constants as cst
from cars.core import inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def get_paths_and_bands(sensor_image, required_bands=None):
    """
    Reformat file paths and bands required from each file to ease reading

    :param sensor_image: input configuration of an image
    :type sensor_image: dict
    :param required_bands: required bands for resampling
    :type required_bands: list
    """
    paths = {}
    if required_bands is None:
        # All bands are required bands
        required_bands = list(sensor_image["bands"].keys())
    for band in required_bands:
        file_path = sensor_image["bands"][band]["path"]
        band_id = sensor_image["bands"][band]["band"] + 1
        if file_path in paths:
            paths[file_path]["band_id"].append(band_id)
            paths[file_path]["band_name"].append(band)
        else:
            paths[file_path] = {"band_id": [band_id], "band_name": [band]}
    return paths


def get_sensors_bounds(sensor_image_left, sensor_image_right):
    """
    Get bounds of sensor images
    Bounds: BoundingBox(left, bottom, right, top)

    :param sensor_image_left: left sensor
    :type sensor_image_left: dict
    :param sensor_image_right: right sensor
    :type sensor_image_right: dict

    :return: left image bounds, right image bounds
    :rtype: tuple(list, list)
    """

    left_sensor_bounds = list(
        inputs.rasterio_get_bounds(
            sensor_image_left[sens_cst.INPUT_IMG][sens_cst.MAIN_FILE],
            apply_resolution_sign=True,
        )
    )

    right_sensor_bounds = list(
        inputs.rasterio_get_bounds(
            sensor_image_right[sens_cst.INPUT_IMG][sens_cst.MAIN_FILE],
            apply_resolution_sign=True,
        )
    )

    return left_sensor_bounds, right_sensor_bounds


def check_tiles_in_sensor(
    sensor_image_left,
    sensor_image_right,
    image_tiling,
    grid_left,
    grid_right,
    geom_plugin,
):
    """
    Check if epipolar tiles will be used.
    A tile is not used if is outside sensor bounds

    :param sensor_image_left: left sensor
    :type sensor_image_left: dict
    :param sensor_image_right: right sensor
    :type sensor_image_right: dict
    :param image_tiling: epipolar tiling grid
    :type image_tiling: np.array
    :param grid_left: left epipolar grid
    :type grid_left: CarsDataset
    :param grid_right: right epipolar grid
    :type grid_right: CarsDataset

    :return: left in sensor, right in sensor
    :rtype: np.array(bool), np.array(bool)

    """

    # Get sensor image bounds
    # BoundingBox: left, bottom, right, top:
    left_sensor_bounds, right_sensor_bounds = get_sensors_bounds(
        sensor_image_left, sensor_image_right
    )

    # Get tile epipolar corners
    interpolation_margin = 20  # arbitrary

    # add margin
    tiling_grid = copy.copy(image_tiling)
    tiling_grid[:, :, 0] -= interpolation_margin
    tiling_grid[:, :, 1] += interpolation_margin
    tiling_grid[:, :, 2] -= interpolation_margin
    tiling_grid[:, :, 3] += interpolation_margin

    # Generate matches
    matches = np.empty((4 * tiling_grid.shape[0] * tiling_grid.shape[1], 2))
    nb_row = tiling_grid.shape[0]
    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            y_min, y_max, x_min, x_max = tiling_grid[row, col]
            matches[
                4 * nb_row * col + 4 * row : 4 * nb_row * col + 4 * row + 4, :
            ] = np.array(
                [
                    [x_min, y_min],
                    [x_min, y_max],
                    [x_max, y_max],
                    [x_max, y_min],
                ]
            )

    # create artificial matches
    tiles_coords_as_matches = np.concatenate([matches, matches], axis=1)

    # Compute sensors positions
    # Transform to sensor coordinates
    (
        sensor_pos_left,
        sensor_pos_right,
    ) = geom_plugin.matches_to_sensor_coords(
        grid_left,
        grid_right,
        tiles_coords_as_matches,
        cst.MATCHES_MODE,
    )

    in_sensor_left_array = np.ones(
        (image_tiling.shape[0], image_tiling.shape[1]), dtype=bool
    )
    in_sensor_right_array = np.ones(
        (image_tiling.shape[0], image_tiling.shape[1]), dtype=bool
    )

    for row in range(tiling_grid.shape[0]):
        for col in range(tiling_grid.shape[1]):
            # Get sensors position for tile
            left_sensor_tile = sensor_pos_left[
                4 * nb_row * col + 4 * row : 4 * nb_row * col + 4 * row + 4, :
            ]
            right_sensor_tile = sensor_pos_right[
                4 * nb_row * col + 4 * row : 4 * nb_row * col + 4 * row + 4, :
            ]

            in_sensor_left, in_sensor_right = check_tile_inclusion(
                left_sensor_bounds,
                right_sensor_bounds,
                left_sensor_tile,
                right_sensor_tile,
            )

            in_sensor_left_array[row, col] = in_sensor_left
            in_sensor_right_array[row, col] = in_sensor_right

    nb_tiles = tiling_grid.shape[0] * tiling_grid.shape[1]
    tiles_dumped_left = nb_tiles - np.sum(in_sensor_left_array)
    tiles_dumped_right = nb_tiles - np.sum(in_sensor_right_array)

    logging.info(
        "Number of left epipolar image tiles outside left sensor "
        "image and removed: {}".format(tiles_dumped_left)
    )
    logging.info(
        "Number of right epipolar image tiles outside right sensor "
        "image and removed: {}".format(tiles_dumped_right)
    )

    return in_sensor_left_array, in_sensor_right_array


def check_tile_inclusion(
    left_sensor_bounds,
    right_sensor_bounds,
    sensor_pos_left,
    sensor_pos_right,
):
    """
    Check if tile is in sensor image

    :param left_sensor_bounds: bounds of left sensor
    :type left_sensor_bounds: list
    :param right_sensor_bounds: bounds of right sensor
    :type right_sensor_bounds: list
    :param sensor_pos_left: left sensor position
    :type sensor_pos_left: np.array
    :param sensor_pos_right: right sensor position
    :type sensor_pos_right: np.array

    :return: left tile in sensor image left, right tile in sensor image right
    :rtype: tuple(bool, bool)
    """

    # check if outside of image
    # Do not use tile if the whole tile is outside sensor
    in_sensor_left = True
    if (
        (
            np.all(
                sensor_pos_left[:, 0]
                < min(left_sensor_bounds[0], left_sensor_bounds[2])
            )
        )
        or (
            np.all(
                sensor_pos_left[:, 0]
                > max(left_sensor_bounds[0], left_sensor_bounds[2])
            )
        )
        or (
            np.all(
                sensor_pos_left[:, 1]
                > max(left_sensor_bounds[1], left_sensor_bounds[3])
            )
        )
        or (
            np.all(
                sensor_pos_left[:, 1]
                < min(left_sensor_bounds[1], left_sensor_bounds[3])
            )
        )
    ):
        in_sensor_left = False

    in_sensor_right = True
    if (
        (
            np.all(
                sensor_pos_right[:, 0]
                < min(right_sensor_bounds[0], right_sensor_bounds[2])
            )
        )
        or (
            np.all(
                sensor_pos_right[:, 0]
                > max(right_sensor_bounds[0], right_sensor_bounds[2])
            )
        )
        or (
            np.all(
                sensor_pos_right[:, 1]
                > max(right_sensor_bounds[1], right_sensor_bounds[3])
            )
        )
        or (
            np.all(
                sensor_pos_right[:, 1]
                < min(right_sensor_bounds[1], right_sensor_bounds[3])
            )
        )
    ):
        in_sensor_right = False

    return in_sensor_left, in_sensor_right
