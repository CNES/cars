#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains the abstract direct_localization application class.
"""

import numpy as np
from shareloc.proj_utils import transform_physical_point_to_index

from cars.core import inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def get_ground_truth(
    geom_plugin,
    grid,
    sensor,
    disp_to_alt_ratio,
    target,
    window,
    geom_plugin_dem_median=None,
    reverse=False,
):
    """
    Computes ground truth in epipolar and sensor geometry.

    :param geom_plugin_dem_median: path to initial dem
    :type geom_plugin_dem_median: str
    :param geom_plugin: Geometry plugin with user's DSM used to
        generate epipolar grids.
    :type geom_plugin: GeometryPlugin
    :param grid: Grid left.
    :type grid: CarsDataset
    :param sensor: sensor data
        Dict must contain keys: "image", "color", "geomodel",
        "no_data", "mask". Paths must be absolute.
    :type sensor: dict
    :param disp_to_alt_ratio: Disp to altitude ratio used for performance map.
    :type disp_to_alt_ratio: float
    :param target: sensor, epipolar or both outputs geometry
    :type target: str
    :param window: size of tile
    :type window: np.ndarray
    :param geom_plugin_dem_median: Geometry plugin with dem median
    :type geom_plugin_dem_median: geometry_plugin
    :param reverse: true if right-> left
    :type reverse: bool
    """

    sensor_data_im = sensor[sens_cst.INPUT_IMG]
    geomodel = sensor[sens_cst.INPUT_GEO_MODEL]

    rows = np.arange(window[0], window[1])
    cols = np.arange(window[2], window[3])

    (positions_col, positions_row) = np.meshgrid(cols, rows)

    if target == "epipolar":

        positions = np.stack([positions_col, positions_row], axis=2)
        sensor_positions = geom_plugin.sensor_position_from_grid(
            grid,
            np.reshape(
                positions,
                (
                    positions.shape[0] * positions.shape[1],
                    2,
                ),
            ),
        )

        transform = inputs.rasterio_get_transform(sensor_data_im)
        row, col = transform_physical_point_to_index(
            ~transform, sensor_positions[:, 1], sensor_positions[:, 0]
        )

        _, _, alt = geom_plugin.direct_loc(
            sensor_data_im,
            geomodel,
            col,
            row,
        )

        alt = np.reshape(alt, (rows.shape[0], cols.shape[0]))

        _, _, alt_ref = geom_plugin_dem_median.direct_loc(
            sensor_data_im,
            geomodel,
            col,
            row,
        )

        alt_ref = np.reshape(alt_ref, (rows.shape[0], cols.shape[0]))

        ground_truth = -(alt - alt_ref) / disp_to_alt_ratio
        if reverse:
            ground_truth *= -1

    if target == "sensor":

        _, _, alt = geom_plugin.direct_loc(
            sensor_data_im,
            geomodel,
            positions_col.ravel(),
            positions_row.ravel(),
        )

        ground_truth = np.reshape(alt, (rows.shape[0], cols.shape[0]))

    return ground_truth
