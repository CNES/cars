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
from cars.core.projection import point_cloud_conversion


def get_ground_truth(
    geom_plugin,
    grid,
    sensor_data,
    geomodel,
    disp_to_alt_ratio,
    target,
    window,
    dem=None,
):
    """
    Computes ground truth in epipolar and sensor geometry.

    :param dem: path to initial dem
    :type dem: str
    :param geom_plugin: Geometry plugin with user's DSM used to
        generate epipolar grids.
    :type geom_plugin: GeometryPlugin
    :param grid: Grid left.
    :type grid: CarsDataset
    :param sensor_data: Tiled data.
        Dict must contain keys: "image", "color", "geomodel",
        "no_data", "mask". Paths must be absolute.
    :type sensor_data: CarsDataset
    :param geomodel: Path and attributes for left geomodel.
    :type geomodel: dict
    :param disp_to_alt_ratio: Disp to altitude ratio used for performance map.
    :type disp_to_alt_ratio: float
    :param target: sensor, epipolar or both outputs geometry
    :type target: str
    :param window: size of tile
    :type window: np.ndarray
    :param dem: path to initial elevation
    :type dem: str
    """

    rows = np.arange(window[0], window[1])
    cols = np.arange(window[2], window[3])

    (positions_row, positions_col) = np.meshgrid(cols, rows)

    if target == "epipolar":

        positions = np.stack([positions_row, positions_col], axis=2)
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

        transform = inputs.rasterio_get_transform(sensor_data)
        row, col = transform_physical_point_to_index(
            ~transform, sensor_positions[:, 1], sensor_positions[:, 0]
        )

        lat, lon, alt = geom_plugin.direct_loc(
            sensor_data,
            geomodel,
            col,
            row,
        )

        alt = np.reshape(alt, (rows.shape[0], cols.shape[0]))

        alt_ref = inputs.rasterio_get_values(
            dem, lon, lat, point_cloud_conversion
        )
        alt_ref = np.reshape(alt_ref, (rows.shape[0], cols.shape[0]))

        ground_truth = -(alt - alt_ref) / disp_to_alt_ratio

    if target == "sensor":

        _, _, alt = geom_plugin.direct_loc(
            sensor_data,
            geomodel,
            positions_row.ravel(),
            positions_col.ravel(),
        )

        ground_truth = np.reshape(alt, (rows.shape[0], cols.shape[0]))

    return ground_truth
