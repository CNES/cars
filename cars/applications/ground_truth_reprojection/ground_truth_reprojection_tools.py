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
import rasterio as rio
from rasterio.windows import Window
from scipy.interpolate import interpn
from shareloc.proj_utils import transform_physical_point_to_index

from cars.core import inputs
from cars.core.projection import point_cloud_conversion
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

        lat, lon, alt_ref = geom_plugin_dem_median.direct_loc(
            sensor_data_im,
            geomodel,
            col,
            row,
        )

        alt_ref = np.reshape(alt_ref, (rows.shape[0], cols.shape[0]))

        ground_truth = -(alt - alt_ref) / disp_to_alt_ratio
        if reverse:
            ground_truth *= -1

        direct_loc = np.column_stack((lon, lat, np.ravel(ground_truth)))

    if target == "sensor":

        lat, lon, alt = geom_plugin.direct_loc(
            sensor_data_im,
            geomodel,
            positions_col.ravel(),
            positions_row.ravel(),
        )

        ground_truth = np.reshape(alt, (rows.shape[0], cols.shape[0]))
        direct_loc = np.column_stack((lon, lat, alt))

    return ground_truth, direct_loc


def resample_auxiliary_values(
    ground_position,
    auxiliary_input,
    window,
    interpolation_method="nearest",
    keep_band=False,
):
    """
    Resample classification map in epipolar geometry

    :param ground_position: Direct localization result (lon, lat, alt)
    :type ground_position: 2D np.darray
    :param auxiliary_input: Path to auxiliary_value
    :type auxiliary_input: string
    :param window: the tile window
    :type window: list
    :param interpolation_method: interpolation method
    :type interpolation_method: string
    :param keep_band: bool to see if we keep the band
    :type keep_band: bool
    """

    # get the shape of the tile
    shape = (window[1] - window[0], window[3] - window[2])

    # Convert shareloc output in degree in UTM
    auxiliary_epsg = inputs.rasterio_get_epsg(auxiliary_input)
    utm_array = point_cloud_conversion(ground_position, 4326, auxiliary_epsg)

    # Keep auxiliary input information
    transform = inputs.rasterio_get_transform(auxiliary_input)
    nb_bands = inputs.rasterio_get_nb_bands(auxiliary_input)
    outside_interpolated_value = inputs.rasterio_get_nodata(auxiliary_input)

    # Convert georef coordinates to pixel coordinates (x,y) -> (row,col)
    rows_geo, cols_geo = transform_physical_point_to_index(
        ~transform, utm_array[:, 1], utm_array[:, 0]
    )

    # Find the right window
    with rio.open(auxiliary_input) as src:
        width = src.width
        height = src.height

        # Put extreme values to nan
        rows_geo[(rows_geo > height) | (rows_geo < 0)] = np.nan
        cols_geo[(cols_geo > width) | (cols_geo < 0)] = np.nan

        row_min, row_max = int(np.floor(np.nanmin(rows_geo))), int(
            np.ceil(np.nanmax(rows_geo))
        )
        col_min, col_max = int(np.floor(np.nanmin(cols_geo))), int(
            np.ceil(np.nanmax(cols_geo))
        )

        # put margin to ensure that the window englobe the coord to interp
        margin = 2
        row_min, row_max = max(0, row_min - margin), min(
            src.height, row_max + margin
        )
        col_min, col_max = max(0, col_min - margin), min(
            src.width, col_max + margin
        )

        # Construct the window and read the data
        window = Window(col_min, row_min, col_max - col_min, row_max - row_min)

        if keep_band:
            data = src.read(window=window)
            height = data.shape[1]
            width = data.shape[2]
        else:
            data = src.read(1, window=window)
            height = data.shape[0]
            width = data.shape[1]

        # Construct the grid that corresponds
        grid_rows = np.linspace(row_min, row_max, height)
        grid_cols = np.linspace(col_min, col_max, width)
        grid = (grid_rows, grid_cols)

    index = np.column_stack((rows_geo, cols_geo))

    # Interpolation step
    interpolated_points = interpolate(
        grid,
        data,
        index,
        method=interpolation_method,
        fill_value=outside_interpolated_value,
    )

    if keep_band:
        output_array = interpolated_points.reshape((nb_bands, *shape))
    else:
        output_array = interpolated_points.reshape(shape)

    return output_array


def interpolate(points, values, positions, method="linear", fill_value=None):
    """
    Interpolate position

    :param points: Points defining the grid
    :type points: np.darray
    :param values: Data
    :type values: 2D np.darray, or 3D np.darray (band, row, col)
    :param positions: Positions to interpolate
    :type positions: np.darray
    :param method: interpolation method
    :type method: string : {'linear', 'nearest'}
    :param fill_value: value to use for points outside of the interp domain
    :type fill_value: float
    :return: interpolated positions
    :rtype: 1D np.darray or 2D np.darray (band, interpolated positions)
    """
    if len(values.shape) > 2:
        interp_point = np.zeros((values.shape[0], positions.shape[0]))
        for band in range(values.shape[0]):
            interp_point[band, :] = interpn(
                points,
                values[band, :, :],
                positions,
                method=method,
                bounds_error=False,
                fill_value=fill_value,
            )
    else:
        interp_point = interpn(
            points,
            values,
            positions,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
    return interp_point
