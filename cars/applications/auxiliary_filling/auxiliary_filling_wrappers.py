# !/usr/bin/env python
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
this module contains the AuxiliaryFillingFromSensors application class.
"""

from pyproj import CRS
from shapely.geometry import Polygon

from cars.core.projection import polygon_projection_crs


def compute_sensor_bounds(sensor_inputs, geom_plugin, output_crs):
    """
    Compute bounds of each input sensor that have an associated color or
    classification image

    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param output_crs: crs of the output polygons
    :type output_crs: CRS

    :return: a dictionary containing a Polygon in output geometry for each
        valid input sensor
    """

    sensor_bounds = {}

    for sensor_name, sensor in sensor_inputs.items():
        reference_sensor_image = sensor["image"]["bands"]["b0"]["path"]
        # no data for this sensor, no need to compute its bounds
        if reference_sensor_image is None:
            continue

        u_l, u_r, l_l, l_r = geom_plugin.image_envelope(
            reference_sensor_image, sensor["geomodel"]
        )

        poly_geo = Polygon([u_l, u_r, l_r, l_l, u_l])

        sensor_bounds[sensor_name] = polygon_projection_crs(
            poly_geo, CRS(4326), output_crs
        )

    return sensor_bounds


def filter_sensor_inputs(sensor_inputs, sensor_bounds, ground_polygon):
    """
    Filter input sensors by comparing their bounds to a reference Polygon

    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param sensor_bounds: dictionary containing bounds of input sensors
    :type sensor_bounds: dict
    :param ground_polygon: reference polygon, in ground geometry
    :type ground_polygon: Polygon

    :return: a fitlered version of sensor_inputs
    """

    filtered_sensor_inputs = sensor_inputs.copy()

    for sensor_name, bound in sensor_bounds.items():
        if not bound.intersects(ground_polygon):
            del filtered_sensor_inputs[sensor_name]

    return filtered_sensor_inputs
