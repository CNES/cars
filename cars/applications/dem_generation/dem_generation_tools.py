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
this module contains tools for the dem generation
"""

# Third party imports
import numpy as np
import pandas

from cars.applications.triangulation import triangulation_tools

# CARS imports
from cars.core import constants as cst
from cars.core import preprocessing, projection
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)


def triangulate_sparse_matches(
    sensor_image_left,
    sensor_image_right,
    grid_left,
    grid_right,
    matches,
    geometry_plugin,
):
    """
    Triangulate matches in a metric system

    :param sensor_image_right: sensor image right
    :type sensor_image_right: CarsDataset
    :param sensor_image_left: sensor image left
    :type sensor_image_left: CarsDataset
    :param grid_left: grid left
    :type grid_left: CarsDataset CarsDataset
    :param grid_right: corrected grid right
    :type grid_right: CarsDataset
    :param matches: matches
    :type matches: np.ndarray
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: float
    :param pair_folder: folder used for current pair
    :type pair_folder: str

    :return: disp min and disp max
    :rtype: float, float
    """

    sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
    sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
    geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
    geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

    point_cloud = triangulation_tools.triangulate_matches(
        geometry_plugin,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        np.ascontiguousarray(matches),
    )

    # compute epsg
    epsg = preprocessing.compute_epsg(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        geometry_plugin,
        disp_min=0,
        disp_max=0,
    )
    # Project point cloud to UTM
    projection.points_cloud_conversion_dataset(point_cloud, epsg)

    # Convert point cloud to pandas format to allow statistical filtering
    labels = [cst.X, cst.Y, cst.Z, cst.DISPARITY, cst.POINTS_CLOUD_CORR_MSK]
    cloud_array = []
    cloud_array.append(point_cloud[cst.X].values)
    cloud_array.append(point_cloud[cst.Y].values)
    cloud_array.append(point_cloud[cst.Z].values)
    cloud_array.append(point_cloud[cst.DISPARITY].values)
    cloud_array.append(point_cloud[cst.POINTS_CLOUD_CORR_MSK].values)
    pd_cloud = pandas.DataFrame(
        np.transpose(np.array(cloud_array)), columns=labels
    )

    pd_cloud.attrs["epsg"] = epsg

    return pd_cloud
