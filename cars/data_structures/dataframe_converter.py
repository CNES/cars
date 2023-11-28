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
Contains function to convert the point cloud dataframe to laz format:
"""
import logging
import warnings

import laspy
import laspy.file
import laspy.header
import numpy as np
from pyproj import CRS

import cars.core.constants as cst


def convert_pcl_to_laz(point_clouds, output_filename: str):
    """
    Convert 3d point cloud to laz format
    :param point_clouds: point_clouds dataframe
    :param output_filename: output laz filename (with naming convention)
    :return: the list of point cloud save in las format
    """

    # get all layer : X, Y, Z and color
    coordinates = ["X", "Y", "Z"]
    las_color = ["red", "green", "blue"]
    color_type = point_clouds.attrs["attributes"]["color_type"]
    epsg = point_clouds.attrs["attributes"]["epsg"]
    input_color = get_input_color(point_clouds)
    laz_file_name = output_filename
    arrays_pcl, arrays_color = extract_point_cloud(
        point_clouds, coordinates, input_color, color_type
    )
    # generate laz
    generate_laz(
        laz_file_name, coordinates, las_color, arrays_pcl, arrays_color
    )
    # dump prj file
    generate_prj_file(laz_file_name, epsg)


def get_input_color(point_clouds):
    """
    Retrieve color index from point cloud
    :param point_clouds: point clouds data
    :return: input color list
    """

    input_color = ["color_R", "color_G", "color_B"]
    color_names = [
        name
        for name in point_clouds.columns
        if cst.POINTS_CLOUD_CLR_KEY_ROOT in name
    ]
    nb_color = len(color_names)
    if nb_color == 1:
        logging.warning("No color available for point cloud")
        input_color = color_names
    elif sorted(color_names[:3]) != sorted(input_color):
        logging.warning(
            "Descriptions of color bands {} does not conform to names "
            "'R', 'G', 'B'".format(color_names[:3])
        )
        input_color = color_names[:3]

    return input_color


def extract_point_cloud(point_clouds, coordinates, input_color, color_type):
    """
    Extract point cloud positions (x,y,z) and color layers
    :param point_clouds: point cloud dataframe
    :param coordinate: target coordinates key index
    :param input_color: color key index according to input
    :param color_type: type of color data
    :param nb
    :return: point cloud array location and color
    """

    nb_color = len(input_color)
    arrays_pcl = None
    arrays_color = None
    for name in point_clouds.columns:
        # get coordinates bands
        if name in ["x", "y", "z"]:
            array = point_clouds[name].to_numpy()
            # first step : define np.ndarray to contains the arrays X,Y,Z
            if arrays_pcl is None:
                arrays_pcl = np.ndarray((3, array.reshape(-1).T.shape[0]))
                # get only valid pixels
            array = array.reshape(-1).T
            # get layer into np.ndarray (laspy accept only XYZ in upper case)
            arrays_pcl[coordinates.index(name.upper())] = array
        elif name in input_color:
            array = point_clouds[name].to_numpy()
            if arrays_color is None:
                arrays_color = np.ndarray((3, array.reshape(-1).T.shape[0]))
            array = array.reshape(-1).T
            maxi = 65535
            if color_type:
                if np.issubdtype(np.dtype(color_type), np.integer):
                    maxi = np.min([maxi, np.iinfo(np.dtype(color_type)).max])
            array[array >= maxi] = maxi
            array[array <= 0] = 0

            arrays_color[input_color.index(name)] = array
            if nb_color == 1:
                arrays_color[1] = array
                arrays_color[2] = array
    return arrays_pcl, arrays_color


def generate_prj_file(output_filename, epsg):
    """
    Generate prj file associated to the laz file if projection is UTM or WGS84
    :param output_filename: name of laz file
    :param epsg: code of output epsg
    """
    with warnings.catch_warnings():
        # Ignore some crs warning
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*You will likely lose important projection"
            " information when converting to a PROJ string from "
            "another format.*",
        )
        crs = CRS.from_epsg(epsg)
        proj = crs.to_proj4()
    if crs.is_geographic:
        logging.warning(
            "Coordinate system of points cloud is geographic: "
            "Display of LAZ file may not work"
        )
    with open(output_filename + ".prj", "w", encoding="utf8") as file_prj:
        file_prj.write(proj)


def generate_laz(
    output_filename, coordinates, las_color, arrays_pcl, arrays_color
):
    """
    Generate laz file from location and color arrays
    """
    # Create laz image
    header = laspy.LasHeader(point_format=2)
    scale_factor = 0.01
    header.scales = [scale_factor, scale_factor, scale_factor]
    laz = laspy.LasData(header)
    # fill X,Y,Z into laspy structure, convert to cm
    scale_multiplicator = 1 / scale_factor
    for layer_index in range(arrays_pcl.shape[0]):
        setattr(
            laz,
            [k for i, k in enumerate(coordinates) if i == layer_index][0],
            scale_multiplicator * arrays_pcl[layer_index],
        )
    if arrays_color is not None:
        for color_index in range(arrays_color.shape[0]):
            setattr(
                laz,
                [k for i, k in enumerate(las_color) if i == color_index][0],
                arrays_color[color_index],
            )
    laz.write(output_filename)
