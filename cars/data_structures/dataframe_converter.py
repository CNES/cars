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
import laspy
import laspy.file
import laspy.header
import numpy as np


def utm_epsg_to_proj(epsg: int):
    """
    Convert a UTM EPSG code to a PROJ string
    :param epsg: Integer EPSG code
    :return: PROJ string
    """
    assert epsg > 32600
    assert epsg < 32800
    south = " +south" if ((epsg - 32600) // 100) else ""
    zone = epsg % 100
    return f"+proj=utm +zone={zone}{south} +datum=WGS84 +units=m +no_defs"


def convert_pcl_to_laz(point_clouds, output_filename: str):
    """
    Convert 3d point cloud to laz format
    :param point_clouds: point_clouds dataframe
    :param output_filename: output laz filename (with naming convention)
    :return: the list of point cloud save in las format
    """
    # Create laz image
    header = laspy.LasHeader(point_format=2)
    scale_factor = 0.01
    header.scales = [scale_factor, scale_factor, scale_factor]
    laz = laspy.LasData(header)

    # get all layer : X, Y, Z and color
    band_index = {"X": 0, "Y": 1, "Z": 2}
    input_color_index = {"clr0": 0, "clr1": 1, "clr2": 2}
    las_color_index = {"red": 0, "green": 1, "blue": 2}
    arrays_pcl = None
    arrays_color = None
    color_type = point_clouds.attrs["attributes"]["color_type"]
    nb_color = 0
    for name in point_clouds.columns:
        if name in ["clr0", "clr1", "clr2"]:
            nb_color += 1

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
            arrays_pcl[band_index[name.upper()]] = array
        elif name in ["clr0", "clr1", "clr2"]:
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

            arrays_color[input_color_index[name]] = array
            if nb_color == 1:
                arrays_color[1] = array
                arrays_color[2] = array

    # fill X,Y,Z into laspy structure, convert to cm
    scale_multiplicator = 1 / scale_factor
    for layer_index in range(arrays_pcl.shape[0]):
        setattr(
            laz,
            [k for k, v in band_index.items() if v == layer_index][0],
            scale_multiplicator * arrays_pcl[layer_index],
        )
    if arrays_color is not None:
        for color_index in range(arrays_color.shape[0]):
            setattr(
                laz,
                [k for k, v in las_color_index.items() if v == color_index][0],
                arrays_color[color_index],
            )
    laz.write(output_filename)
    # dump prj file
    proj = utm_epsg_to_proj(point_clouds.attrs["attributes"]["epsg"])
    with open(output_filename + ".prj", "w", encoding="utf8") as file_prj:
        file_prj.write(proj)
