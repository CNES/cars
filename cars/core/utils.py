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
Utils module:
contains some cars global shared general purpose functions
"""
# Standard imports
import errno
import os
import shutil
from typing import Tuple

# Third party imports
import numpy as np
import numpy.linalg as la
import rasterio as rio


def safe_makedirs(directory, cleanup=False):
    """
    Create directories even if they already exist (mkdir -p)

    :param directory: path of the directory to create
    """
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            if cleanup:
                shutil.rmtree(directory)
        else:
            raise


def make_relative_path_absolute(path, directory):
    """
    If path is a valid relative path with respect to directory,
    returns it as an absolute path

    :param  path: The relative path
    :type path: string
    :param directory: The directory path should be relative to
    :type directory: string

    :return: os.path.join(directory,path)
             if path is a valid relative path form directory, else path
    :rtype: string
    """
    out = path
    if not os.path.isabs(path):
        abspath = os.path.join(directory, path)
        if os.path.exists(abspath):
            out = abspath
    return out


def get_elevation_range_from_metadata(
    img: str, default_min: float = 0, default_max: float = 300
) -> Tuple[float, float]:
    """
    This function will try to derive a valid RPC altitude range
    from img metadata.

    It will first try to read metadata with gdal.
    If it fails, it will look for values in the geom file if it exists
    If it fails, it will return the default range

    :param img: Path to the img for which the elevation range is required
    :param default_min: Default minimum value to return if everything else fails
    :param default_max: Default minimum value to return if everything else fails
    :return: (elev_min, elev_max) float tuple
    """
    # First, try to get range from gdal metadata
    with rio.open(img) as descriptor:
        gdal_height_offset = descriptor.get_tag_item("HEIGHT_OFF", "RPC")
        gdal_height_scale = descriptor.get_tag_item("HEIGHT_SCALE", "RPC")

        if gdal_height_scale is not None and gdal_height_offset is not None:
            if isinstance(gdal_height_offset, str):
                gdal_height_offset = float(gdal_height_offset)
            if isinstance(gdal_height_scale, str):
                gdal_height_scale = float(gdal_height_scale)
            return (
                float(gdal_height_offset - gdal_height_scale / 2.0),
                float(gdal_height_offset + gdal_height_scale / 2.0),
            )

    # If we are still here, try to get range from OTB/OSSIM geom file if exists
    geom_file, _ = os.path.splitext(img)
    geom_file = geom_file + ".geom"

    # If geom file exists
    if os.path.isfile(geom_file):
        with open(geom_file, "r", encoding="utf-8") as geom_file_desc:
            geom_height_offset = None
            geom_height_scale = None

            for line in geom_file_desc:
                if line.startswith("height_off:"):
                    geom_height_offset = float(line.split(sep=":")[1])

                if line.startswith("height_scale:"):
                    geom_height_scale = float(line.split(sep=":")[1])
            if geom_height_offset is not None and geom_height_scale is not None:
                return (
                    float(geom_height_offset - geom_height_scale / 2),
                    float(geom_height_offset + geom_height_scale / 2),
                )

    # If we are still here, return a default range:
    return (default_min, default_max)


def angle_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Compute the smallest angle in radians between two angle_vectors
    Use arctan2 more precise than arcos2
    Tan θ = abs(axb)/ (a.b)
    (same: Cos θ = (a.b)/(abs(a)abs(b)))

    :param vector_1: Numpy first vector
    :param vector_2: Numpy second vector
    :return: Smallest angle in radians
    """

    vec_dot = np.dot(vector_1, vector_2)
    vec_norm = la.norm(np.cross(vector_1, vector_2))
    return np.arctan2(vec_norm, vec_dot)
