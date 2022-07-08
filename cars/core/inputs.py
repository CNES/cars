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
Inputs module:
contains some CARS global shared general purpose inputs functions
"""

# Standard imports
import logging
import warnings
from typing import Dict, Tuple

# Third party imports
import fiona
import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import Checker
from shapely.geometry import shape

# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def read_vector(path_to_file):
    """
    Read vector file and returns the corresponding polygon

    :raise Exception when the input file is unreadable

    :param path_to_file: path to the file to open
    :type path_to_file: str
    :return: a shapely polygon
    :rtype: tuple (polygon, epsg)
    """
    try:
        polys = []
        with fiona.open(path_to_file) as vec_file:
            _, epsg = vec_file.crs["init"].split(":")
            for feat in vec_file:
                polys.append(shape(feat["geometry"]))
    except BaseException as base_except:
        raise Exception(
            "Impossible to read {} file".format(path_to_file)
        ) from base_except

    if len(polys) == 1:
        return polys[0], int(epsg)

    if len(polys) > 1:
        logging.info(
            "Multi features files are not supported, "
            "the first feature of {} will be used".format(path_to_file)
        )
        return polys[0], int(epsg)

    logging.info("No feature is present in the {} file".format(path_to_file))
    return None


def rasterio_get_nb_bands(raster_file: str) -> int:
    """
    Get the number of bands in an image file

    :param raster_file: Image file
    :return: The number of bands
    """
    with rio.open(raster_file, "r") as descriptor:
        return descriptor.count


def rasterio_get_size(raster_file: str) -> Tuple[int, int]:
    """
    Get the size of an image (file)

    :param raster_file: Image file
    :return: The size (width, height)
    """
    with rio.open(raster_file, "r") as descriptor:
        return (descriptor.width, descriptor.height)


def rasterio_read_as_array(
    raster_file: str, window: rio.windows.Window = None
) -> Tuple[np.ndarray, dict]:
    """
    Get the data of an image file, and its profile

    :param raster_file: Image file
    :param window: Window to get data from
    :return: The array, its profile
    """
    with rio.open(raster_file, "r") as descriptor:
        if descriptor.count == 1:
            data = descriptor.read(1, window=window)
        else:
            data = descriptor.read(window=window)

        return data, descriptor.profile


def rasterio_get_profile(raster_file: str) -> Dict:
    """
    Get the profile of an image file

    :param raster_file: Image file
    :return: The profile of the given image
    """
    with rio.open(raster_file, "r") as descriptor:
        return descriptor.profile


def rasterio_can_open(raster_file: str) -> bool:
    """
    Test if a file can be open by rasterio

    :param raster_file: File to test
    :return: True if rasterio can open file and False otherwise
    """
    try:
        rio.open(raster_file)
        return True
    except Exception as read_error:
        logging.warning(
            "Impossible to read file {}: {}".format(raster_file, read_error)
        )
        return False


def ncdf_can_open(file_path):
    """
    Checks if the given file can be opened by NetCDF
    :param file_path: file path.
    :type file_path: str
    :return: True if it can be opened, False otherwise.
    :rtype: bool
    """
    try:
        with xr.open_dataset(file_path) as _:
            return True
    except Exception as read_error:
        logging.warning(
            "Exception caught while trying to read file {}: {}".format(
                file_path, read_error
            )
        )
        return False


def check_json(conf, schema):
    """
    Check a dictionary with respect to a schema

    :param conf: The dictionary to check
    :type conf: dict
    :param schema: The schema to use
    :type schema: dict

    :return: conf if check succeeds (else raises CheckerError)
    :rtype: dict
    """
    schema_validator = Checker(schema)
    checked_conf = schema_validator.validate(conf)
    return checked_conf
