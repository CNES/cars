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
import os
import struct
import warnings
from typing import Tuple

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


def read_geoid_file():
    """
    Read geoid height from OTB geoid file
    Geoid is defined by the $OTB_GEOID_FILE global environement variable.

    A default CARS geoid is deployed in setup.py and
    configured in conf/static_conf.py

    Geoid is returned as an xarray.Dataset and height is stored in the `hgt`
    variable, which is indexed by `lat` and `lon` coordinates. Dataset
    attributes contain geoid bounds geodetic coordinates and
    latitude/longitude step spacing.

    :return: the geoid height array in meter.
    :rtype: xarray.Dataset
    """
    # Set geoid path from OTB_GEOID_FILE
    geoid_path = os.environ.get("OTB_GEOID_FILE")

    with open(geoid_path, mode="rb") as in_grd:  # reading binary data
        # first header part, 4 float of 4 bytes -> 16 bytes to read
        # Endianness seems to be Big-Endian.
        lat_min, lat_max, lon_min, lon_max = struct.unpack(
            ">ffff", in_grd.read(16)
        )
        lat_step, lon_step = struct.unpack(">ff", in_grd.read(8))

        n_lats = int(np.ceil((lat_max - lat_min)) / lat_step) + 1
        n_lons = int(np.ceil((lon_max - lon_min)) / lon_step) + 1

        # read height grid.
        geoid_height = np.fromfile(in_grd, ">f4").reshape(n_lats, n_lons)

        # create output Dataset
        geoid = xr.Dataset(
            {"hgt": (("lat", "lon"), geoid_height)},
            coords={
                "lat": np.linspace(lat_max, lat_min, n_lats),
                "lon": np.linspace(lon_min, lon_max, n_lons),
            },
            attrs={
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "d_lat": lat_step,
                "d_lon": lon_step,
            },
        )

        return geoid


def rasterio_get_nb_bands(raster_file: str) -> int:
    """
    Get the number of bands in an image file

    :param f: Image file
    :returns: The number of bands
    """
    with rio.open(raster_file, "r") as descriptor:
        return descriptor.count


def rasterio_get_size(raster_file: str) -> Tuple[int, int]:
    """
    Get the size of an image (file)

    :param raster_file: Image file
    :returns: The size (width, height)
    """
    with rio.open(raster_file, "r") as descriptor:
        return (descriptor.width, descriptor.height)


def rasterio_can_open(raster_file: str) -> bool:
    """
    Test if a file can be open by rasterio

    :param raster_file: File to test
    :returns: True if rasterio can open file and False otherwise
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

    :returns: conf if check succeeds (else raises CheckerError)
    :rtype: dict
    """
    schema_validator = Checker(schema)
    checked_conf = schema_validator.validate(conf)
    return checked_conf
