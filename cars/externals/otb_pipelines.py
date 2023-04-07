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
OTB Pipelines module:
contains functions that builds Orfeo ToolBox pipelines used by CARS
Refacto: Split function in generic externals calls through functional steps
interfaces (epipolar rectification, ...)
"""

# Standard imports
from __future__ import absolute_import

import logging
import os

# Third party imports
import numpy as np
import otbApplication
import xarray as xr

# CARS imports
from cars.core import constants as cst


def get_utm_zone_as_epsg_code(lon, lat):
    """
    Returns the EPSG code of the UTM zone where the lat, lon point falls in
    TODO: refacto with externals (OTB)

    :param lon: longitude of the point
    :type lon: float
    :param lat: latitude of the point
    :type lat: float
    :return: The EPSG code corresponding to the UTM zone
    :rtype: int
    """
    utm_app = otbApplication.Registry.CreateApplication(
        "ObtainUTMZoneFromGeoPoint"
    )
    utm_app.SetParameterFloat("lon", float(lon))
    utm_app.SetParameterFloat("lat", float(lat))
    utm_app.Execute()
    zone = utm_app.GetParameterInt("utm")
    north_south = 600 if lat >= 0 else 700
    return 32000 + north_south + zone


def read_lowres_dem(
    startx,
    starty,
    sizex,
    sizey,
    dem=None,
    default_alt=None,
    geoid=None,
    resolution=0.000277777777778,
):
    """
    Read an extract of the low resolution input DSM and return it as an Array

    :param startx: Upper left x coordinate for grid in WGS84
    :type startx: float
    :param starty: Upper left y coordinate for grid in WGS84
        (remember that values are decreasing in y axis)
    :type starty: float
    :param sizex: Size of grid in x direction
    :type sizex: int
    :param sizey: Size of grid in y direction
    :type sizey: int
    :param dem: DEM directory
    :type dem: string
    :param default_alt: Default altitude above ellipsoid
    :type default_alt: float
    :param geoid: path to geoid file
    :type geoid: str
    :param resolution: Resolution (in degrees) of output raster
    :type resolution: float
    :return: The extract of the lowres DEM as an xarray.Dataset
    :rtype: xarray.Dataset
    """
    # save os env
    env_save = os.environ.copy()

    if "OTB_GEOID_FILE" in os.environ:
        logging.warning(
            "The OTB_GEOID_FILE environment variable is set by the user,"
            " it might disturbed the read_lowres_dem function geoid management"
        )
        del os.environ["OTB_GEOID_FILE"]

    # create OTB application
    app = otbApplication.Registry.CreateApplication("DEMReader")

    if dem is not None:
        app.SetParameterString("elev.dem", dem)

    if default_alt is not None:
        app.SetParameterFloat("elev.default", default_alt)

    if geoid is not None:
        app.SetParameterString("elev.geoid", geoid)

    app.SetParameterFloat("originx", startx)
    app.SetParameterFloat("originy", starty)
    app.SetParameterInt("sizex", sizex)
    app.SetParameterInt("sizey", sizey)
    app.SetParameterFloat("resolution", resolution)
    app.Execute()

    dem_as_array = np.copy(app.GetImageAsNumpyArray("out"))

    x_values_1d = np.linspace(
        startx + 0.5 * resolution,
        startx + resolution * (sizex + 0.5),
        sizex,
        endpoint=False,
    )
    y_values_1d = np.linspace(
        starty - 0.5 * resolution,
        starty - resolution * (sizey + 0.5),
        sizey,
        endpoint=False,
    )

    dims = [cst.Y, cst.X]
    coords = {cst.X: x_values_1d, cst.Y: y_values_1d}
    dsm_as_ds = xr.Dataset(
        {cst.RASTER_HGT: (dims, dem_as_array)}, coords=coords
    )
    dsm_as_ds[cst.EPSG] = 4326
    dsm_as_ds[cst.RESOLUTION] = resolution

    # restore environment variables
    if "OTB_GEOID_FILE" in env_save.keys():
        os.environ["OTB_GEOID_FILE"] = env_save["OTB_GEOID_FILE"]

    return dsm_as_ds
