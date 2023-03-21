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


def build_extract_roi_application(img, region):
    """
    This function builds a ready to use instance of the ExtractROI application

    :param img: Pointer to the OTB image to extract
    :type img: otb::Image pointer
    :param region: Extraction region
    :type region: list of 4 int (xmin, ymin, xmax, ymax)
    :return: (extracted image, roi application) tuple
    :rtype: ready to use instance of the ExtractROI application
    """
    extract_app = otbApplication.Registry.CreateApplication("ExtractROI")
    if isinstance(img, str):
        extract_app.SetParameterString("in", img)
    elif isinstance(img, np.ndarray):
        extract_app.SetImageFromNumpyArray("in", img)
    else:
        extract_app.SetParameterInputImage("in", img)
    extract_app.SetParameterInt("startx", int(region[0]))
    extract_app.SetParameterInt("starty", int(region[1]))
    extract_app.SetParameterInt("sizex", int(region[2]) - int(region[0]))
    extract_app.SetParameterInt("sizey", int(region[3]) - int(region[1]))
    extract_app.Execute()

    return extract_app


def build_mask_pipeline(
    input_img,
    input_mask,
    input_nodata,
    out_nodata,
    out_valid_value,
    grid,
    epipolar_size_x,
    epipolar_size_y,
    roi,
):
    """
    This function builds a pipeline that computes and
    resampled image mask in epipolar geometry

    :param input_img: Path to the left input image
    :type img: string
    :param input_mask:  Path to left image mask or None
    :type mask: string
    :param input_nodata: Pixel value to be treated as nodata in image or None
    :type input_nodata: float
    :param out_nodata: Pixel value used for the output
    :type out_nodata: float
    :param out_valid_value: Pixel value for valid points in mask
    :typ out_valid_value: float
    :param grid: The stereo-rectification rectification grid
    :type grid: otb::Image pointer or string
    :param epipolar_size_x: Size of stereo-rectified images in x
    :type epipolar_size_x: int
    :param epipolar_size_y: Size of stereo-rectified images in y
    :type epipolar_size_y: int
    :param roi: Region over which to compute epipolar mask or None
    :type roi: list of 4 int (xmin, ymin, xmax, ymax)
    :return: mask
    :rtype: resampled mask as numpy array
    """
    mask_app = otbApplication.Registry.CreateApplication("BuildMask")

    mask_app.SetParameterString("in", input_img)
    if input_nodata is not None:
        mask_app.SetParameterFloat("innodata", input_nodata)
    if input_mask is not None:
        mask_app.SetParameterString("inmask", input_mask)
        mask_app.EnableParameter("inmask")
    mask_app.SetParameterFloat("outnodata", out_nodata)
    mask_app.SetParameterFloat("outvalid", out_valid_value)

    mask_app.Execute()

    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling"
    )
    resampling_app.SetParameterInputImage(
        "io.in", mask_app.GetParameterOutputImage("out")
    )

    if isinstance(grid, str):
        resampling_app.SetParameterString("grid.in", grid)
    else:
        resampling_app.SetParameterInputImage("grid.in", grid)

    resampling_app.SetParameterString("grid.type", "def")
    resampling_app.SetParameterInt("out.sizex", epipolar_size_x)
    resampling_app.SetParameterInt("out.sizey", epipolar_size_y)
    resampling_app.SetParameterString("interpolator", "nn")
    resampling_app.SetParameterFloat("out.default", out_nodata)
    resampling_app.Execute()

    # TODO: Dilate nodata mask to ensure that interpolated pixels are not
    # contaminated
    extract_app = build_extract_roi_application(
        resampling_app.GetParameterOutputImage("io.out"), roi
    )
    msk = np.copy(extract_app.GetImageAsNumpyArray("out"))

    return msk


def build_image_resampling_pipeline(
    img, grid, epipolar_size_x, epipolar_size_y, roi
):
    """
    This function builds a pipeline that resamples images in epipolar geometry

    :param img: Path to the left image
    :type img: string
    :param grid: The stereo-rectification rectification grid
    :type grid: otb::Image pointer or string
    :param epipolar_size_x: Size of stereo-rectified images in x
    :type epipolar_size_x: int
    :param epipolar_size_y: Size of stereo-rectified images in y
    :type epipolar_size_y: int
    :param roi: Region over which to compute epipolar images, or None
    :type roi: list of 4 int (xmin, ymin, xmax, ymax)
    :return: resampled image
    :rtype: resampled image as numpy array
    """

    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling"
    )

    if isinstance(img, str):
        resampling_app.SetParameterString("io.in", img)
    else:
        resampling_app.SetParameterInputImage("io.in", img)

    if isinstance(grid, str):
        resampling_app.SetParameterString("grid.in", grid)
    else:
        resampling_app.SetParameterInputImage("grid.in", grid)

    resampling_app.SetParameterString("grid.type", "def")
    resampling_app.SetParameterInt("out.sizex", epipolar_size_x)
    resampling_app.SetParameterInt("out.sizey", epipolar_size_y)

    resampling_app.Execute()
    extract_app = build_extract_roi_application(
        resampling_app.GetParameterOutputImage("io.out"), roi
    )

    # Retrieve data and build left dataset
    resampled = np.copy(extract_app.GetVectorImageAsNumpyArray("out"))

    return resampled


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
