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
Preprocessing module:
contains functions used during cars prepare pipeline step of cars
"""

# Standard imports
from __future__ import absolute_import
import math
import logging
from typing import Union, Tuple

# Third party imports
import numpy as np
import rasterio as rio
from affine import Affine
from scipy import interpolate
from scipy.signal import butter, lfilter, filtfilt, lfilter_zi
import xarray as xr
import otbApplication as otb


# Cars imports
from cars import pipelines
from cars import constants as cst
from cars import projection
from cars import utils

def dataset_matching(ds1, ds2, matching_threshold = 0.6, n_octave = 8,
                     n_scale_per_octave = 3, dog_threshold = 20,
                     edge_threshold = 5, magnification = 2.0,
                     backmatching = True):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param threshold: Threshold for matches
    :type threshold: float
    :param backmatching: Also check that right vs. left gives same match
    :type backmatching: bool
    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    size1 = [int(ds1.attrs['region'][2] - ds1.attrs['region'][0]),
             int(ds1.attrs['region'][3] - ds1.attrs['region'][1])]
    roi1 = [0, 0, size1[0], size1[1]]
    origin1 = [float(ds1.attrs['region'][0]), float(ds1.attrs['region'][1])]

    size2 = [int(ds2.attrs['region'][2] - ds2.attrs['region'][0]),
             int(ds2.attrs['region'][3] - ds2.attrs['region'][1])]
    roi2 = [0, 0, size2[0], size2[1]]
    origin2 = [float(ds2.attrs['region'][0]), float(ds2.attrs['region'][1])]

    # Encode images for OTB
    im1 = pipelines.encode_to_otb(
        ds1[cst.EPI_IMAGE].values, size1, roi1, origin=origin1)
    msk1 = pipelines.encode_to_otb(
        ds1[cst.EPI_MSK].values, size1, roi1, origin=origin1)
    im2 = pipelines.encode_to_otb(
        ds2[cst.EPI_IMAGE].values, size2, roi2, origin=origin2)
    msk2 = pipelines.encode_to_otb(
        ds2[cst.EPI_MSK].values, size2, roi2, origin=origin2)

    # Build sift matching app
    matching_app = otb.Registry.CreateApplication(
        "EpipolarSparseMatching")

    matching_app.ImportImage("in1", im1)
    matching_app.ImportImage("in2", im2)
    matching_app.EnableParameter("inmask1")
    matching_app.ImportImage("inmask1", msk1)
    matching_app.EnableParameter("inmask2")
    matching_app.ImportImage("inmask2", msk2)

    matching_app.SetParameterInt("maskvalue", 0)
    matching_app.SetParameterString("algorithm", "sift")
    matching_app.SetParameterFloat("matching", matching_threshold)
    matching_app.SetParameterInt("octaves", n_octave)
    matching_app.SetParameterInt("scales", n_scale_per_octave)
    matching_app.SetParameterFloat("tdog", dog_threshold)
    matching_app.SetParameterFloat("tedge", edge_threshold)
    matching_app.SetParameterFloat("magnification", magnification)
    matching_app.SetParameterInt("backmatching", backmatching)
    matching_app.Execute()

    # Retrieve number of matches
    nb_matches = matching_app.GetParameterInt("nbmatches")

    matches = np.empty((0, 4))

    if nb_matches > 0:
        # Export result to numpy
        matches = np.copy(
            matching_app.GetVectorImageAsNumpyArray("out")[:, :, -1])

    return matches


def remove_epipolar_outliers(matches, percent=0.1):
    """
    This function will filter the match vector
    according to a quantile of epipolar error

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema
    :type percent: float
    :return: the filtered match array
    :rtype: numpy array
    """
    epipolar_error_min = np.percentile(matches[:, 1] - matches[:, 3], percent)
    epipolar_error_max = np.percentile(
        matches[:, 1] - matches[:, 3], 100 - percent)
    logging.info(
        "Epipolar error range after outlier rejection: [{},{}]".format(
            epipolar_error_min,
            epipolar_error_max))
    out = matches[(matches[:, 1] - matches[:, 3]) < epipolar_error_max]
    out = out[(out[:, 1] - out[:, 3]) > epipolar_error_min]

    return out


def compute_disparity_range(matches, percent=0.1):
    """
    This function will compute the disparity range
    from matches by filtering percent outliers

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema (in %)
    :type percent: float
    :return: the disparity range
    :rtype: float, float
    """
    disparity = matches[:, 2] - matches[:, 0]

    mindisp = np.percentile(disparity, percent)
    maxdisp = np.percentile(disparity, 100 - percent)

    return mindisp, maxdisp


def correct_right_grid(matches, grid, origin, spacing):
    """
    Compute the corrected right epipolar grid

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param grid: right grid for epipolar rectification
    :type grid: 3d numpy array (x, y, l/c)
    :param origin: origin of the grid
    :type origin: (float, float)
    :param spacing: spacing of the grid
    :type spacing: (float, float)
    :return: the corrected grid
    :rtype: 3d numpy array (x, y, l/c)
    """
    right_grid = np.copy(grid)

    # Form 3D array with grid positions
    x_values_1d = np.linspace(
        origin[0],
        origin[0] +
        right_grid.shape[0] *
        spacing[0],
        grid.shape[0])
    y_values_1d = np.linspace(
        origin[1],
        origin[1] +
        grid.shape[1] *
        spacing[1],
        grid.shape[1])
    x_values_2d, y_values_2d = np.meshgrid(y_values_1d, x_values_1d)

    # Compute corresponding point in sensor geometry (grid encodes (x_sensor -
    # x_epi,y_sensor - y__epi)
    source_points = right_grid
    source_points[:, :, 0] += x_values_2d
    source_points[:, :, 1] += y_values_2d

    # Extract matches for convenience
    matches_y1 = matches[:, 1]
    matches_x2 = matches[:, 2]
    matches_y2 = matches[:, 3]

    # Map real matches to sensor geometry
    sensor_matches_raw_x = interpolate.griddata((
        np.ravel(x_values_2d),
        np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 0]), (matches_x2, matches_y2))

    sensor_matches_raw_y = interpolate.griddata((
        np.ravel(x_values_2d),
        np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 1]), (matches_x2, matches_y2))

    # Simulate matches that have no epipolar error (i.e. y2 == y1) and map
    # them to sensor geometry
    sensor_matches_perfect_x = interpolate.griddata((
        np.ravel(x_values_2d),
        np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 0]), (matches_x2, matches_y1))

    sensor_matches_perfect_y = interpolate.griddata((
        np.ravel(x_values_2d),
        np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 1]), (matches_x2, matches_y1))

    # Compute epipolar error in sensor geometry in both direction
    epipolar_error_x = sensor_matches_perfect_x - sensor_matches_raw_x
    epipolar_error_y = sensor_matches_perfect_y - sensor_matches_raw_y

    # Output epipolar error stats for monitoring
    mean_epipolar_error = [
        np.mean(epipolar_error_x),
        np.mean(epipolar_error_y)]
    median_epipolar_error = [
        np.median(epipolar_error_x),
        np.median(epipolar_error_y)]
    std_epipolar_error = [np.std(epipolar_error_x), np.std(epipolar_error_y)]
    rms_epipolar_error = np.mean(
        np.sqrt(
            epipolar_error_x *
            epipolar_error_x +
            epipolar_error_y *
            epipolar_error_y))
    rmsd_epipolar_error = np.std(
        np.sqrt(
            epipolar_error_x *
            epipolar_error_x +
            epipolar_error_y *
            epipolar_error_y))

    in_stats = {
        "mean_epipolar_error": mean_epipolar_error,
        "median_epipolar_error": median_epipolar_error,
        "std_epipolar_error": std_epipolar_error,
        "rms_epipolar_error": rms_epipolar_error,
        "rmsd_epipolar_error": rmsd_epipolar_error}

    logging.debug(
        "Epipolar error before correction: \n"
        "x    = {:.3f} +/- {:.3f} pixels \n"
        "y    = {:.3f} +/- {:.3f} pixels \n"
        "rmse = {:.3f} +/- {:.3f} pixels \n"
        "medianx = {:.3f} pixels \n"
        "mediany = {:.3f} pixels".format(
            mean_epipolar_error[0],
            std_epipolar_error[0],
            mean_epipolar_error[1],
            std_epipolar_error[1],
            rms_epipolar_error,
            rmsd_epipolar_error,
            median_epipolar_error[0],
            median_epipolar_error[1]))

    # Perform bilinear regression for both component of epipolar error
    lstsq_input = np.array([matches_x2 * 0 + 1, matches_x2, matches_y2]).T
    coefsx, residx, __, __ = np.linalg.lstsq(
        lstsq_input, epipolar_error_x, rcond=None)
    coefsy, residy, __, __ = np.linalg.lstsq(
        lstsq_input, epipolar_error_y, rcond=None)

    # Normalize residuals by number of matches
    rmsex = np.sqrt(residx / matches.shape[0])
    rmsey = np.sqrt(residy / matches.shape[1])

    logging.debug(
        "Root Mean Square Error of correction estimation:"
        "rmsex={} pixels, rmsey={} pixels".format(
            rmsex, rmsey))

    # Reshape coefs to 2D (expected by np.polynomial.polyval2d)
    coefsx_2d = np.ndarray((2, 2))
    coefsx_2d[0, 0] = coefsx[0]
    coefsx_2d[1, 0] = coefsx[1]
    coefsx_2d[0, 1] = coefsx[2]
    coefsx_2d[1, 1] = 0.

    coefsy_2d = np.ndarray((2, 2))
    coefsy_2d[0, 0] = coefsy[0]
    coefsy_2d[1, 0] = coefsy[1]
    coefsy_2d[0, 1] = coefsy[2]
    coefsy_2d[1, 1] = 0.

    # Interpolate the regression model at grid position
    correction_grid_x = np.polynomial.polynomial.polyval2d(
        x_values_2d, y_values_2d, coefsx_2d)
    correction_grid_y = np.polynomial.polynomial.polyval2d(
        x_values_2d, y_values_2d, coefsy_2d)

    # Compute corrected grid
    corrected_grid_x = source_points[:, :, 0] - correction_grid_x - x_values_2d
    corrected_grid_y = source_points[:, :, 1] - correction_grid_y - y_values_2d
    corrected_right_grid = np.stack(
        (corrected_grid_x, corrected_grid_y), axis=2)

    # Map corrected matches to sensor geometry
    sensor_matches_corrected_x = sensor_matches_raw_x + \
        np.polynomial.polynomial.polyval2d(matches_x2, matches_y2, coefsx_2d)
    sensor_matches_corrected_y = sensor_matches_raw_y + \
        np.polynomial.polynomial.polyval2d(matches_x2, matches_y2, coefsy_2d)

    # Map corrected matches to epipolar geometry
    epipolar_matches_corrected_x = interpolate.griddata((
        np.ravel(source_points[:, :, 0]),
        np.ravel(source_points[:, :, 1])),
        np.ravel(x_values_2d),
        (sensor_matches_corrected_x, sensor_matches_corrected_y)
    )
    epipolar_matches_corrected_y = interpolate.griddata((
        np.ravel(source_points[:, :, 0]),
        np.ravel(source_points[:, :, 1])),
        np.ravel(y_values_2d),
        (sensor_matches_corrected_x, sensor_matches_corrected_y)
    )

    corrected_matches = np.copy(matches)
    corrected_matches[:, 2] = epipolar_matches_corrected_x
    corrected_matches[:, 3] = epipolar_matches_corrected_y

    # Compute epipolar error in sensor geometry in both direction after
    # correction
    corrected_epipolar_error_x =\
        sensor_matches_perfect_x - sensor_matches_corrected_x
    corrected_epipolar_error_y =\
        sensor_matches_perfect_y - sensor_matches_corrected_y

    # Ouptut corrected epipolar error stats for monitoring
    mean_corrected_epipolar_error = [
        np.mean(corrected_epipolar_error_x),
        np.mean(corrected_epipolar_error_y)]
    median_corrected_epipolar_error = [
        np.median(corrected_epipolar_error_x),
        np.median(corrected_epipolar_error_y)]
    std_corrected_epipolar_error = [
        np.std(corrected_epipolar_error_x),
        np.std(corrected_epipolar_error_y)]
    rms_corrected_epipolar_error = np.mean(
        np.sqrt(
            corrected_epipolar_error_x *
            corrected_epipolar_error_x +
            corrected_epipolar_error_y *
            corrected_epipolar_error_y))
    rmsd_corrected_epipolar_error = np.std(
        np.sqrt(
            corrected_epipolar_error_x *
            corrected_epipolar_error_x +
            corrected_epipolar_error_y *
            corrected_epipolar_error_y))

    out_stats = {
        "mean_epipolar_error": mean_corrected_epipolar_error,
        "median_epipolar_error": median_corrected_epipolar_error,
        "std_epipolar_error": std_corrected_epipolar_error,
        "rms_epipolar_error": rms_corrected_epipolar_error,
        "rmsd_epipolar_error": rmsd_corrected_epipolar_error}

    logging.debug(
        "Epipolar error after  correction: \n"
        "x    = {:.3f} +/- {:.3f} pixels \n"
        "y    = {:.3f} +/- {:.3f} pixels \n"
        "rmse = {:.3f} +/- {:.3f} pixels \n"
        "medianx = {:.3f} pixels \n"
        "mediany = {:.3f} pixels".format(
            mean_corrected_epipolar_error[0],
            std_corrected_epipolar_error[0],
            mean_corrected_epipolar_error[1],
            std_corrected_epipolar_error[1],
            rms_corrected_epipolar_error,
            rmsd_corrected_epipolar_error,
            median_corrected_epipolar_error[0],
            median_corrected_epipolar_error[1]))

    # And return it
    return corrected_right_grid, corrected_matches, in_stats, out_stats


def write_grid(grid, fname, origin, spacing):
    """
    Write an epipolar resampling grid to file

    :param grid: the grid to write
    :type grid: 3D numpy array
    :param fname: the filename to which the grid will be written
    :type fname: string
    :param origin: origin of the grid
    :type origin: (float, float)
    :param spacing: spacing of the grid
    :type spacing: (float, float)
    """

    geotransform = (
        origin[0] - 0.5 * spacing[0],
        spacing[0],
        0.0,
        origin[1] - 0.5 * spacing[1],
        0.0,
        spacing[1])

    transform = Affine.from_gdal(*geotransform)

    with rio.open(fname, 'w', height=grid.shape[0],
                  width=grid.shape[1], count=2, driver='GTiff',
                  dtype=grid.dtype, transform=transform)\
        as dst:
        dst.write_band(1, grid[:, :, 0])
        dst.write_band(2, grid[:, :, 1])


def image_envelope(img, shp, dem=None, default_alt=None):
    """
    Export the image footprint to a shapefile

    :param img: filename to image or OTB pointer to image
    :type img:  string or OTBImagePointer
    :param shp: Path to the output shapefile
    :type shp: string
    :param dem: Directory containing DEM tiles
    :type dem: string
    :param default_alt: Default altitude above ellipsoid
    :type default_alt: float
    """

    app = otb.Registry.CreateApplication("ImageEnvelope")

    if isinstance(img, str):
        app.SetParameterString("in", img)
    else:
        app.SetParameterInputImage("in", img)

    if dem is not None:
        app.SetParameterString("elev.dem", dem)

    if default_alt is not None:
        app.SetParameterFloat("elev.default", default_alt)

    app.SetParameterString("out", shp)
    app.ExecuteAndWriteOutput()


def read_lowres_dem(startx, starty, sizex, sizey,
                    dem=None, default_alt=None,
                    resolution = 0.000277777777778):
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
    :param resolution: Resolution (in degrees) of output raster
    :type resolution: float
    :return: The extract of the lowres DEM as an xarray.Dataset
    :rtype: xarray.Dataset
    """

    app = otb.Registry.CreateApplication("DEMReader")

    if dem is not None:
        app.SetParameterString("elev.dem", dem)

    if default_alt is not None:
        app.SetParameterFloat("elev.default", default_alt)

    app.SetParameterFloat("originx", startx)
    app.SetParameterFloat("originy", starty)
    app.SetParameterInt("sizex", sizex)
    app.SetParameterInt("sizey", sizey)
    app.SetParameterFloat("resolution", resolution)
    app.Execute()

    dem_as_array = np.copy(app.GetImageAsNumpyArray("out"))

    x_values_1d = np.linspace(startx + 0.5 * resolution,
                              startx + resolution * (sizex + 0.5), sizex,
                              endpoint=False)
    y_values_1d = np.linspace(starty - 0.5 * resolution,
                              starty - resolution * (sizey + 0.5), sizey,
                              endpoint=False)

    dims = [cst.Y, cst.X]
    coords = {cst.X: x_values_1d,
              cst.Y: y_values_1d}
    dsm_as_ds =\
        xr.Dataset({cst.RASTER_HGT: (dims, dem_as_array)}, coords=coords)
    dsm_as_ds[cst.EPSG] = 4326
    dsm_as_ds[cst.RESOLUTION] = resolution

    return dsm_as_ds

def get_time_ground_direction(
    img:str, x_loc:float=None, y_loc:float=None,
    y_offset:float=None, dem:str = None)-> np.ndarray:
    """
    For a given image, compute the direction of increasing acquisition
    time on ground.
    Done by two "img" localizations at "y" and "y+y_offset" values.

    :param img: Path to an image
    :param x_loc: x location in image for estimation (default=center)
    :param y_loc: y location in image for estimation (default=1/4)
    :param y_offset: y location in image for estimation (default=1/2)
    :param dem: DEM for direct localisation function
    :return: normalized direction vector as a numpy array
    """
    # Define x: image center,
    #        y: 1/4 of image,
    # y_offset: 3/4 of image if not defined
    img_size_x, img_size_y = utils.rasterio_get_size(img)
    if x_loc is None:
        x_loc = img_size_x/2
    if y_loc is None:
        y_loc = img_size_y/4
    if y_offset is None:
        y_offset = img_size_y/2

    # Check x, y, y_offset to be in image
    assert x_loc >= 0
    assert x_loc <= img_size_x
    assert y_loc >= 0
    assert y_loc <= img_size_y
    assert y_offset > 0
    assert y_loc + y_offset <= img_size_y

    # Get first coordinates of time direction vector
    lat1, lon1, __ = sensor_to_geo(img, x_loc, y_loc, dem=dem)
    # Get second coordinates of time direction vector
    lat2, lon2, __ = sensor_to_geo(img, x_loc, y_loc+y_offset, dem=dem)

    # Create and normalize the time direction vector
    vec = np.array([lon1-lon2, lat1-lat2])
    vec = vec/np.linalg.norm(vec)

    return vec

def sensor_to_geo(
    img:str, x_coord:float, y_coord:float, z_coord:float=None, dem:str=None,
    geoid:str=None, default_elevation:float=None) -> np.ndarray:
    """
    For a given image point, compute the latitude, longitude, altitude

    Be careful: When SRTM is used, the default elevation (altitude)
    doesn't work anymore (OTB function) when ConvertSensorToGeoPointFast
    is called again. Check the values.

    Advice: to be sure, use x,y,z inputs only

    :param img: Path to an image
    :param x_coord: X Coordinate in input image sensor
    :param y_coord: Y Coordinate in input image sensor
    :param z_coord: Z Altitude coordinate to take the image
    :param dem: if z not defined, take this DEM directory input
    :param geoid: if z and dem not defined, take GEOID directory input
    :param elevation: if z, dem, geoid not defined, take default elevation
    :return: Latitude, Longitude, Altitude coordinates as a numpy array
    """
    s2c_app = otb.Registry.CreateApplication("ConvertSensorToGeoPointFast")

    s2c_app.SetParameterString('in', img)
    s2c_app.SetParameterFloat('input.idx', x_coord)
    s2c_app.SetParameterFloat('input.idy', y_coord)

    if z_coord is not None:
        s2c_app.SetParameterFloat("input.idz", z_coord)
    elif dem is not None:
        s2c_app.SetParameterString("elevation.dem", dem)
    elif geoid is not None:
        s2c_app.SetParameterString("elevation.geoid", geoid)
    elif default_elevation is not None:
        s2c_app.SetParameterFloat("elevation.default", default_elevation)
    #else ConvertSensorToGeoPointFast have only X, Y and OTB configured GEOID

    s2c_app.Execute()

    lon = s2c_app.GetParameterFloat("output.idx")
    lat  = s2c_app.GetParameterFloat("output.idy")
    alt = s2c_app.GetParameterFloat("output.idz")

    return np.array([lat, lon , alt ])

def get_ground_direction(img:str, x_coord:float=None, y_coord:float=None,
                         z0_coord:float=None, z_coord:float=None )->np.ndarray:
    """
    For a given image (x,y) point, compute the direction vector to ground
    The function use sensor_to_geo and make a z variation to get
    a ground direction vector.
    By default, (x,y) is put at image center and z0, z at RPC geometric model
    limits.

    :param img: Path to an image
    :param x: X Coordinate in input image sensor
    :param y: Y Coordinate in input image sensor
    :param z0: Z altitude reference coordinate
    :param z: Z Altitude coordinate to take the image
    :return: (lat0,lon0,alt0, lat,lon,alt) origin and end vector coordinates
    """
    # Define x, y in image center if not defined
    img_size_x, img_size_y = utils.rasterio_get_size(img)
    if x_coord is None:
        x_coord = img_size_x/2
    if y_coord is None:
        y_coord = img_size_y/2
    # Check x, y to be in image
    assert x_coord >= 0
    assert x_coord <= img_size_x
    assert y_coord >= 0
    assert y_coord <= img_size_y

    # Define z and z0 from img RPC constraints if not defined
    (min_alt, max_alt) = utils.get_elevation_range_from_metadata(img)
    if z0_coord is None:
        z0_coord = min_alt
    if z_coord is None:
        z_coord = max_alt
    # Check z0 and z to be in RPC constraints
    assert z0_coord >= min_alt
    assert z0_coord <= max_alt
    assert z_coord >= min_alt
    assert z_coord <= max_alt

    # Get origin vector coordinate with z0 altitude
    lat0, lon0, alt0 = sensor_to_geo(img, x_coord, y_coord ,z_coord=z0_coord)
    # Get end vector coordinate with z altitude
    lat, lon, alt = sensor_to_geo(img, x_coord, y_coord, z_coord=z_coord)

    return np.array([lat0, lon0, alt0, lat, lon, alt])

def get_ground_angles(
        img1:str, img2:str,
        x1_coord:float=None, y1_coord:float=None,
        z1_0_coord:float=None, z1_coord:float=None,
        x2_coord:float=None, y2_coord:float=None,
        z2_0_coord:float=None, z2_coord:float=None )\
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given image (x,y) point, compute the Azimuth angle,
    Elevation angle (not the altitude !) and Range from Ground z0 perspective
    for both stereo image (img1: left and img2: right)

    Calculate also the convergence angle between the two satellites from ground.

    The function use get_ground_direction function to have coordinates of
    ground direction vector and compute angles and range.

    Ref: Jeong, Jaehoon. (2017).
    IMAGING GEOMETRY AND POSITIONING ACCURACY OF DUAL SATELLITE STEREO IMAGES:
    A REVIEW. ISPRS Annals of Photogrammetry, Remote Sensing and Spatial
    Information Sciences.
    IV-2/W4. 235-242. 10.5194/isprs-annals-IV-2-W4-235-2017.

    Perspectives: get bisector  elevation (BIE), and asymmetry angle

    :param img1: Path to left image1
    :param img2: Path to right image2
    :param x1_coord: X Coordinate in input left image1  sensor
    :param y1_coord: Y Coordinate in input left image1 sensor
    :param z1_0_coord: Left image1 Z altitude origin coordinate
        for ground direction vector
    :param z1_coord:  Left image1 Z altitude end coordinate
        for ground direction vector
    :param x2_coord: X Coordinate in input right image2 sensor
    :param y2_coord: Y Coordinate in input right image2 sensor
    :param z2_0_coord: Right image2 Z altitude origin coordinate
        for ground direction vector
    :param z2_coord: Right image2 Z altitude end coordinate
        for ground direction vector
    :return: Left Azimuth, Left Elevation Angle,
            Right Azimuth, Right Elevation Angle, Convergence Angle
    """

    # Get image1 <-> satellite vector from image2 metadata geometric model
    lat1_0, lon1_0, alt1_0, lat1, lon1, alt1 =\
        get_ground_direction(img1, x1_coord, y1_coord, z1_0_coord, z1_coord)
    # Get East North Up vector for left image1
    x1_e, y1_n, y1_u = enu1 =\
        projection.geo_to_enu(lat1, lon1, alt1, lat1_0, lon1_0, alt1_0)
    # Convert vector to Azimuth, Elevation, Range (unused)
    az1, elev_angle1, __ =\
        projection.enu_to_aer(x1_e, y1_n, y1_u)

    # Get image2 <-> satellite vector from image2 metadata geometric model
    lat2_0, lon2_0, alt2_0, lat2, lon2, alt2 =\
        get_ground_direction(img2, x2_coord, y2_coord, z2_0_coord, z2_coord)
    # Get East North Up vector for right image2
    x2_e, y2_n, y2_u = enu2 =\
        projection.geo_to_enu(lat2, lon2, alt2, lat2_0, lon2_0, alt2_0)
    # Convert ENU to Azimuth, Elevation, Range (unused)
    az2, elev_angle2, __ = projection.enu_to_aer(x2_e, y2_n, y2_u)

    # Get convergence angle from two enu vectors.
    convergence_angle=np.degrees(utils.angle_vectors(enu1, enu2))

    return az1, elev_angle1, az2, elev_angle2, convergence_angle

def project_coordinates_on_line(
        x_coord: Union[float, np.ndarray],
        y_coord: Union[float, np.ndarray],
        origin:np.ndarray,
        vec:np.ndarray) -> np.ndarray:
    """
    Project coordinates (x,y) on a line starting from origin with a
    direction vector vec, and return the euclidean distances between
    projected points and origin.

    :param x_coord: scalar or vector of coordinates x
    :type x_coord: float or np.array(float) of shape [n]
    :param y_coord: scalar or vector of coordinates x
    :type y_coord: float or np.array(float) of shape [n]
    :param origin: coordinates of origin point for line
    :type origin: list(float) or np.array(float) of size 2
    :param vec: direction vector of line
    :type vec: list(float) or np.array(float) of size 2
    :return: vector of distances of projected points to origin
    :rtype: numpy array of float
    """
    assert len(x_coord) == len(y_coord)
    assert len(origin) == 2
    assert len(vec) == 2

    vec_angle = math.atan2(vec[1],vec[0])
    point_angle = np.arctan2(y_coord-origin[1], x_coord-origin[0])
    proj_angle = point_angle - vec_angle
    dist_to_origin = np.sqrt(\
        np.square(x_coord-origin[0]) + np.square(y_coord-origin[1]))

    return dist_to_origin*np.cos(proj_angle)


def lowres_initial_dem_splines_fit(lowres_dsm_from_matches: xr.Dataset,
                                   lowres_initial_dem: xr.Dataset,
                                   origin: np.ndarray,
                                   time_direction_vector: np.ndarray,
                                   ext: int = 3,
                                   order: int = 3):
    """
    This function takes 2 datasets containing DSM and models the
    difference between the two as an UnivariateSpline along the
    direction given by origin and time_direction_vector. Internally,
    it looks for the highest smoothing factor that satisfies the rmse threshold.

    :param lowres_dsm_from_matches: Dataset containing the low resolution DSM
        obtained from matches, as returned by the
        rasterization.simple_rasterization_dataset function.
    :param lowres_initial_dem: Dataset containing the low resolution DEM,
        obtained by read_lowres_dem function,
        on the same grid as lowres_dsm_from_matches
    :param origin: coordinates of origin point for line
    :type origin: list(float) or np.array(float) of size 2
    :param time_direction_vector: direction vector of line
    :type time_direction_vector: list(float) or np.array(float) of size 2
    :param ext: behavior outside of interpolation domain
    :param order: spline order
    """
    # Initial DSM difference
    dsm_diff = lowres_initial_dem.hgt - lowres_dsm_from_matches.hgt

    # Form arrays of coordinates
    x_values_2d, y_values_2d = np.meshgrid(dsm_diff.x, dsm_diff.y)

    # Project coordinates on the line
    # of increasing acquisition time to form 1D coordinates
    # If there are sensor oscillations, they will occur in this direction
    linear_coords = project_coordinates_on_line(x_values_2d, y_values_2d,
                                                origin, time_direction_vector)

    # Form a 1D array with diff values indexed with linear coords
    linear_diff_array = xr.DataArray(
        dsm_diff.values.ravel(),
        coords={"l" : linear_coords.ravel()},
        dims = ("l")
    )
    linear_diff_array = linear_diff_array.dropna(dim='l')
    linear_diff_array = linear_diff_array.sortby('l')

    # Apply median filtering within cell matching input dsm resolution
    min_l = np.min(linear_diff_array.l)
    max_l = np.max(linear_diff_array.l)
    nbins = int(math.ceil((max_l-min_l)/(lowres_dsm_from_matches.resolution)))
    median_linear_diff_array =\
        linear_diff_array.groupby_bins('l',nbins).median()
    median_linear_diff_array = median_linear_diff_array.rename({'l_bins': 'l'})
    median_linear_diff_array = median_linear_diff_array.assign_coords(
        {'l' : np.array([d.mid for d in median_linear_diff_array.l.data])})

    count_linear_diff_array = linear_diff_array.groupby_bins('l', nbins).count()
    count_linear_diff_array = count_linear_diff_array.rename({'l_bins': 'l'})
    count_linear_diff_array = count_linear_diff_array.assign_coords(
        {'l' : np.array([d.mid for d in count_linear_diff_array.l.data])})

    # Filter measurements with insufficient amount of points
    median_linear_diff_array = median_linear_diff_array.where(
        count_linear_diff_array > 100).dropna(dim='l')

    if len(median_linear_diff_array) < 100:
        logging.warning("Insufficient amount of points along time direction "
                        "after measurements filtering to estimate correction "
                        "to fit initial DEM")
        return None

    # Apply butterworth lowpass filter to retrieve only the low frequency
    # (from example of doc: https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.signal.butter.html#scipy.signal.butter )
    b, a = butter(3, 0.05)
    zi_filter  = lfilter_zi(b, a)
    z_filter, _ = lfilter(b, a,
        median_linear_diff_array.values,
        zi=zi_filter*median_linear_diff_array.values[0])
    lfilter(b, a, z_filter, zi = zi_filter*z_filter[0])
    filtered_median_linear_diff_array = xr.DataArray(
        filtfilt(b, a,median_linear_diff_array.values),
        coords=median_linear_diff_array.coords
    )

    # Estimate performances of spline s = 100 * length of diff array
    smoothing_factor = 100 * len(filtered_median_linear_diff_array.l)
    splines = interpolate.UnivariateSpline(
        filtered_median_linear_diff_array.l,
        filtered_median_linear_diff_array.values,
        ext=ext, k=order, s=smoothing_factor
    )
    estimated_correction = xr.DataArray(
        splines(filtered_median_linear_diff_array.l),
        coords=filtered_median_linear_diff_array.coords
    )
    rmse = (filtered_median_linear_diff_array-estimated_correction).std(dim='l')

    target_rmse = 0.3

    # If RMSE is too high, try to decrease smoothness until it fits
    while rmse > target_rmse and smoothing_factor > 0.001:
        # divide s by 2
        smoothing_factor/=2

        # Estimate splines
        splines = interpolate.UnivariateSpline(
            filtered_median_linear_diff_array.l,
            filtered_median_linear_diff_array.values,
            ext=ext, k=order, s=smoothing_factor)

        # Compute RMSE
        estimated_correction = xr.DataArray(
            splines(filtered_median_linear_diff_array.l),
            coords=filtered_median_linear_diff_array.coords
        )

        rmse = (filtered_median_linear_diff_array \
            - estimated_correction).std(dim='l')

    logging.debug(
        "Best smoothing factor for splines regression: "
        "{} (rmse={})".format(smoothing_factor, rmse))

    # Return estimated spline
    return splines
