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
Grids module:
contains functions used for epipolar grid creation and correction
"""

# TODO Refactor : we should have an abstract class definition with otb,
#      shareloc & libgeo as possible instances

# Standard imports
from __future__ import absolute_import

import logging
import math

# Third party imports
import numpy as np
from scipy import interpolate

# CARS imports
from cars.conf import output_prepare
from cars.core import constants as cst
from cars.core import projection

# TODO depends on another step (and a later one) : make it independent
from cars.lib.steps.triangulation import triangulate_matches


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


def compute_epipolar_grid_min_max(grid,
                                  epsg,
                                  conf,
                                  disp_min = None,
                                  disp_max = None):
    """
    Compute ground terrain location of epipolar grids at disp_min and disp_max

    :param grid: The epipolar grid to project
    :type grid: np.ndarray of shape (N,M,2)
    :param epsg: EPSG code of the terrain projection
    :type epsg: Int
    :param conf: Configuration dictionnary from prepare step
    :type conf: Dict
    :param disp_min: Minimum disparity
                     (if None, read from configuration dictionnary)
    :type disp_min: Float or None
    :param disp_max: Maximum disparity
                     (if None, read from configuration dictionnary)
    :type disp_max: Float or None
    :returns: a tuple of location grid at disp_min and disp_max
    :rtype: Tuple(np.ndarray, np.ndarray) same shape as grid param
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_configuration = conf\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_configuration\
                        [output_prepare.MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_configuration\
                        [output_prepare.MAXIMUM_DISPARITY_TAG]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Generate disp_min and disp_max matches
    matches_min = np.stack((grid[:,:,0].flatten(),
                            grid[:,:,1].flatten(),
                            grid[:,:,0].flatten()+disp_min,
                            grid[:,:,1].flatten()), axis=1)
    matches_max = np.stack((grid[:,:,0].flatten(),
                            grid[:,:,1].flatten(),
                            grid[:,:,0].flatten()+disp_max,
                            grid[:,:,1].flatten()), axis=1)

    # Generate corresponding points clouds
    pc_min = triangulate_matches(conf, matches_min)
    pc_max = triangulate_matches(conf, matches_max)

    # Convert to correct EPSG
    projection.points_cloud_conversion_dataset(pc_min, epsg)
    projection.points_cloud_conversion_dataset(pc_max, epsg)

    # Form grid_min and grid_max
    grid_min = np.concatenate((pc_min[cst.X].values,
                               pc_min[cst.Y].values),
                              axis=1)
    grid_max = np.concatenate((pc_max[cst.X].values,
                               pc_max[cst.Y].values),
                              axis=1)

    return grid_min, grid_max
