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
contains functions used for epipolar grid  correction
"""

# Standard imports
from __future__ import absolute_import

import logging
import os

# Third party imports
import numpy as np
import pandas
from scipy import interpolate

import cars.applications.grid_generation.grid_constants as grid_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants

# CARS imports
from cars.data_structures import cars_dataset


def correct_grid_from_1d(grid, grid_correction_coef):
    """
    Correct grid from correction given in 1d

    param grid: grid to correct
    :type grid: CarsDataset
    :param grid_correction_coef: grid correction to apply
    :param grid_correction_coef: list(float), size 6
    """

    coefs_x = grid_correction_coef[:3]
    coefs_x.append(0.0)
    coefs_y = grid_correction_coef[3:6]
    coefs_y.append(0.0)
    grid_correction_coef = (
        np.array(coefs_x).reshape((2, 2)),
        np.array(coefs_y).reshape((2, 2)),
    )

    # Correct grid right with provided epipolar a priori
    corrected_grid_right = correct_grid(grid, grid_correction_coef)

    return corrected_grid_right


def correct_grid(grid, grid_correction):
    """
    Correct grid

    :param grid: grid to correct
    :type grid: CarsDataset
    :param grid_correction: grid correction to apply
    :param grid_correction: Tuple(np.ndarray, np.ndarray)
            (coefsx_2d, coefsy_2d) , each of size (2,2)

    """

    coefsx_2d, coefsy_2d = grid_correction

    right_grid = np.copy(grid[0, 0])
    origin = grid.attributes["grid_origin"]
    spacing = grid.attributes["grid_spacing"]

    # Form 3D array with grid positions
    x_values_1d = np.linspace(
        origin[0],
        origin[0] + right_grid.shape[0] * spacing[0],
        right_grid.shape[0],
    )
    y_values_1d = np.linspace(
        origin[1],
        origin[1] + right_grid.shape[1] * spacing[1],
        right_grid.shape[1],
    )
    x_values_2d, y_values_2d = np.meshgrid(y_values_1d, x_values_1d)

    # Compute corresponding point in sensor geometry (grid encodes (x_sensor -
    # x_epi,y_sensor - y__epi)
    source_points = right_grid

    # Interpolate the regression model at grid position
    correction_grid_x = np.polynomial.polynomial.polyval2d(
        x_values_2d, y_values_2d, coefsx_2d
    )
    correction_grid_y = np.polynomial.polynomial.polyval2d(
        x_values_2d, y_values_2d, coefsy_2d
    )

    # Compute corrected grid
    corrected_grid_x = source_points[:, :, 0] - correction_grid_x
    corrected_grid_y = source_points[:, :, 1] - correction_grid_y
    corrected_right_grid = np.stack(
        (corrected_grid_x, corrected_grid_y), axis=2
    )

    # create new cars ds
    corrected_grid_right = cars_dataset.CarsDataset("arrays")
    corrected_grid_right.attributes = grid.attributes
    corrected_grid_right.tiling_grid = grid.tiling_grid
    corrected_grid_right[0, 0] = corrected_right_grid

    return corrected_grid_right


def estimate_right_grid_correction(
    matches,
    grid_right,
    initial_cars_ds=None,
    save_matches=False,
    pair_folder="",
    pair_key="pair_0",
    orchestrator=None,
):
    """
    Estimates grid correction, and correct matches

    :param matches: matches
    :type matches: np.ndarray
    :param grid_right: grid to correct
    :type grid_right: CarsDataset
    :param save_matches: true is matches needs to be saved
    :type save_matches: bool
    :param pair_folder: folder used for current pair
    :type pair_folder: str

    :return: grid_correction to apply, corrected_matches, info before,
             info after
    :rtype: Tuple(np.ndarray, np.ndarray) , np.ndarray, dict, dict
            grid_correction is : (coefsx_2d, coefsy_2d) , each of size (2,2)

    """
    # Default orchestrator
    if orchestrator is None:
        # Create default sequential orchestrator for current application
        # be awere, no out_json will be shared between orchestrators
        # No files saved
        cars_orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )
    else:
        cars_orchestrator = orchestrator

    if matches.shape[0] < 100:
        logging.error(
            "Insufficient amount of matches found (< 100), can not safely "
            "estimate epipolar error correction"
        )

        raise ValueError(
            "Insufficient amount of matches found (< 100), can not safely "
            "estimate epipolar error correction"
        )

    # Get grids attributes
    right_grid = np.copy(grid_right[0, 0])
    origin = grid_right.attributes["grid_origin"]
    spacing = grid_right.attributes["grid_spacing"]

    # Form 3D array with grid positions
    x_values_1d = np.linspace(
        origin[0],
        origin[0] + right_grid.shape[0] * spacing[0],
        right_grid.shape[0],
    )
    y_values_1d = np.linspace(
        origin[1],
        origin[1] + right_grid.shape[1] * spacing[1],
        right_grid.shape[1],
    )
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
    sensor_matches_raw_x = interpolate.griddata(
        (np.ravel(x_values_2d), np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 0]),
        (matches_x2, matches_y2),
    )

    sensor_matches_raw_y = interpolate.griddata(
        (np.ravel(x_values_2d), np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 1]),
        (matches_x2, matches_y2),
    )

    # Simulate matches that have no epipolar error (i.e. y2 == y1) and map
    # them to sensor geometry
    sensor_matches_perfect_x = interpolate.griddata(
        (np.ravel(x_values_2d), np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 0]),
        (matches_x2, matches_y1),
    )

    sensor_matches_perfect_y = interpolate.griddata(
        (np.ravel(x_values_2d), np.ravel(y_values_2d)),
        np.ravel(source_points[:, :, 1]),
        (matches_x2, matches_y1),
    )

    # Compute epipolar error in sensor geometry in both direction
    epipolar_error_x = sensor_matches_perfect_x - sensor_matches_raw_x
    epipolar_error_y = sensor_matches_perfect_y - sensor_matches_raw_y

    # Output epipolar error stats for monitoring
    mean_epipolar_error = [np.mean(epipolar_error_x), np.mean(epipolar_error_y)]
    median_epipolar_error = [
        np.median(epipolar_error_x),
        np.median(epipolar_error_y),
    ]
    std_epipolar_error = [np.std(epipolar_error_x), np.std(epipolar_error_y)]
    rms_epipolar_error = np.mean(
        np.sqrt(
            epipolar_error_x * epipolar_error_x
            + epipolar_error_y * epipolar_error_y
        )
    )
    rmsd_epipolar_error = np.std(
        np.sqrt(
            epipolar_error_x * epipolar_error_x
            + epipolar_error_y * epipolar_error_y
        )
    )

    in_stats = {
        "mean_epipolar_error": mean_epipolar_error,
        "median_epipolar_error": median_epipolar_error,
        "std_epipolar_error": std_epipolar_error,
        "rms_epipolar_error": rms_epipolar_error,
        "rmsd_epipolar_error": rmsd_epipolar_error,
    }

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
            median_epipolar_error[1],
        )
    )

    # Perform bilinear regression for both component of epipolar error
    lstsq_input = np.array([matches_x2 * 0 + 1, matches_x2, matches_y2]).T
    coefsx, residx, __, __ = np.linalg.lstsq(
        lstsq_input, epipolar_error_x, rcond=None
    )
    coefsy, residy, __, __ = np.linalg.lstsq(
        lstsq_input, epipolar_error_y, rcond=None
    )

    # Normalize residuals by number of matches
    rmsex = np.sqrt(residx / matches.shape[0])
    rmsey = np.sqrt(residy / matches.shape[1])

    logging.debug(
        "Root Mean Square Error of correction estimation:"
        "rmsex={} pixels, rmsey={} pixels".format(rmsex, rmsey)
    )

    # Reshape coefs to 2D (expected by np.polynomial.polyval2d)
    coefsx_2d = np.ndarray((2, 2))
    coefsx_2d[0, 0] = coefsx[0]
    coefsx_2d[1, 0] = coefsx[1]
    coefsx_2d[0, 1] = coefsx[2]
    coefsx_2d[1, 1] = 0.0

    coefsy_2d = np.ndarray((2, 2))
    coefsy_2d[0, 0] = coefsy[0]
    coefsy_2d[1, 0] = coefsy[1]
    coefsy_2d[0, 1] = coefsy[2]
    coefsy_2d[1, 1] = 0.0

    grid_correction = (coefsx_2d, coefsy_2d)

    # Map corrected matches to sensor geometry
    sensor_matches_corrected_x = (
        sensor_matches_raw_x
        + np.polynomial.polynomial.polyval2d(matches_x2, matches_y2, coefsx_2d)
    )
    sensor_matches_corrected_y = (
        sensor_matches_raw_y
        + np.polynomial.polynomial.polyval2d(matches_x2, matches_y2, coefsy_2d)
    )

    # Map corrected matches to epipolar geometry
    epipolar_matches_corrected_x = interpolate.griddata(
        (np.ravel(source_points[:, :, 0]), np.ravel(source_points[:, :, 1])),
        np.ravel(x_values_2d),
        (sensor_matches_corrected_x, sensor_matches_corrected_y),
    )
    epipolar_matches_corrected_y = interpolate.griddata(
        (np.ravel(source_points[:, :, 0]), np.ravel(source_points[:, :, 1])),
        np.ravel(y_values_2d),
        (sensor_matches_corrected_x, sensor_matches_corrected_y),
    )

    corrected_matches = np.copy(matches)
    corrected_matches[:, 2] = epipolar_matches_corrected_x
    corrected_matches[:, 3] = epipolar_matches_corrected_y

    # Compute epipolar error in sensor geometry in both direction after
    # correction
    corrected_epipolar_error_x = (
        sensor_matches_perfect_x - sensor_matches_corrected_x
    )
    corrected_epipolar_error_y = (
        sensor_matches_perfect_y - sensor_matches_corrected_y
    )

    # Output corrected epipolar error stats for monitoring
    mean_corrected_epipolar_error = [
        np.mean(corrected_epipolar_error_x),
        np.mean(corrected_epipolar_error_y),
    ]
    median_corrected_epipolar_error = [
        np.median(corrected_epipolar_error_x),
        np.median(corrected_epipolar_error_y),
    ]
    std_corrected_epipolar_error = [
        np.std(corrected_epipolar_error_x),
        np.std(corrected_epipolar_error_y),
    ]
    rms_corrected_epipolar_error = np.mean(
        np.sqrt(
            corrected_epipolar_error_x * corrected_epipolar_error_x
            + corrected_epipolar_error_y * corrected_epipolar_error_y
        )
    )
    rmsd_corrected_epipolar_error = np.std(
        np.sqrt(
            corrected_epipolar_error_x * corrected_epipolar_error_x
            + corrected_epipolar_error_y * corrected_epipolar_error_y
        )
    )

    out_stats = {
        "mean_epipolar_error": mean_corrected_epipolar_error,
        "median_epipolar_error": median_corrected_epipolar_error,
        "std_epipolar_error": std_corrected_epipolar_error,
        "rms_epipolar_error": rms_corrected_epipolar_error,
        "rmsd_epipolar_error": rmsd_corrected_epipolar_error,
    }

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
            median_corrected_epipolar_error[1],
        )
    )

    corrected_epipolar_error = corrected_matches[:, 1] - corrected_matches[:, 3]
    logging.info(
        "Epipolar error after correction: mean = {:.3f} pix., "
        "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
            np.mean(corrected_epipolar_error),
            np.std(corrected_epipolar_error),
            np.max(np.fabs(corrected_epipolar_error)),
        )
    )

    # Export filtered matches
    matches_array_path = None
    if save_matches:
        logging.info("Writing matches file")
        if pair_folder is None:
            logging.error("Pair folder not provided")
        else:
            current_out_dir = pair_folder
        matches_array_path = os.path.join(
            current_out_dir, "corrected_filtered_matches.npy"
        )
        np.save(matches_array_path, corrected_matches)

    # Create CarsDataset containing corrected matches, with same tiling as input
    corrected_matches_cars_ds = None
    if initial_cars_ds is not None:
        corrected_matches_cars_ds = create_matches_cars_ds(
            corrected_matches, initial_cars_ds
        )

    # Update orchestrator out_json
    corrected_matches_infos = {
        application_constants.APPLICATION_TAG: {
            pair_key: {
                grid_cst.GRID_CORRECTION_TAG: {
                    grid_cst.CORRECTED_MATCHES_TAG: matches_array_path
                }
            }
        }
    }
    cars_orchestrator.update_out_info(corrected_matches_infos)

    return (
        grid_correction,
        corrected_matches,
        corrected_matches_cars_ds,
        in_stats,
        out_stats,
    )


def create_matches_cars_ds(corrected_matches, initial_cars_ds):
    """
    Create CarsDataset representing matches, from numpy matches.
    Matches are split into tiles, stored in pandas DataFrames

    Right CarsDataset is filled with Nones

    :param corrected_matches: matches
    :type corrected_matches: numpy array
    :param initial_cars_ds: cars dataset to use tiling from
    :type initial_cars_ds: CarsDataset

    :return new_matches_cars_ds
    :rtype: CarsDataset
    """

    # initialize CarsDataset
    new_matches_cars_ds = cars_dataset.CarsDataset("points")
    new_matches_cars_ds.create_empty_copy(initial_cars_ds)

    for row in range(new_matches_cars_ds.shape[0]):
        for col in range(new_matches_cars_ds.shape[1]):
            [
                row_min,
                row_max,
                col_min,
                col_max,
            ] = new_matches_cars_ds.tiling_grid[row, col, :]

            # Get corresponding matches
            tile_matches = corrected_matches[corrected_matches[:, 1] > row_min]
            tile_matches = tile_matches[tile_matches[:, 1] < row_max]
            tile_matches = tile_matches[tile_matches[:, 0] > col_min]
            tile_matches = tile_matches[tile_matches[:, 0] < col_max]

            # Create pandas DataFrame
            new_matches_cars_ds[row, col] = pandas.DataFrame(tile_matches)

    return new_matches_cars_ds
