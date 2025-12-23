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
this module contains the epipolar grid correction application class.
"""
# Standard imports
from __future__ import absolute_import

# Standard imports
import logging
import os

import numpy as np

# Third party imports
import rasterio as rio

# Third party imports
from json_checker import And, Checker
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay  # pylint: disable=E0611

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_correction.abstract_grid_correction_app import (
    GridCorrection,
)
from cars.applications.grid_generation import (
    grid_generation_algo,
)
from cars.applications.grid_generation import (
    grid_generation_constants as grid_constants,
)

# CARS imports
from cars.core.utils import safe_makedirs
from cars.orchestrator.cluster.log_wrapper import cars_profile


class GridCorrectionApp(GridCorrection, short_name="default"):
    """
    EpipolarGridGeneration
    """

    def __init__(self, conf=None):
        """
        Init function of EpipolarGridGeneration

        :param conf: configuration for grid generation
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.nb_matches = self.used_config["nb_matches"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # Init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "epipolar")
        overloaded_conf["nb_matches"] = conf.get("nb_matches", 90)
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        grid_generation_schema = {
            "method": str,
            "nb_matches": And(int, lambda x: x > 0),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(grid_generation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_minimum_nb_matches(self):
        """
        Get the minimul number of matches required for grid correction
        """

        return self.nb_matches

    @cars_profile(name="Correct grid")
    def correct_grid(self, grid, grid_correction, pair_folder, save_grid=None):
        """
        Correct grid

        :param grid: grid to correct
        :type grid: dict
        :param grid_correction: grid correction to apply
        :type grid_correction: Tuple(np.ndarray, np.ndarray)
                (coefsx_2d, coefsy_2d) , each of size (2,2)
        :param pair_folder: directory where grids are saved: either in
            pair_folder/tmp, or at the root of pair_folder if save_grid is True
        :type pair_folder: str
        :param save_grid: if True grids are saved in root of pair_folder,
            instead of tmp
        :type save_grid: bool
        """

        coefsx_2d, coefsy_2d = grid_correction

        with rio.open(grid["path"]) as right_grid:
            right_grid_row = right_grid.read(1)
            right_grid_col = right_grid.read(2)

        origin = grid["grid_origin"]
        spacing = grid["grid_spacing"]

        # Form 3D array with grid positions
        x_values_1d = np.linspace(
            origin[0],
            origin[0] + right_grid_row.shape[0] * spacing[0],
            right_grid_row.shape[0],
        )
        y_values_1d = np.linspace(
            origin[1],
            origin[1] + right_grid_row.shape[1] * spacing[1],
            right_grid_row.shape[1],
        )
        x_values_2d, y_values_2d = np.meshgrid(y_values_1d, x_values_1d)

        # Compute corresponding point in sensor geometry
        # (grid encodes (x_sensor -
        # x_epi,y_sensor - y__epi)

        # Interpolate the regression model at grid position
        correction_grid_x = np.polynomial.polynomial.polyval2d(
            x_values_2d, y_values_2d, coefsx_2d
        )
        correction_grid_y = np.polynomial.polynomial.polyval2d(
            x_values_2d, y_values_2d, coefsy_2d
        )

        # Compute corrected grid
        corrected_grid_x = right_grid_row - correction_grid_x
        corrected_grid_y = right_grid_col - correction_grid_y
        corrected_right_grid = np.stack(
            (corrected_grid_x, corrected_grid_y), axis=2
        )

        # create new grid dict
        corrected_grid_right = grid.copy()

        # Dump corrected grid
        grid_origin = grid["grid_origin"]
        grid_spacing = grid["grid_spacing"]

        # Get save folder (permanent or temporay according to
        # save_grid parameter)
        if save_grid:
            safe_makedirs(pair_folder)
            save_folder = os.path.join(
                pair_folder, "corrected_right_epi_grid.tif"
            )
        else:
            safe_makedirs(os.path.join(pair_folder, "tmp"))
            save_folder = os.path.join(
                pair_folder, "tmp", "corrected_right_epi_grid.tif"
            )

        grid_generation_algo.write_grid(
            corrected_right_grid, save_folder, grid_origin, grid_spacing
        )

        corrected_grid_right["path"] = save_folder

        return corrected_grid_right

    @cars_profile(name="Epi Grid Generation")
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        matches,
        grid_right,
        save_matches=False,
        minimum_nb_matches=100,
        pair_folder="",
        pair_key="pair_0",
        orchestrator=None,
        save_corrected_grid=None,
    ):
        """
        Estimates grid correction, and correct matches

        :param matches: matches
        :type matches: np.ndarray
        :param grid_right: grid to correct
        :type grid_right: dict
        :param save_matches: true is matches needs to be saved
        :type save_matches: bool
        :param minimum_nb_matches: minimum number of matches required
        :type minimum_nb_matches: int
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

        if matches.shape[0] < minimum_nb_matches:
            logging.error(
                "Insufficient amount of matches found"
                ", can not safely estimate epipolar error correction"
            )

            raise ValueError(
                f"Insufficient amount of matches found (< {minimum_nb_matches})"
                ", can not safely estimate epipolar error correction"
            )

        # Get grids attributes
        with rio.open(grid_right["path"]) as right_grid:
            right_grid_row = right_grid.read(1)
            right_grid_col = right_grid.read(2)

        origin = grid_right["grid_origin"]
        spacing = grid_right["grid_spacing"]

        # Form 3D array with grid positions
        x_values_1d = np.arange(
            origin[0],
            origin[0] + right_grid_row.shape[0] * spacing[0],
            spacing[0],
        )
        y_values_1d = np.arange(
            origin[1],
            origin[1] + right_grid_row.shape[1] * spacing[1],
            spacing[1],
        )
        x_values_2d, y_values_2d = np.meshgrid(y_values_1d, x_values_1d)

        # Compute corresponding point in sensor geometry
        # (grid encodes (x_sensor -
        # x_epi,y_sensor - y__epi)

        # Extract matches for convenience
        matches_y1 = matches[:, 1]
        matches_x2 = matches[:, 2]
        matches_y2 = matches[:, 3]

        # Map real matches to sensor geometry
        points = np.column_stack((np.ravel(x_values_2d), np.ravel(y_values_2d)))

        triangulation = Delaunay(points)

        values = np.ravel(right_grid_row)

        interpolator = LinearNDInterpolator(triangulation, values)
        sensor_matches_raw_x = interpolator(matches_x2, matches_y2)

        # Simulate matches that have no epipolar error (i.e. y2 == y1) and map
        # them to sensor geometry
        sensor_matches_perfect_x = interpolator(matches_x2, matches_y1)

        values = np.ravel(right_grid_col)
        interpolator = LinearNDInterpolator(triangulation, values)
        sensor_matches_raw_y = interpolator(matches_x2, matches_y2)

        sensor_matches_perfect_y = interpolator(matches_x2, matches_y1)

        # Compute epipolar error in sensor geometry in both direction
        epipolar_error_x = sensor_matches_perfect_x - sensor_matches_raw_x
        epipolar_error_y = sensor_matches_perfect_y - sensor_matches_raw_y

        # Output epipolar error stats for monitoring
        mean_epipolar_error = [
            np.mean(epipolar_error_x),
            np.mean(epipolar_error_y),
        ]
        median_epipolar_error = [
            np.median(epipolar_error_x),
            np.median(epipolar_error_y),
        ]
        std_epipolar_error = [
            np.std(epipolar_error_x),
            np.std(epipolar_error_y),
        ]
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
        nan_mask = np.logical_and(
            ~np.isnan(epipolar_error_x), ~np.isnan(epipolar_error_y)
        )
        lstsq_input = np.array(
            [
                matches_x2[nan_mask] * 0 + 1,
                matches_x2[nan_mask],
                matches_y2[nan_mask],
            ]
        ).T
        coefsx, residx, __, __ = np.linalg.lstsq(
            lstsq_input, epipolar_error_x[nan_mask], rcond=None
        )
        coefsy, residy, __, __ = np.linalg.lstsq(
            lstsq_input, epipolar_error_y[nan_mask], rcond=None
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
            + np.polynomial.polynomial.polyval2d(
                matches_x2, matches_y2, coefsx_2d
            )
        )
        sensor_matches_corrected_y = (
            sensor_matches_raw_y
            + np.polynomial.polynomial.polyval2d(
                matches_x2, matches_y2, coefsy_2d
            )
        )

        # Map corrected matches to epipolar geometry
        points = np.column_stack(
            (np.ravel(right_grid_row), np.ravel(right_grid_col))
        )
        triangulation = Delaunay(points)

        values = np.ravel(x_values_2d)
        interpolator = LinearNDInterpolator(triangulation, values)
        epipolar_matches_corrected_x = interpolator(
            sensor_matches_corrected_x, sensor_matches_corrected_y
        )

        values = np.ravel(y_values_2d)
        interpolator = LinearNDInterpolator(triangulation, values)
        epipolar_matches_corrected_y = interpolator(
            sensor_matches_corrected_x, sensor_matches_corrected_y
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

        corrected_epipolar_error = (
            corrected_matches[:, 1] - corrected_matches[:, 3]
        )
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
        current_out_dir = None
        if save_matches:
            logging.info("Writing matches file")
            if pair_folder is None:
                logging.error("Pair folder not provided")
            else:
                safe_makedirs(pair_folder)
                current_out_dir = pair_folder
            matches_array_path = os.path.join(
                current_out_dir, "corrected_filtered_matches.npy"
            )
            np.save(matches_array_path, corrected_matches)

        # Update orchestrator out_json
        corrected_matches_infos = {
            application_constants.APPLICATION_TAG: {
                grid_constants.GRID_CORRECTION_TAG: {pair_key: {}}
            }
        }
        cars_orchestrator.update_out_info(corrected_matches_infos)

        corrected_grid_right = self.correct_grid(
            grid_right,
            grid_correction,
            pair_folder,
            save_corrected_grid,
        )

        return (
            corrected_grid_right,
            corrected_matches,
            in_stats,
            out_stats,
        )
