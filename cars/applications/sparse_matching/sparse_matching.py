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
this module contains the abstract matching application class.
"""
import logging
import math
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core import constants as cst
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import safe_makedirs


@Application.register("sparse_matching")
class SparseMatching(ApplicationTemplate, metaclass=ABCMeta):
    """
    SparseMatching
    """

    available_applications: Dict = {}
    default_application = "sift"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        matching_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Sparse Matching method not specified, default "
                " {} is used".format(matching_method)
            )
        else:
            matching_method = conf.get("method", cls.default_application)

        if matching_method not in cls.available_applications:
            logging.error(
                "No matching application named {} registered".format(
                    matching_method
                )
            )
            raise KeyError(
                "No matching application named {} registered".format(
                    matching_method
                )
            )

        logging.info(
            "The SparseMatching({}) application will be used".format(
                matching_method
            )
        )

        return super(SparseMatching, cls).__new__(
            cls.available_applications[matching_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_applications[name] = cls

    def __init__(self, conf=None):
        """
        Init function of SparseMatching

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def get_disparity_margin(self):
        """
        Get disparity margin corresponding to sparse matches

        :return: margin in percent

        """

    @abstractmethod
    def get_strip_margin(self):
        """
        Get strip margin corresponding to sparse matches

        :return: margin

        """

    @abstractmethod
    def get_epipolar_error_upper_bound(self):
        """
        Get epipolar error upper bound corresponding to sparse matches

        :return: margin

        """

    @abstractmethod
    def get_epipolar_error_maximum_bias(self):
        """
        Get epipolar error lower bound corresponding to sparse matches

        :return: margin

        """

    @abstractmethod
    def get_matches_filter_knn(self):
        """
        Get matches_filter_knn :
        number of neighboors used to measure isolation of matches

        :return: matches_filter_knn

        """

    @abstractmethod
    def get_matches_filter_dev_factor(self):
        """
        Get matches_filter_dev_factor :
        factor of deviation in the formula
        to compute threshold of outliers

        :return: matches_filter_dev_factor

        """

    @abstractmethod
    def get_minimum_nb_matches(self):
        """
        Get minimum_nb_matches :
        get the minimum number of matches

        :return: minimum_nb_matches

        """

    def get_margins_fun(self, disp_min=None, disp_max=None, method="sift"):
        """
        Get margins function to use in resampling

        :param disp_min: disp min for info
        :param disp_max: disp max for info
        :param method: method for the margins

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

        # Compute margins
        corner = ["left", "up", "right", "down"]
        data = np.zeros(len(corner))
        col = np.arange(len(corner))
        margins = xr.Dataset(
            {"left_margin": (["col"], data)}, coords={"col": col}
        )
        margins["right_margin"] = xr.DataArray(data, dims=["col"])

        left_margin = self.get_strip_margin()

        if method == "sift":
            right_margin = self.get_strip_margin() + int(
                math.floor(
                    self.get_epipolar_error_upper_bound()
                    + self.get_epipolar_error_maximum_bias()
                )
            )
        else:
            right_margin = left_margin

        # Compute margins for left region
        margins["left_margin"].data = [0, left_margin, 0, left_margin]

        # Compute margins for right region
        margins["right_margin"].data = [0, right_margin, 0, right_margin]

        # add disp range info
        margins.attrs["disp_min"] = disp_min
        margins.attrs["disp_max"] = disp_max

        logging.info(
            "Margins added to left region for matching: {}".format(
                margins["left_margin"].data
            )
        )

        logging.info(
            "Margins added to right region for matching: {}".format(
                margins["right_margin"].data
            )
        )

        def margins_wrapper(  # pylint: disable=unused-argument
            row_min, row_max, col_min, col_max
        ):
            """
            Generates margins Dataset used in resampling

            :param row_min: row min
            :param row_max: row max
            :param col_min: col min
            :param col_max: col max

            :return: margins
            :rtype: xr.Dataset
            """

            # Constant margins for all tiles
            return margins

        return margins_wrapper

    def filter_matches(
        self,
        epipolar_matches_left,
        grid_left,
        grid_right,
        orchestrator=None,
        pair_key="pair_0",
        pair_folder=None,
        save_matches=False,
    ):
        """
        Transform matches CarsDataset to numpy matches, and filters matches

        :param cars_orchestrator: orchestrator
        :param epipolar_matches_left: matches. CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future pandas DataFrame containing:

                - data : (L, 4) shape matches
            - attributes containing "disp_lower_bound",  "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"
        :type epipolar_matches_left: CarsDataset
        :param grid_left: left epipolar grid
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid
        :type grid_right: CarsDataset
        :param save_matches: true is matches needs to be saved
        :type save_matches: bool

        :return filtered matches
        :rtype: np.ndarray

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

        if pair_folder is None:
            pair_folder = os.path.join(cars_orchestrator.out_dir, "tmp")

        epipolar_error_upper_bound = self.get_epipolar_error_upper_bound()
        epipolar_error_maximum_bias = self.get_epipolar_error_maximum_bias()

        # Compute grid correction

        # Concatenated matches
        list_matches = []
        for row in range(epipolar_matches_left.shape[0]):
            for col in range(epipolar_matches_left.shape[1]):
                # CarsDataset containing Pandas DataFrame, not Delayed anymore
                if epipolar_matches_left[row, col] is not None:
                    epipolar_matches = epipolar_matches_left[
                        row, col
                    ].to_numpy()

                    sensor_matches = AbstractGeometry.matches_to_sensor_coords(
                        grid_left,
                        grid_right,
                        epipolar_matches,
                        cst.MATCHES_MODE,
                    )
                    sensor_matches = np.concatenate(sensor_matches, axis=1)
                    matches = np.concatenate(
                        [
                            epipolar_matches,
                            sensor_matches,
                        ],
                        axis=1,
                    )
                    list_matches.append(matches)

        matches = np.concatenate(list_matches)

        raw_nb_matches = matches.shape[0]

        logging.info(
            "Raw number of matches found: {} matches".format(raw_nb_matches)
        )

        # Export matches
        raw_matches_array_path = None
        if save_matches:
            safe_makedirs(pair_folder)

            logging.info("Writing raw matches file")
            raw_matches_array_path = os.path.join(
                pair_folder, "raw_matches.npy"
            )
            np.save(raw_matches_array_path, matches)

        # Filter matches that are out of margin
        if epipolar_error_maximum_bias == 0:
            epipolar_median_shift = 0
        else:
            epipolar_median_shift = np.median(matches[:, 3] - matches[:, 1])

        matches = matches[
            ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
            >= -epipolar_error_upper_bound
        ]
        matches = matches[
            ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
            <= epipolar_error_upper_bound
        ]

        matches_discarded_message = (
            "{} matches discarded because their epipolar error "
            "is greater than --epipolar_error_upper_bound = {} pix"
        ).format(raw_nb_matches - matches.shape[0], epipolar_error_upper_bound)

        if epipolar_error_maximum_bias != 0:
            matches_discarded_message += (
                " considering a shift of {} pix".format(epipolar_median_shift)
            )

        logging.info(matches_discarded_message)

        filtered_matches_array_path = None
        if save_matches:
            logging.info("Writing filtered matches file")
            filtered_matches_array_path = os.path.join(
                pair_folder, "filtered_matches.npy"
            )
            np.save(filtered_matches_array_path, matches)

        # Retrieve number of matches
        nb_matches = matches.shape[0]

        # Check if we have enough matches
        # TODO: we could also make it a warning and continue
        # with uncorrected grid
        # and default disparity range
        if nb_matches < self.get_minimum_nb_matches():
            error_message_matches = (
                "Insufficient amount of matches found ({} < {}), "
                "can not safely estimate epipolar error correction "
                " and disparity range".format(
                    nb_matches, self.get_minimum_nb_matches()
                )
            )
            logging.error(error_message_matches)
            raise ValueError(error_message_matches)

        logging.info(
            "Number of matches kept for epipolar "
            "error correction: {} matches".format(nb_matches)
        )

        # Compute epipolar error
        epipolar_error = matches[:, 1] - matches[:, 3]
        epi_error_mean = np.mean(epipolar_error)
        epi_error_std = np.std(epipolar_error)
        epi_error_max = np.max(np.fabs(epipolar_error))
        logging.info(
            "Epipolar error before correction: mean = {:.3f} pix., "
            "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
                epi_error_mean,
                epi_error_std,
                epi_error_max,
            )
        )

        # Update orchestrator out_json
        raw_matches_infos = {
            application_constants.APPLICATION_TAG: {
                sm_cst.MATCH_FILTERING_TAG: {
                    pair_key: {
                        sm_cst.NUMBER_MATCHES_TAG: nb_matches,
                        sm_cst.RAW_NUMBER_MATCHES_TAG: raw_nb_matches,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MEAN: epi_error_mean,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_STD: epi_error_std,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MAX: epi_error_max,
                    }
                }
            }
        }
        cars_orchestrator.update_out_info(raw_matches_infos)

        return matches

    @abstractmethod
    def get_save_matches(self):
        """
        Get save_matches parameter

        :return: true is save_matches activated
        :rtype: bool
        """

    @abstractmethod
    def run(self, epipolar_image_left, epipolar_image_right, **kwargs):
        """
        Run Matching application.

        Create left and right CarsDataset filled with pandas.DataFrame ,
        corresponding to epipolar 2D disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        :param epipolar_image_left: tiled left epipolar
        :type epipolar_image_left: CarsDataset
        :param epipolar_image_right: tiled right epipolar
        :type epipolar_image_right: CarsDataset
        :param disp_to_alt_ratio: disp to alti ratio
        :type disp_to_alt_ratio: float
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair key id
        :type pair_key: str
        :param mask1_ignored_by_sift: values used in left mask to ignore
         in correlation
        :type mask1_ignored_by_sift: list
        :param mask2_ignored_by_sift: values used in right mask to ignore
         in correlation
        :type mask2_ignored_by_sift: list

        :return left matches, right matches
        :rtype: Tuple(CarsDataset, CarsDataset)
        """
