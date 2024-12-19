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
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


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
