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
this module contains the abstract pandora sparse matching application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("pandora_sparse_matching")
class PandoraSparseMatching(ApplicationTemplate, metaclass=ABCMeta):
    """
    PandoraSparseMatching
    """

    available_applications: Dict = {}
    default_application = "pandora"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: an application_to_use object
        """

        matching_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Pandora Sparse Matching method not specified, "
                "default {} is used".format(matching_method)
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
            "The AbstractPandoraSparseMatching({}) "
            "application will be used".format(matching_method)
        )

        return super(PandoraSparseMatching, cls).__new__(
            cls.available_applications[matching_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PandoraSparseMatching

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    def get_save_matches(self):
        """
        Get save_matches parameter

        :return: true is save_matches activated
        :rtype: bool
        """

    def get_margins_fun(self, disp_min=None, disp_max=None):
        """
        Get margins function to use in resampling

        :param disp_min: disp min for info
        :param disp_max: disp max for info

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

    def get_connection_val(self):
        """
        Get connection_val :
        distance to use to consider that two points are connected

        :return: connection_val

        """

    def get_nb_pts_threshold(self):
        """
        Get nb_pts_threshold :
        number of points to use to identify small clusters to filter

        :return: nb_pts_threshold

        """

    def get_matches_filter_knn(self):
        """
        Get matches_filter_knn :
        number of neighboors used to measure isolation of matches

        :return: matches_filter_knn

        """

    def get_matches_filter_dev_factor(self):
        """
        Get matches_filter_dev_factor :
        factor of deviation in the formula
        to compute threshold of outliers

        :return: matches_filter_dev_factor

        """

    @abstractmethod
    def run(
        self,
        epipolar_image_left,
        epipolar_image_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        disp_to_alt_ratio=None,
    ):
        """
        Run PandoraSparseMatching application.

        Get matches using pandora in low resolution

        :param epipolar_image_left: tiled left epipolar CarsDataset contains:

               - N x M Delayed tiles. \
                   Each tile will be a future xarray Dataset containing:

                   - data with keys : "im", "msk", "color"
                   - attrs with keys: "margins" with "disp_min" and "disp_max"\
                       "transform", "crs", "valid_pixels", "no_data_mask",\
                       "no_data_img"
               - attributes containing:
                   "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_image_left: CarsDataset
        :param epipolar_image_right: tiled right epipolar CarsDataset contains:

               - N x M Delayed tiles. \
                   Each tile will be a future xarray Dataset containing:

                   - data with keys : "im", "msk", "color"
                   - attrs with keys: "margins" with "disp_min" and "disp_max"
                       "transform", "crs", "valid_pixels", "no_data_mask",
                       "no_data_img"
               - attributes containing:
                   "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_image_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param disp_to_alt_ratio: disp to alti ratio used for performance map
        :type disp_to_alt_ratio: float

        :return: disparity map in lo resolution: \
           The CarsDataset contains:

           - N x M Delayed tiles.\
             Each tile will be a future xarray Dataset containing:
               - data with keys : "disp", "disp_msk"
               - attrs with keys: profile, window, overlaps
           - attributes containing:
               "largest_epipolar_region","opt_epipolar_tile_size",
                "disp_min_tiling", "disp_max_tiling"

        :rtype: CarsDataset
        """
