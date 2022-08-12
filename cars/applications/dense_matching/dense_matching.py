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
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("dense_matching")
class DenseMatching(ApplicationTemplate, metaclass=ABCMeta):
    """
    AbstractDenseMatching
    """

    available_applications: Dict = {}
    default_application = "census_sgm"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        matching_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Dense Matching method not specified, "
                "default {} is used".format(matching_method)
            )
        else:
            matching_method = conf["method"]

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
            "The AbstractDenseMatching {} application will be used".format(
                matching_method
            )
        )

        return super(DenseMatching, cls).__new__(
            cls.available_applications[matching_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302

        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_applications[name] = cls

    @abstractmethod
    def get_optimal_tile_size(self, disp_min, disp_max):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_min: minimum disparity
        :param disp_max: maximum disparity

        :return: optimal tile size

        """

    @abstractmethod
    def get_margins(self, grid_left, disp_min=None, disp_max=None):
        """
        Get Margins needeed by matching method, to use during resampling

        :param grid_left: left epipolar grid
        :param disp_min: minimum disparity
        :param disp_max: maximum disparity

        :return: margins, updated disp_min, updated disp_max

        """

    @abstractmethod
    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        mask1_ignored_by_corr: List[int] = None,
        mask2_ignored_by_corr: List[int] = None,
        mask1_set_to_ref_alt: List[int] = None,
        mask2_set_to_ref_alt: List[int] = None,
        disp_min=None,
        disp_max=None,
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with xarray.Dataset ,
        corresponding to epipolar disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        :param epipolar_images_left: tiled left epipolar
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar
        :type epipolar_images_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param mask1_ignored_by_corr: values used in left mask to ignore
         in correlation
        :type mask1_ignored_by_corr: list
        :param mask2_ignored_by_corr: values used in right mask to ignore
         in correlation
        :type mask2_ignored_by_corr: list
        :param mask1_set_to_ref_alt: values used in left mask to altitude
         to ref
        :type mask1_set_to_ref_alt: list
        :param mask2_set_to_ref_alt: values used in right mask to altitude
         to ref
        :type mask2_set_to_ref_alt: list
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int

        :return: left disparity map, right disparity map
        :rtype: Tupple(CarsDataset, CarsDataset)
        """
