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
        if bool(conf) is False:
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
        cls.available_applications[short_name] = cls

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
        factor ofdeviation in the formula
        to compute threshold of outliers

        :return: matches_filter_dev_factor

        """

    @abstractmethod
    def get_margins_fun(self):
        """
        Get margins function to use in resampling

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

    @abstractmethod
    def get_save_matches(self):
        """
        Get save_matches parameter

        :return: true is save_matches activated
        :rtype: bool
        """

    @abstractmethod
    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        disp_to_alt_ratio,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        mask1_ignored_by_sift: List[int] = None,
        mask2_ignored_by_sift: List[int] = None,
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with pandas.DataFrame ,
        corresponding to epipolar 2D disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        :param epipolar_images_left: tiled left epipolar
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar
        :type epipolar_images_right: CarsDataset
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
