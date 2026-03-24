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
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.applications.sparse_matching.methods import (
    abstract_sparse_matching_method as asmm,
)

AbstractSparseMatchingMethod = asmm.AbstractSparseMatchingMethod


@Application.register("sparse_matching")
class SparseMatching(ApplicationTemplate, metaclass=ABCMeta):
    """
    SparseMatching
    """

    available_applications: Dict = {}
    default_application = "basic"
    default_method = "sift"

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

        if conf is None:
            conf = {}

        # init the method before the application
        conf["method"] = conf.get("method", self.default_method)
        # pylint: disable=abstract-class-instantiated
        self.sparse_matching_method = AbstractSparseMatchingMethod(conf)

        super().__init__(conf=conf)

    @abstractmethod
    def get_required_bands(self):
        """
        Get bands required by this application.

        :return: required bands for left and right image
        :rtype: dict
        """

    @abstractmethod
    def get_margins_strip_fun(
        self, disp_min=None, disp_max=None, method="sift"
    ):
        """
        Get margins function to use in resampling

        :param disp_min: disp min for info
        :param disp_max: disp max for info
        :param method: method for the margins

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

    @abstractmethod
    def get_margins_tile_fun(self, grid_left, disp_range_grid, method="sift"):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :type grid_left: dict
        :param disp_range_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi

        """

    @abstractmethod
    def filter_matches(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_matches_left,
        grid_left,
        grid_right,
        geom_plugin,
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
            - attributes containing "disp_lower_bound", "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"
        :type epipolar_matches_left: CarsDataset
        :param grid_left: left epipolar grid dict
        :type grid_left: dict
        :param grid_right: right epipolar grid dict
        :type grid_right: dict
        :param save_matches: true is matches needs to be saved
        :type save_matches: bool

        :return filtered matches
        :rtype: np.ndarray

        """

    @abstractmethod
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_image_left,
        epipolar_image_right,
        disp_to_alt_ratio,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        classif_bands_to_mask=None,
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with pandas.DataFrame ,
        corresponding to epipolar 2D disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        """
