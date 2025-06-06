#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains the abstract ground_truth_reprojection
application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("ground_truth_reprojection")
class GroundTruthReprojection(ApplicationTemplate, metaclass=ABCMeta):
    """
    Epipolar matches ground truth computation

    """

    available_applications: Dict = {}
    default_application = "direct_loc"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises: KeyError when the required application is not registered
        :param conf: configuration for grid generation
        :return: an application_to_use object

        """

        ground_truth_computation_method = cls.default_application

        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Ground truth reprojection method not specified"
                ", default {} is used".format(ground_truth_computation_method)
            )
        else:
            ground_truth_computation_method = conf.get(
                "method", cls.default_application
            )

        if ground_truth_computation_method not in cls.available_applications:
            logging.error(
                "No GroundTruthReprojection application named {} "
                "registered".format(ground_truth_computation_method)
            )
            raise KeyError(
                "No GroundTruthReprojection application named {} "
                "registered".format(ground_truth_computation_method)
            )

        logging.info(
            "The GroundTruthReprojection({}) application will be "
            "used".format(ground_truth_computation_method)
        )

        return super(GroundTruthReprojection, cls).__new__(
            cls.available_applications[ground_truth_computation_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of Epipolar matches ground truth computation

        :param conf: configuration
        :return: an application_to_use object

        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        sensor_left,
        sensor_right,
        grid_left,
        grid_right,
        geom_plugin,
        geom_plugin_dem_median,
        disp_to_alt_ratio,
        auxiliary_values,
        auxiliary_interp,
        orchestrator=None,
        pair_folder=None,
    ):  # noqa: C901
        """
        Compute disparity maps from a DSM. This function will be run
        as a delayed task. If user want to correctly save dataset, the user must
        provide saving_info_left and right.  See cars_dataset.fill_dataset.

        :param geom_plugin_dem_median: Geometry plugin with dem median
        :type geom_plugin_dem_median: geometry_plugin
        :param sensor_left: Tiled sensor left image.
            Dict must contain keys: "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolute.
        :type sensor_left: CarsDataset
        :param grid_left: Grid left.
        :type grid_left: CarsDataset
        :param geom_plugin: Geometry plugin with user's DSM used to generate
            epipolar grids.
        :type geom_plugin: GeometryPlugin
        :param disp_to_alt_ratio: Disp to altitude ratio used for
            performance map.
        :type disp_to_alt_ratio: float
        :param orchestrator: Orchestrator used.
        :type orchestrator: Any
        :param pair_folder: Folder used for current pair.
        :type pair_folder: str
        """
