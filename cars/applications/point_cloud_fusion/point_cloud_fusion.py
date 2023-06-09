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
this module contains the abstract PointsCloudFusion application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("point_cloud_fusion")
class PointCloudFusion(ApplicationTemplate, metaclass=ABCMeta):
    """
    PointsCloudFusion
    """

    available_applications: Dict = {}
    default_application = "mapping_to_terrain_tiles"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for fusion
        :return: a application_to_use object
        """

        fusion_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Fusion method not specified, "
                "default {} is used".format(fusion_method)
            )
        else:
            fusion_method = conf["method"]

        if fusion_method not in cls.available_applications:
            logging.error(
                "No Fusion application named {} registered".format(
                    fusion_method
                )
            )
            raise KeyError(
                "No Fusion application named {} registered".format(
                    fusion_method
                )
            )

        logging.info(
            "The PointCloudFusion {} application will be used".format(
                fusion_method
            )
        )

        return super(PointCloudFusion, cls).__new__(
            cls.available_applications[fusion_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PointCloudFusion

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        list_epipolar_points_cloud,
        bounds,
        epsg,
        orchestrator=None,
        margins=0,
        optimal_terrain_tile_width=500,
    ):
        """
        Run EpipolarCloudFusion application.

        Creates a CarsDataset corresponding to the merged points clouds,
        tiled with the terrain grid used during rasterization.

        :param list_epipolar_points_cloud: list with points clouds
        :type list_epipolar_points_cloud: list(CarsDataset) filled with
          xr.Dataset
        :param bounds: terrain bounds
        :type bounds: list
        :param epsg: epsg to use
        :type epsg: str
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param margins: margins to add to tiles
        :type margins: float
        :param optimal_terrain_tile_width: optimal terrain tile width
        :type optimal_terrain_tile_width: int

        :return: Merged points clouds
        :rtype: CarsDataset filled with pandas.DataFrame

        """
