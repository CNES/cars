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
this module contains the abstract PointsCloudOutlierRemoving application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("point_cloud_outliers_removing")
class PointCloudOutliersRemoving(ApplicationTemplate, metaclass=ABCMeta):
    """
    PointCloudOutliersRemoving
    """

    available_applications: Dict = {}
    default_application = "statistical"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for points removing
        :return: a application_to_use object
        """

        points_removing_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Points removing method not specified, "
                "default {} is used".format(points_removing_method)
            )
        else:
            points_removing_method = conf["method"]

        if points_removing_method not in cls.available_applications:
            logging.error(
                "No Points removing application named {} registered".format(
                    points_removing_method
                )
            )
            raise KeyError(
                "No Points removing application named {} registered".format(
                    points_removing_method
                )
            )

        logging.info(
            "The PointCloudOutliersRemoving {} application"
            " will be used".format(points_removing_method)
        )

        return super(PointCloudOutliersRemoving, cls).__new__(
            cls.available_applications[points_removing_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PointCloudOutliersRemoving

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def get_on_ground_margin(self, resolution=0.5):
        """
        Get margins to use during point clouds fusion

        :return: margin
        :rtype: float

        """

    @abstractmethod
    def get_method(self):
        """
        Get margins to use during point clouds fusion

        :return: algorithm method
        :rtype: string

        """

    @abstractmethod
    def run(
        self,
        merged_points_cloud,
        orchestrator=None,
    ):
        """
        Run PointCloudOutliersRemoving application.

        Creates a CarsDataset filled with new point cloud tiles.

        :param merged_points_cloud: merged point cloud
        :type merged_points_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used

        :return: filtered merged points cloud
        :rtype: CarsDataset filled with xr.Dataset
        """
