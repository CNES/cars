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
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.data_structures import cars_dataset


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
        # init orchestrator
        cls.orchestrator = None
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
    def get_optimal_tile_size(
        self,
        max_ram_per_worker,
        superposing_point_clouds=1,
        point_cloud_resolution=0.5,
    ):
        """
        Get the optimal tile size to use, depending on memory available

        :param max_ram_per_worker: maximum ram available
        :type max_ram_per_worker: int
        :param superposing_point_clouds: number of point clouds superposing
        :type superposing_point_clouds: int
        :param point_cloud_resolution: resolution of point cloud
        :type point_cloud_resolution: float

        :return: optimal tile size in meter
        :rtype: float

        """

    def __register_dataset__(
        self,
        merged_points_cloud,
        save_points_cloud_as_laz,
        save_points_cloud_as_csv,
        app_name=None,
    ):
        """
        Create dataset and registered the output in the orchestrator

        :param merged_points_cloud:  Merged point cloud
        :type merged_points_cloud: CarsDataset
        :param save_points_cloud_as_laz: true if save as laz
        :type save_points_cloud_as_laz: bool
        :param save_points_cloud_as_csv: true if save as csv
        :type save_points_cloud_as_csv: bool
        :param app_name: application name for file names
        :type app_name: str

        :return: Filtered point cloud
        :rtype: CarsDataset

        """
        if app_name is None:
            app_name = ""

        # Create CarsDataset
        filtered_point_cloud = cars_dataset.CarsDataset("points")

        # Get tiling grid
        filtered_point_cloud.tiling_grid = merged_points_cloud.tiling_grid
        filtered_point_cloud.generate_none_tiles()
        filtered_point_cloud.attributes = merged_points_cloud.attributes.copy()

        # Save objects
        if save_points_cloud_as_laz or save_points_cloud_as_csv:
            # Points cloud file name
            # TODO in input conf file
            pc_file_name = os.path.join(
                self.orchestrator.out_dir,
                "points_cloud_post_" + app_name + "_removing",
            )
            self.orchestrator.add_to_save_lists(
                pc_file_name,
                None,
                filtered_point_cloud,
                cars_ds_name="filtered_merged_pc_" + app_name,
            )

        return filtered_point_cloud

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
