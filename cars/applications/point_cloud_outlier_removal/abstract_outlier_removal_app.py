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
this module contains the abstract PointsCloudOutlierRemoval application class.
"""

import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np

from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.applications.point_cloud_outlier_removal import (
    outlier_removal_constants as pr_cst,
)
from cars.core import constants as cst
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


@Application.register("point_cloud_outlier_removal")
class PointCloudOutlierRemoval(ApplicationTemplate, metaclass=ABCMeta):
    """
    PointCloudOutlierRemoval
    """

    available_applications: Dict = {}
    default_application = "statistical"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for points removal
        :return: a application_to_use object
        """

        points_removal_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Points removal method not specified, "
                "default {} is used".format(points_removal_method)
            )
        else:
            points_removal_method = conf.get("method", cls.default_application)

        if points_removal_method not in cls.available_applications:
            logging.error(
                "No Points removal application named {} registered".format(
                    points_removal_method
                )
            )
            raise KeyError(
                "No Points removal application named {} registered".format(
                    points_removal_method
                )
            )

        logging.info(
            "The PointCloudOutlierRemoval({}) application"
            " will be used".format(points_removal_method)
        )

        return super(PointCloudOutlierRemoval, cls).__new__(
            cls.available_applications[points_removal_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        # init orchestrator
        cls.orchestrator = None
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PointCloudOutlierRemoval

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
    def get_epipolar_margin(self):
        """
        Get epipolar margin to use

        :return: margin
        :rtype: int
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

    def __register_epipolar_dataset__(
        self,
        epipolar_point_cloud,
        depth_map_dir=None,
        dump_dir=None,
        app_name="",
        pair_key="PAIR_0",
    ):
        """
        Create dataset and registered the output in the orchestrator. the output
        X, Y and Z ground coordinates will be saved in depth_map_dir if the
        parameter is no None. Alternatively it will be saved to dump_dir if
        save_intermediate_data is set and depth_map_dir is None.

        :param epipolar_point_cloud:  Merged point cloud
        :type epipolar_point_cloud: CarsDataset
        :param depth_map_dir: output depth map directory. If None output will be
            written in dump_dir if intermediate data is requested
        :type depth_map_dir: str
        :param dump_dir: dump dir for output (except depth map) if intermediate
            data is requested
        :type dump_dir: str
        :param app_name: application name for file names
        :type app_name: str
        :param pair_key: name of current pair for index registration
        :type pair_key: str

        :return: Filtered point cloud
        :rtype: CarsDataset

        """

        # Create epipolar point cloud CarsDataset
        filtered_point_cloud = cars_dataset.CarsDataset(
            epipolar_point_cloud.dataset_type, name=app_name
        )

        filtered_point_cloud.create_empty_copy(epipolar_point_cloud)

        # Update attributes to get epipolar info
        filtered_point_cloud.attributes.update(epipolar_point_cloud.attributes)

        if depth_map_dir or self.used_config.get(
            application_constants.SAVE_INTERMEDIATE_DATA
        ):
            filtered_dir = (
                depth_map_dir if depth_map_dir is not None else dump_dir
            )
            safe_makedirs(filtered_dir)
            self.orchestrator.add_to_save_lists(
                os.path.join(filtered_dir, "X.tif"),
                cst.X,
                filtered_point_cloud,
                cars_ds_name="depth_map_x_filtered_" + app_name,
                dtype=np.float64,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(filtered_dir, "Y.tif"),
                cst.Y,
                filtered_point_cloud,
                cars_ds_name="depth_map_y_filtered_" + app_name,
                dtype=np.float64,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(filtered_dir, "Z.tif"),
                cst.Z,
                filtered_point_cloud,
                cars_ds_name="depth_map_z_filtered_" + app_name,
                dtype=np.float64,
            )

        # update depth map index if required
        if depth_map_dir:
            index = {
                cst.INDEX_DEPTH_MAP_X: os.path.join(pair_key, "X.tif"),
                cst.INDEX_DEPTH_MAP_Y: os.path.join(pair_key, "Y.tif"),
                cst.INDEX_DEPTH_MAP_Z: os.path.join(pair_key, "Z.tif"),
            }
            self.orchestrator.update_index({"depth_map": {pair_key: index}})

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos(
            [filtered_point_cloud]
        )

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pr_cst.CLOUD_OUTLIER_REMOVAL_RUN_TAG: {app_name: {}},
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        return filtered_point_cloud, saving_info

    def __register_pc_dataset__(
        self,
        merged_point_cloud=None,
        point_cloud_dir=None,
        dump_dir=None,
        app_name=None,
    ):
        """
        Create dataset and registered the output in the orchestrator. The
        point cloud dataset can be saved as laz using the save_laz_output
        option. Alternatively, the point cloud will be saved as laz and csv
        in the dump directory if the application save_intermediate data
        configuration parameter is set.

        :param merged_point_cloud:  Merged point cloud
        :type merged_point_cloud: CarsDataset
        :param point_cloud_dir: output depth map directory. If None output will
            be written in dump_dir if intermediate data is requested
        :type point_cloud_dir: str
        :param dump_dir: dump dir for output (except depth map) if intermediate
            data is requested
        :type dump_dir: str
        :param app_name: application name for file names
        :type app_name: str

        :return: Filtered point cloud
        :rtype: CarsDataset

        """
        if app_name is None:
            app_name = ""

        save_point_cloud_as_csv = self.used_config.get(
            application_constants.SAVE_INTERMEDIATE_DATA, False
        )
        # Save laz point cloud if save_intermediate_date is activated (dump_dir)
        # or if point_cloud_dir is provided (save as official product)
        save_point_cloud_as_laz = (
            point_cloud_dir is not None
            or self.used_config.get(
                application_constants.SAVE_INTERMEDIATE_DATA, False
            )
        )

        # Create CarsDataset
        filtered_point_cloud = cars_dataset.CarsDataset(
            "points", name="point_cloud_removal_" + app_name
        )

        # Get tiling grid
        filtered_point_cloud.create_empty_copy(merged_point_cloud)
        filtered_point_cloud.attributes = merged_point_cloud.attributes.copy()

        laz_pc_dir_name = None
        if save_point_cloud_as_laz:
            if point_cloud_dir is not None:
                laz_pc_dir_name = point_cloud_dir
            else:
                laz_pc_dir_name = os.path.join(dump_dir, "laz")
            safe_makedirs(laz_pc_dir_name)
            self.orchestrator.add_to_compute_lists(
                filtered_point_cloud,
                cars_ds_name="filtered_point_cloud_laz_" + app_name,
            )
        csv_pc_dir_name = None
        if save_point_cloud_as_csv:
            csv_pc_dir_name = os.path.join(dump_dir, "csv")
            safe_makedirs(csv_pc_dir_name)
            self.orchestrator.add_to_compute_lists(
                filtered_point_cloud,
                cars_ds_name="filtered_point_cloud_csv_" + app_name,
            )

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos(
            [filtered_point_cloud]
        )

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pr_cst.CLOUD_OUTLIER_REMOVAL_RUN_TAG: {},
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        return (
            filtered_point_cloud,
            laz_pc_dir_name,
            csv_pc_dir_name,
            saving_info,
        )

    @abstractmethod
    def run(
        self,
        merged_point_cloud,
        orchestrator=None,
        save_laz_output=False,
        depth_map_dir=None,
        point_cloud_dir=None,
        dump_dir=None,
        epsg=None,
    ):
        """
        Run PointCloudOutlierRemoval application.

        Creates a CarsDataset filled with new point cloud tiles.

        :param merged_point_cloud: merged point cloud
        :type merged_point_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used
        :param save_laz_output: save output point cloud as laz
        :type save_laz_output: bool
        :param output_dir: output depth map directory. If None output will be
            written in dump_dir if intermediate data is requested
        :type output_dir: str
        :param dump_dir: dump dir for output (except depth map) if intermediate
            data is requested
        :type dump_dir: str
        :param epsg: cartographic reference for the point cloud (array input)
        :type epsg: int

        :return: filtered merged point cloud
        :rtype: CarsDataset filled with xr.Dataset
        """
