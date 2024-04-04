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

from json_checker import Checker

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("pc_denoising")
class PCDenoising(ApplicationTemplate, metaclass=ABCMeta):
    """
    denoising_method
    """

    available_applications: Dict = {}
    default_application = "none"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for denoising
        :return: a application_to_use object
        """

        denoising_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "PC denoising_method method not specified, "
                "default {} is used".format(denoising_method)
            )
        else:
            denoising_method = conf.get("method", cls.default_application)

        if denoising_method not in cls.available_applications:
            logging.error(
                "No denoising application named {} registered".format(
                    denoising_method
                )
            )
            raise KeyError(
                "No denoising application named {} registered".format(
                    denoising_method
                )
            )

        logging.info(
            "The denoising_method({}) application will be used".format(
                denoising_method
            )
        )

        return super(PCDenoising, cls).__new__(
            cls.available_applications[denoising_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PCDenoising

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    def get_triangulation_overload(self):
        """
        Get function to apply to overload point cloud dataset,
            in triangulation wrapper. This function must add layers with
            key root : cst.EPI_DENOISING_INFO_KEY_ROOT , to be propagated

        :return: fun(
                    xr_pc_dataset,
                    sensor_image_left,
                    sensor_image_right,
                    geomodel_left,
                    geomodel_right,
                    grid_left,
                    grid_right,
                    geometry_plugin,
                    disparity_map
            ):
                :param xr_pc_dataset: point cloud dataset
                :type xr_pc_dataset: xarray dataset
                :param sensor_image_left: sensor image left
                :type sensor_image_left: str
                :param sensor_image_right: sensor image right
                :type sensor_image_right: str
                :param geomodel_left: left geomodel
                :type geomodel_left: dict
                :param geomodel_right: right geomodel
                :type geomodel_right: dict
                :param grid_left: left grid
                :type: grid_left: CarsDataset
                :param grid_right: right grid
                :type: grid_right: CarsDataset
                :param geometry_plugin: geometry plugin
                :type: geometry_plugin: GeometryPlugin
                :param disparity_map: disparity plugin
                :type disparity_map: xarray Dataset
        """

        return identity_func

    @abstractmethod
    def run(
        self,
        point_cloud,
        orchestrator=None,
        pair_key="default",
        pair_folder=None,
    ):
        """
        Run denoising

        :param point_cloud: point cloud to denoise
        :param orchestrator: orchestrator
        :param pair_key: pair_key
        :param pair_folder: pair_folder

        :return: denoised point cloud
        """


def identity_func(
    xr_pc_dataset,  # pylint: disable=unused-argument
    sensor_image_left,  # pylint: disable=unused-argument
    sensor_image_right,  # pylint: disable=unused-argument
    geomodel_left,  # pylint: disable=unused-argument
    geomodel_right,  # pylint: disable=unused-argument
    grid_left,  # pylint: disable=unused-argument
    grid_right,  # pylint: disable=unused-argument
    geometry_plugin,  # pylint: disable=unused-argument
    disparity_map,  # pylint: disable=unused-argument
):
    """

    :param xr_pc_dataset: point cloud dataset
    :type xr_pc_dataset: xarray dataset
    :param sensor_image_left: sensor image left
    :type sensor_image_left: str
    :param sensor_image_right: sensor image right
    :type sensor_image_right: str
    :param geomodel_left: left geomodel
    :type geomodel_left: dict
    :param geomodel_right: right geomodel
    :type geomodel_right: dict
    :param grid_left: left grid
    :type: grid_left: CarsDataset
    :param grid_right: right grid
    :type: grid_right: CarsDataset
    :param geometry_plugin: geometry plugin
    :type: geometry_plugin: GeometryPlugin
    :param disparity_map: disparity plugin
    :type disparity_map: xarray Dataset

    """

    # no modification of dataset


class NonePCDenoising(PCDenoising, short_name="none"):
    """
    PCDenoising
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of PCDenoising

        :param conf: configuration for point cloud denoising
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf

        # get rasterization parameter
        overloaded_conf["method"] = conf.get("method", "bilateral")

        pc_denoising_schema = {
            "method": str,
        }

        # Check conf
        checker = Checker(pc_denoising_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        point_cloud,
        orchestrator=None,
        pair_key="default",
        pair_folder=None,
    ):
        """
        Run Denoising

        :param point_cloud: point cloud
        :type point_cloud: CarsDatasetet

        :return: denoised point cloud
        :rtype: CarsDatas
        """

        return point_cloud
