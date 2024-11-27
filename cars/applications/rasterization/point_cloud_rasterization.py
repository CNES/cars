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
this module contains the abstract PointCloudRasterization application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("point_cloud_rasterization")
class PointCloudRasterization(ApplicationTemplate, metaclass=ABCMeta):
    """
    PointCloudRasterization
    """

    available_applications: Dict = {}
    default_application = "simple_gaussian"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for rasterization
        :return: a application_to_use object
        """

        rasterization_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Rasterisation method not specified, "
                "default {} is used".format(rasterization_method)
            )
        else:
            rasterization_method = conf.get("method", cls.default_application)

        if rasterization_method not in cls.available_applications:
            logging.error(
                "No rasterization application named {} registered".format(
                    rasterization_method
                )
            )
            raise KeyError(
                "No rasterization application named {} registered".format(
                    rasterization_method
                )
            )

        logging.info(
            "The PointCloudRasterization({}) application will be used".format(
                rasterization_method
            )
        )

        return super(PointCloudRasterization, cls).__new__(
            cls.available_applications[rasterization_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PointCloudRasterization

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def get_margins(self, resolution):
        """
        Get the margin to use for terrain tiles

        :param resolution: resolution of raster data (in target CRS unit)
        :type epsg: float

        :return: margin in meters or degrees
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

    @abstractmethod
    def run(
        self,
        point_clouds,
        epsg,
        resolution,
        orchestrator=None,
        dsm_file_name=None,
        color_file_name=None,
        mask_file_name=None,
        classif_file_name=None,
        performance_map_file_name=None,
        contributing_pair_file_name=None,
        filling_file_name=None,
        color_dtype=None,
        dump_dir=None,
    ):
        """
        Run PointCloudRasterisation application.

        Creates a CarsDataset filled with dsm tiles.

        :param point_clouds: merged point cloud
        :type point_clouds: CarsDataset filled with pandas.DataFrame
        :param epsg: epsg of raster data
        :type epsg: str
        :param resolution: resolution of raster data (in target CRS unit)
        :type epsg: float
        :param orchestrator: orchestrator used
        :param dsm_file_name: path of dsm
        :type dsm_file_name: str
        :param color_file_name: path of color
        :type color_file_name: str
        :param mask_file_name: path of color
        :type mask_file_name: str
        :param classif_file_name: path of color
        :type classif_file_name: str
        :param performance_map_file_name: path of confidence file
        :type performance_map_file_name: str
        :param contributing_pair_file_name: path of contributing pair file
        :type contributing_pair_file_name: str
        :param filling_file_name: path of filling file
        :type filling_file_name: str
        :param color_dtype: output color image type
        :type color_dtype: str (numpy type)
        :param dump_dir: directory used for outputs with no associated filename
        :type dump_dir: str

        :return: raster DSM
        :rtype: CarsDataset filled with xr.Dataset
        """
