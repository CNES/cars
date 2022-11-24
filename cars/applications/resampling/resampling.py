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
this module contains the abstract resampling application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("resampling")
class Resampling(ApplicationTemplate, metaclass=ABCMeta):
    """
    Resampling
    """

    available_applications: Dict = {}
    default_application = "bicubic"

    def __new__(cls, orchestrator=None, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param orchestrator: orchestrator used
        :param conf: configuration for resampling
        :return: an application_to_use object
        """

        resampling_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Resampling method not specified, default"
                " {} is used".format(resampling_method)
            )
        else:
            resampling_method = conf["method"]

        if resampling_method not in cls.available_applications:
            logging.error(
                "No resampling application named {} registered".format(
                    resampling_method
                )
            )
            raise KeyError(
                "No resampling application named {} registered".format(
                    resampling_method
                )
            )

        logging.info(
            "The Resampling {} application will be used".format(
                resampling_method
            )
        )

        return super(Resampling, cls).__new__(
            cls.available_applications[resampling_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302

        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    @abstractmethod
    def run(
        self,
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        margins=None,
        optimum_tile_size=None,
        add_color=True,
    ):
        """
        Run resampling application.

        Creates left and right CarsDataset filled with xarray.Dataset,
        corresponding to sensor images resampled in epipolar geometry.

        :param sensor_images_left: tiled sensor left image
        :type sensor_images_left: CarsDataset
        :param sensor_images_right: tiled sensor right image
        :type sensor_images_right: CarsDataset
        :param grid_left: left epipolar grid
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid
        :type grid_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: directory to save files to
        :param pair_key: pair  id
        :type pair_key: str
        :param margins: margins to use
        :type margins: xr.Dataset
        :param optimum_tile_size: optimum tile size to use
        :type optimum_tile_size: int
        :param add_color: add color image to dataset
        :type add_color: bool

        :return: left epipolar image, right epipolar image
        :rtype: Tuple(CarsDataset, CarsDataset)
        """
