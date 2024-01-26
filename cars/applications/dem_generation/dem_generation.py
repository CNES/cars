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


@Application.register("dem_generation")
class DemGeneration(ApplicationTemplate, metaclass=ABCMeta):
    """
    DemGeneration
    """

    available_applications: Dict = {}
    default_application = "dichotomic"

    def __new__(cls, orchestrator=None, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param orchestrator: orchestrator used
        :param conf: configuration for resampling
        :return: an application_to_use object
        """

        dem_generation_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "MntGeneration method not specified, default"
                " {} is used".format(dem_generation_method)
            )
        else:
            dem_generation_method = conf["method"]

        if dem_generation_method not in cls.available_applications:
            logging.error(
                "No dem_generation application named {} registered".format(
                    dem_generation_method
                )
            )
            raise KeyError(
                "No dem_generation application named {} registered".format(
                    dem_generation_method
                )
            )

        logging.info(
            "The DemGeneration {} application will be used".format(
                dem_generation_method
            )
        )

        return super(DemGeneration, cls).__new__(
            cls.available_applications[dem_generation_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of MntGeneration

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(self, triangulated_matches_list, output_dir):
        """
        Run dem generation using matches

        :param triangulated_matches_list: list of triangulated matches
            positions must be in a metric system
        :type triangulated_matches_list: list(pandas.Dataframe)
        :param output_dir: directory to save dem
        :type output_dir: str

        :return: dem data computed with mean, min and max.
            dem is also saved in disk, and paths are available in attributes.
            (DEM_MEDIAN_PATH, DEM_MIN_PATH, DEM_MAX_PATH)
        :rtype: CarsDataset
        """
