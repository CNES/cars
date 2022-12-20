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
this module contains the abstract grid generation application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("grid_generation")
class GridGeneration(ApplicationTemplate, metaclass=ABCMeta):
    """
    AbstractGridGeneration
    """

    available_applications: Dict = {}
    default_application = "epipolar"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for grid generation
        :return: a application_to_use object
        """

        grid_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Grid generation method not specified, default "
                " {} is used".format(grid_method)
            )
        else:
            grid_method = conf["method"]

        if grid_method not in cls.available_applications:
            logging.error(
                "No GridGeneration application named {} registered".format(
                    grid_method
                )
            )
            raise KeyError(
                "No GridGeneration application named {} registered".format(
                    grid_method
                )
            )

        logging.info(
            "The GridGeneration {} application will be used".format(grid_method)
        )

        return super(GridGeneration, cls).__new__(
            cls.available_applications[grid_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of GridGeneration

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        image_left,
        image_right,
        orchestrator=None,
        pair_folder=None,
        srtm_dir=None,
        default_alt=None,
        geoid_path=None,
        pair_key="PAIR_0",
    ):
        """
        Run EpipolarGridGeneration application

        Create left and right grid CarsDataset filled with xarray.Dataset ,
        corresponding to left and right epipolar grids.

        :param image_left: left image
        :type image_left: dict
        :param image_right: right image
        :type image_right: dict
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator used
        :param srtm_dir: srtm directory
        :type srtm_dir: str
        :param default_alt: default altitude
        :type default_alt: float
        :param geoid_path: geoid path
        :type geoid_path: str
        :param pair_key: pair configuration id
        :type pair_key: str

        :return: left grid, right grid
        :rtype: Tuple(CarsDataset, CarsDataset)
        """
