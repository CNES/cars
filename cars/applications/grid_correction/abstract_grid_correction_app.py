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


@Application.register("grid_correction")
class GridCorrection(ApplicationTemplate, metaclass=ABCMeta):
    """
    AbstractGridCorrection
    """

    available_applications: Dict = {}
    default_application = "default"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for grid generation
        :return: a application_to_use object
        """

        grid_method = cls.default_application

        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Grid correction method not specified, default "
                " {} is used".format(grid_method)
            )
        else:
            grid_method = conf.get("method", cls.default_application)

        if grid_method not in cls.available_applications:
            logging.error(
                "No GridCorrection application named {} registered".format(
                    grid_method
                )
            )
            raise KeyError(
                "No GridCorrection application named {} registered".format(
                    grid_method
                )
            )

        logging.info(
            "The GridCorrection({}) application will be used".format(
                grid_method
            )
        )

        return super(GridCorrection, cls).__new__(
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
    def run(  # pylint: disable=too-many-positional-arguments
        self,
    ):
        """
        Run GridCorrection application
        """
