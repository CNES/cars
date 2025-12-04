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

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("sensors_subsampling")
class SensorsSubsampling(ApplicationTemplate, metaclass=ABCMeta):
    """
    SensorsSubsampling
    """

    available_applications: Dict = {}
    default_application = "rasterio"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        subsampling_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "subsampling method not specified, default "
                " {} is used".format(subsampling_method)
            )
        else:
            subsampling_method = conf.get("method", cls.default_application)

        if subsampling_method not in cls.available_applications:
            logging.error(
                "No subsampling application named {} registered".format(
                    subsampling_method
                )
            )
            raise KeyError(
                "No subsampling application named {} registered".format(
                    subsampling_method
                )
            )

        logging.info(
            "The subsampling({}) application will be used".format(
                subsampling_method
            )
        )

        return super(SensorsSubsampling, cls).__new__(
            cls.available_applications[subsampling_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_applications[name] = cls

    def __init__(self, conf=None):
        """
        Init function of SensorsSubsampling

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        id_image,
        sensor_dict,
        resolution,
        out_directory,
        orchestrator,
    ):
        """
        Run Subsampling application.

        """
