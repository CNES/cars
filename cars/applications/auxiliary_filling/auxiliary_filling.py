# !/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
this module contains the abstract AuxiliaryFilling application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("auxiliary_filling")
class AuxiliaryFilling(ApplicationTemplate, metaclass=ABCMeta):
    """
    AuxiliaryFilling abstract class
    """

    available_applications: Dict = {}
    default_application = "auxiliary_filling_from_sensors"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for auxiliary_filling
        :return: an application_to_use object
        """

        auxiliary_filling_method = cls.default_application

        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Auxiliary filling method not specified, "
                "default {} is used".format(auxiliary_filling_method)
            )
        else:
            auxiliary_filling_method = conf["method"]

        if auxiliary_filling_method not in cls.available_applications:
            logging.error(
                "No auxiliary_filling application named {} registered".format(
                    auxiliary_filling_method
                )
            )
            raise KeyError(
                "No auxiliary_filling application named {} registered".format(
                    auxiliary_filling_method
                )
            )

        logging.info(
            "The AuxiliaryFilling({}) application will be used".format(
                auxiliary_filling_method
            )
        )

        return super(AuxiliaryFilling, cls).__new__(
            cls.available_applications[auxiliary_filling_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    @abstractmethod
    def run(
        self, dsm_file, color_file, classif_file, dump_dir, orchestrator=None
    ):
        """
        Run Auxiliary filling
        """
