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
this module contains the abstract dsm filling application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("dsm_filling")
class DsmFilling(ApplicationTemplate, metaclass=ABCMeta):
    """
    DsmFilling
    """

    available_applications: Dict = {}
    default_application = "bulldozer"

    def __new__(cls, orchestrator=None, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param orchestrator: orchestrator used
        :param conf: configuration for filling
        :return: an application_to_use object
        """

        dsm_filling_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "dsm_filling method not specified, default"
                " {} is used".format(dsm_filling_method)
            )
        else:
            dsm_filling_method = conf["method"]

        if dsm_filling_method not in cls.available_applications:
            logging.error(
                "No dsm_filling application named {} registered".format(
                    dsm_filling_method
                )
            )
            raise KeyError(
                "No dsm_filling application named {} registered".format(
                    dsm_filling_method
                )
            )

        logging.info(
            "The DsmFilling {} application will be used".format(
                dsm_filling_method
            )
        )

        return super(DsmFilling, cls).__new__(
            cls.available_applications[dsm_filling_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of DSM Filling

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(self):
        """
        Run dsm filling using initial elevation and the current dsm
        """
