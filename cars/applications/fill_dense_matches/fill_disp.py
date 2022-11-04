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
from abc import ABCMeta
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("fill_disp")
class FillDisp(ApplicationTemplate, metaclass=ABCMeta):
    """
    AbstractFillDisp
    """

    available_applications: Dict = {}
    default_application = "plane"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        fill_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Fill Disparity method not specified, "
                "default {} is used".format(fill_method)
            )
        else:
            fill_method = conf["method"]

        if fill_method not in cls.available_applications:
            logging.error(
                "No fill disp application named {} registered".format(
                    fill_method
                )
            )
            raise KeyError(
                "No fill disp application named {} registered".format(
                    fill_method
                )
            )

        logging.info(
            "[The AbstractFillDisp {} application will be used".format(
                fill_method
            )
        )

        return super(FillDisp, cls).__new__(
            cls.available_applications[fill_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302

        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_applications[name] = cls
