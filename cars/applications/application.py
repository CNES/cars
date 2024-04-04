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
This module contains class application factory.
"""

# Standard imports
import logging

# CARS imports
from cars.conf.input_parameters import ConfigType


class Application:
    """
    Application factory:
    A class designed for registered all available Cars application and
    instantiate when needed.

    """

    # Dict (application_name:str, class: object) containing registered
    # applications
    available_applications = {}

    def __new__(
        cls,
        app_name: str,
        cfg: ConfigType = None,
    ):
        """
        Return the instance of application associated with the application
        name given as parameter

        :param app_name: name of the application.
        :type app_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        """

        return cls.create_app(app_name, cfg)

    @classmethod
    def create_app(cls, name: str, cfg: ConfigType):
        """Factory command to create the application
        Return the instance of application associated with the application
        name given as parameter

        :param app_name: name of the application.
        :type app_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        """

        app = None

        try:
            app_class = cls.available_applications[name]
        except KeyError:
            logging.error("No application named {0} supported".format(name))
            return None

        app = app_class(cfg)
        return app

    @classmethod
    def print_applications(cls):
        """
        Print all registered applications
        """

        for app_name in cls.available_applications:
            print(app_name)

    @classmethod
    def register(cls, app_name: str):
        """
        Allows to register the application with its name
        :param app_name: the application to be registered
        :type app_name: string
        """

        def decorator(app):
            """
            Registers the class in the available methods
            :param app: the app class to be registered
            :type app: object
            """
            cls.available_applications[app_name] = app
            return app

        return decorator
