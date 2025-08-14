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
This module contains sensor loader factory.
"""

import logging


class SensorLoader:
    """
    SensorLoader factory:
    A class designed for registered all available Cars sensor loader and
    instantiate when needed.

    """

    # Dict (loader_name:str, class: object) containing registered
    # sensor loaders
    available_loaders = {}

    def __new__(
        cls, app_name: str, cfg: dict, input_type: str, config_dir: str
    ):
        """
        Return the instance of sensor loader associated with the sensor loader
        name given as parameter

        :param app_name: name of the sensor loader.
        :type app_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        """

        return cls.create_app(app_name, cfg, input_type, config_dir)

    @classmethod
    def create_app(cls, name: str, cfg: dict, input_type: str, config_dir: str):
        """
        Factory command to create the sensor loader
        Return the instance of sensor loader associated with the sensor loader
        name given as parameter

        :param app_name: name of the sensor loader.
        :type app_name: str
        :param cfg: sensor loader configuration
        :type cfg: dictionary
        """
        loader = None

        try:
            loader_class = cls.available_loaders[name]
        except KeyError:
            logging.error("No sensor loader named {0} supported".format(name))
            return None
        loader = loader_class(cfg, input_type, config_dir)
        return loader

    @classmethod
    def register(cls, app_name: str):
        """
        Allows to register the sensor loader with its name
        :param app_name: the sensor loader to be registered
        :type app_name: string
        """

        def decorator(app):
            """
            Registers the class in the available methods
            :param app: the app class to be registered
            :type app: object
            """
            cls.available_loaders[app_name] = app
            return app

        return decorator
