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
This module contains class application template for
templating the application concept.
Useful for shared parameters and functions in all applications
Beware: argument-differ is activated in pylintrc for run parameters different
in sub application classes
"""

# Standard imports
import logging
import pprint
from abc import ABCMeta, abstractmethod

# CARS imports
from cars.conf.input_parameters import ConfigType


class ApplicationTemplate(metaclass=ABCMeta):
    """
    Class for general specification of an application
    Empty for the moment because there is no any common method or function
    """

    # Define abstract attributes
    used_config: ConfigType

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Generic run() function to be defined in subclasses
        """

    def __init__(self, conf=None):  # pylint: disable=W0613
        """
        Init function of ApplicationTemplate

        :param conf: configuration for application

        """
        # Check conf
        try:
            self.used_config = self.check_conf(conf)
        except Exception:
            logging.error(
                "The {} application checking has been failed!".format(
                    self.__class__.__bases__[0].__name__
                )
            )
            raise

    @abstractmethod
    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

    def print_config(self):
        """
        Print used application configuration

        """
        pretty_printer = pprint.PrettyPrinter(indent=4)

        try:
            pretty_printer.pprint(self.used_config)
        except Exception:
            logging.error("self.used_config not filled by application")

    def get_conf(self):
        """
        Get used conf

        :return: used conf
        """

        return self.used_config
