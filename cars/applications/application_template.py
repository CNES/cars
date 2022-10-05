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
templating the application concept. Useless for the moment
but will be useful for shared functions in all applications
"""

import logging
import pprint
from abc import ABCMeta

# Disable pylint error: too few public method


class ApplicationTemplate(  # noqa: B024
    metaclass=ABCMeta
):  # pylint: disable=R0903
    """
    Class for general specification of an application
    Empty for the moment because there is no any common method or function
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
