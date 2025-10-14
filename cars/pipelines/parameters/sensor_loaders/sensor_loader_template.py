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
this module contains the AbstractSensorLoader class.
"""

from abc import abstractmethod


class SensorLoaderTemplate:
    """
    Class for general specification of a sensor loader
    """

    def __init__(self, conf, config_dir):
        """
        Init function of SensorLoaderTemplate

        :param conf: configuration for sensor loader
        """
        self.config_dir = config_dir
        self.used_config = self.check_conf(conf)

        self.pivot_format = None

    @abstractmethod
    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """

    def get_pivot_format(self):
        """
        Return sensor configuration as pivot format

        :return: pivot format configuration
        :rtype: dict
        """
        if self.pivot_format is None:
            self.set_pivot_format()

        return self.pivot_format

    @abstractmethod
    def set_pivot_format(self):
        """
        Transform input configuration to pivot format and store it
        """
