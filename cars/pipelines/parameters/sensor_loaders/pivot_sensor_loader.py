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
this module contains the PivotSensorLoader class.
"""

from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader

@SensorLoader.register("pivot")
class PivotSensorLoader(SensorLoaderTemplate):
    """
    AbstractSensorLoader
    """


    def check_conf(self, conf, input_type):
        overloaded_conf = conf.copy()

        return overloaded_conf


    def set_pivot_format(self):
        self.pivot_format = self.used_config
