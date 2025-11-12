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
This module contains the ClassifSensorLoader class.
"""

from json_checker import Checker

from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_loaders.pivot_classif_loader import (
    PivotClassifSensorLoader,
)
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)


@SensorLoader.register("slurp_classification")
class SlurpClassifSensorLoader(SensorLoaderTemplate):
    """
    SLURP sensor loader
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """
        if isinstance(conf, str):
            overloaded_conf = {}
            image_path = make_relative_path_absolute(conf, self.config_dir)
            overloaded_conf[sens_cst.INPUT_PATH] = image_path
        else:
            raise TypeError(f"Input {conf} is not a string")

        sensor_schema = {
            sens_cst.INPUT_PATH: str,
        }

        # Check conf
        checker = Checker(sensor_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def set_pivot_format(self):
        """
        Transform input configuration to pivot format and store it
        """
        pivot_config = {
            sens_cst.INPUT_PATH: self.used_config[sens_cst.INPUT_PATH],
        }
        pivot_config["values"] = inputs.rasterio_get_classif_values(
            self.used_config[sens_cst.INPUT_PATH]
        )
        # Remove value 0 because it corresponds to unclassified data
        pivot_config["values"].remove(0)
        pivot_sensor_loader = PivotClassifSensorLoader(
            pivot_config, self.config_dir
        )
        self.pivot_format = pivot_sensor_loader.get_pivot_format()
