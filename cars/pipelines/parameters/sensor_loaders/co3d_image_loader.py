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
This module contains the CO3DImageSensorLoader class.
"""

from json_checker import Checker, Or

from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_loaders.pivot_image_loader import (
    PivotImageSensorLoader,
)
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)

RGB = "RGB"


@SensorLoader.register("CO3D_image")
class CO3DImageSensorLoader(SensorLoaderTemplate):
    """
    Default sensor loader for image (used when no sensor loader is specified)
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
        elif isinstance(conf, dict):
            overloaded_conf = conf.copy()
            image_path = make_relative_path_absolute(
                conf["RGB"], self.config_dir
            )
        else:
            raise TypeError(f"Input {conf} is not a string ot dict")
        overloaded_conf["RGB"] = image_path
        overloaded_conf["no_data"] = 0

        sensor_schema = {
            "RGB": str,
            sens_cst.INPUT_NODATA: Or(None, int),
        }

        # Check conf
        checker = Checker(sensor_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def set_pivot_format(self):
        """
        Transform input configuration to pivot format and store it
        """
        pivot_config = {"bands": {}}
        pivot_config["bands"]["b0"] = {
            sens_cst.INPUT_PATH: self.used_config[RGB],
            "band": 1,
        }
        pivot_config["bands"]["b1"] = {
            sens_cst.INPUT_PATH: self.used_config[RGB],
            "band": 0,
        }
        pivot_config["bands"]["b2"] = {
            sens_cst.INPUT_PATH: self.used_config[RGB],
            "band": 2,
        }
        pivot_config[sens_cst.SENSOR_TYPE] = "CO3D"
        pivot_sensor_loader = PivotImageSensorLoader(
            pivot_config, self.config_dir
        )
        self.pivot_format = pivot_sensor_loader.get_pivot_format()
