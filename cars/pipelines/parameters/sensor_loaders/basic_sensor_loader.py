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
this module contains the BasicSensorLoader class.
"""

from cars.core import inputs
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    AbstractSensorLoader,
)
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.pivot_sensor_loader import PivotSensorLoader


@SensorLoader.register("basic")
class BasicSensorLoader(AbstractSensorLoader):
    """
    BasicSensorLoader
    """

    def __init__(self, input_config, input_type):
        super().__init__(conf=input_config, input_type=input_type)

    def check_conf(conf, input_type):
        if isinstance(conf, str):
            overloaded_conf = {}
            overloaded_conf["path"] = conf
            if input_type == "image":
                overloaded_conf["nodata"] = 0
        elif isinstance(conf, dict):
            overloaded_conf = conf.copy()
            overloaded_conf["path"] = conf["path"]
            if input_type == "image":
                overloaded_conf["nodata"] = conf.get("nodata", 0)
        else:
            raise TypeError("TODO")

        return overloaded_conf

    def transform_config_to_pivot_format(self):
        """
        Transforme une conf "image" ou "classification" dans le format pivot
        """
        pivot_config = {"main_file_path": self.used_config["path"]}
        pivot_config["bands"] = {}
        for band_id in range(inputs.rasterio_get_nb_bands(self.used_config["path"])):
            band_name = "b" + str(band_id)
            pivot_config["bands"][band_name] = {
                "path": self.input_path,
                "band": band_id,
            }
        pivot_sensor_loader = PivotSensorLoader(pivot_config)
        self.pivot_config = pivot_sensor_loader.get_pivot_format()

    def get_pivot_format(self):
        if self.pivot_config is None:
            self.transform_config_to_pivot_format()

        return self.pivot_config

    def get_main_file_path(self):
        return self.used_config["path"]
