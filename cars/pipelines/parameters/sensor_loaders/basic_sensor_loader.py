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
from cars.pipelines.parameters.sensor_loaders.pivot_sensor_loader import (
    PivotSensorLoader,
)
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)


@SensorLoader.register("basic")
class BasicSensorLoader(SensorLoaderTemplate):
    """
    Default sensor loader (used when no sensor loader is specified)
    """

    def check_conf(self, conf, input_type):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """
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
            raise TypeError(f"Input {input_type} is not a string ot dict")

        return overloaded_conf

    def set_pivot_format(self):
        """
        Transform input configuration to pivot format and store it
        """
        pivot_config = {"main_file_path": self.used_config["path"]}
        pivot_config["bands"] = {}
        for band_id in range(inputs.rasterio_get_nb_bands(self.used_config["path"])):
            band_name = "b" + str(band_id)
            pivot_config["bands"][band_name] = {
                "path": self.used_config["path"],
                "band": band_id,
            }
        pivot_sensor_loader = PivotSensorLoader(pivot_config, self.input_type)
        self.pivot_format = pivot_sensor_loader.get_pivot_format()
