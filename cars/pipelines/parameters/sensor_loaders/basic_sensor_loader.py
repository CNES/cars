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

from json_checker import Checker, Or

from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
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
            overloaded_conf["path"] = image_path
            overloaded_conf["loader"] = "basic"
            if self.input_type == "image":
                overloaded_conf[sens_cst.INPUT_NODATA] = 0
            else:
                overloaded_conf[sens_cst.INPUT_NODATA] = None
        elif isinstance(conf, dict):
            overloaded_conf = conf.copy()
            image_path = make_relative_path_absolute(
                conf["path"], self.config_dir
            )
            overloaded_conf["path"] = image_path
            overloaded_conf["loader"] = conf.get("loader", "basic")
            if self.input_type == "image":
                overloaded_conf[sens_cst.INPUT_NODATA] = conf.get(
                    sens_cst.INPUT_NODATA, 0
                )
            else:
                overloaded_conf[sens_cst.INPUT_NODATA] = None
        else:
            raise TypeError(f"Input {self.input_type} is not a string ot dict")

        sensor_schema = {"loader": str, "path": str, "no_data": Or(None, int)}

        # Check conf
        checker = Checker(sensor_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def set_pivot_format(self):
        """
        Transform input configuration to pivot format and store it
        """
        pivot_config = {
            "loader": "pivot",
            "main_file": self.used_config["path"],
        }
        pivot_config["bands"] = {}
        for band_id in range(
            inputs.rasterio_get_nb_bands(self.used_config["path"])
        ):
            band_name = "b" + str(band_id)
            pivot_config["bands"][band_name] = {
                "path": self.used_config["path"],
                "band": band_id,
            }
        pivot_config["texture_bands"] = None
        pivot_sensor_loader = PivotSensorLoader(
            pivot_config, self.input_type, self.config_dir
        )
        self.pivot_format = pivot_sensor_loader.get_pivot_format()
