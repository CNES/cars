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

import logging

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
    Default sensor loader (used when no sensor loader is specified)
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """
        slurp_filling = {
            "fill_with_geoid": 8,
            "interpolate_from_borders": 9,
            "fill_with_endogenous_dem": 10,
            "fill_with_exogenous_dem": 6,
        }
        if isinstance(conf, dict):
            overloaded_conf = conf.copy()
            image_path = make_relative_path_absolute(
                conf[sens_cst.INPUT_PATH], self.config_dir
            )
            overloaded_conf[sens_cst.INPUT_PATH] = image_path
            if sens_cst.INPUT_FILLING in conf:
                logging.warning(
                    "A filling dictionary has been defined but "
                    "the slurp_classification loader is selected : filling "
                    "values will be overriden according to SLURP conventions"
                )
            overloaded_conf[sens_cst.INPUT_FILLING] = slurp_filling
        else:
            raise TypeError(f"Input {conf} is not a string ot dict")

        sensor_schema = {
            sens_cst.INPUT_LOADER: str,
            sens_cst.INPUT_PATH: str,
            sens_cst.INPUT_FILLING: dict,
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
            sens_cst.INPUT_LOADER: "basic_classification",
            sens_cst.INPUT_PATH: self.used_config[sens_cst.INPUT_PATH],
            sens_cst.INPUT_FILLING: self.used_config[sens_cst.INPUT_FILLING],
        }
        pivot_config["values"] = inputs.rasterio_get_classif_values(
            self.used_config[sens_cst.INPUT_PATH]
        )
        # Remove value 0 because it corresponds to unclassified
        pivot_config["values"].remove(0)
        pivot_sensor_loader = PivotClassifSensorLoader(
            pivot_config, self.config_dir
        )
        self.pivot_format = pivot_sensor_loader.get_pivot_format()
