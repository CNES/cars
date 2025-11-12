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
this module contains the PivotImageSensorLoader class.
"""

from json_checker import Checker

from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)


@SensorLoader.register("pivot_classification")
class PivotClassifSensorLoader(SensorLoaderTemplate):
    """
    Pivot image sensor loader : used by CARS to read inputs
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """
        overloaded_conf = conf.copy()
        # Make relative path absolute
        overloaded_conf["path"] = make_relative_path_absolute(
            overloaded_conf["path"], self.config_dir
        )
        overloaded_conf[sens_cst.INPUT_VALUES] = conf.get(
            sens_cst.INPUT_VALUES, []
        )
        # Check dtype and number of bands
        classif_file = overloaded_conf["path"]
        classif_dtype = inputs.rasterio_get_dtype(classif_file)
        if classif_dtype != "uint8":
            raise TypeError(
                "Classification file {} has type {} which is not supported "
                "for classification : type must be uint8".format(
                    classif_file, classif_dtype
                )
            )
        classif_nb_bands = inputs.rasterio_get_nb_bands(classif_file)
        if classif_nb_bands != 1:
            raise TypeError(
                "Classification file {} has {} bands but only mono-band "
                "classification is allowed".format(
                    classif_file, classif_nb_bands
                )
            )

        sensor_schema = {
            sens_cst.INPUT_PATH: str,
            sens_cst.INPUT_VALUES: list,
        }

        # Check conf
        checker = Checker(sensor_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def set_pivot_format(self):
        self.pivot_format = self.used_config
