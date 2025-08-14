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

from json_checker import Checker, Or

from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader
from cars.pipelines.parameters.sensor_loaders.sensor_loader_template import (
    SensorLoaderTemplate,
)


@SensorLoader.register("pivot")
class PivotSensorLoader(SensorLoaderTemplate):
    """
    Pivot sensor loader : used by CARS to read inputs
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check

        :return: overloaded configuration
        :rtype: dict
        """
        overloaded_conf = conf.copy()
        # Make relative paths absolutes
        for band in overloaded_conf["bands"]:
            overloaded_conf["bands"][band]["path"] = (
                make_relative_path_absolute(
                    overloaded_conf["bands"][band]["path"], self.config_dir
                )
            )
        # Check consistency between files
        b0_path = overloaded_conf["bands"]["b0"]["path"]
        b0_size = inputs.rasterio_get_size(b0_path)
        b0_transform = inputs.rasterio_get_size(b0_path)
        for band in overloaded_conf["bands"]:
            band_path = overloaded_conf["bands"][band]["path"]
            band_id = overloaded_conf["bands"][band]["band"]
            nb_bands = inputs.rasterio_get_nb_bands(band_path)
            if band_id >= nb_bands:
                raise RuntimeError(
                    "Band id {} is not valid for sensor which "
                    "has only {} bands".format(band_id, nb_bands)
                )
            if band_path != b0_path:
                band_size = inputs.rasterio_get_size(band_path)
                band_transform = inputs.rasterio_get_size(band_path)
                if b0_size != band_size:
                    raise RuntimeError(
                        "The files {} and {} do not have the same size"
                        "but are in the same image".format(b0_path, band_path)
                    )
                if b0_transform != band_transform:
                    raise RuntimeError(
                        "The files {} and {} do not have the same size"
                        "but are in the same image".format(
                            b0_transform,
                            band_transform,
                        )
                    )
        overloaded_conf["main_file"] = conf.get("main_file", None)
        if overloaded_conf["main_file"] is None:
            overloaded_conf["main_file"] = overloaded_conf["bands"]["b0"][
                "path"
            ]
        else:
            overloaded_conf["main_file"] = make_relative_path_absolute(
                overloaded_conf["main_file"], self.config_dir
            )
        overloaded_conf["texture_bands"] = conf.get("texture_bands", None)
        if overloaded_conf["texture_bands"] is not None:
            for texture_band in overloaded_conf["texture_bands"]:
                if texture_band not in overloaded_conf["bands"]:
                    raise RuntimeError(
                        "Texture band {} not found in bands {} "
                        "of sensor image".format(
                            texture_band, overloaded_conf["bands"]
                        )
                    )

        if self.input_type == "image":
            overloaded_conf[sens_cst.INPUT_NODATA] = conf.get(
                sens_cst.INPUT_NODATA, 0
            )
        else:
            overloaded_conf[sens_cst.INPUT_NODATA] = None

        sensor_schema = {
            "loader": str,
            "main_file": str,
            "bands": dict,
            "texture_bands": Or(None, [str]),
            "no_data": Or(None, int),
        }

        # Check conf
        checker = Checker(sensor_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def set_pivot_format(self):
        self.pivot_format = self.used_config
