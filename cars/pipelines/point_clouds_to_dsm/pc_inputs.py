#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
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
CARS point cloud inputs
"""

import logging

from json_checker import Checker, Or

import cars.pipelines.point_clouds_to_dsm.pc_constants as pc_cst

# CARS imports
from cars.core import constants as cst
from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)


def check_point_clouds_inputs(conf, config_json_dir=None):
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_json_dir: directory of used json, if
        user filled paths with relative paths
    :type config_json_dir: str

    :return: overloader inputs
    :rtype: dict
    """

    overloaded_conf = {}

    # Overload some optional parameters
    overloaded_conf[sens_cst.EPSG] = conf.get(sens_cst.EPSG, None)
    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)
    overloaded_conf[pc_cst.POINT_CLOUDS] = {}

    # Validate inputs
    inputs_schema = {
        pc_cst.POINT_CLOUDS: dict,
        sens_cst.EPSG: Or(int, None),
        sens_cst.ROI: Or(str, dict, None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate point clouds

    pc_schema = {
        cst.X: str,
        cst.Y: str,
        cst.Z: str,
        cst.POINTS_CLOUD_CLASSIF_KEY_ROOT: Or(str, None),
        cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT: Or(dict, None),
        cst.POINTS_CLOUD_CLR_KEY_ROOT: str,
        cst.POINTS_CLOUD_FILLING_KEY_ROOT: Or(str, None),
        cst.POINTS_CLOUD_MSK: Or(str, None),
        cst.PC_EPSG: Or(str, int, None),
    }
    checker_pc = Checker(pc_schema)
    confidence_conf_ref = None
    for point_cloud_key in conf[pc_cst.POINT_CLOUDS]:
        # Get point clouds with default
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key] = {}
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.X] = conf[
            pc_cst.POINT_CLOUDS
        ][point_cloud_key].get("x", None)
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Y] = conf[
            pc_cst.POINT_CLOUDS
        ][point_cloud_key].get("y", None)
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Z] = conf[
            pc_cst.POINT_CLOUDS
        ][point_cloud_key].get("z", None)
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
            cst.POINTS_CLOUD_CLR_KEY_ROOT
        ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("color", None)
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
            cst.POINTS_CLOUD_MSK
        ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("mask", None)
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
            cst.POINTS_CLOUD_CLASSIF_KEY_ROOT
        ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get(
            "classification", None
        )
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
            cst.POINTS_CLOUD_FILLING_KEY_ROOT
        ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("filling", None)
        confidence_conf = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get(
            "confidence", None
        )
        if confidence_conf:
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
            ] = {}
            if (
                confidence_conf_ref
                and confidence_conf.keys() != confidence_conf_ref
            ):
                raise KeyError(
                    "The confidence keys are not the same: \n",
                    confidence_conf.keys(),
                    "\n",
                    confidence_conf_ref,
                )

            confidence_conf_ref = confidence_conf.keys()
            for confidence_name in confidence_conf:
                output_confidence_name = confidence_name
                if (
                    cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
                    not in output_confidence_name
                ):
                    output_confidence_name = (
                        cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
                        + "_"
                        + output_confidence_name
                    )
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                    cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
                ][output_confidence_name] = confidence_conf[confidence_name]
        else:
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
            ] = None
        overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.PC_EPSG] = (
            conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("epsg", 4326)
        )
        # validate
        checker_pc.validate(
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key]
        )

    # Modify to absolute path
    if config_json_dir is not None:
        modify_to_absolute_path(config_json_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    for point_cloud_key in conf[pc_cst.POINT_CLOUDS]:
        # check sizes
        check_input_size(
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.X],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Y],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Z],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_MSK
            ],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CLR_KEY_ROOT
            ],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CLASSIF_KEY_ROOT
            ],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_FILLING_KEY_ROOT
            ],
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
            ],
        )

    return overloaded_conf


def check_input_size(
    x_path, y_path, z_path, mask, color, classif, filling, confidence
):
    """
    Check x, y, z, mask, color, classif and confidence given

    Images must have same size

    :param x_path: x path
    :type x_path: str
    :param y_path: y path
    :type y_path: str
    :param z_path: z path
    :type z_path: str
    :param mask: mask path
    :type mask: str
    :param color: color path
    :type color: str
    :param classif: classif path
    :type classif: str
    :param filling: filling path
    :type filling: str
    :param confidence: confidence dict path
    :type confidence: dict[str]
    """

    for path in [x_path, y_path, z_path]:
        if inputs.rasterio_get_nb_bands(path) != 1:
            raise RuntimeError("{} is not mono-band image".format(path))

    for path in [mask, color, classif, filling]:
        if path is not None:
            if inputs.rasterio_get_size(x_path) != inputs.rasterio_get_size(
                path
            ):
                raise RuntimeError(
                    "The image {} and {} "
                    "do not have the same size".format(x_path, path)
                )
    if confidence:
        for key in confidence:
            path = confidence[key]
            if path is not None:
                if inputs.rasterio_get_size(x_path) != inputs.rasterio_get_size(
                    path
                ):
                    raise RuntimeError(
                        "The image {} and {} "
                        "do not have the same size".format(x_path, path)
                    )


def modify_to_absolute_path(config_json_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_json_dir: directory of the json configuration
    :type config_json_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for point_cloud_key in overloaded_conf[pc_cst.POINT_CLOUDS]:
        point_cloud = overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key]
        for tag in [
            cst.X,
            cst.Y,
            cst.Z,
            cst.POINTS_CLOUD_CLR_KEY_ROOT,
            cst.POINTS_CLOUD_MSK,
            cst.POINTS_CLOUD_CLASSIF_KEY_ROOT,
            cst.POINTS_CLOUD_FILLING_KEY_ROOT,
            cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT,
        ]:
            if tag != cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT:
                if point_cloud[tag] is not None:
                    point_cloud[tag] = make_relative_path_absolute(
                        point_cloud[tag], config_json_dir
                    )
            else:
                if point_cloud[tag] is not None:
                    for confidence_name in point_cloud[tag]:
                        if point_cloud[tag][confidence_name] is not None:
                            point_cloud[tag][confidence_name] = (
                                make_relative_path_absolute(
                                    point_cloud[tag][confidence_name],
                                    config_json_dir,
                                )
                            )

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_json_dir
            )
