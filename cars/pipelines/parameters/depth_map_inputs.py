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
CARS depth map inputs
"""

import logging

from json_checker import Checker, Or

import cars.pipelines.parameters.depth_map_inputs_constants as depth_map_cst

# CARS imports
from cars.core import constants as cst
from cars.core import inputs
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs as sens_inp
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def check_depth_maps_inputs(conf, config_json_dir=None):
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
    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)
    overloaded_conf[depth_map_cst.DEPTH_MAPS] = {}

    overloaded_conf[sens_cst.INITIAL_ELEVATION] = (
        sens_inp.get_initial_elevation(
            conf.get(sens_cst.INITIAL_ELEVATION, None)
        )
    )

    # Validate inputs
    inputs_schema = {
        depth_map_cst.DEPTH_MAPS: dict,
        sens_cst.ROI: Or(str, dict, None),
        sens_cst.INITIAL_ELEVATION: Or(dict, None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate depth maps

    pc_schema = {
        cst.X: str,
        cst.Y: str,
        cst.Z: str,
        cst.Z_INF: Or(str, None),
        cst.Z_SUP: Or(str, None),
        cst.POINT_CLOUD_CLASSIF_KEY_ROOT: Or(str, None),
        cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT: Or(dict, None),
        cst.POINT_CLOUD_CLR_KEY_ROOT: str,
        cst.POINT_CLOUD_FILLING_KEY_ROOT: Or(str, None),
        cst.POINT_CLOUD_MSK: Or(str, None),
        cst.POINT_CLOUD_PERFORMANCE_MAP: Or(str, None),
        cst.PC_EPSG: Or(str, int, None),
    }
    checker_pc = Checker(pc_schema)
    confidence_conf_ref = None
    for depth_map_key in conf[depth_map_cst.DEPTH_MAPS]:
        # Get depth maps with default
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key] = {}
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.X] = conf[
            depth_map_cst.DEPTH_MAPS
        ][depth_map_key].get("x", None)
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Y] = conf[
            depth_map_cst.DEPTH_MAPS
        ][depth_map_key].get("y", None)
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Z] = conf[
            depth_map_cst.DEPTH_MAPS
        ][depth_map_key].get("z", None)

        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Z_INF] = (
            conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("z_inf", None)
        )
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Z_SUP] = (
            conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("z_sup", None)
        )

        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.POINT_CLOUD_CLR_KEY_ROOT
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("color", None)

        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.POINT_CLOUD_MSK
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("mask", None)

        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.POINT_CLOUD_CLASSIF_KEY_ROOT
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get(
            "classification", None
        )
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.POINT_CLOUD_PERFORMANCE_MAP
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get(
            "performance_map", None
        )

        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.POINT_CLOUD_FILLING_KEY_ROOT
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("filling", None)

        confidence_conf = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get(
            "confidence", None
        )
        if confidence_conf:
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
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
                    cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
                    not in output_confidence_name
                ):
                    output_confidence_name = (
                        cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
                        + "_"
                        + output_confidence_name
                    )
                overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                    cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
                ][output_confidence_name] = confidence_conf[confidence_name]
        else:
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
            ] = None
        overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
            cst.PC_EPSG
        ] = conf[depth_map_cst.DEPTH_MAPS][depth_map_key].get("epsg", 4326)
        # validate
        checker_pc.validate(
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key]
        )

    # Modify to absolute path
    if config_json_dir is not None:
        modify_to_absolute_path(config_json_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    for depth_map_key in conf[depth_map_cst.DEPTH_MAPS]:
        # check sizes
        check_input_size(
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.X],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Y],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][cst.Z],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_MSK
            ],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_CLR_KEY_ROOT
            ],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_CLASSIF_KEY_ROOT
            ],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_FILLING_KEY_ROOT
            ],
            overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key][
                cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
            ],
        )

    # Check srtm dir
    sens_inp.check_srtm(
        overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
    )

    return overloaded_conf


def check_geometry_plugin(conf_inputs, conf_geom_plugin):
    """
    Check the geometry plugin with inputs
    :param conf_geom_plugin: name of geometry plugin
    :type conf_geom_plugin: str
    :param conf_inputs: checked configuration of inputs
    :type conf_inputs: type

    :return: geometry plugin with dem
    """
    if conf_geom_plugin is None:
        conf_geom_plugin = "SharelocGeometry"

    dem_path = conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]

    if dem_path is None:
        return None

    # Initialize a geometry plugin with elevation information
    geom_plugin_with_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            dem=dem_path,
            geoid=conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
            default_alt=sens_cst.CARS_DEFAULT_ALT,
        )
    )

    return geom_plugin_with_dem_and_geoid


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
    for depth_map_key in overloaded_conf[depth_map_cst.DEPTH_MAPS]:
        depth_map = overloaded_conf[depth_map_cst.DEPTH_MAPS][depth_map_key]
        for tag in [
            cst.X,
            cst.Y,
            cst.Z,
            cst.POINT_CLOUD_CLR_KEY_ROOT,
            cst.POINT_CLOUD_MSK,
            cst.POINT_CLOUD_CLASSIF_KEY_ROOT,
            cst.POINT_CLOUD_FILLING_KEY_ROOT,
            cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT,
        ]:
            if tag != cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT:
                if depth_map[tag] is not None:
                    depth_map[tag] = make_relative_path_absolute(
                        depth_map[tag], config_json_dir
                    )
            else:
                if depth_map[tag] is not None:
                    for confidence_name in depth_map[tag]:
                        if depth_map[tag][confidence_name] is not None:
                            depth_map[tag][confidence_name] = (
                                make_relative_path_absolute(
                                    depth_map[tag][confidence_name],
                                    config_json_dir,
                                )
                            )

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_json_dir
            )
