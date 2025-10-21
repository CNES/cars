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
This module contains the output definition
"""

import logging
import os

from json_checker import And, Checker, Or
from pyproj import CRS

import cars.core.constants as cst
from cars.core.utils import safe_makedirs
from cars.pipelines.parameters import output_constants
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def is_valid_epsg(epsg) -> bool:
    """
    Check if the given EPSG code is valid using pyproj.
    """
    if epsg is None:
        return True

    try:
        # Try creating a CRS
        CRS(f"EPSG:{epsg}")
        return True
    except Exception:
        return False


def check_output_parameters(  # noqa: C901 : too complex
    inputs, conf, scaling_coeff
):
    """
    Check the output json configuration and fill in default values

    :param conf: configuration of output
    :type conf: dict
    :param scaling_coeff: scaling factor for resolution
    :type scaling_coeff: float
    :param pipeline_name: name of corresponding pipeline
    :type pipeline_name: str
    """
    overloaded_conf = conf.copy()
    out_dir = conf[output_constants.OUT_DIRECTORY]
    out_dir = os.path.abspath(out_dir)
    # Ensure that output directory and its subdirectories exist
    safe_makedirs(out_dir)

    # Overload some parameters
    overloaded_conf[output_constants.OUT_DIRECTORY] = out_dir

    overloaded_conf[output_constants.PRODUCT_LEVEL] = overloaded_conf.get(
        output_constants.PRODUCT_LEVEL, "dsm"
    )
    if isinstance(overloaded_conf[output_constants.PRODUCT_LEVEL], str):
        overloaded_conf[output_constants.PRODUCT_LEVEL] = [
            overloaded_conf[output_constants.PRODUCT_LEVEL]
        ]
    for level in overloaded_conf[output_constants.PRODUCT_LEVEL]:
        if level not in ["dsm", "depth_map", "point_cloud"]:
            raise RuntimeError("Unknown product level {}".format(level))

    overloaded_conf[output_constants.OUT_GEOID] = overloaded_conf.get(
        output_constants.OUT_GEOID, True
    )
    overloaded_conf[output_constants.EPSG] = overloaded_conf.get(
        output_constants.EPSG, None
    )

    resolution = None
    overloaded_scaling_coeff = scaling_coeff
    if overloaded_conf.get(output_constants.RESOLUTION, None) is not None:
        resolution = overloaded_conf[output_constants.RESOLUTION]
        # update scaling coeff so the parameters are right for the dsm
        # overloaded_scaling_coeff = 2*resolution

        if resolution < 0.5 * scaling_coeff:
            logging.warning(
                "The requested DSM resolution of "
                f"{overloaded_conf[output_constants.RESOLUTION]} seems "
                "too low for the sensor images' resolution. "
                "The pipeline will still continue with it."
            )

    else:
        resolution = float(0.5 * scaling_coeff)
        logging.info(
            "The resolution of the output DSM will be " f"{resolution} meters. "
        )

    overloaded_conf[output_constants.RESOLUTION] = resolution

    overloaded_conf[output_constants.SAVE_BY_PAIR] = overloaded_conf.get(
        output_constants.SAVE_BY_PAIR, False
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY] = overloaded_conf.get(
        output_constants.AUXILIARY, {}
    )

    # Load auxiliary and subfields
    overloaded_conf[output_constants.AUXILIARY][output_constants.AUX_IMAGE] = (
        overloaded_conf[output_constants.AUXILIARY].get(
            output_constants.AUX_IMAGE, True
        )
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_DEM_MIN
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_DEM_MIN, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_DEM_MAX
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_DEM_MAX, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_DEM_MEDIAN
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_DEM_MEDIAN, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_WEIGHTS
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_WEIGHTS, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CLASSIFICATION
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CLASSIFICATION, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_PERFORMANCE_MAP
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_PERFORMANCE_MAP, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CONTRIBUTING_PAIR
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_CONTRIBUTING_PAIR, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_FILLING
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_FILLING, False
    )
    overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_AMBIGUITY
    ] = overloaded_conf[output_constants.AUXILIARY].get(
        output_constants.AUX_AMBIGUITY, False
    )

    # Check schema
    output_schema = {
        output_constants.OUT_DIRECTORY: str,
        output_constants.PRODUCT_LEVEL: list,
        output_constants.OUT_GEOID: Or(bool, str),
        output_constants.EPSG: And(Or(int, str, None), is_valid_epsg),
        output_constants.RESOLUTION: Or(int, float),
        output_constants.SAVE_BY_PAIR: bool,
        output_constants.AUXILIARY: dict,
    }
    checker_output = Checker(output_schema)
    checker_output.validate(overloaded_conf)

    # check auxiliary keys
    auxiliary_schema = {
        output_constants.AUX_IMAGE: Or(bool, str, list),
        output_constants.AUX_WEIGHTS: bool,
        output_constants.AUX_CLASSIFICATION: Or(bool, dict, list),
        output_constants.AUX_PERFORMANCE_MAP: Or(bool, list),
        output_constants.AUX_CONTRIBUTING_PAIR: bool,
        output_constants.AUX_FILLING: bool,
        output_constants.AUX_AMBIGUITY: bool,
        output_constants.AUX_DEM_MIN: bool,
        output_constants.AUX_DEM_MAX: bool,
        output_constants.AUX_DEM_MEDIAN: bool,
    }

    # Check and overload classification parameter
    check_classification_parameter(inputs, overloaded_conf)

    # Check and overload image parameter
    check_texture_bands(inputs, overloaded_conf)

    # Check and overload performance_map parameter
    check_performance_classes(overloaded_conf)

    checker_auxiliary = Checker(auxiliary_schema)
    checker_auxiliary.validate(overloaded_conf[output_constants.AUXILIARY])

    if "epsg" in overloaded_conf and overloaded_conf["epsg"]:
        spatial_ref = CRS.from_epsg(overloaded_conf["epsg"])
        if spatial_ref.is_geographic:
            if overloaded_conf[output_constants.RESOLUTION] > 10e-3:
                logging.warning(
                    "The resolution of the "
                    + "point_cloud_rasterization should be "
                    + "fixed according to the epsg"
                )

    return overloaded_conf, overloaded_scaling_coeff


def check_texture_bands(inputs, overloaded_conf):
    """
    Check and overload texture bands
    """
    texture_bands = overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_IMAGE
    ]

    if inputs[sens_cst.SENSORS] is not None:
        if isinstance(texture_bands, list):
            for elem in texture_bands:
                if not isinstance(elem, str):
                    raise RuntimeError(
                        "The image parameter of auxiliary should "
                        "be a boolean, a string or a list of string"
                    )
        elif isinstance(texture_bands, str):
            overloaded_conf[output_constants.AUXILIARY][
                output_constants.AUX_IMAGE
            ] = [texture_bands]
        elif texture_bands is True:
            first_key = list(inputs[sens_cst.SENSORS].keys())[0]
            image = inputs[sens_cst.SENSORS][first_key][sens_cst.INPUT_IMG]
            bands = set(image["bands"].keys())

            overloaded_conf[output_constants.AUXILIARY][
                output_constants.AUX_IMAGE
            ] = sorted(bands)


def check_classification_parameter(inputs, overloaded_conf):
    """
    Check and overload classification parameter
    """
    classification_formatting = overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_CLASSIFICATION
    ]

    if inputs[sens_cst.SENSORS] is not None:
        first_key = list(inputs[sens_cst.SENSORS].keys())[0]

        if "classification" in inputs[sens_cst.SENSORS][first_key]:
            classif = inputs[sens_cst.SENSORS][first_key][
                sens_cst.INPUT_CLASSIFICATION
            ]
            bands_classif = sorted(set(classif["bands"].keys()))

            if isinstance(classification_formatting, list):
                overloaded_conf[output_constants.AUXILIARY][
                    output_constants.AUX_CLASSIFICATION
                ] = {val: str(val) for val in classification_formatting}

                for elem in classification_formatting:
                    if not isinstance(elem, int):
                        raise RuntimeError(
                            "The image parameter of auxiliary should "
                            "be a boolean, a string or a list of int"
                        )

                    if elem > len(bands_classif):
                        raise RuntimeError(
                            f"If you want to use {elem} as a band num, "
                            f"you should use a dictionary, not a list"
                        )
            elif classification_formatting is True:
                overloaded_conf[output_constants.AUXILIARY][
                    output_constants.AUX_CLASSIFICATION
                ] = {val + 1: name for val, name in enumerate(bands_classif)}
            elif isinstance(classification_formatting, dict):
                for _, value in classification_formatting.items():
                    if value not in bands_classif:
                        raise RuntimeError(
                            f"The band {value} is "
                            f"not an existing band of "
                            f"the input classification"
                        )


def check_performance_classes(overloaded_conf):
    """
    Check performance classes
    """
    performance_map_classes = overloaded_conf[output_constants.AUXILIARY][
        output_constants.AUX_PERFORMANCE_MAP
    ]

    if isinstance(performance_map_classes, list):
        for elem in performance_map_classes:
            if not isinstance(elem, (int, float)):
                raise RuntimeError(
                    "The performance_map parameter of auxiliary should"
                    "be a boolean or a list of float/int"
                )
        if len(performance_map_classes) < 2:
            raise RuntimeError("Not enough step for performance_map_classes")
        if performance_map_classes:
            previous_step = -1
            for step in performance_map_classes:
                if step < 0:
                    raise RuntimeError(
                        "All step in performance_map_classes must be >=0"
                    )
                if step <= previous_step:
                    raise RuntimeError(
                        "performance_map_classes list must be ordered."
                    )
                previous_step = step
    elif performance_map_classes is True:
        # default classes, in meters:
        default_performance_classes = [
            0,
            0.968,
            1.13375,
            1.295,
            1.604,
            2.423,
            3.428,
        ]

        overloaded_conf[output_constants.AUXILIARY][
            output_constants.AUX_PERFORMANCE_MAP
        ] = default_performance_classes


def intialize_product_index(orchestrator, product_levels, input_pairs):
    """
    Initialize the index dictionary according to requested levels with None
    values for all paths.

    :param orchestrator: cars orchestrator
    :type orchestrator: Orchestrator
    :param product_levels: name of corresponding pipeline
    :type product_levels: list
    :param input_pairs: list containing the pair names
    :type input_pairs: list
    """

    index = {}

    if "dsm" in product_levels:
        index["dsm"] = {
            cst.INDEX_DSM_ALT: None,
            cst.INDEX_DSM_COLOR: None,
            cst.INDEX_DSM_MASK: None,
            cst.INDEX_DSM_CLASSIFICATION: None,
            cst.INDEX_DSM_PERFORMANCE_MAP: None,
            cst.INDEX_DSM_CONTRIBUTING_PAIR: None,
            cst.INDEX_DSM_FILLING: None,
            cst.INDEX_DSM_WEIGHTS: None,
        }

    if "point_cloud" in product_levels:
        # Initialize an empty index for point cloud because its content is
        # unknown at this point (tile name, save by pair or not)
        index["point_cloud"] = {}

    if "depth_map" in product_levels:
        index["depth_map"] = {}
        for pair in input_pairs:
            index["depth_map"][pair] = {
                cst.INDEX_DEPTH_MAP_X: None,
                cst.INDEX_DEPTH_MAP_Y: None,
                cst.INDEX_DEPTH_MAP_Z: None,
                cst.INDEX_DEPTH_MAP_COLOR: None,
                cst.INDEX_DEPTH_MAP_MASK: None,
                cst.INDEX_DEPTH_MAP_CLASSIFICATION: None,
                cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP: None,
                cst.INDEX_DEPTH_MAP_FILLING: None,
                cst.INDEX_DEPTH_MAP_EPSG: None,
            }

    orchestrator.update_index(index)
