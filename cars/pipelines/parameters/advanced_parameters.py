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
CARS module containing functions to check advanced parameters configuration
"""
import logging
import os

import rasterio as rio
from json_checker import And, Checker, OptionalKey, Or

from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import dsm_inputs
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_inputs import CARS_GEOID_PATH


def get_resolutions(conf):
    """
    Get the epipolar resolutions from the configuration

    :param conf: configuration of advanced parameters
    :type conf: dict

    :return: list of epipolar resolutions
    :rtype: list
    """
    if adv_cst.RESOLUTIONS in conf:
        return conf[adv_cst.RESOLUTIONS]

    return [16, 4, 1]


def check_advanced_parameters(inputs, conf, output_dem_dir=None):
    """
    Check the advanced parameters consistency

    :param inputs: configuration of inputs
    :type inputs: dict
    :param conf: configuration of advanced parameters
    :type conf: dict

    :return: overloaded configuration
    :rtype: dict
    """

    overloaded_conf = conf.copy()

    overloaded_conf[adv_cst.SAVE_INTERMEDIATE_DATA] = conf.get(
        adv_cst.SAVE_INTERMEDIATE_DATA, False
    )

    overloaded_conf[adv_cst.LAND_COVER_MAP] = conf.get(
        adv_cst.LAND_COVER_MAP, "global_land_cover_map.tif"
    )

    overloaded_conf[adv_cst.KEEP_LOW_RES_DIR] = conf.get(
        adv_cst.KEEP_LOW_RES_DIR,
        bool(overloaded_conf[adv_cst.SAVE_INTERMEDIATE_DATA]),
    )

    overloaded_conf[adv_cst.DEBUG_WITH_ROI] = conf.get(
        adv_cst.DEBUG_WITH_ROI, False
    )

    overloaded_conf[adv_cst.CLASSIFICATION_TO_CONFIGURATION_MAPPING] = conf.get(
        adv_cst.CLASSIFICATION_TO_CONFIGURATION_MAPPING, "config_mapping.json"
    )

    overloaded_conf[adv_cst.PHASING] = conf.get(adv_cst.PHASING, None)

    # use endogenous dm when generated
    overloaded_conf[adv_cst.USE_ENDOGENOUS_DEM] = conf.get(
        adv_cst.USE_ENDOGENOUS_DEM,
        inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] is None,
    )

    overloaded_conf[adv_cst.DSM_MERGING_TILE_SIZE] = conf.get(
        adv_cst.DSM_MERGING_TILE_SIZE, 4000
    )

    overloaded_conf[adv_cst.GROUND_TRUTH_DSM] = conf.get(
        adv_cst.GROUND_TRUTH_DSM, {}
    )

    # Validate ground truth DSM
    if overloaded_conf[adv_cst.GROUND_TRUTH_DSM]:
        overloaded_conf[adv_cst.GROUND_TRUTH_DSM][adv_cst.INPUT_AUX_PATH] = (
            conf[adv_cst.GROUND_TRUTH_DSM].get(adv_cst.INPUT_AUX_PATH, None)
        )
        overloaded_conf[adv_cst.GROUND_TRUTH_DSM][adv_cst.INPUT_AUX_INTERP] = (
            conf[adv_cst.GROUND_TRUTH_DSM].get(adv_cst.INPUT_AUX_INTERP, None)
        )
        check_ground_truth_dsm_data(overloaded_conf[adv_cst.GROUND_TRUTH_DSM])

    # Check geometry plugin
    geom_plugin_without_dem_and_geoid = None
    geom_plugin_with_dem_and_geoid = None

    scaling_coeff = None

    if inputs[sens_cst.SENSORS] is not None:
        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            inputs,
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_without_dem_and_geoid,
            geom_plugin_with_dem_and_geoid,
            scaling_coeff,
        ) = sensor_inputs.check_geometry_plugin(
            inputs,
            conf.get(adv_cst.GEOMETRY_PLUGIN, None),
            output_dem_dir,
        )
    elif dsm_cst.DSMS in inputs:
        # assume the input comes from 0.5m sensor images
        scaling_coeff = 1
        # If there's an initial elevation with
        # point clouds as inputs, generate a plugin (used in dsm_filling)
        (
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_with_dem_and_geoid,
        ) = dsm_inputs.check_geometry_plugin(
            inputs, conf.get(adv_cst.GEOMETRY_PLUGIN, None)
        )

    # Check pipeline
    overloaded_conf[adv_cst.PIPELINE] = conf.get(adv_cst.PIPELINE, "default")

    # Validate inputs
    schema = {
        adv_cst.DEBUG_WITH_ROI: bool,
        adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
        adv_cst.KEEP_LOW_RES_DIR: bool,
        adv_cst.GROUND_TRUTH_DSM: Or(dict, str),
        adv_cst.PHASING: Or(dict, None),
        adv_cst.GEOMETRY_PLUGIN: Or(str, dict),
        adv_cst.PIPELINE: str,
        adv_cst.DSM_MERGING_TILE_SIZE: And(int, lambda x: x > 0),
        adv_cst.LAND_COVER_MAP: str,
        adv_cst.CLASSIFICATION_TO_CONFIGURATION_MAPPING: str,
        adv_cst.USE_ENDOGENOUS_DEM: bool,
    }

    checker_advanced_parameters = Checker(schema)
    checker_advanced_parameters.validate(overloaded_conf)

    return (
        inputs,
        overloaded_conf,
        overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
        geom_plugin_without_dem_and_geoid,
        geom_plugin_with_dem_and_geoid,
        scaling_coeff,
        overloaded_conf[adv_cst.LAND_COVER_MAP],
        overloaded_conf[adv_cst.CLASSIFICATION_TO_CONFIGURATION_MAPPING],
    )


def check_ground_truth_dsm_data(conf):
    """
    Check data of the image ground truth

    :param conf: ground truth dsm configuration
    :type conf: str
    """
    if isinstance(conf, str):
        with rio.open(conf) as img_reader:
            trans = img_reader.transform
            if trans.e < 0:
                logging.warning(
                    "{} seems to have an incoherent pixel size. "
                    "Input images has to be in sensor geometry.".format(conf)
                )

    conf[adv_cst.INPUT_GEOID] = conf.get(adv_cst.INPUT_GEOID, None)

    if isinstance(conf, dict):
        ground_truth_dsm_schema = {
            adv_cst.INPUT_GROUND_TRUTH_DSM: str,
            OptionalKey(adv_cst.INPUT_AUX_PATH): Or(dict, None),
            OptionalKey(adv_cst.INPUT_AUX_INTERP): Or(dict, None),
            adv_cst.INPUT_GEOID: Or(None, str, bool),
        }

        checker_ground_truth_dsm_schema = Checker(ground_truth_dsm_schema)
        checker_ground_truth_dsm_schema.validate(conf)

        gt_dsm_path = conf[adv_cst.INPUT_GROUND_TRUTH_DSM]
        with rio.open(gt_dsm_path) as img_reader:
            trans = img_reader.transform
            if trans.e < 0:
                logging.warning(
                    "{} seems to have an incoherent pixel size. "
                    "Input images has to be in sensor geometry.".format(
                        gt_dsm_path
                    )
                )

        # Update geoid
        if isinstance(conf[adv_cst.INPUT_GEOID], bool):
            if conf[adv_cst.INPUT_GEOID]:
                # Use CARS geoid
                logging.info(
                    "CARS will use its own internal file as geoid reference"
                )
                package_path = os.path.dirname(__file__)
                geoid_path = os.path.join(
                    package_path, "..", "..", "conf", CARS_GEOID_PATH
                )
                conf[adv_cst.INPUT_GEOID] = geoid_path
            else:
                conf[adv_cst.INPUT_GEOID] = None

        path_dict = conf.get(adv_cst.INPUT_AUX_PATH, None)
        if path_dict is not None:
            for key in path_dict.keys():
                if not isinstance(path_dict[key], str):
                    raise RuntimeError("Path should be a string")
                if not os.path.exists(path_dict[key]):
                    raise RuntimeError("Path doesn't exist")

        path_interp = conf.get(adv_cst.INPUT_AUX_INTERP, None)
        if path_interp is not None:
            for key in path_interp.keys():
                if not isinstance(path_interp[key], str):
                    raise RuntimeError("interpolator should be a string")
                if path_interp[key] not in (
                    "nearest",
                    "linear",
                    "cubic",
                    "slinear",
                    "quintic",
                ):
                    raise RuntimeError("interpolator does not exist")
