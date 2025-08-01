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

import numpy as np
import rasterio as rio
from json_checker import And, Checker, OptionalKey, Or

from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import depth_map_inputs
from cars.pipelines.parameters import depth_map_inputs_constants as depth_cst
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_inputs import CARS_GEOID_PATH
from cars.pipelines.pipeline_constants import ADVANCED


def check_advanced_parameters(inputs, conf, check_epipolar_a_priori=True):
    """
    Check the advanced parameters consistency

    :param inputs: configuration of inputs
    :type inputs: dict
    :param conf: configuration of advanced parameters
    :type conf: dict
    :param check_epipolar_a_priori: use epipolar a priori parameters
    :type check_epipolar_a_priori: bool

    :return: overloaded configuration
    :rtype: dict
    """

    overloaded_conf = conf.copy()

    overloaded_conf[adv_cst.SAVE_INTERMEDIATE_DATA] = conf.get(
        adv_cst.SAVE_INTERMEDIATE_DATA, False
    )

    overloaded_conf[adv_cst.KEEP_LOW_RES_DIR] = conf.get(
        adv_cst.KEEP_LOW_RES_DIR, True
    )

    overloaded_conf[adv_cst.DEBUG_WITH_ROI] = conf.get(
        adv_cst.DEBUG_WITH_ROI, False
    )

    overloaded_conf[adv_cst.PHASING] = conf.get(adv_cst.PHASING, None)

    overloaded_conf[adv_cst.EPIPOLAR_RESOLUTIONS] = conf.get(
        adv_cst.EPIPOLAR_RESOLUTIONS, [16, 4, 1]
    )

    overloaded_conf[adv_cst.MERGING] = conf.get(adv_cst.MERGING, False)
    overloaded_conf[adv_cst.DSM_MERGING_TILE_SIZE] = conf.get(
        adv_cst.DSM_MERGING_TILE_SIZE, 4000
    )

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
    overloaded_conf[adv_cst.PERFORMANCE_MAP_CLASSES] = conf.get(
        adv_cst.PERFORMANCE_MAP_CLASSES, default_performance_classes
    )
    if overloaded_conf[adv_cst.PERFORMANCE_MAP_CLASSES] is not None:
        check_performance_classes(
            overloaded_conf[adv_cst.PERFORMANCE_MAP_CLASSES]
        )

    (overloaded_conf[adv_cst.TEXTURE_BANDS]) = check_texture_bands(
        inputs, conf.get(adv_cst.TEXTURE_BANDS, None)
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

    if check_epipolar_a_priori:
        # Check conf use_epipolar_a_priori
        overloaded_conf[adv_cst.USE_EPIPOLAR_A_PRIORI] = conf.get(
            adv_cst.USE_EPIPOLAR_A_PRIORI, False
        )
        # Retrieve epipolar_a_priori if it is provided
        overloaded_conf[adv_cst.EPIPOLAR_A_PRIORI] = conf.get(
            adv_cst.EPIPOLAR_A_PRIORI, {}
        )
        # Retrieve terrain_a_priori if it is provided
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI] = conf.get(
            adv_cst.TERRAIN_A_PRIORI, {}
        )

    # Check geometry plugin
    geom_plugin_without_dem_and_geoid = None
    geom_plugin_with_dem_and_geoid = None
    dem_generation_roi = None

    # If use a priori, override initial elevation with dem_median
    if adv_cst.USE_EPIPOLAR_A_PRIORI in overloaded_conf:
        if overloaded_conf[adv_cst.USE_EPIPOLAR_A_PRIORI]:
            if adv_cst.DEM_MEDIAN in overloaded_conf[adv_cst.TERRAIN_A_PRIORI]:
                inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] = (
                    overloaded_conf[adv_cst.TERRAIN_A_PRIORI][
                        adv_cst.DEM_MEDIAN
                    ]
                )

    if inputs[sens_cst.SENSORS] is not None:
        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            inputs,
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_without_dem_and_geoid,
            geom_plugin_with_dem_and_geoid,
            dem_generation_roi,
        ) = sensor_inputs.check_geometry_plugin(
            inputs, conf.get(adv_cst.GEOMETRY_PLUGIN, None)
        )
    elif depth_cst.DEPTH_MAPS in inputs or dsm_cst.DSMS in inputs:
        # If there's an initial elevation with
        # point clouds as inputs, generate a plugin (used in dsm_filling)
        (
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_with_dem_and_geoid,
        ) = depth_map_inputs.check_geometry_plugin(
            inputs, conf.get(adv_cst.GEOMETRY_PLUGIN, None)
        )

    # Check pipeline
    overloaded_conf[adv_cst.PIPELINE] = conf.get(adv_cst.PIPELINE, "default")

    # Validate inputs
    schema = {
        adv_cst.DEBUG_WITH_ROI: bool,
        adv_cst.MERGING: bool,
        adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
        adv_cst.KEEP_LOW_RES_DIR: bool,
        adv_cst.GROUND_TRUTH_DSM: Or(dict, str),
        adv_cst.PHASING: Or(dict, None),
        adv_cst.PERFORMANCE_MAP_CLASSES: Or(None, list),
        adv_cst.GEOMETRY_PLUGIN: Or(str, dict),
        adv_cst.PIPELINE: str,
        adv_cst.DSM_MERGING_TILE_SIZE: And(int, lambda x: x > 0),
        adv_cst.TEXTURE_BANDS: list,
        adv_cst.EPIPOLAR_RESOLUTIONS: Or(int, list),
    }
    if check_epipolar_a_priori:
        schema[adv_cst.USE_EPIPOLAR_A_PRIORI] = bool
        schema[adv_cst.EPIPOLAR_A_PRIORI] = dict
        schema[adv_cst.TERRAIN_A_PRIORI] = dict

    checker_advanced_parameters = Checker(schema)
    checker_advanced_parameters.validate(overloaded_conf)

    # Validate epipolar schema
    epipolar_schema = {
        adv_cst.GRID_CORRECTION: Or(list, None),
        adv_cst.DISPARITY_RANGE: list,
    }
    checker_epipolar = Checker(epipolar_schema)

    # Check terrain a priori
    if check_epipolar_a_priori and overloaded_conf[adv_cst.TERRAIN_A_PRIORI]:
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI][adv_cst.DEM_MEDIAN] = (
            overloaded_conf[adv_cst.TERRAIN_A_PRIORI].get(
                adv_cst.DEM_MEDIAN, None
            )
        )
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI][adv_cst.DEM_MIN] = (
            overloaded_conf[adv_cst.TERRAIN_A_PRIORI].get(adv_cst.DEM_MIN, None)
        )
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI][adv_cst.DEM_MAX] = (
            overloaded_conf[adv_cst.TERRAIN_A_PRIORI].get(adv_cst.DEM_MAX, None)
        )
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI][
            adv_cst.ALTITUDE_DELTA_MIN
        ] = overloaded_conf[adv_cst.TERRAIN_A_PRIORI].get(
            adv_cst.ALTITUDE_DELTA_MIN, None
        )
        overloaded_conf[adv_cst.TERRAIN_A_PRIORI][
            adv_cst.ALTITUDE_DELTA_MAX
        ] = overloaded_conf[adv_cst.TERRAIN_A_PRIORI].get(
            adv_cst.ALTITUDE_DELTA_MAX, None
        )
        terrain_a_priori_schema = {
            adv_cst.DEM_MEDIAN: str,
            adv_cst.DEM_MIN: Or(str, None),  # TODO mandatory with local disp
            adv_cst.DEM_MAX: Or(str, None),
            adv_cst.ALTITUDE_DELTA_MIN: Or(int, None),
            adv_cst.ALTITUDE_DELTA_MAX: Or(int, None),
        }
        checker_terrain = Checker(terrain_a_priori_schema)
        checker_terrain.validate(overloaded_conf[adv_cst.TERRAIN_A_PRIORI])

    # check epipolar a priori for each image pair
    if (
        check_epipolar_a_priori
        and overloaded_conf[adv_cst.USE_EPIPOLAR_A_PRIORI]
    ):
        validate_epipolar_a_priori(overloaded_conf, checker_epipolar)

    return (
        inputs,
        overloaded_conf,
        overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
        geom_plugin_without_dem_and_geoid,
        geom_plugin_with_dem_and_geoid,
        dem_generation_roi,
    )


def check_performance_classes(performance_map_classes):
    """
    Check performance classes

    :param performance_map_classes: list for step defining border of class
    :type performance_map_classes: list or None
    """
    if len(performance_map_classes) < 2:
        raise RuntimeError("Not enough step for performance_map_classes")
    if performance_map_classes is not None:
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


def check_texture_bands(inputs, texture_bands_in_advanced_cfg):
    """
    Define bands used for texture and put it on advanced and
    inputs configuration
    """
    if inputs[sens_cst.SENSORS] is None:
        # Parameter "texture_band" is used only when inputs are sensors
        return []
    texture_bands_dict = {}
    bands_set = None
    if texture_bands_in_advanced_cfg is not None:
        texture_bands_dict[tuple(texture_bands_in_advanced_cfg)] = "advanced"
    for [sensor_left, _] in inputs[sens_cst.PAIRING]:
        image = inputs[sens_cst.SENSORS][sensor_left][sens_cst.INPUT_IMG]
        bands = set(image["bands"].keys())
        if bands_set is None:
            bands_set = bands
        else:
            bands_set = bands_set & bands
        texture_bands = image["texture_bands"]
        if texture_bands is not None:
            for texture_band in texture_bands:
                if texture_band not in bands:
                    raise RuntimeError(
                        "Band {} not found in sensor {}".format(
                            texture_band,
                            sensor_left,
                        )
                    )
            texture_bands_dict[tuple(texture_bands)] = "sensor " + sensor_left
            if len(texture_bands_dict) == 2:
                keys = list(texture_bands_dict.keys())
                raise RuntimeError(
                    "Texture bands defined in {} ({}) and {} ({}) "
                    "are not the same".format(
                        texture_bands_dict[keys[0]],
                        keys[0],
                        texture_bands_dict[keys[1]],
                        keys[1],
                    )
                )
    if len(texture_bands_dict) == 0:
        if len(bands_set) == 0:
            logging.warning("No common band found between sensors")
            return []
        logging.info(
            "Texture bands are not defined, "
            "taking all possible bands : {}".format(bands_set)
        )
        return sorted(bands_set)
    return list(next(iter(texture_bands_dict)))


def validate_epipolar_a_priori(
    overloaded_conf,
    checker_epipolar,
):
    """
    Validate inner epipolar configuration

    :param conf : input configuration json
    :type conf: dict
    :param overloaded_conf : overloaded configuration json
    :type overloaded_conf: dict
    :param checker_epipolar : json checker
    :type checker_epipolar: Checker
    """

    for key_image_pair in overloaded_conf[adv_cst.EPIPOLAR_A_PRIORI]:
        checker_epipolar.validate(
            overloaded_conf[adv_cst.EPIPOLAR_A_PRIORI][key_image_pair]
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


def update_conf(
    conf,
    grid_correction_coef=None,
    dmin=None,
    dmax=None,
    pair_key=None,
    dem_median=None,
    dem_min=None,
    dem_max=None,
    altitude_delta_max=None,
    altitude_delta_min=None,
):
    """
    Update the conf with grid correction and disparity range
    :param grid_correction_coef: grid correction coefficient
    :type grid_correction_coef: list
    :param dmin: disparity range minimum
    :type dmin: float
    :param dmax: disparity range maximum
    :type dmax: float
    :param pair_key: name of the inputs key pair
    :type pair_key: str
    """

    if pair_key is not None:
        if pair_key not in conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI]:
            conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI][pair_key] = {}
        if grid_correction_coef is not None:
            if len(grid_correction_coef) == 2:
                conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI][pair_key][
                    "grid_correction"
                ] = (
                    np.concatenate(grid_correction_coef[0], axis=0).tolist()[
                        :-1
                    ]
                    + np.concatenate(grid_correction_coef[1], axis=0).tolist()[
                        :-1
                    ]
                )
            else:
                conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI][pair_key][
                    "grid_correction"
                ] = list(grid_correction_coef)
        if None not in (dmin, dmax):
            conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI][pair_key][
                "disparity_range"
            ] = [
                dmin,
                dmax,
            ]

    if dem_median is not None:
        conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI]["dem_median"] = dem_median
    if dem_min is not None:
        conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI]["dem_min"] = dem_min
    if dem_max is not None:
        conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI]["dem_max"] = dem_max
    if altitude_delta_max is not None:
        conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
            "altitude_delta_max"
        ] = altitude_delta_max
    if altitude_delta_min is not None:
        conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
            "altitude_delta_min"
        ] = altitude_delta_min
