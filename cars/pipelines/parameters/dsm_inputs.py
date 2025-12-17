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
# pylint: disable=too-many-lines
"""
CARS dsm inputs
"""

import logging

import numpy as np
from json_checker import Checker, Or

# CARS imports
import cars.pipelines.parameters.dsm_inputs_constants as dsm_cst
from cars.core import constants as cst
from cars.core import inputs
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import sensor_inputs as sens_inp
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def check_dsm_inputs(conf, config_dir=None):
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_dir: directory of used json/yaml, if
        user filled paths with relative paths
    :type config_dir: str

    :return: overloader inputs
    :rtype: dict
    """

    overloaded_conf = {}

    # Overload some optional parameters
    overloaded_conf[dsm_cst.DSMS] = {}

    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)

    overloaded_conf[sens_cst.INITIAL_ELEVATION] = (
        sens_inp.get_initial_elevation(
            conf.get(sens_cst.INITIAL_ELEVATION, None)
        )
    )

    overloaded_conf[sens_cst.SENSORS] = conf.get(sens_cst.SENSORS, None)

    overloaded_conf[sens_cst.PAIRING] = conf.get(sens_cst.PAIRING, None)

    # Validate inputs
    inputs_schema = {
        dsm_cst.DSMS: dict,
        sens_cst.ROI: Or(str, dict, None),
        sens_cst.INITIAL_ELEVATION: Or(dict, None),
        sens_cst.SENSORS: Or(dict, None),
        sens_cst.PAIRING: Or([[str]], None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate depth maps

    dsm_schema = {
        cst.DSM_CLASSIF: Or(str, None),
        cst.DSM_ALT: Or(str, None),
        cst.DSM_ALT_INF: Or(str, None),
        cst.DSM_ALT_SUP: Or(str, None),
        cst.DSM_WEIGHTS_SUM: Or(str, None),
        cst.DSM_MSK: Or(str, None),
        cst.DSM_NB_PTS: Or(str, None),
        cst.DSM_NB_PTS_IN_CELL: Or(str, None),
        cst.DSM_MEAN: Or(str, None),
        cst.DSM_STD_DEV: Or(str, None),
        cst.DSM_INF_MEAN: Or(str, None),
        cst.DSM_INF_STD: Or(str, None),
        cst.DSM_SUP_MEAN: Or(str, None),
        cst.DSM_SUP_STD: Or(str, None),
        cst.DSM_AMBIGUITY: Or(str, None),
        cst.DSM_PERFORMANCE_MAP: Or(str, None),
        cst.DSM_SOURCE_PC: Or(str, None),
        cst.DSM_FILLING: Or(str, None),
        cst.DSM_COLOR: Or(str, None),
    }

    checker_pc = Checker(dsm_schema)
    for dsm_key in conf[dsm_cst.DSMS]:
        # Get depth maps with default
        overloaded_conf[dsm_cst.DSMS][dsm_key] = {}
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_CLASSIF] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("merging_classification", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_COLOR] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("image", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_MSK] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("mask", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT_INF] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT_SUP] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_WEIGHTS_SUM] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("weights", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_NB_PTS] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_n_pts", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_NB_PTS_IN_CELL] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_pts_in_cell", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_STD_DEV] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_INF_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_INF_STD] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SUP_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SUP_STD] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_AMBIGUITY] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("ambiguity", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_PERFORMANCE_MAP] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("performance_map", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SOURCE_PC] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("contributing_pair", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_FILLING] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("merging_filling", None)

        # validate
        checker_pc.validate(overloaded_conf[dsm_cst.DSMS][dsm_key])

    # Modify to absolute path
    if config_dir is not None:
        modify_to_absolute_path(config_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    for dsm_key in conf[dsm_cst.DSMS]:
        # check sizes
        check_input_size(
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_ALT],
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_CLASSIF],
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_COLOR],
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_MASK],
        )

    # Check srtm dir
    sens_inp.check_srtm(
        overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
    )

    check_phasing(conf[dsm_cst.DSMS])

    overloaded_conf[sens_cst.LOADERS] = sens_inp.check_loaders(
        conf.get(sens_cst.LOADERS, {})
    )

    classif_loader = overloaded_conf[sens_cst.LOADERS][
        sens_cst.INPUT_CLASSIFICATION
    ]

    overloaded_conf[sens_cst.FILLING] = sens_inp.check_filling(
        conf.get(sens_cst.FILLING, {}), classif_loader
    )

    if sens_cst.SENSORS in conf and conf[sens_cst.SENSORS] is not None:
        sens_inp.check_sensors(conf, overloaded_conf, config_dir)

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
        return conf_geom_plugin, None

    # Initialize a geometry plugin with elevation information
    geom_plugin_with_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            dem=dem_path,
            geoid=conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
            default_alt=sens_cst.CARS_DEFAULT_ALT,
        )
    )

    return conf_geom_plugin, geom_plugin_with_dem_and_geoid


def check_input_size(dsm, classif, color, mask):
    """
    Check dsm, mask, color, classif given

    Images must have same size

    :param dsm: phased dsm path
    :type dsm: str
    :param classif: classif path
    :type classif: str
    :param color: color path
    :type color: str
    :param mask: mask path
    :type mask: str
    """

    if inputs.rasterio_get_nb_bands(dsm) != 1:
        raise RuntimeError("{} is not mono-band image".format(dsm))

    for path in [mask, color, classif]:
        if path is not None:
            if inputs.rasterio_get_size(dsm) != inputs.rasterio_get_size(path):
                raise RuntimeError(
                    "The image {} and {} "
                    "do not have the same size".format(dsm, path)
                )


def modify_to_absolute_path(config_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_dir: directory of the json configuration
    :type config_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for dsm_key in overloaded_conf[dsm_cst.DSMS]:
        dsms = overloaded_conf[dsm_cst.DSMS][dsm_key]
        for tag in [
            cst.INDEX_DSM_ALT,
            cst.DSM_CLASSIF,
            cst.INDEX_DSM_COLOR,
            cst.INDEX_DSM_MASK,
            cst.DSM_FILLING,
        ]:
            if dsms[tag] is not None:
                dsms[tag] = make_relative_path_absolute(dsms[tag], config_dir)

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_dir
            )


def check_phasing(dsm_dict):
    """
    Check if the dsm are phased, and if resolution and epsg code are equivalent

    :param dsm_dict: list of phased dsm
    :type dsm_dict: dict
    """

    ref_key = next(iter(dsm_dict))
    ref_epsg = inputs.rasterio_get_epsg_code(dsm_dict[ref_key]["dsm"])
    ref_profile = inputs.rasterio_get_profile(dsm_dict[ref_key]["dsm"])
    ref_transform = list(ref_profile["transform"])
    ref_res_x = ref_transform[0]
    ref_res_y = ref_transform[4]
    ref_bounds = inputs.rasterio_get_bounds(dsm_dict[ref_key]["dsm"])

    for dsm_key in dsm_dict:
        if dsm_key == ref_key:
            continue

        epsg = inputs.rasterio_get_epsg_code(dsm_dict[dsm_key]["dsm"])
        profile = inputs.rasterio_get_profile(dsm_dict[ref_key]["dsm"])
        transform = list(profile["transform"])
        res_x = transform[0]
        res_y = transform[4]
        bounds = inputs.rasterio_get_bounds(dsm_dict[dsm_key]["dsm"])

        if epsg != ref_epsg:
            raise RuntimeError(
                f"EPSG mismatch: DSM {dsm_key} has EPSG {epsg}, "
                f"expected {ref_epsg}."
            )

        if ref_res_x != res_x or ref_res_y != res_y:
            raise RuntimeError(
                f"Resolution mismatch: DSM {dsm_key} has resolution "
                f"{(res_x, res_y)}, expected {(ref_res_x, ref_res_y)}."
            )

        # Compare the left_bottom corner
        diff = ref_bounds[0:2] - bounds[0:2]
        resolution = np.array([ref_res_x, -ref_res_y])
        res_ratio = diff / resolution

        if ~np.all(np.equal(res_ratio, res_ratio.astype(int))) and ~np.all(
            np.equal(1 / res_ratio, (1 / res_ratio).astype(int))
        ):
            raise RuntimeError(f"DSM {dsm_key} and {ref_key} are not phased")
