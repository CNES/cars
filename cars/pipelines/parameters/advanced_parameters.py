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

import numpy as np
from json_checker import Checker, Or

from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.pipeline_constants import ADVANCED


def check_advanced_parameters(conf, check_epipolar_a_priori=True):
    """
    Check the advanced parameters consistency

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

    overloaded_conf[adv_cst.DEBUG_WITH_ROI] = conf.get(
        adv_cst.DEBUG_WITH_ROI, False
    )

    overloaded_conf[adv_cst.MERGING] = conf.get(adv_cst.MERGING, False)

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

    # Validate inputs
    schema = {
        adv_cst.DEBUG_WITH_ROI: bool,
        adv_cst.MERGING: bool,
        adv_cst.SAVE_INTERMEDIATE_DATA: bool,
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

    return overloaded_conf


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
