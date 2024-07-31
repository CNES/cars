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
Test module for cars/pipelines/sensors_inputs.py
"""

import os

import pytest

from cars.pipelines.sensor_to_dense_dsm import sensors_inputs

from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_check_full_conf():
    """
    Test configuration check for sensors inputs
    """
    input_json = absolute_data_path("input/phr_ventoux/input.json")
    json_dir_path = os.path.dirname(input_json)
    conf = {
        "sensors": {
            "left": {
                "image": "left_image.tif",
                "color": "color_image.tif",
                "geomodel": {"path": "left_image.geom"},
                "no_data": 0,
                "mask": None,
                "classification": None,
            },
            "right": {
                "image": "right_image.tif",
                "geomodel": {"path": "right_image.geom"},
                "color": "right_image.tif",
                "no_data": 0,
                "mask": None,
                "classification": None,
            },
        },
        "pairing": [["left", "right"]],
        "initial_elevation": {
            "dem": "srtm/N44E005.hgt",
            "default_alt": 0,
        },
        "use_endogenous_elevation": True,
        "roi": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [
                            [
                                [5.194, 44.2064],
                                [5.194, 44.2059],
                                [5.195, 44.2059],
                                [5.195, 44.2064],
                                [5.194, 44.2064],
                            ]
                        ],
                        "type": "Polygon",
                    },
                }
            ],
        },
        "debug_with_roi": False,
        "check_inputs": False,
        "use_epipolar_a_priori": False,
        "epipolar_a_priori": {
            "left_right": {
                "grid_correction": [
                    4.288258198116454,
                    -5.126994797702842e-05,
                    -0.0001868504592756687,
                    0.7633338209640763,
                    0.00017551039816977951,
                    7.416961236000769e-05,
                ],
                "disparity_range": [-26.157764557028997, 26.277429517242638],
            }
        },
        "terrain_a_priori": {},
    }
    _ = sensors_inputs.sensors_check_inputs(conf, config_json_dir=json_dir_path)
