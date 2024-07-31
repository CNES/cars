#!/usr/bin/env python  pylint: disable=too-many-lines
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
Test pipeline inputs
"""

import json
import os

import pytest
from json_checker.core.exceptions import MissKeyCheckerError

# CARS imports
from cars.pipelines.sensor_to_dense_dsm import sensors_inputs

# CARS Tests imports
from .helpers import absolute_data_path


@pytest.mark.unit_tests
def test_input_full_sensors():
    """
    Test with full input
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # Basic
    input_test = {
        "sensors": {
            "one": {
                "image": "img1_crop.tif",
                "geomodel": {"path": "img1_crop.geom"},
            },
            "two": {
                "image": "img2_crop.tif",
                "geomodel": {"path": "img2_crop.geom"},
            },
        },
        "pairing": [["one", "two"]],
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
        "initial_elevation": "srtm_dir/N29E031_KHEOPS.tif",
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_geom_squeezed_sensors():
    """
    Test with 2 levels squeezed goemodel:

    - path on top level
    - no geomodel
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_geom_squezed_sensors():
    """
    Test with 2 levels squeezed goemodel:

    - path on top level
    - no geomodel
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_minimal_sensors():
    """
    Test minimal config
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {"sensors": {"one": "img1_crop.tif", "two": "img2_crop.tif"}}

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_dem_with_geoid_sensors():
    """
    Test input with geoid
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
        "initial_elevation": {
            "dem": "srtm_dir/N29E031_KHEOPS.tif",
            "geoid": absolute_data_path("../../cars/conf/geoid/egm96.grd.hrd"),
        },
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_dem_no_geoid_sensors():
    """
    Test input with geoid
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
        "initial_elevation": {"dem": "srtm_dir/N29E031_KHEOPS.tif"},
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_dem_no_geoid_squeezed_sensors():
    """
    Test input with initial_elevation as string
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
        "initial_elevation": "srtm_dir/N29E031_KHEOPS.tif",
    }

    sensors_inputs.sensors_check_inputs(
        input_test, config_json_dir=os.path.dirname(input_json)
    )


@pytest.mark.unit_tests
def test_input_dem_epsg_exit_sensors():
    """
    Invalid configuration : epsg in inputs
    """

    input_json = absolute_data_path(
        "input/data_gizeh_crop/configfile_crop.json"
    )

    with open(input_json, encoding="utf-8") as dsc:
        inpput_dict = json.load(dsc)

    print(inpput_dict)

    # Generate new inputs

    # squeezed on two levels
    input_test = {
        "sensors": {
            "one": {"image": "img1_crop.tif"},
            "two": {"image": "img2_crop.tif"},
        },
        "pairing": [["one", "two"]],
        "epsg": 4326,
    }

    # epsg should not be defined in inputs
    with pytest.raises(MissKeyCheckerError):
        sensors_inputs.sensors_check_inputs(
            input_test, config_json_dir=os.path.dirname(input_json)
        )
