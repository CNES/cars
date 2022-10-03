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
Pytest conftest.py for automatically sharing
See info : https://docs.pytest.org/en/6.2.x/fixture.html
"""

# Standard imports
import json

# Third party imports
import pytest

# CARS Tests imports
from .helpers import absolute_data_path

# Local testing function pytest fixtures (mainly stereo)
# Ease following stereo tests readability


@pytest.fixture(scope="module")
def images_and_grids_conf():  # pylint: disable=redefined-outer-name
    """
    Returns images (img1 and img2) and grids (left, right) configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["images_and_grids"]

    for tag in ["img1", "img2"]:
        configuration["input"][tag] = absolute_data_path(
            configuration["input"][tag]
        )

    for tag in ["left_epipolar_grid", "right_epipolar_grid"]:
        configuration["preprocessing"]["output"][tag] = absolute_data_path(
            configuration["preprocessing"]["output"][tag]
        )

    return configuration


@pytest.fixture(scope="module")
def color1_conf():  # pylint: disable=redefined-outer-name
    """
    Returns color1 configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["color1"]

    configuration["input"]["color1"] = absolute_data_path(
        configuration["input"]["color1"]
    )

    return configuration


@pytest.fixture(scope="module")
def color_pxs_conf():  # pylint: disable=redefined-outer-name
    """
    Returns color_pxs configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["color_pxs"]

    configuration["input"]["color1"] = absolute_data_path(
        configuration["input"]["color1"]
    )

    return configuration


@pytest.fixture(scope="module")
def no_data_conf():  # pylint: disable=redefined-outer-name
    """
    Returns no data configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["no_data"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_sizes_conf():  # pylint: disable=redefined-outer-name
    """
    Returns epipolar size configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["epipolar_sizes"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_origins_spacings_conf():  # pylint: disable=redefined-outer-name
    """
    Returns epipolar spacing configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["epipolar_origins_spacings"]

    return configuration


@pytest.fixture(scope="module")
def disparities_conf():  # pylint: disable=redefined-outer-name
    """
    Returns disparities configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(
        absolute_data_path(json_path), "r", encoding="utf-8"
    ) as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["disparities"]

    return configuration
