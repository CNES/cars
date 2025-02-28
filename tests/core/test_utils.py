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
Test module for cars/core/utils.py
"""
# TODO: refacto/clean/dispatch with utils cars module

# Third party imports
import numpy as np
import pytest

# CARS imports
from cars.core import utils

# CARS Tests imports
from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_get_elevation_range_from_metadata():
    """
    Test the get_elevation_range_from_metadata function
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")

    (min_elev, max_elev) = utils.get_elevation_range_from_metadata(img)

    assert min_elev == 632.5
    assert max_elev == 1517.5


@pytest.mark.unit_tests
def test_angle_vectors():
    """
    Testing vectors and angle result reference
    """
    vector_1 = [1, 1, 1]
    vector_2 = [-1, -1, -1]
    angle_ref = np.pi

    angle_result = utils.angle_vectors(vector_1, vector_2)

    assert angle_result == angle_ref


@pytest.mark.unit_tests
def test_safe_cast_float():
    """
    Testing  safe_cast_float
    """

    data = "1256.36586 meters"
    transformed_data = utils.safe_cast_float(data)

    assert transformed_data == 1256.36586
