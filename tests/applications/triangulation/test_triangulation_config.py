#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
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
Test module for config of
cars/applications/triangulation/line_of_sight_intersection_app.py
"""

# Third party imports
import pytest

# CARS imports
from cars.applications.triangulation.line_of_sight_intersection_app import (
    LineOfSightIntersection,
)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "conf_normalization, expected_normalization",
    [
        (None, [1, 0]),
        ([0.5, -2.0], [0.5, -2.0]),
    ],
)
def test_performance_map_affine_normalization_conf(
    conf_normalization, expected_normalization
):
    """
    Test performance map affine normalization configuration.
    """
    conf = {
        "method": "line_of_sight_intersection",
        "snap_to_img1": False,
        "save_intermediate_data": False,
    }

    if conf_normalization is not None:
        conf["performance_map_affine_normalization"] = conf_normalization

    application = LineOfSightIntersection(conf)

    assert (
        application.performance_map_affine_normalization
        == expected_normalization
    )


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "normalization",
    [
        [0, 1],
        [-1, 1],
        [-0.5, -2],
    ],
)
def test_invalid_performance_map_affine_normalization_conf(normalization):
    """
    Test invalid affine normalization coefficient.
    """
    conf = {
        "method": "line_of_sight_intersection",
        "snap_to_img1": False,
        "save_intermediate_data": False,
        "performance_map_affine_normalization": normalization,
    }

    with pytest.raises(
        ValueError,
        match="first coefficient must be strictly positive",
    ):
        LineOfSightIntersection(conf)
