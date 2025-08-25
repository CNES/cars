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
Test pipeline output
"""

import os
import tempfile

import pytest
from json_checker.core.exceptions import DictCheckerError

from cars.pipelines.parameters import output_parameters

from ..helpers import temporary_dir


@pytest.mark.unit_tests
def test_output_full():
    """
    Test output
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "product_level": "dsm",
            "auxiliary": {
                "performance_map": False,
                "mask": False,
                "texture": True,
                "classification": False,
                "contributing_pair": False,
            },
            "epsg": 4326,
            "resolution": 0.5,
            "geoid": "path/to/geoid",
            "save_by_pair": False,
        }

        print(f"config {config}")
        output_parameters.check_output_parameters(config, 1)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "case",
    [
        {"epsg": None, "expected": "valid"},
        {"epsg": 4326, "expected": "valid"},
        {"epsg": "4326", "expected": "valid"},
        {"epsg": "4326+5773", "expected": "valid"},
        {"epsg": "3857", "expected": "valid"},
        {"epsg": 999999, "expected": "invalid"},
        {"epsg": 0, "expected": "invalid"},
        {"epsg": "not_a_code", "expected": "invalid"},
    ],
)
def test_output_epsg(case):
    """
    Test output_parameters.check_output_parameters with different EPSG inputs
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "product_level": "dsm",
            "auxiliary": {
                "performance_map": False,
                "mask": False,
                "texture": True,
                "classification": False,
                "contributing_pair": False,
            },
            "epsg": case["epsg"],
            "resolution": 0.5,
            "geoid": "path/to/geoid",
            "save_by_pair": False,
        }

        if case["expected"] == "valid":
            # Should succeed without raising
            output_parameters.check_output_parameters(config)
        else:
            with pytest.raises(DictCheckerError):
                # Expecting some sort of failure
                output_parameters.check_output_parameters(config)


@pytest.mark.unit_tests
def test_output_minimal():
    """
    Test output
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {"directory": os.path.join(directory, "outdir")}

        print(f"config {config}")
        overload = output_parameters.check_output_parameters(config, 1)
        print(overload)
