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
Test module for config of cars/applications/dsm_filling/bulldozer_filling.py
"""

# Third party imports
import pytest

from cars.applications.dsm_filling.border_interpolation import (
    BorderInterpolation,
)
from cars.applications.dsm_filling.bulldozer_filling import BulldozerFilling

# CARS imports
from cars.applications.dsm_filling.exogenous_filling import ExogenousFilling


@pytest.mark.unit_tests
def test_check_full_conf_exogenous_filling():
    """
    Test configuration check for dsm filling application
    """
    conf = {
        "method": "exogenous_filling",
        "activated": True,
        "classification": ["cloud", "lake", "sea"],
        "fill_with_geoid": ["sea"],
        "save_intermediate_data": False,
    }
    _ = ExogenousFilling(conf)


@pytest.mark.unit_tests
def test_check_full_conf_bulldozer_filling():
    """
    Test configuration check for dsm filling application
    """
    conf = {
        "method": "bulldozer",
        "activated": True,
        "classification": ["cloud", "lake", "sea"],
        "save_intermediate_data": False,
    }
    _ = BulldozerFilling(conf)


@pytest.mark.unit_tests
def test_check_full_conf_border_interpolation():
    """
    Test configuration check for dsm filling application
    """
    conf = {
        "method": "border_interpolation",
        "activated": True,
        "classification": ["cloud", "lake", "sea"],
        "component_min_size": 5,
        "border_size": 10,
        "percentile": 10,
        "save_intermediate_data": False,
    }
    _ = BorderInterpolation(conf)
