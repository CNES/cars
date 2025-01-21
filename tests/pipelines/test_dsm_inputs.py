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
Test module for cars/pipelines/parameters/dsm_inputs.py
"""

import pytest

from cars.pipelines.parameters import dsm_inputs

from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_input_none_phased_dsm():
    """
    Test with non phased dsms
    """

    dsm_one = absolute_data_path("ref_output/dsm_end2end_ventoux_split.tif")

    dsm_two = absolute_data_path("ref_output/dsm_end2end_paca_bulldozer.tif")

    input_test = {
        "one": {
            "dsm": dsm_one,
        },
        "two": {
            "dsm": dsm_two,
        },
    }

    with pytest.raises(RuntimeError) as exception:
        dsm_inputs.check_phasing(input_test)

    assert str(exception.value) == "DSM two and one are not phased"


@pytest.mark.unit_tests
def test_input_same_dsm():
    """
    Test with same dsms
    """

    dsm_one = absolute_data_path("ref_output/dsm_end2end_ventoux_split.tif")

    dsm_two = absolute_data_path("ref_output/dsm_end2end_ventoux_split.tif")

    input_test = {
        "one": {
            "dsm": dsm_one,
        },
        "two": {
            "dsm": dsm_two,
        },
    }

    dsm_inputs.check_phasing(input_test)
