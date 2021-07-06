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
Test module for cars/plugins/matching/correlator_configuratoin/
                correlator_configuration.py
"""

# Standard imports
import os
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.plugins.matching.correlator_configuration import corr_conf

# CARS Tests imports
from ....helpers import temporary_dir


@pytest.mark.unit_tests
def test_configure_pandora_default():
    """
    Test configure pandora correlator (default configuration)
    """
    corr_config = corr_conf.configure_correlator()
    assert (
        corr_config["pipeline"]["matching_cost"]["matching_cost_method"]
        == "census"
    )
    assert (
        corr_config["pipeline"]["optimization"]["optimization_method"] == "sgm"
    )


@pytest.mark.unit_tests
def test_configure_pandora_with_file():
    """
    Test configure pandora correlator
    """
    json_content = """{
    "input":{
        "nodata_left":"NaN",
        "nodata_right":"NaN"
    },
    "pipeline": {
        "right_disp_map": {
            "method": "accurate"
        },
        "matching_cost" : {
            "matching_cost_method": "census",
            "window_size": 5,
            "subpix": 1
        },
        "optimization" : {
            "optimization_method": "sgm",
            "P1": 8,
            "P2": 24,
            "p2_method": "constant",
            "penalty_method": "sgm_penalty",
            "overcounting": false,
            "min_cost_paths": false
        },
        "disparity": {
            "disparity_method": "wta",
            "invalid_disparity": "NaN"
        },
        "refinement": {
            "refinement_method": "vfit"
        },
        "filter" : {
            "filter_method": "median",
            "filter_size": 3
        },
        "validation" : {
            "validation_method": "cross_checking"
        }
    }
    }
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        json_path = os.path.join(directory, "corr.json")
        with open(json_path, "w") as json_file:
            json_file.write(json_content)

        corr_config = corr_conf.configure_correlator(json_path)
        assert corr_config["pipeline"]["optimization"]["P2"] == 24
