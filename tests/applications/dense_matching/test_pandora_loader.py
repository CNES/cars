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
Test module for cars pandora loadr
"""

# Standard imports

# Third party imports
import copy

import pytest

# CARS imports
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
    overload_pandora_conf_with_confidence,
)

# CARS Tests imports


@pytest.mark.unit_tests
def test_configure_pandora_default():
    """
    Test configure pandora correlator (default configuration)
    """

    pandora_loader = PandoraLoader(conf=None, method_name="census_sgm")
    corr_config = pandora_loader.get_conf()
    assert (
        corr_config["pipeline"]["matching_cost"]["matching_cost_method"]
        == "census"
    )
    assert (
        corr_config["pipeline"]["optimization"]["optimization_method"] == "sgm"
    )


@pytest.mark.unit_tests
def test_configure_pandora_config():
    """
    Test configure pandora correlator
    """

    pandora_config = {
        "input": {"nodata_left": "NaN", "nodata_right": "NaN"},
        "pipeline": {
            "right_disp_map": {"method": "accurate"},
            "matching_cost": {
                "matching_cost_method": "census",
                "window_size": 5,
                "subpix": 1,
            },
            "optimization": {
                "optimization_method": "sgm",
                "penalty": {
                    "P1": 8,
                    "P2": 24,
                    "p2_method": "constant",
                    "penalty_method": "sgm_penalty",
                },
                "overcounting": False,
                "min_cost_paths": False,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": "NaN",
            },
            "refinement": {"refinement_method": "vfit"},
            "filter": {"filter_method": "median", "filter_size": 3},
            "validation": {"validation_method": "cross_checking"},
        },
    }

    pandora_loader = PandoraLoader(conf=pandora_config, method_name=None)
    corr_config = pandora_loader.get_conf()

    assert corr_config["pipeline"]["optimization"]["penalty"]["P2"] == 24


@pytest.mark.unit_tests
def test_configure_cross_validation():
    """
    Test configure pandora correlator cross validation
    """

    pandora_config = {
        "input": {"nodata_left": "NaN", "nodata_right": "NaN"},
        "pipeline": {
            "right_disp_map": {"method": "accurate"},
            "matching_cost": {
                "matching_cost_method": "census",
                "window_size": 5,
                "subpix": 1,
            },
            "optimization": {
                "optimization_method": "sgm",
                "penalty": {
                    "P1": 8,
                    "P2": 24,
                    "p2_method": "constant",
                    "penalty_method": "sgm_penalty",
                },
                "overcounting": False,
                "min_cost_paths": False,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": "NaN",
            },
            "refinement": {"refinement_method": "vfit"},
            "filter": {"filter_method": "median", "filter_size": 3},
        },
    }

    # test 1, validation as input already
    conf_with_validation = copy.deepcopy(pandora_config)
    conf_with_validation["pipeline"].update(
        {"validation": {"validation_method": "cross_checking"}}
    )
    pandora_loader = PandoraLoader(
        conf=conf_with_validation, use_cross_validation=False
    )
    corr_config = pandora_loader.get_conf()
    assert "validation" in corr_config["pipeline"]

    # test 2: no validation as input, add it
    pandora_loader = PandoraLoader(
        conf=copy.deepcopy(pandora_config), use_cross_validation=True
    )
    corr_config = pandora_loader.get_conf()
    assert "validation" in corr_config["pipeline"]

    # test 3: no validation as input, do not add it
    pandora_loader = PandoraLoader(
        conf=copy.deepcopy(pandora_config), use_cross_validation=False
    )
    corr_config = pandora_loader.get_conf()
    assert "validation" not in corr_config["pipeline"]


@pytest.mark.unit_tests
def test_overload_pandora_conf_with_confidence():
    """
    Test overload_pandora_conf_with_confidence

    """

    conf = {"a": 1, "b": 2, "disparity": 3, "c": 4, "d": 5}

    confidence_conf = {"confidence1": 8, "confidence2": 9}

    # run

    new_conf = overload_pandora_conf_with_confidence(conf, confidence_conf)

    # test

    ref = {
        "a": 1,
        "b": 2,
        "confidence1": 8,
        "confidence2": 9,
        "disparity": 3,
        "c": 4,
        "d": 5,
    }

    # transform to string to test order
    assert " ".join(ref.keys()) == " ".join(new_conf.keys())
