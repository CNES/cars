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

import pytest
import tempfile
import os

from cars import configuration_correlator as corr_cfg
from utils import temporary_dir


@pytest.mark.unit_tests
def test_configure_pandora_default():
    """
    Test configure pandora correlator (default configuration)
    """
    corr_config = corr_cfg.configure_correlator()
    assert corr_config["stereo"]["stereo_method"] == "census"
    assert corr_config["optimization"]["optimization_method"] == "sgm"


@pytest.mark.unit_tests
def test_configure_pandora_with_file():
    """
    Test configure pandora correlator
    """
    json_content = """{
    "image":{
        "nodata1":"np.nan",
        "nodata2":"np.nan"
    },
    "stereo" : {
        "stereo_method": "census",
        "window_size": 5,
        "subpix": 1
    },
    "aggregation" : {
        "aggregation_method": "none"
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
    "refinement": {
        "refinement_method": "vfit"
    },
    "filter" : {
        "filter_method": "median",
        "filter_size": 3
    },
    "validation" : {
        "validation_method": "cross_checking",
        "interpolated_disparity": "none"
    },
    "invalid_disparity": "np.nan"
    } 
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        json_path = os.path.join(directory, "corr.json")
        with open(json_path, "w") as f:
            f.write(json_content)

        corr_config = corr_cfg.configure_correlator(json_path)
        assert corr_config["optimization"]["P2"] == 24
