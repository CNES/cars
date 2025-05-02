#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
Test module for cars/bundleadjustment.py
"""

# Standard imports
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.bundleadjustment import cars_bundle_adjustment

# CARS Tests imports
from .helpers import absolute_data_path, generate_input_json, temporary_dir

NB_WORKERS = 2


@pytest.mark.end2end_tests
def test_cars_bundle_adjustment():
    """
    Test to check non regression api
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf_path, __ = generate_input_json(
            absolute_data_path("input/phr_gizeh/input_bundle_adjustment.json"),
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )
        cars_bundle_adjustment(conf_path, no_run_sparse=False)
