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
import json
import os
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.bundleadjustment import cars_bundle_adjustment

# CARS Tests imports
from .helpers import absolute_data_path, generate_input_json, temporary_dir

NB_WORKERS = 2


@pytest.mark.end2end_tests
def test_cars_bundle_adjustment_output_and_rectification_grids():
    """
    Test bundle adjustment with additional output parameters
    and predefined rectification grids.
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf_path, _ = generate_input_json(
            absolute_data_path("input/phr_gizeh/input_bundle_adjustment.json"),
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
            pipeline="tie_points",
        )

        with open(conf_path, encoding="utf-8") as file:
            config = json.load(file)

        # Check that additional output parameters do not interfere
        # with the sparse matching pipeline configuration.
        config["output"]["product_level"] = ["dsm"]
        config["output"]["resolution"] = 0.6

        # Check that bundle adjustment can reuse predefined
        # rectification grids with their associated metadata.
        grid_metadata = {
            "grid_spacing": [30.0, 30.0],
            "grid_origin": [-150.0, -150.0],
            "epipolar_size_x": 867,
            "epipolar_size_y": 816,
            "epipolar_origin_x": -150.0,
            "epipolar_origin_y": -150.0,
            "epipolar_spacing_x": 30.0,
            "epipolar_spacing": 30.0,
            "disp_to_alt_ratio": 6.063405070767783,
            "epipolar_step": 30,
        }

        grid_paths = {
            key: absolute_data_path(f"input/phr_gizeh/{key}_epi_grid.tif")
            for key in ["one", "two", "three"]
        }

        config["input"]["rectification_grids"] = {
            key: {
                **grid_metadata,
                "path": path,
            }
            for key, path in grid_paths.items()
        }

        config["input"]["pairing"] = [
            ["one", "two"],
            ["one", "three"],
        ]

        config["tie_points"]["applications"]["bundle_adjustment"]["pairing"] = [
            ["one", "two"],
            ["one", "three"],
        ]

        for path in grid_paths.values():
            assert os.path.isfile(path), f"Missing rectification grid: {path}"

        with open(conf_path, "w", encoding="utf-8") as file:
            json.dump(config, file)

        cars_bundle_adjustment(conf_path, no_run_sparse=False)


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
            pipeline="tie_points",
        )
        cars_bundle_adjustment(conf_path, no_run_sparse=False)
