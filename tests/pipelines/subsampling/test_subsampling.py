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
Test module for cars/stereo.py
Important : Uses conftest.py for shared pytest fixtures
"""

import argparse
import copy
import json
import os
import tempfile

import pytest

from cars.cars import main_cli
from cars.pipelines.subsampling import subsampling

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_images,
)
from tests.helpers import cars_copy2 as copy2
from tests.helpers import (
    generate_input_json,
    temporary_dir,
)

NB_WORKERS = 2

DEFAULT_TOL = 0.1
CARS_GITHUB_ACTIONS = (
    os.getenv("CARS_GITHUB_ACTIONS", "false").lower() == "true"
)


@pytest.mark.parametrize(
    "resolution",
    [
        [2],
        [8, 4],
        [16, 4, 1],
        [64, 32],
    ],
)
@pytest.mark.unit_tests
def test_subsampling(resolution):
    """Test subsampling pipeline"""

    atol = DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001
    rtol = DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6

    intermediate_dir = absolute_data_path("intermediate_data")
    ref_output_dir = absolute_data_path("ref_output")

    file_map = {
        "left_image": ("left/left_image.tif", "img1_phr_ventoux_res_{res}.tif"),
        "right_image": (
            "right/right_image.tif",
            "img2_phr_ventoux_res_{res}.tif",
        ),
        "color": ("left/color_image.tif", "color1_phr_ventoux_res_{res}.tif"),
        "left_classif": (
            "left/left_classif.tif",
            "classif1_phr_ventoux_res_{res}.tif",
        ),
        "right_classif": (
            "right/right_classif.tif",
            "classif2_phr_ventoux_res_{res}.tif",
        ),
        "left_mask": (
            "left/left_mask.tif",
            "left_mask_phr_ventoux_res_{res}.tif",
        ),
    }

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = absolute_data_path(
            "input/phr_ventoux/input_with_color_and_classif.json"
        )

        _, input_conf = generate_input_json(
            conf,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )

        input_conf["subsampling"] = {"advanced": {"resolutions": resolution}}

        out_dirs = {
            "api": input_conf["output"]["directory"],
            "cli": os.path.join(
                input_conf["output"]["directory"], "..", "output_cli"
            ),
        }

        # Prepare CLI configuration
        cli_conf = copy.deepcopy(input_conf)
        cli_conf["pipeline"] = "subsampling"
        cli_conf["output"]["directory"] = out_dirs["cli"]

        conf_path = os.path.join(directory, "subsampling_conf_dump.json")
        with open(conf_path, "w", encoding="utf8") as f:
            json.dump(cli_conf, f)

        # Run API
        subsampling.SubsamplingPipeline(
            input_conf, absolute_data_path(directory)
        ).run()

        # Run CLI
        main_cli(argparse.Namespace(conf=conf_path))

        # Validation
        for out_dir in out_dirs.values():
            for res in resolution:
                if res == 1:
                    continue

                res_dir = os.path.join(out_dir, f"subsampling/res_{res}")

                for _, (rel_path, ref_name) in file_map.items():
                    src = os.path.join(res_dir, rel_path)
                    ref = os.path.join(ref_output_dir, ref_name.format(res=res))
                    inter = os.path.join(
                        intermediate_dir, ref_name.format(res=res)
                    )

                    # Copy intermediate/reference files
                    copy2(src, inter)

                    # Compare output with reference
                    assert_same_images(
                        src,
                        ref,
                        atol=atol,
                        rtol=rtol,
                    )
