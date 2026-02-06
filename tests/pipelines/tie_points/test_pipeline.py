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
Test pipeline tie points
"""

import argparse
import copy
import json
import os
import tempfile

import numpy as np
import pytest

from cars.cars import main_cli
from cars.pipelines.tie_points.tie_points import TiePointsPipeline

from ...helpers import absolute_data_path, temporary_dir


def _check_tie_points_output(outdir, expected):
    raw_matches = np.load(os.path.join(outdir, "raw_matches.npy"))
    filtered_matches = np.load(os.path.join(outdir, "filtered_matches.npy"))

    disparity_raw = raw_matches[:, 2] - raw_matches[:, 0]
    epipolar_raw = np.abs(raw_matches[:, 3] - raw_matches[:, 1])

    disparity_filtered = filtered_matches[:, 2] - filtered_matches[:, 0]
    epipolar_filtered = np.abs(filtered_matches[:, 3] - filtered_matches[:, 1])

    assert len(raw_matches) == expected["raw_count"]
    assert np.mean(disparity_raw) == pytest.approx(
        expected["raw_disp"], abs=expected["disp_tol"]
    )
    assert np.mean(epipolar_raw) == pytest.approx(
        expected["raw_epi"], abs=expected["epi_tol"]
    )

    assert len(filtered_matches) == expected["filtered_count"]
    assert np.mean(disparity_filtered) == pytest.approx(
        expected["filtered_disp"], abs=expected["disp_tol"]
    )
    assert np.mean(epipolar_filtered) == pytest.approx(
        expected["filtered_epi"], abs=expected["epi_tol"]
    )


def _run_and_check(conf_api, expected):

    # init cli conf
    conf_cli = copy.deepcopy(conf_api)
    conf_cli["pipeline"] = "tie_points"
    conf_cli["output"]["directory"] = os.path.join(
        conf_api["output"]["directory"], "..", "out_cli"
    )

    # out dirs
    api_outdir = os.path.join(conf_api["output"]["directory"], "image1_image2")
    cli_outdir = os.path.join(conf_cli["output"]["directory"], "image1_image2")

    # write cli conf to file
    conf_path = os.path.join(
        conf_api["output"]["directory"], "tie_points_conf.json"
    )
    with open(conf_path, "w", encoding="utf8") as f:
        json.dump(conf_cli, f)

    # run pipelines
    TiePointsPipeline(conf_api).run()
    main_cli(argparse.Namespace(conf=conf_path))

    _check_tie_points_output(api_outdir, expected)
    _check_tie_points_output(cli_outdir, expected)


@pytest.mark.end2end_tests
def test_pipeline_ventoux():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                    },
                }
            },
            "tie_points": {"advanced": {"save_intermediate_data": False}},
            "output": {"directory": directory},
        }

        expected = {
            "raw_count": 92,
            "filtered_count": 88,
            "raw_disp": -358,
            "filtered_disp": -378,
            "raw_epi": 6,
            "filtered_epi": 4.75,
            "disp_tol": 2,
            "epi_tol": 0.5,
        }

        _run_and_check(conf, expected)


@pytest.mark.end2end_tests
def test_pipeline_ventoux_with_dem():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                    },
                },
                "initial_elevation": absolute_data_path(
                    "input/phr_ventoux/srtm/N44E005.hgt"
                ),
            },
            "tie_points": {"advanced": {"save_intermediate_data": False}},
            "output": {"directory": directory},
        }

        expected = {
            "raw_count": 117,
            "filtered_count": 114,
            "raw_disp": 6,
            "filtered_disp": 4,
            "raw_epi": 6,
            "filtered_epi": 4.75,
            "disp_tol": 0.5,
            "epi_tol": 0.5,
        }

        _run_and_check(conf, expected)


@pytest.mark.end2end_tests
def test_pipeline_ventoux_with_mask():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                        "mask": absolute_data_path(
                            "input/phr_ventoux/left_mask.tif"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                        "mask": absolute_data_path(
                            "input/phr_ventoux/right_mask.tif"
                        ),
                    },
                },
            },
            "tie_points": {"advanced": {"save_intermediate_data": False}},
            "output": {"directory": directory},
        }

        expected = {
            "raw_count": 92,
            "filtered_count": 88,
            "raw_disp": -358,
            "filtered_disp": -378,
            "raw_epi": 6,
            "filtered_epi": 4.75,
            "disp_tol": 2,
            "epi_tol": 0.5,
        }

        _run_and_check(conf, expected)
