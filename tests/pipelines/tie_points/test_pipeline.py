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
Test  pipeline
"""

import os
import tempfile

import numpy as np
import pytest

from cars.pipelines.tie_points.tie_points import TiePointsPipeline

from ...helpers import absolute_data_path, temporary_dir


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
        outdir = os.path.join(conf["output"]["directory"], "image1_image2")
        tie_points_pipeline = TiePointsPipeline(conf)
        tie_points_pipeline.run()
        raw_matches = np.load(os.path.join(outdir, "raw_matches.npy"))
        filtered_matches = np.load(os.path.join(outdir, "filtered_matches.npy"))
        disparity_raw_matches = raw_matches[:, 2] - raw_matches[:, 0]
        epipolar_error_raw_matches = np.abs(
            raw_matches[:, 3] - raw_matches[:, 1]
        )
        disparity_filtered_matches = (
            filtered_matches[:, 2] - filtered_matches[:, 0]
        )
        epipolar_error_filtered_matches = np.abs(
            filtered_matches[:, 3] - filtered_matches[:, 1]
        )
        assert len(raw_matches) == 92
        assert np.mean(disparity_raw_matches) == pytest.approx(-358, abs=2)
        assert np.mean(epipolar_error_raw_matches) == pytest.approx(6, abs=0.5)
        assert len(filtered_matches) == 88
        assert np.mean(disparity_filtered_matches) == pytest.approx(-378, abs=2)
        assert np.mean(epipolar_error_filtered_matches) == pytest.approx(
            4.75, abs=0.5
        )


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
        outdir = os.path.join(conf["output"]["directory"], "image1_image2")
        tie_points_pipeline = TiePointsPipeline(conf)
        tie_points_pipeline.run()
        raw_matches = np.load(os.path.join(outdir, "raw_matches.npy"))
        filtered_matches = np.load(os.path.join(outdir, "filtered_matches.npy"))
        disparity_raw_matches = raw_matches[:, 2] - raw_matches[:, 0]
        epipolar_error_raw_matches = np.abs(
            raw_matches[:, 3] - raw_matches[:, 1]
        )
        disparity_filtered_matches = (
            filtered_matches[:, 2] - filtered_matches[:, 0]
        )
        epipolar_error_filtered_matches = np.abs(
            filtered_matches[:, 3] - filtered_matches[:, 1]
        )
        assert len(raw_matches) == 117
        assert np.mean(disparity_raw_matches) == pytest.approx(6, abs=0.5)
        assert np.mean(epipolar_error_raw_matches) == pytest.approx(6, abs=0.5)
        assert len(filtered_matches) == 114
        assert np.mean(disparity_filtered_matches) == pytest.approx(4, abs=0.5)
        assert np.mean(epipolar_error_filtered_matches) == pytest.approx(
            4.75, abs=0.5
        )


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
        outdir = os.path.join(conf["output"]["directory"], "image1_image2")
        tie_points_pipeline = TiePointsPipeline(conf)
        tie_points_pipeline.run()
        raw_matches = np.load(os.path.join(outdir, "raw_matches.npy"))
        filtered_matches = np.load(os.path.join(outdir, "filtered_matches.npy"))
        disparity_raw_matches = raw_matches[:, 2] - raw_matches[:, 0]
        epipolar_error_raw_matches = np.abs(
            raw_matches[:, 3] - raw_matches[:, 1]
        )
        disparity_filtered_matches = (
            filtered_matches[:, 2] - filtered_matches[:, 0]
        )
        epipolar_error_filtered_matches = np.abs(
            filtered_matches[:, 3] - filtered_matches[:, 1]
        )
        assert len(raw_matches) == 92
        assert np.mean(disparity_raw_matches) == pytest.approx(-358, abs=2)
        assert np.mean(epipolar_error_raw_matches) == pytest.approx(6, abs=0.5)
        assert len(filtered_matches) == 88
        assert np.mean(disparity_filtered_matches) == pytest.approx(-378, abs=2)
        assert np.mean(epipolar_error_filtered_matches) == pytest.approx(
            4.75, abs=0.5
        )
