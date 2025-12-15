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
import tempfile

import pytest

from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

from ...helpers import absolute_data_path, temporary_dir


@pytest.mark.end2end_tests
def test_pipeline_with_low_res_dsm():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path("input/phr_gizeh/img1.tif"),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path("input/phr_gizeh/img2.tif"),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
                "low_res_dsm": absolute_data_path(
                    "input/phr_gizeh/low_res_dsm.tif"
                ),
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "surface_modeling": {"advanced": {"save_intermediate_data": True}},
            "tie_points": {"advanced": {"save_intermediate_data": True}},
            "output": {"directory": directory},
        }
        # outdir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
