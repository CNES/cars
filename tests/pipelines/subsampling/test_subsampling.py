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

import os
import tempfile

import pytest

from cars.pipelines.subsampling import subsampling

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_images,
)
from tests.helpers import cars_copy2 as copy2
from tests.helpers import (
    temporary_dir,
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
    """
    Test subsampling pipeline
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        path = "tutorials/data_gizeh_small/"
        conf = {
            "input": {
                "loaders": {"image": "pivot"},
                "sensors": {
                    "one": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": os.path.join(path, "img1.tif"),
                                    "band": 0,
                                },
                                "b1": {
                                    "path": os.path.join(path, "color1.tif"),
                                    "band": 1,
                                },
                                "b2": {
                                    "path": os.path.join(path, "color1.tif"),
                                    "band": 2,
                                },
                                "b3": {
                                    "path": os.path.join(path, "color1.tif"),
                                    "band": 2,
                                },
                            }
                        },
                        "classification": os.path.join(path, "classif1.tif"),
                    },
                    "two": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": os.path.join(path, "img2.tif"),
                                    "band": 0,
                                }
                            }
                        },
                        "classification": os.path.join(path, "classif2.tif"),
                    },
                },
            },
            "output": {"directory": os.path.join(directory, "test")},
            "advanced": {"epipolar_resolutions": resolution},
        }
        dense_dsm_pipeline = subsampling.SubsamplingPipeline(
            conf, absolute_data_path(directory)
        )
        dense_dsm_pipeline.run()

        out_dir = os.path.dirname(os.path.join(conf["output"]["directory"]))

        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        for res in resolution:
            copy2(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/img1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "img1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "two/img2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "img2_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/color1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "color1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/classif1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classif1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "two/classif2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classif2_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
            )

            assert_same_images(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/img1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "img1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "two/img2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "img2_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/color1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "color1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "one/classif1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classif1_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir, "subsampling/res_" + str(res), "two/classif2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classif2_data_gizeh_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
