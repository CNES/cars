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
    generate_input_json,
    temporary_dir,
)

NB_WORKERS = 2


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
        conf = absolute_data_path(
            "input/phr_ventoux/input_with_color_and_classif.json"
        )

        # Run dense dsm pipeline
        _, input_conf = generate_input_json(
            conf,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )

        input_conf["advanced"] = {"epipolar_resolutions": resolution}

        dense_dsm_pipeline = subsampling.SubsamplingPipeline(
            input_conf, absolute_data_path(directory)
        )
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_conf["output"]["directory"])

        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        for res in resolution:
            copy2(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/left_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "img1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "right/right_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "img2_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/color_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "color1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/left_classif.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classif1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
            )

            copy2(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "right/right_classif.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classif2_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
            )

            assert_same_images(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/left_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "img1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "right/right_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "img2_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/color_image.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "color1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "left/left_classif.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classif1_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir,
                    "subsampling/res_" + str(res),
                    "right/right_classif.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classif2_phr_ventoux_res_" + str(res) + ".tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
