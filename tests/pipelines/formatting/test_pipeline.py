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

import pytest

from cars.pipelines.default import default_pipeline

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


@pytest.mark.end2end_tests
def test_pipeline():
    """
    Test filling pipeline
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

        input_conf["subsampling"] = {}
        input_conf["subsampling"]["advanced"] = {}
        input_conf["subsampling"]["advanced"]["epipolar_resolutions"] = 1

        # without formatting
        input_conf["pipeline"] = ["subsampling", "surface_modeling"]

        pipeline = default_pipeline.DefaultPipeline(
            input_conf, absolute_data_path(directory)
        )
        pipeline.run()

        out_dir = os.path.join(input_conf["output"]["directory"])

        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(
                out_dir,
                "intermediate_data/surface_modeling/res1/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_phr_ventoux_pipeline_formating.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "intermediate_data/surface_modeling/res1/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_phr_ventoux_pipeline_formating.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data/surface_modeling/res1/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_phr_ventoux_pipeline_formating.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data/surface_modeling/res1/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "image_phr_ventoux_pipeline_formating.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        # with formatting

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

        input_conf["subsampling"] = {}
        input_conf["subsampling"]["advanced"] = {}
        input_conf["subsampling"]["advanced"]["epipolar_resolutions"] = 1

        input_conf["pipeline"] = [
            "subsampling",
            "surface_modeling",
            "formatting",
        ]

        input_conf["input"].pop("filling", None)

        pipeline = default_pipeline.DefaultPipeline(
            input_conf, absolute_data_path(directory)
        )
        pipeline.run()

        copy2(
            os.path.join(
                out_dir,
                "dsm/dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm__phr_ventoux_pipeline_formating.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "dsm/image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_phr_ventoux_pipeline_formating.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "dsm/dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_phr_ventoux_pipeline_formating.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "dsm/image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "image_phr_ventoux_pipeline_formating.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
