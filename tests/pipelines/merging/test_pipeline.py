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

from cars.pipelines.merging.merging import MergingPipeline

from ...helpers import absolute_data_path, assert_same_images
from ...helpers import cars_copy2 as copy2
from ...helpers import temporary_dir


@pytest.mark.end2end_tests
def test_pipeline_phased_dsm():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "dsms": {
                    "dsm1": {
                        "dsm": absolute_data_path(
                            "input/phr_gizeh/dsm1_phased.tif"
                        ),
                        "weights": absolute_data_path(
                            "input/phr_gizeh/weights1_phased.tif"
                        ),
                    },
                    "dsm2": {
                        "dsm": absolute_data_path(
                            "input/phr_gizeh/dsm2_phased.tif"
                        ),
                        "weights": absolute_data_path(
                            "input/phr_gizeh/weights2_phased.tif"
                        ),
                    },
                }
            },
            "merging": {
                "applications": {"dsm_merging": {"method": "weighted_fusion"}},
                "advanced": {"save_intermediate_data": True},
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        merging_pipeline = MergingPipeline(conf)
        merging_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dsm_test_pipeline_merging.tif"
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_test_pipeline_merging.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_pipeline_unphased_dsm():
    """
    End to end pipeline processing
    """
    conf = {
        "input": {
            "dsms": {
                "dsm1": {
                    "dsm": absolute_data_path(
                        "input/phr_gizeh/dsm1_unphased.tif"
                    ),
                    "weights": absolute_data_path(
                        "input/phr_gizeh/weights1_unphased.tif"
                    ),
                },
                "dsm2": {
                    "dsm": absolute_data_path(
                        "input/phr_gizeh/dsm2_unphased.tif"
                    ),
                    "weights": absolute_data_path(
                        "input/phr_gizeh/weights2_unphased.tif"
                    ),
                },
            }
        },
        "merging": {
            "applications": {"dsm_merging": {"method": "weighted_fusion"}},
            "advanced": {"save_intermediate_data": True},
        },
        "output": {"directory": "test_merging_pipeline_unphased_dsm"},
    }

    with pytest.raises(RuntimeError) as error:
        _ = MergingPipeline(conf)

    assert str(error.value) == "DSM dsm2 and dsm1 are not phased"
