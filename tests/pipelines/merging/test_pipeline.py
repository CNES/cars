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
Test pipeline merging
"""

import os
import tempfile

import pytest

from cars.pipelines.merging.merging import MergingPipeline

from ...helpers import absolute_data_path, assert_same_images
from ...helpers import cars_copy2 as copy2
from ...helpers import temporary_dir

DEFAULT_TOL = 0.1
CARS_GITHUB_ACTIONS = (
    os.getenv("CARS_GITHUB_ACTIONS", "false").lower() == "true"
)


@pytest.mark.end2end_tests
def test_phased_dsm():
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
                os.path.join(intermediate_output_dir, "dsm_test_merging.tif")
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_test_merging.tif")
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_unphased_dsm():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
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
            "output": {"directory": directory},
        }

        with pytest.raises(RuntimeError) as error:
            _ = MergingPipeline(conf)

        assert str(error.value) == "DSM dsm2 and dsm1 are not phased"


@pytest.mark.end2end_tests
def test_auxiliary():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "dsms": {
                    "dsm1": {
                        "dsm": absolute_data_path(
                            "input/phr_gizeh_small/dsm.tif"
                        ),
                        "weights": absolute_data_path(
                            "input/phr_gizeh_small/weights.tif"
                        ),
                        "ambiguity": absolute_data_path(
                            "input/phr_gizeh_small/ambiguity.tif"
                        ),
                        "merging_classification": absolute_data_path(
                            "input/phr_gizeh_small/classification.tif"
                        ),
                        "contributing_pair": absolute_data_path(
                            "input/phr_gizeh_small/contributing_pair.tif"
                        ),
                        "merging_filling": absolute_data_path(
                            "input/phr_gizeh_small/filling.tif"
                        ),
                        "image": absolute_data_path(
                            "input/phr_gizeh_small/image.tif"
                        ),
                        "performance_map": absolute_data_path(
                            "input/phr_gizeh_small/performance_map.tif"
                        ),
                    },
                    "dsm2": {
                        "dsm": absolute_data_path(
                            "input/phr_gizeh_small/dsm.tif"
                        ),
                        "weights": absolute_data_path(
                            "input/phr_gizeh_small/weights.tif"
                        ),
                        "ambiguity": absolute_data_path(
                            "input/phr_gizeh_small/ambiguity.tif"
                        ),
                        "merging_classification": absolute_data_path(
                            "input/phr_gizeh_small/classification.tif"
                        ),
                        "contributing_pair": absolute_data_path(
                            "input/phr_gizeh_small/contributing_pair.tif"
                        ),
                        "merging_filling": absolute_data_path(
                            "input/phr_gizeh_small/filling.tif"
                        ),
                        "image": absolute_data_path(
                            "input/phr_gizeh_small/image.tif"
                        ),
                        "performance_map": absolute_data_path(
                            "input/phr_gizeh_small/performance_map.tif"
                        ),
                    },
                }
            },
            "merging": {
                "applications": {"dsm_merging": {"method": "weighted_fusion"}},
                "advanced": {"save_intermediate_data": True},
            },
            "output": {
                "directory": directory,
                "auxiliary": {
                    "ambiguity": True,
                    "classification": True,
                    "contributing_pair": True,
                    "filling": True,
                    "image": True,
                    "performance_map": True,
                },
            },
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
                    intermediate_output_dir,
                    "dsm_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ambiguity_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "contributing_pair_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_merging_auxiliary.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_test_merging_auxiliary.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_test_merging_auxiliary.tif")
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ambiguity_test_merging_auxiliary.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_test_merging_auxiliary.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "contributing_pair_test_merging_auxiliary.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "filling_test_merging_auxiliary.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "image_test_merging_auxiliary.tif")
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "performance_map_test_merging_auxiliary.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
