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
from cars.pipelines.filling import filling

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


@pytest.mark.end2end_tests
def test_pipeline_filling_end2end_global():
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

        input_conf["input"]["filling"] = {
            "fill_with_exogenous_dem": 4,
            "interpolate_from_borders": 2,
            "fill_with_endogenous_dem": 1,
            "fill_with_geoid": 3,
        }

        input_conf["subsampling"] = {}
        input_conf["subsampling"]["advanced"] = {}
        input_conf["subsampling"]["advanced"]["resolutions"] = [1]

        input_conf["pipeline"] = ["subsampling", "surface_modeling", "filling"]

        input_conf["filling"] = {}

        input_conf["filling"]["applications"] = {
            "auxiliary_filling": {
                "method": "auxiliary_filling_from_sensors",
                "activated": True,
                "mode": "full",
                "save_intermediate_data": True,
            }
        }

        input_conf["output"]["auxiliary"] = {
            "filling": True,
            "classification": True,
        }

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
                "intermediate_data",
                "filling/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_filled_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_filled_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "classification.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "filling.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_filled_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "image_filled_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "classification.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "filling/dsm",
                "filling.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "filling_phr_ventoux_pipeline_filling_global.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_pipeline():
    """
    Test filling pipeline
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_conf = {
            "input": {
                "loaders": {"image": "pivot"},
                "sensors": {
                    "image1": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/left_image.tif"
                                    ),
                                    "band": 0,
                                },
                                "b1": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/color_image.tif"
                                    ),
                                    "band": 0,
                                },
                                "b2": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/color_image.tif"
                                    ),
                                    "band": 1,
                                },
                                "b3": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/color_image.tif"
                                    ),
                                    "band": 2,
                                },
                                "b4": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/color_image.tif"
                                    ),
                                    "band": 3,
                                },
                            }
                        },
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/left_classif.tif"
                        ),
                    },
                    "image2": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": absolute_data_path(
                                        "input/phr_ventoux/right_image.tif"
                                    ),
                                    "band": 0,
                                }
                            }
                        },
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/right_classif.tif"
                        ),
                    },
                },
                "initial_elevation": absolute_data_path(
                    "input/phr_ventoux/srtm/N44E005.hgt"
                ),
            },
            "output": {
                "directory": directory,
            },
        }

        input_conf["input"]["filling"] = {
            "fill_with_exogenous_dem": 4,
            "interpolate_from_borders": 2,
            "fill_with_endogenous_dem": 1,
            "fill_with_geoid": 3,
        }

        path_input = "tests/data/input/input_filling_pipeline/"
        classif_path = "classif_test_surface_modeling_ventoux.tif"

        input_conf["input"]["dsm_to_fill"] = {
            "dsm": path_input + "dsm_test_surface_modeling_ventoux.tif",
            "image": path_input + "image_test_surface_modeling_ventoux.tif",
            "classification": path_input + classif_path,
            "filling": path_input + "filling_test_surface_modeling_ventoux.tif",
        }

        input_conf["subsampling"] = {}
        input_conf["subsampling"]["advanced"] = {}
        input_conf["subsampling"]["advanced"]["resolutions"] = 1

        input_conf["filling"] = {}

        input_conf["filling"]["applications"] = {
            "auxiliary_filling": {
                "method": "auxiliary_filling_from_sensors",
                "activated": True,
                "mode": "full",
                "save_intermediate_data": True,
            }
        }

        input_conf["output"]["auxiliary"] = {
            "filling": True,
            "classification": True,
            "image": ["b1", "b2", "b3"],
        }

        pipeline = filling.FillingPipeline(
            input_conf, absolute_data_path(directory)
        )
        pipeline.run()

        out_dir = os.path.join(input_conf["output"]["directory"])

        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(
                out_dir,
                "filling/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_filled_phr_ventoux_pipeline_filling.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "filling/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_filled_phr_ventoux_pipeline_filling.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "filling/dsm",
                "classification.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_phr_ventoux_pipeline_filling.tif",
                )
            ),
        )

        copy2(
            os.path.join(
                out_dir,
                "filling/dsm",
                "filling.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_phr_ventoux_pipeline_filling.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "filling/dsm",
                "dsm.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_filled_phr_ventoux_pipeline_filling.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "filling/dsm",
                "image.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "image_filled_phr_ventoux_pipeline_filling.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "filling/dsm",
                "classification.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_phr_ventoux_pipeline_filling.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )

        assert_same_images(
            os.path.join(
                out_dir,
                "filling/dsm",
                "filling.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "filling_phr_ventoux_pipeline_filling.tif",
                )
            ),
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )
