#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars_geometry_plugin_otb
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
End2end tests
"""

# Standard imports
from __future__ import absolute_import

import os
import tempfile
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

# Third party imports
import pytest

# CARS imports
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_to_dense_dsm_pipeline as sensor_to_dense_dsm,
)

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    temporary_dir,
)


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique_shareloc():
    """
    End to end processing with shareloc geometry plugin
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input_shareloc.json")

        # Run dense dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {
                "method": "epipolar",
                "epi_step": 30,
            },
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "save_points_cloud": True,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": False,
            },
            "point_cloud_outliers_removing.1": {
                "method": "small_components",
                "activated": True,
            },
            "point_cloud_outliers_removing.2": {
                "method": "statistical",
                "activated": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {
                "method": "census_sgm",
                "loader_conf": {
                    "input": {},
                    "pipeline": {
                        "right_disp_map": {"method": "accurate"},
                        "matching_cost": {
                            "matching_cost_method": "census",
                            "window_size": 5,
                            "subpix": 1,
                        },
                        "cost_volume_confidence.before": {
                            "confidence_method": "ambiguity",
                            "eta_max": 0.7,
                            "eta_step": 0.01,
                        },
                        "cost_volume_confidence.std_intensity_before": {
                            "confidence_method": "std_intensity"
                        },
                        "cost_volume_confidence.risk_before": {
                            "confidence_method": "risk"
                        },
                        "optimization": {
                            "optimization_method": "sgm",
                            "penalty": {
                                "P1": 8,
                                "P2": 32,
                                "p2_method": "constant",
                                "penalty_method": "sgm_penalty",
                            },
                            "overcounting": False,
                            "min_cost_paths": False,
                        },
                        "cost_volume_confidence": {
                            "confidence_method": "ambiguity",
                            "eta_max": 0.7,
                            "eta_step": 0.01,
                        },
                        "cost_volume_confidence.std_intensity": {
                            "confidence_method": "std_intensity"
                        },
                        "cost_volume_confidence.risk": {
                            "confidence_method": "risk"
                        },
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": "NaN",
                        },
                        "refinement": {"refinement_method": "vfit"},
                        "filter": {"filter_method": "median", "filter_size": 3},
                        "validation": {
                            "validation_method": "cross_checking",
                            "cross_checking_threshold": 1.0,
                        },
                    },
                },
            },
        }

        input_config_dense_dsm["applications"] = application_config
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"
        # update geometry plugin
        input_config_dense_dsm["geometry_plugin"] = "SharelocGeometry"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Check used_conf for dense dsm
        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path("ref_output/dsm_end2end_ventoux_shareloc.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux_shareloc.tif"))
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_shareloc.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_shareloc.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
