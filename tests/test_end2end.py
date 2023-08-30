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
Test module End to End:
Prepare and Compute DSM run user tests through pipelines run() functions
TODO: Cars_cli is not tested
TODO: Refactor in several files and remove too-many-lines
"""

# Standard imports
from __future__ import absolute_import

import json
import math
import os
import shutil
import tempfile
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

# Third party imports
import pyproj
import pytest
import rasterio
from shapely.ops import transform

from cars.conf.input_parameters import read_input_parameters

# CARS imports
from cars.core import roi_tools
from cars.pipelines.point_clouds_to_dsm import (
    point_cloud_to_dsm_pipeline as pipeline_dsm,
)
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_to_dense_dsm_pipeline as sensor_to_dense_dsm,
)
from cars.pipelines.sensor_to_sparse_dsm import (
    sensor_to_sparse_dsm_pipeline as sensor_to_sparse_dsm,
)

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    temporary_dir,
)


@pytest.mark.end2end_tests
def test_end2end_gizeh_rectangle_epi_image_performance_map():
    """
    End to end processing

    Test pipeline with a non square epipolar image, and the generation
    of the performance map
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/data_gizeh_crop/configfile_crop.json"
        )

        # Run dense dsm pipeline
        _, input_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 300,
            },
        )
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "dense_matching": {
                "method": "census_sgm",
                "generate_performance_map": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "save_msk": True,
                "save_confidence": True,
            },
        }
        input_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm.tif"),
        #     absolute_data_path("ref_output/dsm_end2end_gizeh_crop.tif"),
        # )
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_gizeh_crop.tif"))
        # copy2(os.path.join(out_dir, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_gizeh_crop.tif"))
        # copy2(os.path.join(out_dir, 'confidence_performance_map.tif'),
        #      absolute_data_path(
        #           "ref_output/performance_map_end2end_gizeh_crop.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_gizeh_crop.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_gizeh_crop.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_gizeh_crop.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_performance_map.tif"),
            absolute_data_path(
                "ref_output/performance_map_end2end_gizeh_crop.tif"
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique():
    """
    End to end processing
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run sparse dsm pipeline
        _, input_config_sparse_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
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
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                -20
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["minimum_disparity"]
                < -18
            )
            assert (
                14
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 15
            )

            # check matches file exists
            assert os.path.isfile(
                out_json["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Check used_conf for sparse res

        gt_used_conf_orchestrator = {
            "orchestrator": {
                "mode": "local_dask",
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
                "activate_dashboard": False,
                "profiling": {
                    "activated": False,
                    "mode": "time",
                    "loop_testing": False,
                },
                "python": None,
                "use_memory_logger": False,
                "config_name": "unknown",
            }
        }

        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensors_to_sparse_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["sparse_matching"]["disparity_margin"]
                == 0.25
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            assert (
                os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675240.0_4897185.0.laz"
                    )
                )
            ) is True

            for k in range(0, 8):
                nb_str = str(k)
                assert (
                    os.path.exists(
                        os.path.join(
                            out_dir,
                            "left_right",
                            "epi_pc",
                            nb_str + ".csv",
                        )
                    )
                ) is True
                assert (
                    os.path.exists(
                        os.path.join(
                            out_dir,
                            "left_right",
                            "epi_pc",
                            nb_str + "_attrs.json",
                        )
                    )
                ) is True
                assert (
                    os.path.exists(
                        os.path.join(
                            out_dir,
                            "left_right",
                            "epi_pc",
                            nb_str + ".laz",
                        )
                    )
                ) is True
                assert (
                    os.path.exists(
                        os.path.join(
                            out_dir,
                            "left_right",
                            "epi_pc",
                            nb_str + ".laz.prj",
                        )
                    )
                ) is True

            # check used_conf reentry
            _ = sensor_to_sparse_dsm.SensorSparseDsmPipeline(used_conf)

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
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
                "save_confidence": True,
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
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        # Check used_conf for dense dsm
        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensors_to_dense_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["point_cloud_rasterization"]["sigma"]
                == 0.3
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            # check used_conf reentry
            _ = sensor_to_dense_dsm.SensorToDenseDsmPipeline(used_conf)
        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux.tif"))
        # copy2(
        #     os.path.join(out_dir, "confidence_from_ambiguity.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_ambiguity_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,
        #         "confidence_from_intensity_std_std_intensity.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_intensity_std_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "confidence_from_risk_min_risk.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_risk_min_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "confidence_from_risk_max_risk.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_risk_max_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,
        #         "confidence_from_ambiguity_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #         "ref_output",
        #         "confidence_from_ambiguity_"+"before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,
        #         "confidence_from_intensity_std"+
        #         "_std_intensity_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #         "ref_output",
        #         "confidence_from_intensity_std_before_end2end_ventoux.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,
        #         "confidence_from_risk_min_risk_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_risk_min_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,
        #         "confidence_from_risk_max_risk_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_risk_max_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_ambiguity.tif"),
            absolute_data_path(
                "ref_output/confidence_from_ambiguity_end2end_ventoux.tif"
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir, "confidence_from_intensity_std_std_intensity.tif"
            ),
            absolute_data_path(
                "ref_output/confidence_from_intensity_std_end2end_ventoux.tif"
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_risk_min_risk.tif"),
            absolute_data_path(
                "ref_output/confidence_from_risk_min_end2end_ventoux.tif"
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_risk_max_risk.tif"),
            absolute_data_path(
                "ref_output/confidence_from_risk_max_end2end_ventoux.tif"
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_ambiguity_before.tif"),
            absolute_data_path(
                os.path.join(
                    "ref_output",
                    "confidence_from_ambiguity_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "confidence_from_intensity_std_std_intensity_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    "ref_output",
                    "confidence_from_intensity_std_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_risk_min_risk_before.tif"),
            absolute_data_path(
                os.path.join(
                    "ref_output",
                    "confidence_from_risk_min_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_risk_max_risk_before.tif"),
            absolute_data_path(
                os.path.join(
                    "ref_output",
                    "confidence_from_risk_max_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    input_json = absolute_data_path(
        "input/phr_ventoux/input_without_color.json"
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # Run sparse dsm pipeline
        _, input_config_sparse_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
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
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test we have the same results with multiprocessing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run sparse dsm pipeline
        _, input_config_sparse_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "mp",
            orchestrator_parameters={
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
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
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique_split():
    """
    End to end processing
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_classif.json"
        )
        # Run sensors_to_dense_point_clouds pipeline
        _, input_config_pc = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_point_clouds",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": False,
            },
            "dense_matching": {
                "method": "census_sgm",
                "save_disparity_map": True,
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
            "dense_matches_filling.1": {
                "method": "plane",
                "classification": ["forest"],
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "save_points_cloud": True,
            },
        }

        input_config_pc["applications"].update(application_config)
        pc_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_pc
        )
        pc_pipeline.run()

        out_dir = input_config_pc["output"]["out_dir"]

        # Create input json for pc to dsm pipeline
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            epi_pc_path = os.path.join(out_dir, "left_right")
            output_path = os.path.join(directory2, "outresults_dsm_from_pc")

            input_dsm_config = {
                "inputs": {
                    "point_clouds": {
                        "one": {
                            "x": os.path.join(epi_pc_path, "epi_pc_X.tif"),
                            "y": os.path.join(epi_pc_path, "epi_pc_Y.tif"),
                            "z": os.path.join(epi_pc_path, "epi_pc_Z.tif"),
                            "color": os.path.join(
                                epi_pc_path, "epi_pc_color.tif"
                            ),
                            "classification": os.path.join(
                                epi_pc_path, "epi_classification.tif"
                            ),
                            "filling": os.path.join(
                                epi_pc_path, "epi_filling.tif"
                            ),
                            "confidence": {
                                "confidence_from_ambiguity2": os.path.join(
                                    epi_pc_path,
                                    "epi_confidence_from" + "_ambiguity.tif",
                                ),
                                "confidence_from_ambiguity1": os.path.join(
                                    epi_pc_path,
                                    "epi_confidence_from"
                                    + "_ambiguity_before.tif",
                                ),
                            },
                        }
                    },
                    "roi": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "coordinates": [
                                        [
                                            [5.194, 44.2064],
                                            [5.194, 44.2059],
                                            [5.195, 44.2059],
                                            [5.195, 44.2064],
                                            [5.194, 44.2064],
                                        ]
                                    ],
                                    "type": "Polygon",
                                },
                            }
                        ],
                    },
                },
                "output": {"out_dir": output_path},
                "pipeline": "dense_point_clouds_to_dense_dsm",
                "applications": {
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
                        "save_classif": True,
                        "save_filling": True,
                        "save_confidence": True,
                        "save_color": True,
                        "save_source_pc": True,
                    },
                },
            }

            dsm_pipeline = pipeline_dsm.PointCloudsToDsmPipeline(
                input_dsm_config
            )
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["out_dir"]

            # Uncomment the 2 following instructions to update reference data
            # copy2(os.path.join(out_dir_dsm, 'dsm.tif'),
            #     absolute_data_path(
            #       "ref_output/dsm_end2end_ventoux_split.tif")
            #  )
            # copy2(os.path.join(out_dir_dsm, 'clr.tif'),
            #     absolute_data_path(
            #       "ref_output/clr_end2end_ventoux_split.tif")
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "classif.tif"),
            #     absolute_data_path(
            #         "ref_output/classif_end2end_ventoux_split.tif"
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "filling.tif"),
            #     absolute_data_path(
            #         "ref_output/filling_end2end_ventoux_split.tif"
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "confidence_from_ambiguity2.tif"),
            #     absolute_data_path(
            #         "ref_output/confidence_from"
            #         + "_ambiguity2_end2end_ventoux_split.tif"
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "confidence_from_ambiguity1.tif"),
            #     absolute_data_path(
            #         "ref_output/confidence_from"
            #         + "_ambiguity1_end2end_ventoux_split.tif"
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "source_pc.tif"),
            #     absolute_data_path(
            #         "ref_output/source_pc_end2end_ventoux_split.tif"
            #     ),
            # )

            assert_same_images(
                os.path.join(out_dir_dsm, "dsm.tif"),
                absolute_data_path("ref_output/dsm_end2end_ventoux_split.tif"),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "clr.tif"),
                absolute_data_path("ref_output/clr_end2end_ventoux_split.tif"),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "classif.tif"),
                absolute_data_path(
                    "ref_output/classif_end2end_ventoux_split.tif"
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(out_dir_dsm, "confidence_from_ambiguity1.tif"),
                absolute_data_path(
                    "ref_output/confidence_from"
                    + "_ambiguity1_end2end_ventoux_split.tif"
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "confidence_from_ambiguity2.tif"),
                absolute_data_path(
                    "ref_output/confidence_from"
                    + "_ambiguity2_end2end_ventoux_split.tif"
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )


@pytest.mark.end2end_tests
def test_end2end_use_epipolar_a_priori():
    """
    End to end processing sparse dsm pipeline
    and use prepared refined dense dsm pipeline conf
    to compute the dense dsm pipeline, without strm
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run sparse dsm pipeline
        _, input_config_sparse_res = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        # no srtm
        input_config_sparse_res["inputs"]["initial_elevation"] = None

        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)
        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                15
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["minimum_disparity"]
                < 16
            )
            assert (
                60
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 61
            )

            # check matches file exists
            assert os.path.isfile(
                out_json["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

            # Uncomment the 2 following instructions to update reference data
            # copy2(os.path.join(out_dir, 'dtm_mean.tif'),
            #     absolute_data_path("ref_output/dtm_mean_end2end_ventoux.tif"))
            # copy2(os.path.join(out_dir, 'dtm_min.tif'),
            #     absolute_data_path("ref_output/dtm_min_end2end_ventoux.tif"))
            # copy2(os.path.join(out_dir, 'dtm_max.tif'),
            #     absolute_data_path("ref_output/dtm_max_end2end_ventoux.tif"))

            assert_same_images(
                os.path.join(out_dir, "dtm_mean.tif"),
                absolute_data_path("ref_output/dtm_mean_end2end_ventoux.tif"),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir, "dtm_min.tif"),
                absolute_data_path("ref_output/dtm_min_end2end_ventoux.tif"),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir, "dtm_max.tif"),
                absolute_data_path("ref_output/dtm_max_end2end_ventoux.tif"),
                atol=0.0001,
                rtol=1e-6,
            )

        # Check used_conf for low res

        gt_used_conf_orchestrator = {
            "orchestrator": {
                "mode": "local_dask",
                "walltime": "00:10:00",
                "nb_workers": 4,
                "profiling": {
                    "activated": False,
                    "mode": "time",
                    "loop_testing": False,
                },
                "python": None,
                "use_memory_logger": False,
                "activate_dashboard": False,
                "max_ram_per_worker": 1000,
                "config_name": "unknown",
            }
        }

        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check refined_config_dense_dsm_json file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensors_to_sparse_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["sparse_matching"]["disparity_margin"]
                == 0.25
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            # check used_conf reentry
            _ = sensor_to_sparse_dsm.SensorSparseDsmPipeline(used_conf)

        refined_config_dense_dsm_json = os.path.join(
            out_dir, "refined_config_dense_dsm.json"
        )
        assert os.path.isfile(refined_config_dense_dsm_json)
        with open(
            refined_config_dense_dsm_json, "r", encoding="utf-8"
        ) as json_file:
            refined_config_dense_dsm_json = json.load(json_file)
            # check refined_config_dense_dsm_json inputs conf exists
            assert "inputs" in refined_config_dense_dsm_json
            assert "sensors" in refined_config_dense_dsm_json["inputs"]
            # check refined_config_dense_dsm_json pipeline
            assert (
                refined_config_dense_dsm_json["pipeline"]
                == "sensors_to_dense_dsm"
            )
            # check refined_config_dense_dsm_json sparse_matching configuration
            assert (
                "use_epipolar_a_priori"
                in refined_config_dense_dsm_json["inputs"]
            )
            assert (
                refined_config_dense_dsm_json["inputs"]["use_epipolar_a_priori"]
                is True
            )
            assert (
                "epipolar_a_priori" in refined_config_dense_dsm_json["inputs"]
            )
            assert (
                "grid_correction"
                in refined_config_dense_dsm_json["inputs"]["epipolar_a_priori"][
                    "left_right"
                ]
            )
            assert (
                "dtm_mean"
                in refined_config_dense_dsm_json["inputs"]["terrain_a_priori"]
            )
            assert (
                "dtm_min"
                in refined_config_dense_dsm_json["inputs"]["terrain_a_priori"]
            )
            assert (
                "dtm_max"
                in refined_config_dense_dsm_json["inputs"]["terrain_a_priori"]
            )

            # check if orchestrator conf is the same as gt
            assert (
                refined_config_dense_dsm_json["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = refined_config_dense_dsm_json.copy()
        # update applications
        input_config_dense_dsm["applications"] = input_config_sparse_res[
            "applications"
        ]
        dense_dsm_applications = {
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
                "save_confidence": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631
        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        # Check used_conf for dense_dsm
        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensors_to_dense_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["point_cloud_rasterization"]["sigma"]
                == 0.3
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            # check used_conf reentry
            _ = sensor_to_dense_dsm.SensorToDenseDsmPipeline(used_conf)
        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path("ref_output/dsm_end2end_ventoux_no_srtm.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux_no_srtm.tif"))
        # copy2(
        #     os.path.join(out_dir, "confidence_from_ambiguity.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output",
        #             "confidence_from_ambiguity_end2end_ventoux_no_srtm.tif",
        #         )
        #     ),
        # )
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_no_srtm.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "confidence_from_ambiguity.tif"),
            absolute_data_path(
                "ref_output/"
                "confidence_from_ambiguity_end2end_ventoux_no_srtm.tif"
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_no_srtm.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_prepare_ventoux_bias():
    """
    Dask prepare with bias geoms
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input_bias.json")
        # Run sparse dsm pipeline
        _, input_config_sparse_res = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 2000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "epipolar_error_maximum_bias": 50.0,
                "elevation_delta_lower_bound": -120.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -84
            assert out_disp_compute["minimum_disparity"] < -82
            assert out_disp_compute["maximum_disparity"] > -46
            assert out_disp_compute["maximum_disparity"] < -44

            # check matches file exists
            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )


@pytest.mark.end2end_tests
def test_end2end_ventoux_with_color():
    """
    End to end processing with color
    """

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input_with_color.json")
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_color.json"
        )
        # Run sparse dsm pipeline
        _, input_config_sparse_res = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {
                "method": "bicubic",
                "epi_tile_size": 250,
                "save_epipolar_image": True,
                "save_epipolar_color": False,
            },
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
                "save_points_cloud_as_csv": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -21
            assert out_disp_compute["minimum_disparity"] < -17
            assert out_disp_compute["maximum_disparity"] > 13
            assert out_disp_compute["maximum_disparity"] < 16

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

            assert (
                os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675240.0_4897185.0.laz"
                    )
                )
                and os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675375.0_4897185.0.csv"
                    )
                )
            ) is True
            assert (
                os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675375.0_4897185.0.laz"
                    )
                )
                and os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675240.0_4897185.0.csv"
                    )
                )
            ) is True

        # Run dense_dsm dsm pipeline
        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_res.copy()
        # update applications
        dense_dsm_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_confidence": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "loader": "pandora",
                "save_disparity_map": True,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
            "point_cloud_outliers_removing.1": {
                "method": "small_components",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
            "point_cloud_outliers_removing.2": {
                "method": "statistical",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631

        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        assert (
            os.path.exists(
                os.path.join(out_dir, "confidence_from_ambiguity.tif")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", "675431.5_4897173.0.laz")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", "675431.5_4897173.0.csv")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    "675248.0_4897173.0.laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    "675248.0_4897173.0.csv",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    "675248.0_4897173.0.laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    "675248.0_4897173.0.csv",
                )
            )
            is True
        )

        # Uncomment the following instruction to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux_4bands.tif"))

        # assert_same_images(
        #     os.path.join(out_dir, "dsm.tif"),
        #     absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
        #     atol=0.0001,
        #     rtol=1e-6,
        # )
        # assert_same_images(
        #     os.path.join(out_dir, "clr.tif"),
        #     absolute_data_path("ref_output/clr_end2end_ventoux_4bands.tif"),
        #     rtol=1.0e-7,
        #     atol=1.0e-7,
        # )


@pytest.mark.end2end_tests
def test_end2end_ventoux_with_classif():
    """
    End to end processing with p+xs fusion
    """

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input_with_classif.json")
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_classif.json"
        )
        # Run sparse dsm pipeline
        _, input_config_sparse_res = generate_input_json(
            input_json,
            directory,
            "sensors_to_sparse_dsm",
            "local_dask",
            orchestrator_parameters={
                "walltime": "00:10:00",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {
                "method": "bicubic",
                "epi_tile_size": 250,
                "save_epipolar_image": True,
                "save_epipolar_color": False,
            },
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
                "save_points_cloud_as_csv": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -21
            assert out_disp_compute["minimum_disparity"] < -17
            assert out_disp_compute["maximum_disparity"] > 13
            assert out_disp_compute["maximum_disparity"] < 16

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

            assert (
                os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675240.0_4897185.0.laz"
                    )
                )
                and os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675375.0_4897185.0.csv"
                    )
                )
            ) is True
            assert (
                os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675375.0_4897185.0.laz"
                    )
                )
                and os.path.exists(
                    os.path.join(
                        out_dir, "points_cloud", "675240.0_4897185.0.csv"
                    )
                )
            ) is True

        # Run dense_dsm dsm pipeline
        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_res.copy()
        # update applications
        dense_dsm_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_classif": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "loader": "pandora",
                "save_disparity_map": True,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
            "point_cloud_outliers_removing.1": {
                "method": "small_components",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
            "point_cloud_outliers_removing.2": {
                "method": "statistical",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["inputs"]["epsg"] = 32631

        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_res["output"]["out_dir"]

        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", "675431.5_4897173.0.laz")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", "675431.5_4897173.0.csv")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    "675248.0_4897173.0.laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    "675248.0_4897173.0.csv",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    "675248.0_4897173.0.laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    "675248.0_4897173.0.csv",
                )
            )
            is True
        )

        # Uncomment the following instruction to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'classif.tif'),
        #      absolute_data_path("ref_output/classif_end2end_ventoux.tif"))

        assert_same_images(
            os.path.join(out_dir, "classif.tif"),
            absolute_data_path("ref_output/classif_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_compute_dsm_with_roi_ventoux():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_color.json"
        )
        # Run sparse dsm pipeline
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
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        # Update roi
        roi_geo_json = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [
                            [
                                [5.194, 44.2064],
                                [5.194, 44.2059],
                                [5.195, 44.2059],
                                [5.195, 44.2064],
                                [5.194, 44.2064],
                            ]
                        ],
                        "type": "Polygon",
                    },
                }
            ],
        }

        input_config_dense_dsm["inputs"]["roi"] = roi_geo_json

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_end2end_ventoux_with_roi.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #      absolute_data_path(
        #      "ref_output/clr_end2end_ventoux_with_roi.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_with_roi.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_with_roi.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

        # check final bounding box
        # create reference
        # Transform to shapely polygon, epsg
        roi_poly, roi_epsg = roi_tools.geojson_to_shapely(roi_geo_json)

        project = pyproj.Transformer.from_proj(
            pyproj.Proj(init="epsg:{}".format(roi_epsg)),
            pyproj.Proj(init="epsg:{}".format(final_epsg)),
        )
        ref_roi_poly = transform(project.transform, roi_poly)

        [ref_xmin, ref_ymin, ref_xmax, ref_ymax] = ref_roi_poly.bounds

        # retrieve bounding box of computed dsm
        data = rasterio.open(os.path.join(out_dir, "dsm.tif"))
        xmin = min(data.bounds.left, data.bounds.right)
        ymin = min(data.bounds.bottom, data.bounds.top)
        xmax = max(data.bounds.left, data.bounds.right)
        ymax = max(data.bounds.bottom, data.bounds.top)

        assert math.floor(ref_xmin / resolution) * resolution == xmin
        assert math.ceil(ref_xmax / resolution) * resolution == xmax
        assert math.floor(ref_ymin / resolution) * resolution == ymin
        assert math.ceil(ref_ymax / resolution) * resolution == ymax


@pytest.mark.end2end_tests
def test_compute_dsm_with_snap_to_img1():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run sparse dsm pipeline
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
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
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
                "snap_to_img1": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path(
        #    "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path(
        #     "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path(
                "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path(
                "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_quality_stats():
    """
    End to end processing, with no srtm
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run sparse dsm pipeline
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

        # no srtm
        input_config_dense_dsm["inputs"]["initial_elevation"] = None
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "disparity_margin": 0.25,
                "save_matches": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_stats": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > 15
            assert out_disp_compute["minimum_disparity"] < 16
            assert out_disp_compute["maximum_disparity"] > 60
            assert out_disp_compute["maximum_disparity"] < 62

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dtm_mean.tif'),
        #       absolute_data_path("ref_output/"
        #     "dtm_mean_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dtm_min.tif'),
        #       absolute_data_path("ref_output/"
        #     "dtm_min_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dtm_max.tif'),
        #       absolute_data_path("ref_output/"
        #     "dtm_max_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #       absolute_data_path("ref_output/"
        #     "dsm_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #      absolute_data_path("ref_output/"
        #     "clr_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dsm_mean.tif'),
        #      absolute_data_path("ref_output/"
        #     "dsm_mean_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dsm_std.tif'),
        #      absolute_data_path("ref_output/"
        #     "dsm_std_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dsm_n_pts.tif'),
        #      absolute_data_path("ref_output/"
        #     "dsm_n_pts_end2end_ventoux_quality_stats.tif"))
        # copy2(os.path.join(out_dir, 'dsm_pts_in_cell.tif'),
        #      absolute_data_path("ref_output/"
        #     "dsm_pts_in_cell_end2end_ventoux_quality_stats.tif"))

        assert_same_images(
            os.path.join(out_dir, "dtm_mean.tif"),
            absolute_data_path(
                "ref_output/dtm_mean_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dtm_min.tif"),
            absolute_data_path(
                "ref_output/dtm_min_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dtm_max.tif"),
            absolute_data_path(
                "ref_output/dtm_max_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path(
                "ref_output/dsm_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path(
                "ref_output/clr_end2end_ventoux_quality_stats.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_mean.tif"),
            absolute_data_path(
                "ref_output/dsm_mean_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_std.tif"),
            absolute_data_path(
                "ref_output/dsm_std_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_n_pts.tif"),
            absolute_data_path(
                "ref_output/dsm_n_pts_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_pts_in_cell.tif"),
            absolute_data_path(
                "ref_output/dsm_pts_in_cell_end2end_ventoux_quality_stats.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_ventoux_egm96_geoid():
    """
    End to end processing
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run sparse dsm pipeline
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
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
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
                "use_geoid_alt": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_stats": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -20
            assert out_disp_compute["minimum_disparity"] < -18
            assert out_disp_compute["maximum_disparity"] > 14
            assert out_disp_compute["maximum_disparity"] < 15

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_ventoux_egm96.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_egm96.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
    assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_without_color.json"
        )
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
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
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
                "use_geoid_alt": True,
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_stats": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_egm96.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_paca_with_mask():
    """
    End to end processing
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_paca/input.json")

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
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
                "minimum_nb_matches": 10,
            },
            "dense_matches_filling.2": {
                "method": "zero_padding",
                "classification": ["water", "road"],
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "save_msk": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_paca.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_paca.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )

    # Test we have the same results with multiprocessing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_paca/input.json")

        # Run sparse dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "mp",
            orchestrator_parameters={
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "dense_matches_filling.2": {
                "method": "zero_padding",
                "classification": ["water", "road"],
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "save_msk": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm.tif"),
        #     absolute_data_path("ref_output/dsm_end2end_paca.tif"),
        # )
        # copy2(
        #     os.path.join(out_dir, "clr.tif"),
        #     absolute_data_path("ref_output/clr_end2end_paca.tif"),
        # )
        # copy2(
        #     os.path.join(out_dir, "msk.tif"),
        #     absolute_data_path("ref_output/msk_end2end_paca.tif"),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_paca.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )


@pytest.mark.end2end_tests
def test_end2end_disparity_filling():
    """
    End to end processing, test with mask and fill holes
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_gizeh/input_msk_fill.json")

        # Run dense dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "mp",
            orchestrator_parameters={
                "nb_workers": 4,
                "max_ram_per_worker": 200,
            },
        )
        resolution = 0.5
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm",
                "save_disparity_map": True,
            },
            "dense_matches_filling.1": {
                "method": "plane",
                "save_disparity_map": True,
                "classification": ["shadow"],
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "save_msk": True,
                "save_filling": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_gizeh_fill.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_gizeh_fill.tif"))
        # copy2(os.path.join(out_dir, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_gizeh_fill.tif"))
        # copy2(os.path.join(out_dir, 'filling.tif'),
        #      absolute_data_path("ref_output/filling_end2end_gizeh_fill.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_gizeh_fill.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_gizeh_fill.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_gizeh_fill.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "filling.tif"),
            absolute_data_path("ref_output/filling_end2end_gizeh_fill.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )


@pytest.mark.end2end_tests
def test_end2end_disparity_filling_with_zeros():
    """
    End to end processing, test with mask and
    fill holes with zero_padding method
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_gizeh/input_msk_fill.json")

        # Run dense dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "local_dask",
        )
        resolution = 0.5
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm",
                "save_disparity_map": True,
            },
            "dense_matches_filling.2": {
                "method": "zero_padding",
                "save_disparity_map": True,
                "classification": ["bat"],
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
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "save_msk": True,
                "save_filling": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["inputs"]["epsg"] = final_epsg

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output", "dsm_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "clr.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output", "clr_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "msk.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output", "msk_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "filling.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             "ref_output", "filling_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path(
                "ref_output/dsm_end2end_gizeh_fill_with_zero.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path(
                "ref_output/clr_end2end_gizeh_fill_with_zero.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path(
                "ref_output/msk_end2end_gizeh_fill_with_zero.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "filling.tif"),
            absolute_data_path(
                "ref_output/filling_end2end_gizeh_fill_with_zero.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
