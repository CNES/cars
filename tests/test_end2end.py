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

    Test defaut conf use_endogenous_elevation :
        use srtm for prepare, and compute
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
                "max_ram_per_worker": 500,
            },
        )
        resolution = 0.5
        dense_dsm_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 254,
            },
        }
        input_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_dense_dsm["output"]["epsg"] = final_epsg
        input_dense_dsm["output"]["auxiliary"] = {
            "mask": True,
            "performance_map": True,
        }

        # use endogenous dem
        input_dense_dsm["inputs"]["use_endogenous_elevation"] = False

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dsm_end2end_gizeh_crop.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "color_end2end_gizeh_crop.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "rasterization", "mask.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "mask_end2end_gizeh_crop.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "rasterization",
        #                            "performance_map.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "performance_map_end2end_gizeh_crop.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_gizeh_crop.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_gizeh_crop.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "mask_end2end_gizeh_crop.tif")
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dsm",
                "performance_map.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "performance_map_end2end_gizeh_crop.tif"
                )
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        # launch no mergin pipeline
        input_dense_dsm["pipeline"] = "sensors_to_dense_dsm_no_merging"
        input_dense_dsm["output"]["directory"] = out_dir + "no_merging"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_dense_dsm
        )
        dense_dsm_pipeline.run()
        out_dir = input_dense_dsm["output"]["directory"]

        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end_gizeh_crop_no_merging.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "color_end2end_gizeh_crop_no_merging.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "rasterization", "mask.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "mask_end2end_gizeh_crop_no_merging.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                        "confidence_performance_map.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "performance_map_end2end_gizeh_crop_no_merging.tif",
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_gizeh_crop_no_merging.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_gizeh_crop_no_merging.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "mask_end2end_gizeh_crop_no_merging.tif"
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dsm",
                "performance_map.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "performance_map_end2end_gizeh_crop_no_merging.tif",
                )
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_sparse_dsm_8bits():
    """
    End to end processing

    Check sensors_to_sparse_dsm pipeline with 8 bits images input
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input_8bits.json")
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
            "resampling": {"method": "bicubic"},
            "sparse_matching": {
                "method": "sift",
                # Uncomment the following line to update dsm reference data
                # "sift_peak_threshold":1,
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "save_intermediate_data": False,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
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
                < -13
            )
            assert (
                17
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 22
            )

        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"
        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                    "dem_median.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dem_median_end2end_ventoux_8bit.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "dem_generation",
        #                    "dem_min.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir,
        #         "dem_min_end2end_ventoux_8bit.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                               "dem_max.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir,
        #         "dem_max_end2end_ventoux_8bit.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dem_median_end2end_ventoux_8bit.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_min.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_min_end2end_ventoux_8bit.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_max_end2end_ventoux_8bit.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique():
    """
    End to end processing with ventoux data
    1 run sparse dsm pipeline: check config, check data presence,
       check used conf reentry
    2 run dense dsm pipeline + Baseline checking
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
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
                -29
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["minimum_disparity"]
                < -27
            )
            assert (
                26
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 30
            )

            # check matches file exists
            assert os.path.isfile(
                out_json["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                             "dem_median.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dem_median_end2end_ventoux.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                                  "dem_min.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dem_min_end2end_ventoux.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                                     "dem_max.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dem_max_end2end_ventoux.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_median_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_min.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_min_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_max_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
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
                    "mode": "cars_profiling",
                    "loop_testing": False,
                },
                "python": None,
                "task_timeout": 600,
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

            # check used_conf reentry
            _ = sensor_to_sparse_dsm.SensorSparseDsmPipeline(used_conf)

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline (keep geometry_plugin)
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
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
                "loader_conf": {
                    "input": {},
                    "pipeline": {
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
                            "validation_method": "cross_checking_accurate",
                            "cross_checking_threshold": 1.0,
                        },
                    },
                },
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "save_intermediate_data": True,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_intermediate_data": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

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
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                   "confidence_from_ambiguity.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_ambiguity_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(
        #         out_dir,  "dump_dir", "rasterization",
        #                   "confidence_from_intensity_std_std_intensity.tif"
        #     ),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_intensity_std_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                             "confidence_from_risk_min_risk.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_risk_min_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                               "confidence_from_risk_max_risk.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_risk_max_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                           "confidence_from_ambiguity_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_ambiguity_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(
        #         out_dir, "dump_dir", "rasterization",
        #         "confidence_from_intensity_std_std_intensity_before.tif",
        #     ),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_intensity"
        #             + "_std_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                      "confidence_from_risk_min_risk_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_risk_min_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                         "confidence_from_risk_max_risk_before.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_risk_max_before_end2end_ventoux.tif",
        #         )
        #     ),
        # )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_ambiguity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_ambiguity_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_intensity_std_std_intensity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_intensity_std_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-5,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_min_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_risk_min_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_max_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_risk_max_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_ambiguity_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_ambiguity_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_intensity_std_std_intensity_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_intensity_std_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_min_risk_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_risk_min_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_max_risk_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_risk_max_before_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_dsm["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_dsm
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_dsm["output"]["directory"]

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique_split_epsg_4326():
    """
    Splitted sensor to dsm pipeline with ROI on ventoux data
    1 run sensor to dense point clouds(PC) pipeline -> PC outputs
    2 run PC outputs with cloud to dsm pipeline + Baseline checking
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
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
        # use endogenous dem
        input_config_pc["inputs"]["use_endogenous_elevation"] = True
        pc_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_pc
        )

        input_config_pc["output"]["epsg"] = 4326

        pc_pipeline.run()

        out_dir = input_config_pc["output"]["directory"]
        geometry_plugin_name = input_config_pc["geometry_plugin"]

        # Create input json for pc to dsm pipeline
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            epi_pc_path = os.path.join(
                out_dir, "dump_dir", "triangulation", "left_right"
            )
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
                "geometry_plugin": geometry_plugin_name,
                "output": {
                    "directory": output_path,
                    "epsg": 4326,
                },
                "pipeline": "dense_point_clouds_to_dense_dsm",
                "applications": {
                    "point_cloud_rasterization": {
                        "method": "simple_gaussian",
                        "resolution": 0.000005,
                        "save_intermediate_data": True,
                    }
                },
            }

            dsm_pipeline = pipeline_dsm.PointCloudsToDsmPipeline(
                input_dsm_config
            )
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            # Ref output dir dependent from geometry plugin chosen
            ref_output_dir = "ref_output"

            # Uncomment the following instructions to update reference data
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "dsm_end2end_ventoux_split_4326.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "color.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "color_end2end_ventoux_split_4326.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                                                 "source_pc.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "source_pc_end2end" + "_ventoux_split_4326.tif",
            #         )
            #     ),
            # )

            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dsm_end2end_ventoux_split_4326.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "color.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "color_end2end_ventoux_split_4326.tif"
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "source_pc_end2end_ventoux_split_4326.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )

            # launch with no merging pipeline
            input_dsm_config["pipeline"] = (
                "dense_point_clouds_to_dense_dsm_no_merging"
            )
            input_dsm_config["output"]["directory"] = output_path + "no_merging"

            dsm_pipeline = pipeline_dsm.PointCloudsToDsmPipeline(
                input_dsm_config
            )
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dsm_end2end_ventoux_split_4326.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "color.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "color_end2end_ventoux_split_4326.tif"
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "source_pc_end2end_ventoux_split_4326.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique_split():
    """
    Splitted sensor to dsm pipeline with ROI on ventoux data
    1 run sensor to dense point clouds(PC) pipeline -> PC outputs
    2 run PC outputs with cloud to dsm pipeline + Baseline checking
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_classif_and_mask.json"
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
            "resampling": {"method": "bicubic", "strip_height": 200},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": False,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
                "save_intermediate_data": True,
                "generate_confidence_intervals": False,
                "loader_conf": {
                    "input": {},
                    "pipeline": {
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
                            "validation_method": "cross_checking_accurate",
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
                "save_intermediate_data": True,
            },
        }

        input_config_pc["applications"].update(application_config)
        # use endogenous dem
        input_config_pc["inputs"]["use_endogenous_elevation"] = True

        pc_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_pc
        )
        pc_pipeline.run()

        out_dir = input_config_pc["output"]["directory"]
        geometry_plugin_name = input_config_pc["geometry_plugin"]

        # Create input json for pc to dsm pipeline
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            epi_pc_path = os.path.join(
                out_dir, "dump_dir", "triangulation", "left_right"
            )
            dense_matching_path = os.path.join(
                out_dir, "dump_dir", "dense_matching", "left_right"
            )
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
                                epi_pc_path, "epi_pc_classification.tif"
                            ),
                            "filling": os.path.join(
                                epi_pc_path, "epi_pc_filling.tif"
                            ),
                            "confidence": {
                                "confidence_from_ambiguity2": os.path.join(
                                    dense_matching_path,
                                    "epi_confidence_from" + "_ambiguity.tif",
                                ),
                                "confidence_from_ambiguity1": os.path.join(
                                    dense_matching_path,
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
                "geometry_plugin": geometry_plugin_name,
                "output": {"directory": output_path},
                "pipeline": "dense_point_clouds_to_dense_dsm",
                "applications": {
                    "point_cloud_outliers_removing.1": {
                        "method": "small_components",
                        "activated": True,
                        "save_points_cloud_as_laz": True,
                        "save_points_cloud_by_pair": True,
                    },
                    "point_cloud_outliers_removing.2": {
                        "method": "statistical",
                        "activated": True,
                        "save_points_cloud_as_laz": True,
                        "save_points_cloud_by_pair": True,
                    },
                    "point_cloud_rasterization": {
                        "method": "simple_gaussian",
                        "dsm_radius": 3,
                        "resolution": 0.5,
                        "sigma": 0.3,
                        "dsm_no_data": -999,
                        "color_no_data": 0,
                        "save_intermediate_data": True,
                    },
                },
            }

            dsm_pipeline = pipeline_dsm.PointCloudsToDsmPipeline(
                input_dsm_config
            )
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            # Ref output dir dependent from geometry plugin chosen
            ref_output_dir = "ref_output"

            assert (
                os.path.exists(
                    os.path.join(
                        out_dir_dsm,
                        "points_cloud_post_small_components_removing",
                        "675292.3110543193_4897140.457149682_one.laz",
                    )
                )
                is True
            )
            # Uncomment the following instructions to update reference data
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "dsm_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "color.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "color_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                                                   "mask.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "mask_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                         "classification.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "classification_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                                                 "filling.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "filling_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                               "confidence_from_ambiguity1.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "confidence_from_ambiguity1"
            #             + "_end2end_ventoux_split.tif",
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                               "confidence_from_ambiguity2.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "confidence_from_ambiguity2"
            #             + "_end2end_ventoux_split.tif",
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm,  "dump_dir", "rasterization",
            #                                                "source_pc.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir, "source_pc_end2end_ventoux_split.tif"
            #         )
            #     ),
            # )

            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dsm_end2end_ventoux_split.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "color.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "color_end2end_ventoux_split.tif"
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "mask.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "mask_end2end_ventoux_split.tif"
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "classification.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classification_end2end_ventoux_split.tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "filling.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "filling_end2end_ventoux_split.tif"
                    )
                ),
                atol=1.0e-7,
                rtol=1.0e-7,
            )

            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "confidence_from_ambiguity1.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "confidence_from"
                        + "_ambiguity1_end2end_ventoux_split.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "confidence_from_ambiguity2.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "confidence_from"
                        + "_ambiguity2_end2end_ventoux_split.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "source_pc_end2end_ventoux_split.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )

            # Run no merging pipeline

            input_dsm_config["output"]["directory"] = (
                input_dsm_config["output"]["directory"] + "_no_merging"
            )
            del input_dsm_config["applications"][
                "point_cloud_outliers_removing.1"
            ]
            del input_dsm_config["applications"][
                "point_cloud_outliers_removing.2"
            ]
            input_dsm_config["pipeline"] = (
                "dense_point_clouds_to_dense_dsm_no_merging"
            )

            # launch
            dsm_pipeline = pipeline_dsm.PointCloudsToDsmPipeline(
                input_dsm_config
            )
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            # Uncomment the following instructions to update reference data
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "dsm_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dsm", "color.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "clr_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dump_dir", "rasterization",
            #                   "mask.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "msk_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dump_dir", "rasterization",
            #                  "classification.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "classif_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dump_dir", "rasterization",
            #                      "filling.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "filling_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dump_dir", "rasterization",
            #                     "confidence_from_ambiguity1.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "confidence_from_ambiguity1"
            #             + "_end2end_ventoux_split_no_merging.tif",
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "dump_dir", "rasterization",
            #                              "confidence_from_ambiguity2.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "confidence_from_ambiguity2"
            #             + "_end2end_ventoux_split_no_merging.tif",
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir_dsm, "source_pc.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "source_pc_end2end_ventoux_split_no_merging.tif"
            #         )
            #     ),
            # )

            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "dsm_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(out_dir_dsm, "dsm", "color.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "clr_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "mask.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "msk_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "classification.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "classif_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "filling.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "filling_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                atol=1.0e-7,
                rtol=1.0e-7,
            )

            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "confidence_from_ambiguity1.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "confidence_from"
                        + "_ambiguity1_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "confidence_from_ambiguity2.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "confidence_from"
                        + "_ambiguity2_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                rtol=1.0e-7,
                atol=1.0e-7,
            )
            assert_same_images(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "source_pc_end2end_ventoux_split_no_merging.tif",
                    )
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
            "resampling": {"method": "bicubic", "strip_height": 200},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)
        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
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
                -27
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["minimum_disparity"]
                < -25
            )
            assert (
                23
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 28
            )

            # check matches file exists
            assert os.path.isfile(
                out_json["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

            # Ref output dir dependent from geometry plugin chosen
            ref_output_dir = "ref_output"

            # Uncomment the 2 following instructions to update reference data
            # copy2(
            #     os.path.join(out_dir, "dump_dir", "dem_generation",
            #                      "dem_median.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "dem_median_end2end_ventoux_no_srtm.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir, "dump_dir", "dem_generation",
            #                                                "dem_min.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "dem_min_end2end_ventoux_no_srtm.tif"
            #         )
            #     ),
            # )
            # copy2(
            #     os.path.join(out_dir, "dump_dir", "dem_generation",
            #                                              "dem_max.tif"),
            #     absolute_data_path(
            #         os.path.join(
            #             ref_output_dir,
            #             "dem_max_end2end_ventoux_no_srtm.tif"
            #         )
            #     ),
            # )

            assert_same_images(
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_median.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dem_median_end2end_ventoux_no_srtm.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_min.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dem_min_end2end_ventoux_no_srtm.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_max.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dem_max_end2end_ventoux_no_srtm.tif"
                    )
                ),
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
                    "mode": "cars_profiling",
                    "loop_testing": False,
                },
                "python": None,
                "task_timeout": 600,
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
            assert "advanced" in refined_config_dense_dsm_json
            assert (
                "use_epipolar_a_priori"
                in refined_config_dense_dsm_json["advanced"]
            )

            assert (
                refined_config_dense_dsm_json["advanced"][
                    "use_epipolar_a_priori"
                ]
                is True
            )
            assert (
                "epipolar_a_priori" in refined_config_dense_dsm_json["advanced"]
            )
            assert (
                "grid_correction"
                in refined_config_dense_dsm_json["advanced"][
                    "epipolar_a_priori"
                ]["left_right"]
            )
            assert (
                "dem_median"
                in refined_config_dense_dsm_json["advanced"]["terrain_a_priori"]
            )
            assert (
                "dem_min"
                in refined_config_dense_dsm_json["advanced"]["terrain_a_priori"]
            )
            assert (
                "dem_max"
                in refined_config_dense_dsm_json["advanced"]["terrain_a_priori"]
            )

            # check if orchestrator conf is the same as gt
            assert (
                refined_config_dense_dsm_json["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )

        # dense dsm pipeline
        input_config_dense_dsm = refined_config_dense_dsm_json.copy()
        # update applications
        input_config_dense_dsm["applications"] = input_config_sparse_res[
            "applications"
        ]
        dense_dsm_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True
        # Update outdir, write new dir
        input_config_dense_dsm["output"]["directory"] += "dense"
        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )

        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

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

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"
        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end" + "_ventoux_no_srtm.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "color_end2end" + "_ventoux_no_srtm.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                   "confidence_from_ambiguity.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "confidence_from_ambiguity_end2end_ventoux_no_srtm.tif",
        #         )
        #     ),
        # )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_no_srtm.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_no_srtm.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_ambiguity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "confidence_from_ambiguity_end2end_ventoux_no_srtm.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False


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
            "resampling": {"method": "bicubic", "strip_height": 100},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "epipolar_error_maximum_bias": 50.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 120.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
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
            assert out_disp_compute["minimum_disparity"] > -28
            assert out_disp_compute["minimum_disparity"] < -26
            assert out_disp_compute["maximum_disparity"] > 24
            assert out_disp_compute["maximum_disparity"] < 29

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
                "strip_height": 80,
                "save_intermediate_data": True,
            },
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
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
            assert out_disp_compute["minimum_disparity"] > -29
            assert out_disp_compute["minimum_disparity"] < -27
            assert out_disp_compute["maximum_disparity"] > 26
            assert out_disp_compute["maximum_disparity"] < 30

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

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
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "loader": "pandora",
                "save_intermediate_data": True,
                "use_global_disp_range": False,
                "generate_performance_map": False,
                "generate_confidence_intervals": False,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_intermediate_data": True,
                "save_points_cloud_by_pair": True,
            },
            "point_cloud_outliers_removing.1": {
                "method": "small_components",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
                "save_points_cloud_by_pair": True,
            },
            "point_cloud_outliers_removing.2": {
                "method": "statistical",
                "activated": True,
                "save_points_cloud_as_laz": True,
                "save_points_cloud_as_csv": True,
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "save_intermediate_data": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "rasterization",
                    "confidence_from_ambiguity.tif",
                )
            )
            is True
        )

        print(os.listdir(os.path.join(out_dir, "points_cloud")))
        if input_config_dense_dsm["geometry_plugin"] == "OTBGeometry":
            pc1 = "675436.5_4897170.5"
            pc2 = "675248.0_4897170.5"
        else:
            pc1 = "675437.0_4897170.0"
            pc2 = "675248.5_4897170.0"

        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", pc1 + "_left_right.laz")
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(out_dir, "points_cloud", pc1 + "_left_right.csv")
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    pc1 + "_left_right.laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    pc1 + "_left_right.csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    pc2 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    pc2 + ".csv",
                )
            )
            is True
        )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"
        # Uncomment the following instruction to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end_ventoux_with_color.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "color_end2end_ventoux_with_color.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_with_color.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_with_color.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_with_classif():
    """
    End to end processing with p+xs fusion
    and input classification to test
    """

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
                "strip_height": 80,
                "save_intermediate_data": True,
            },
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
        }

        input_config_sparse_res["applications"].update(application_config)

        sparse_res_pipeline = sensor_to_sparse_dsm.SensorSparseDsmPipeline(
            input_config_sparse_res
        )
        sparse_res_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
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
            assert out_disp_compute["minimum_disparity"] > -29
            assert out_disp_compute["minimum_disparity"] < -27
            assert out_disp_compute["maximum_disparity"] > 26
            assert out_disp_compute["maximum_disparity"] < 30

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

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
            },
            "dense_matching": {
                "method": "census_sgm",
                "loader": "pandora",
                "save_intermediate_data": True,
                "use_global_disp_range": False,
            },
            "point_cloud_fusion": {
                "method": "mapping_to_terrain_tiles",
                "save_intermediate_data": True,
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
            "triangulation": {
                "method": "line_of_sight_intersection",
                "save_intermediate_data": True,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631

        # Save classif
        input_config_dense_dsm["output"]["auxiliary"] = {"classification": True}
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        # update pipeline
        input_config_dense_dsm["pipeline"] = "sensors_to_dense_dsm"

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_sparse_res["output"]["directory"]

        print(os.listdir(os.path.join(out_dir, "points_cloud")))
        if input_config_dense_dsm["geometry_plugin"] == "OTBGeometry":
            pc1 = "675436.5_4897170.5"
        else:
            pc1 = "675437.0_4897170.0"

        assert (
            os.path.exists(os.path.join(out_dir, "points_cloud", pc1 + ".laz"))
            is True
        )
        assert (
            os.path.exists(os.path.join(out_dir, "points_cloud", pc1 + ".csv"))
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_small_components_removing",
                    pc1 + ".csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "points_cloud_post_statistical_removing",
                    pc1 + ".csv",
                )
            )
            is True
        )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the following instruction to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end_ventoux_with_classif.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dsm", "classification.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "classification_end2end_ventoux_with_classif.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_with_classif.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_end2end_ventoux_with_classif.tif",
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
            },
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -1000,  # -20.0,
                "elevation_delta_upper_bound": 1000,  # 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg

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

        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True
        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"
        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end" + "_ventoux_with_roi.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "color_end2end" + "_ventoux_with_roi.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end" + "_ventoux_with_roi.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_with_roi.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

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
        data = rasterio.open(os.path.join(out_dir, "dsm", "dsm.tif"))
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
    test sensor to dense dsm pipeline with snap_to_img1 triangulation option
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dsm_end2end_ventoux_with_snap_to_img1.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "color_end2end_ventoux_with_snap_to_img1.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_with_snap_to_img1.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "color_end2end_ventoux_with_snap_to_img1.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False


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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "disparity_margin": 0.25,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # Save all intermediate data
        input_config_dense_dsm["advanced"] = {"save_intermediate_data": True}

        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "dense_matching_run"
            ]
            assert out_disp_compute["global_disp_min"] > -31
            assert out_disp_compute["global_disp_min"] < -30
            assert out_disp_compute["global_disp_max"] > 25
            assert out_disp_compute["global_disp_max"] < 32

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                                "dem_median.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dem_median_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                                    "dem_min.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dem_min_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "dem_generation",
        #                                                     "dem_max.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dem_max_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dsm_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "color_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                                   "dsm_mean.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dsm_mean_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                                     "dsm_std.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dsm_std_end2end_ventoux_quality_stats.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                                 "dsm_n_pts.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "dsm_n_pts_end2end_ventoux_quality_stats.tif",
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dump_dir", "rasterization",
        #                                           "dsm_pts_in_cell.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir,
        #             "" "dsm_pts_in_cell_end2end_ventoux_quality_stats.tif",
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_median_end2end_ventoux_quality_stats.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dem_min_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dem_max_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_quality_stats.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_mean.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_mean_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_std.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_std_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_n_pts.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_n_pts_end2end_ventoux_quality_stats.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "rasterization", "dsm_pts_in_cell.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_pts_in_cell_end2end_ventoux_quality_stats.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False


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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
            },
            "triangulation": {"method": "line_of_sight_intersection"},
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        input_config_dense_dsm["output"]["geoid"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "dense_matching_run"
            ]
            # global_disp_min   -21 shareloc
            assert out_disp_compute["global_disp_min"] > -34
            assert out_disp_compute["global_disp_min"] < -32
            # global max: 86 shareloc
            assert out_disp_compute["global_disp_max"] > 30
            assert out_disp_compute["global_disp_max"] < 31

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dsm_end2end_ventoux_egm96.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #       os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_egm96.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
    assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
            },
            "triangulation": {"method": "line_of_sight_intersection"},
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        input_config_dense_dsm["output"]["geoid"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_egm96.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

    # Test with custom geoid
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_custom_geoid.json"
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
            },
            "triangulation": {"method": "line_of_sight_intersection"},
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
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        input_config_dense_dsm["output"]["geoid"] = absolute_data_path(
            "input/geoid/egm96_15_modified.tif"
        )

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "dense_matching_run"
            ]
            # global_disp_min   -21 shareloc
            assert out_disp_compute["global_disp_min"] > -34
            assert out_disp_compute["global_disp_min"] < -32
            # global max: 86 shareloc
            assert out_disp_compute["global_disp_max"] > 30
            assert out_disp_compute["global_disp_max"] < 31

            assert os.path.isfile(
                out_data["applications"]["left_right"]["grid_correction"][
                    "corrected_filtered_matches"
                ]
            )

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir,
        #             "dsm_end2end_ventoux_egm96_custom_geoid.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir,
        #             "color_end2end_ventoux_egm96_custom_geoid.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_egm96_custom_geoid.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "color_end2end_ventoux_egm96_custom_geoid.tif",
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )


@pytest.mark.end2end_tests
def test_end2end_paca_with_mask():
    """
    End to end processing sensor to dense pipeline with a mask on paca data.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # paca config contains mask config
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
                "minimum_nb_matches": 10,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
                "msk_no_data": 254,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        input_config_dense_dsm["output"]["auxiliary"] = {"mask": True}
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dsm_end2end_paca.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "color_end2end_paca.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir,  "dsm", "mask.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "mask_end2end_paca.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_paca.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_paca.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "mask_end2end_paca.tif")
            ),
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
            "resampling": {"method": "bicubic", "strip_height": 80},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm",
                "use_global_disp_range": False,
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
                "msk_no_data": 254,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        input_config_dense_dsm["output"]["auxiliary"] = {"mask": True}
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Uncomment the above instructions of first run to update reference data
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_paca.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_paca.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "mask_end2end_paca.tif")
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )


@pytest.mark.end2end_tests
def test_end2end_disparity_filling():
    """
    End to end processing, test with mask and fill holes
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_gizeh/input_mask_fill.json")

        # Run dense dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "mp",
            orchestrator_parameters={
                "nb_workers": 4,
                "max_ram_per_worker": 300,
            },
        )
        resolution = 0.5
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm",
                "min_epi_tile_size": 100,
                "save_intermediate_data": True,
                "use_global_disp_range": False,
            },
            "dense_matches_filling.1": {
                "method": "plane",
                "save_intermediate_data": True,
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
                "msk_no_data": 254,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # Save mask and filling
        input_config_dense_dsm["output"]["auxiliary"] = {
            "filling": True,
            "mask": True,
        }
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "dsm_end2end_gizeh_fill.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "color_end2end_gizeh_fill.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "mask.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir, "mask_end2end_gizeh_fill.tif")
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "filling.tif"),
        #     absolute_data_path(
        #         os.path.join(ref_output_dir,
        #         "filling_end2end_gizeh_fill.tif")
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_gizeh_fill.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_gizeh_fill.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "mask_end2end_gizeh_fill.tif")
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "filling_end2end_gizeh_fill.tif")
            ),
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
        input_json = absolute_data_path("input/phr_gizeh/input_mask_fill.json")

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
                "save_intermediate_data": True,
                "use_global_disp_range": True,
            },
            "dense_matches_filling.2": {
                "method": "zero_padding",
                "save_intermediate_data": True,
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
                "msk_no_data": 254,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        # use endogenous dem
        input_config_dense_dsm["inputs"]["use_endogenous_elevation"] = True

        # Save mask and filling
        input_config_dense_dsm["output"]["auxiliary"] = {
            "filling": True,
            "mask": True,
        }

        dense_dsm_pipeline = sensor_to_dense_dsm.SensorToDenseDsmPipeline(
            input_config_dense_dsm
        )
        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Ref output dir dependent from geometry plugin chosen
        ref_output_dir = "ref_output"

        # Uncomment the 2 following instructions to update reference data
        # copy2(
        #     os.path.join(out_dir, "dsm", "dsm.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "dsm_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dsm", "color.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "color_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir", "rasterization",  "mask.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "mask_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )
        # copy2(
        #     os.path.join(out_dir, "dump_dir",
        #                                 "rasterization",  "filling.tif"),
        #     absolute_data_path(
        #         os.path.join(
        #             ref_output_dir, "filling_end2end_gizeh_fill_with_zero.tif"
        #         )
        #     ),
        # )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_gizeh_fill_with_zero.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "color.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_gizeh_fill_with_zero.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "mask_end2end_gizeh_fill_with_zero.tif"
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "filling_end2end_gizeh_fill_with_zero.tif"
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )


@pytest.mark.end2end_tests
def test_end2end_gizeh_dry_run_of_used_conf():
    """
    End to end processing

    Test pipeline with a non square epipolar image, and the generation
    of the performance map
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/data_gizeh_crop/configfile_crop.json"
        )

        # Run sensors pipeline with simple config
        _, sensors_input_config_first_run = generate_input_json(
            input_json,
            directory,
            "sensors_to_dense_dsm",
            "multiprocessing",
        )

        applications = {
            "triangulation": {
                "save_intermediate_data": True,
            }
        }

        sensors_input_config_first_run["applications"].update(applications)

        sensors_pipeline_first_run = (
            sensor_to_dense_dsm.SensorToDenseDsmPipeline(
                sensors_input_config_first_run
            )
        )
        sensors_pipeline_first_run.run()
        sensors_out_dir_first_run = sensors_input_config_first_run["output"][
            "directory"
        ]

        # Run sensors pipeline with generated config
        used_conf = os.path.join(sensors_out_dir_first_run, "used_conf.json")
        with open(used_conf, "r", encoding="utf8") as fstream:
            sensors_input_config_second_run = json.load(fstream)

        sensors_input_config_second_run["output"][
            "directory"
        ] += "_from_used_conf"

        sensors_pipeline_second_run = (
            sensor_to_dense_dsm.SensorToDenseDsmPipeline(
                sensors_input_config_second_run
            )
        )
        sensors_pipeline_second_run.run()
        sensors_out_dir_second_run = sensors_input_config_second_run["output"][
            "directory"
        ]

        assert_same_images(
            os.path.join(sensors_out_dir_first_run, "dsm", "dsm.tif"),
            os.path.join(sensors_out_dir_second_run, "dsm", "dsm.tif"),
            atol=0.0001,
            rtol=1e-6,
        )

        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            # Run pc pipeline with simple config
            epi_pc_path = os.path.join(
                sensors_out_dir_first_run,
                "dump_dir",
                "triangulation",
                "one_two",
            )
            pc_input_config_first_run = {
                "inputs": {
                    "point_clouds": {
                        "one": {
                            "x": os.path.join(epi_pc_path, "epi_pc_X.tif"),
                            "y": os.path.join(epi_pc_path, "epi_pc_Y.tif"),
                            "z": os.path.join(epi_pc_path, "epi_pc_Z.tif"),
                            "color": os.path.join(
                                epi_pc_path, "epi_pc_color.tif"
                            ),
                        }
                    }
                },
                "output": {"directory": directory2},
            }

            pc_pipeline_first_run = pipeline_dsm.PointCloudsToDsmPipeline(
                pc_input_config_first_run
            )
            pc_pipeline_first_run.run()
            pc_out_dir_first_run = pc_input_config_first_run["output"][
                "directory"
            ]

            # Run pc pipeline with generated config
            used_conf = os.path.join(pc_out_dir_first_run, "used_conf.json")
            with open(used_conf, "r", encoding="utf8") as fstream:
                pc_input_config_second_run = json.load(fstream)

            pc_input_config_second_run["output"][
                "directory"
            ] += "_from_used_conf"

            pc_pipeline_second_run = pipeline_dsm.PointCloudsToDsmPipeline(
                pc_input_config_second_run
            )
            pc_pipeline_second_run.run()
            pc_out_dir_second_run = pc_input_config_second_run["output"][
                "directory"
            ]

            assert_same_images(
                os.path.join(pc_out_dir_first_run, "dsm", "dsm.tif"),
                os.path.join(pc_out_dir_second_run, "dsm", "dsm.tif"),
                atol=0.0001,
                rtol=1e-6,
            )
