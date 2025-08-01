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
from cars.core import inputs, roi_tools
from cars.pipelines.default import default_pipeline as default
from cars.pipelines.unit import unit_pipeline as unit

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    temporary_dir,
)

NB_WORKERS = 2


@pytest.mark.end2end_tests
def test_end2end_dsm_fusion():
    """
    End to end processing

    test the phased dsm re-entrance
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_color_and_classif.json"
        )

        # Run dense dsm pipeline
        _, input_dense_dsm_lr = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                    "msk_no_data": 254,
                    "save_intermediate_data": True,
                },
            }
        }

        input_dense_dsm_lr["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_dense_dsm_lr["output"]["epsg"] = final_epsg
        resolution = 0.5
        input_dense_dsm_lr["output"]["resolution"] = resolution
        input_dense_dsm_lr["output"]["auxiliary"] = {
            "mask": True,
            "performance_map": True,
            "weights": True,
            "filling": True,
            "texture": True,
            "contributing_pair": True,
            "classification": True,
        }

        input_dense_dsm_lr["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_dense_dsm_lr)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_dense_dsm_lr["output"]["directory"])

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "phased_dsm_end2end_ventoux_lr.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_ventoux_lr.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_end2end_ventoux_lr.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "classif_end2end_ventoux_lr.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "contributing_pair_end2end_ventoux_lr.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "weights_end2end_ventoux_lr.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir/rasterization/", "dsm_inf.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dsm_inf_end2end_ventoux_lr.tif"
                )
            ),
        )

        in_dsm = {
            "dsm": absolute_data_path(
                "ref_output/phased_dsm_end2end_ventoux_lr.tif"
            ),
            "weights": absolute_data_path(
                "ref_output/weights_end2end_ventoux_lr.tif"
            ),
            "texture": absolute_data_path(
                "ref_output/color_end2end_ventoux_lr.tif"
            ),
            "classification": absolute_data_path(
                "ref_output/classif_end2end_ventoux_lr.tif"
            ),
            "performance_map": absolute_data_path(
                "ref_output/performance_map_end2end_ventoux_lr.tif"
            ),
            "dsm_inf": absolute_data_path(
                "ref_output/dsm_inf_end2end_ventoux_lr.tif"
            ),
            "contributing_pair": absolute_data_path(
                "ref_output/contributing_pair_end2end_ventoux_lr.tif"
            ),
        }

        input_dsm_config = {
            "inputs": {
                "dsms": {
                    "one": in_dsm,
                    "two": in_dsm,
                }
            }
        }

        input_dsm_config["advanced"] = {}
        input_dsm_config["advanced"]["dsm_merging_tile_size"] = 100
        input_dsm_config["advanced"]["epipolar_resolutions"] = 1

        input_dsm_config["output"] = {}
        input_dsm_config["output"]["directory"] = directory
        input_dsm_config["output"]["auxiliary"] = {
            "mask": True,
            "performance_map": True,
            "weights": True,
            "filling": True,
            "texture": True,
            "contributing_pair": True,
            "classification": True,
        }

        dsm_merging_pipeline = default.DefaultPipeline(input_dsm_config)
        dsm_merging_pipeline.run()

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "phased_dsm_end2end_ventoux_fusion.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "weights_end2end_ventoux_fusion.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_ventoux_fusion.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "contributing_pair_end2end_ventoux_fusion.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_end2end_ventoux_fusion.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_end2end_ventoux_fusion.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "phased_dsm_end2end_ventoux_fusion.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "weights_end2end_ventoux_fusion.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_fusion.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "contributing_pair_end2end_ventoux_fusion.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "classification_end2end_ventoux_fusion.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "performance_map_end2end_ventoux_fusion.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        # assertion on descriptions and classes

        assert inputs.get_descriptions_bands(
            os.path.join(out_dir, "dsm", "texture.tif")
        ) == (None, None, None, None)
        assert inputs.get_descriptions_bands(
            os.path.join(out_dir, "dsm", "classification.tif")
        ) == ("b0", "b1", "b2")
        assert inputs.get_descriptions_bands(
            os.path.join(out_dir, "dsm", "contributing_pair.tif")
        ) == ("left_right",)
        assert str(
            inputs.rasterio_get_tags(
                os.path.join(out_dir, "dsm", "performance_map.tif")
            )["CLASSES"]
        ) == str(
            {
                0: (0, 0.968),
                1: (0.968, 1.13375),
                2: (1.13375, 1.295),
                3: (1.295, 1.604),
                4: (1.604, 2.423),
                5: (2.423, 3.428),
                6: (3.428, math.inf),
            }
        )

        # Run the same mergin, for only basic data
        in_dsm_base = {
            "dsm": absolute_data_path(
                "ref_output/phased_dsm_end2end_ventoux_lr.tif"
            ),
            "weights": absolute_data_path(
                "ref_output/weights_end2end_ventoux_lr.tif"
            ),
        }

        input_dsm_config_base = {
            "inputs": {
                "dsms": {
                    "one": in_dsm_base,
                    "two": in_dsm_base,
                }
            }
        }

        input_dsm_config_base["output"] = {}
        input_dsm_config_base["output"]["directory"] = os.path.join(
            directory, "other"
        )
        os.makedirs(input_dsm_config_base["output"]["directory"], exist_ok=True)
        input_dsm_config_base["advanced"] = {}
        input_dsm_config_base["advanced"]["epipolar_resolutions"] = 1

        dsm_merging_pipeline = unit.UnitPipeline(input_dsm_config_base)
        dsm_merging_pipeline.run()

        assert_same_images(
            os.path.join(
                input_dsm_config_base["output"]["directory"], "dsm", "dsm.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "phased_dsm_end2end_ventoux_fusion.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_color_after_dsm_reentrance():
    """
    End to end processing

    test the colorisation after depth_map re entrance
    """

    in_dsm = {
        "dsm": absolute_data_path(
            "ref_output/phased_dsm_end2end_ventoux_lr.tif"
        ),
        "weights": absolute_data_path(
            "ref_output/weights_end2end_ventoux_lr.tif"
        ),
        "texture": absolute_data_path(
            "ref_output/color_end2end_ventoux_lr.tif"
        ),
    }
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_dsm_config = {
            "inputs": {
                "dsms": {"one": in_dsm, "two": in_dsm},
                "sensors": {
                    "one": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": {
                            "path": absolute_data_path(
                                "input/phr_ventoux/left_image.geom"
                            ),
                        },
                    },
                    "two": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": {
                            "path": absolute_data_path(
                                "input/phr_ventoux/right_image.geom"
                            ),
                        },
                    },
                },
                "pairing": [["one", "two"]],
                "initial_elevation": absolute_data_path(
                    "input/phr_ventoux/srtm/N44E005.hgt"
                ),
            },
            "applications": {
                "auxiliary_filling": {"save_intermediate_data": True}
            },
        }

        input_dsm_config["output"] = {}
        input_dsm_config["output"]["directory"] = directory

        out_dir = input_dsm_config["output"]["directory"]

        input_dsm_config["advanced"] = {}
        input_dsm_config["advanced"]["epipolar_resolutions"] = 1

        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        dsm_merging_pipeline = unit.UnitPipeline(input_dsm_config)
        dsm_merging_pipeline.run()

        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "colorisation_end2end_gizeh_reentrance.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "colorisation_end2end_gizeh_reentrance.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_gizeh_rectangle_epi_image_performance_map():
    """
    End to end processing

    Test pipeline with a non square epipolar image, and the generation
    of the performance map and the ground truth reprojection
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/data_gizeh_crop/configfile_crop.json"
        )

        # Run dense dsm pipeline
        _, input_dense_dsm = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "dem_generation": {"resolution": 30},
                "sparse_matching.sift": {
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": True,
                },
                "ground_truth_reprojection": {
                    "method": "direct_loc",
                    "target": "all",
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                    "msk_no_data": 254,
                },
            }
        }

        input_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_dense_dsm["output"]["epsg"] = final_epsg
        resolution = 0.5
        input_dense_dsm["output"]["resolution"] = resolution
        input_dense_dsm["output"]["auxiliary"] = {
            "mask": True,
            "performance_map": True,
        }

        # Ground truth generation
        dsm_gt = input_dense_dsm["inputs"]["initial_elevation"]["dem"]

        input_dense_dsm["advanced"]["ground_truth_dsm"] = {"dsm": dsm_gt}
        input_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_dense_dsm["output"]["directory"])

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_gizeh_crop_no_merging.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_gizeh_crop_no_merging.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "mask_end2end_gizeh_crop_no_merging.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_end2end_gizeh_crop_no_merging.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "epipolar_disp_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ref_epipolar_disp_ground_truth_left.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "epipolar_disp_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ref_epipolar_disp_ground_truth_right.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "sensor_dsm_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ref_sensor_dsm_ground_truth_left.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "sensor_dsm_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ref_sensor_dsm_ground_truth_right.tif",
                )
            ),
        )
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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "epipolar_disp_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ref_epipolar_disp_ground_truth_left.tif",
                )
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "one_two",
                "epipolar_disp_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ref_epipolar_disp_ground_truth_right.tif",
                )
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir/",
                "ground_truth_reprojection/one_two/"
                "sensor_dsm_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ref_sensor_dsm_ground_truth_right.tif",
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )

        application_config = {
            "resolution_4": {
                "sparse_matching.sift": {
                    "method": "sift",
                    # Uncomment the following line to update dsm reference data
                    # "sift_peak_threshold":1,
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "save_intermediate_data": False,
                    "decimation_factor": 100,
                },
            },
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic"},
                "sparse_matching.sift": {
                    "method": "sift",
                    # Uncomment the following line to update dsm reference data
                    # "sift_peak_threshold":1,
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "save_intermediate_data": False,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }

        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_dsm["applications"] = application_config
        input_config_sparse_dsm["output"].update(output_config)

        input_config_sparse_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                -65
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["minimum_disparity"]
                < -11
            )
            assert (
                0
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["maximum_disparity"]
                < 46
            )

        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        out_dir_res4 = os.path.join(
            input_config_sparse_dsm["output"]["directory"],
            "intermediate_res/out_res4",
        )

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_median_end2end_ventoux_8bit.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dem_min_end2end_ventoux_8bit.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dem_max_end2end_ventoux_8bit.tif"
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dem_median_end2end_ventoux_8bit.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_min_end2end_ventoux_8bit.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True,
                    "use_cross_validation": False,
                    "save_intermediate_data": True,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }
        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_dsm["applications"] = application_config
        input_config_sparse_dsm["output"].update(output_config)

        input_config_sparse_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])
        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                -66
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["minimum_disparity"]
                < -17
            )
            assert (
                0
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["maximum_disparity"]
                < 46
            )

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        out_dir_res4 = os.path.join(
            input_config_sparse_dsm["output"]["directory"],
            "intermediate_res/out_res4",
        )

        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dem_median_end2end_ventoux.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dem_min_end2end_ventoux.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dem_max_end2end_ventoux.tif"
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_median_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_min_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dem_max_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        # Check used_conf for sparse res
        gt_used_conf_orchestrator = {
            "orchestrator": {
                "mode": "multiprocessing",
                "mp_mode": "forkserver",
                "nb_workers": NB_WORKERS,
                "profiling": {
                    "mode": "cars_profiling",
                    "loop_testing": False,
                },
                "max_ram_per_worker": 1000,
                "task_timeout": 600,
                "max_tasks_per_worker": 10,
                "dump_to_disk": True,
                "per_job_timeout": 600,
                "factorize_tasks": True,
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
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["sparse_matching.sift"][
                    "disparity_margin"
                ]
                == 0.25
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )

            # check used_conf reentry
            _ = unit.UnitPipeline(used_conf)

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline (keep geometry_plugin)
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "activated": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "activated": True,
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "use_median": False,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": "accurate",
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
        }
        input_config_dense_dsm["applications"]["resolution_1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # update output product
        input_config_dense_dsm["output"]["product_level"] = ["dsm"]

        input_config_dense_dsm["output"]["auxiliary"] = {"ambiguity": True}
        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5

        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Check used_conf for dense dsm
        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
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
            _ = unit.UnitPipeline(used_conf)

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(intermediate_output_dir, "dsm_end2end_ventoux.tif")
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_ventoux.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "confidence_from_ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_ambiguity_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_intensity_std_std_intensity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_intensity_std_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_min_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_risk_min_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_max_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_risk_max_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dsm", "confidence_from_ambiguity_before.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_ambiguity_before_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_intensity_std_std_intensity_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_intensity"
                    + "_std_before_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_min_risk_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_risk_min_before_end2end_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "rasterization",
                "confidence_from_risk_max_risk_before.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_risk_max_before_end2end_ventoux.tif",
                )
            ),
        )
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
                "dsm",
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
                "dsm",
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
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

    # Test that we have the same results without setting the texture
    input_json = absolute_data_path(
        "input/phr_ventoux/input_without_color.json"
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # Run sparse dsm pipeline
        _, input_config_sparse_dsm = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True,
                    "use_cross_validation": False,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }

        input_config_sparse_dsm["applications"] = application_config

        input_config_sparse_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm_default",
                "use_global_disp_range": False,
                "use_cross_validation": False,
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
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "activated": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "activated": True,
                "use_median": False,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
            },
        }
        input_config_dense_dsm["applications"]["resolution_1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5

        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "mp",
            orchestrator_parameters={
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True,
                    "use_cross_validation": False,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }

        input_config_sparse_dsm["applications"] = application_config

        input_config_sparse_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # dense dsm pipeline
        input_config_dense_dsm = input_config_sparse_dsm.copy()
        # update applications
        dense_dsm_applications = {
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": "accurate",
                "use_global_disp_range": False,
            },
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "activated": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "activated": True,
                "use_median": False,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
            },
        }
        input_config_dense_dsm["applications"]["resolution_1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5

        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.005,
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
        # Run sensors_to_dense_depth_maps pipeline
        _, input_config_pc = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        input_config_pc["applications"] = {
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": True,
            },
        }

        input_config_pc["output"]["product_level"] = ["depth_map"]

        input_config_pc["advanced"]["epipolar_resolutions"] = [4, 1]

        pc_pipeline = default.DefaultPipeline(input_config_pc)

        input_config_pc["output"]["epsg"] = 4326

        pc_pipeline.run()

        out_dir = os.path.join(input_config_pc["output"]["directory"])
        geometry_plugin_name = input_config_pc["advanced"]["geometry_plugin"]

        # Create input json for pc to dsm pipeline
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            depth_map_path = os.path.join(out_dir, "depth_map", "left_right")
            output_path = os.path.join(directory2, "outresults_dsm_from_pc")

            input_dsm_config = {
                "inputs": {
                    "depth_maps": {
                        "one": {
                            "x": os.path.join(depth_map_path, "X.tif"),
                            "y": os.path.join(depth_map_path, "Y.tif"),
                            "z": os.path.join(depth_map_path, "Z.tif"),
                            "texture": os.path.join(
                                depth_map_path, "texture.tif"
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
                    "sensors": {
                        "left": {
                            "image": absolute_data_path(
                                "input/phr_ventoux/left_image.tif"
                            ),
                            "geomodel": {
                                "path": absolute_data_path(
                                    "input/phr_ventoux/left_image.geom"
                                )
                            },
                        },
                        "right": {
                            "image": absolute_data_path(
                                "input/phr_ventoux/right_image.tif"
                            ),
                            "geomodel": {
                                "path": absolute_data_path(
                                    "input/phr_ventoux/left_image.geom"
                                ),
                            },
                        },
                    },
                    "pairing": [["left", "right"]],
                    "initial_elevation": absolute_data_path(
                        "input/phr_ventoux/srtm/N44E005.hgt"
                    ),
                },
                "advanced": {"geometry_plugin": geometry_plugin_name},
                "output": {
                    "directory": output_path,
                    "resolution": 0.000005,
                    "epsg": 4326,
                },
                "applications": {
                    "point_cloud_rasterization": {
                        "method": "simple_gaussian",
                        "save_intermediate_data": True,
                    },
                    "dsm_filling.1": {
                        "method": "exogenous_filling",
                        "activated": True,
                    },
                    "dsm_filling.2": {"method": "bulldozer", "activated": True},
                    "auxiliary_filling": {"activated": True},
                },
            }

            input_dsm_config["advanced"]["epipolar_resolutions"] = 1

            dsm_pipeline = unit.UnitPipeline(input_dsm_config)
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            # Ref output dir dependent from geometry plugin chosen
            intermediate_output_dir = "intermediate_data"
            ref_output_dir = "ref_output"

            copy2(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "dsm_end2end_ventoux_split_4326.tif",
                    )
                ),
            )
            copy2(
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "color_end2end_ventoux_split_4326.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "source_pc_end2end" + "_ventoux_split_4326.tif",
                    )
                ),
            )

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
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
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
        # Run sensors_to_dense_depth_maps pipeline
        _, input_config_pc = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 200},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": False,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                    "save_intermediate_data": True,
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
                            "filter": {
                                "filter_method": "disparity_denoiser",
                                "filter_size": 3,
                            },
                            "validation": {
                                "validation_method": "cross_checking_accurate",
                                "cross_checking_threshold": 1.0,
                            },
                        },
                    },
                },
                "dense_match_filling.1": {
                    "method": "plane",
                    "classification": ["b0"],
                },
                "triangulation": {
                    "method": "line_of_sight_intersection",
                    "save_intermediate_data": True,
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                    "save_intermediate_data": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "save_intermediate_data": True,
                    "use_median": False,
                },
            },
        }

        input_config_pc["applications"] = application_config

        input_config_pc["output"]["product_level"] = ["depth_map"]
        input_config_pc["output"]["auxiliary"] = {
            "classification": True,
            "filling": True,
            "texture": True,
            "performance_map": True,
            "ambiguity": True,
        }

        input_config_pc["advanced"]["epipolar_resolutions"] = [4, 1]

        pc_pipeline = default.DefaultPipeline(input_config_pc)
        pc_pipeline.run()

        out_dir = os.path.join(input_config_pc["output"]["directory"])
        geometry_plugin_name = input_config_pc["advanced"]["geometry_plugin"]

        # Create input json for pc to dsm pipeline
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory2:
            depth_map_path = os.path.join(out_dir, "depth_map", "left_right")

            output_path = os.path.join(directory2, "outresults_dsm_from_pc")

            input_dsm_config = {
                "inputs": {
                    "depth_maps": {
                        "one": {
                            "x": os.path.join(depth_map_path, "X.tif"),
                            "y": os.path.join(depth_map_path, "Y.tif"),
                            "z": os.path.join(depth_map_path, "Z.tif"),
                            "texture": os.path.join(
                                depth_map_path, "texture.tif"
                            ),
                            "classification": os.path.join(
                                depth_map_path, "classification.tif"
                            ),
                            "filling": os.path.join(
                                depth_map_path, "filling.tif"
                            ),
                            "performance_map": os.path.join(
                                depth_map_path, "performance_map.tif"
                            ),
                            "confidence": {
                                "confidence_from_ambiguity2": os.path.join(
                                    depth_map_path,
                                    "confidence_from_ambiguity.tif",
                                ),
                                "confidence_from_ambiguity1": os.path.join(
                                    depth_map_path,
                                    "confidence_from_ambiguity_before.tif",
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
                "output": {
                    "directory": output_path,
                    "resolution": 0.5,
                    "auxiliary": {"ambiguity": True},
                },
                "applications": {
                    "point_cloud_rasterization": {
                        "method": "simple_gaussian",
                        "dsm_radius": 3,
                        "sigma": 0.3,
                        "dsm_no_data": -999,
                        "texture_no_data": 0,
                        "save_intermediate_data": True,
                    },
                },
                "advanced": {
                    "geometry_plugin": geometry_plugin_name,
                    "merging": True,
                },
            }

            input_dsm_config["advanced"]["epipolar_resolutions"] = 1

            dsm_pipeline = unit.UnitPipeline(input_dsm_config)
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            # Ref output dir dependent from geometry plugin chosen
            intermediate_output_dir = "intermediate_data"
            ref_output_dir = "ref_output"

            assert (
                os.path.exists(
                    os.path.join(
                        out_dir,
                        "dump_dir",
                        "triangulation",
                        "left_right",
                        "laz",
                        "0_0.laz",
                    )
                )
                is True
            )

            copy2(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir, "dsm_end2end_ventoux_split.tif"
                    )
                ),
            )
            copy2(
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "color_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "mask.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "mask_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "classification.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classification_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "performance_map.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "performance_map_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "filling.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "filling_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dsm", "confidence_from_ambiguity1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "confidence_from_ambiguity1"
                        + "_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dsm", "confidence_from_ambiguity2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "confidence_from_ambiguity2"
                        + "_end2end_ventoux_split.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "source_pc_end2end_ventoux_split.tif",
                    )
                ),
            )

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
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
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
                    "performance_map.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "performance_map_end2end_ventoux_split.tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )

            assert_same_images(
                os.path.join(
                    out_dir_dsm,
                    "dsm",
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
                    "dsm",
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

            # avoid deleting other variables in advanced if it exists
            if "advanced" in input_dsm_config:
                input_dsm_config["advanced"]["merging"] = False
            else:
                input_dsm_config["advanced"] = {"merging": False}

            input_dsm_config["output"]["product_level"] = ["dsm"]
            input_dsm_config["output"]["auxiliary"] = {"ambiguity": True}

            input_dsm_config["output"]["directory"] = (
                input_dsm_config["output"]["directory"] + "_no_merging"
            )
            input_dsm_config["advanced"]["merging"] = False

            # launch
            dsm_pipeline = unit.UnitPipeline(input_dsm_config)
            dsm_pipeline.run()

            out_dir_dsm = input_dsm_config["output"]["directory"]

            copy2(
                os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "dsm_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "clr_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "mask.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "msk_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "classification.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "classif_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm,
                    "dump_dir",
                    "rasterization",
                    "performance_map.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "performance_map_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "filling.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "filling_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dsm", "confidence_from_ambiguity1.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "confidence_from_ambiguity1"
                        + "_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dsm", "confidence_from_ambiguity2.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "confidence_from_ambiguity2"
                        + "_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )
            copy2(
                os.path.join(
                    out_dir_dsm, "dump_dir", "rasterization", "source_pc.tif"
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "source_pc_end2end_ventoux_split_no_merging.tif",
                    )
                ),
            )

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
                os.path.join(out_dir_dsm, "dsm", "texture.tif"),
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
                    "performance_map.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir,
                        "performance_map_end2end_ventoux_split_no_merging.tif",
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
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
                    "dsm",
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
                    "dsm",
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        # no srtm
        input_config_sparse_res["inputs"]["initial_elevation"] = None

        application_config = {
            "resolution_2": {
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "nb_points_threshold": 150,
                    "connection_distance": 3.0,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 100,
                },
            },
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 200},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 100,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "denoise_disparity_map": True,
                    "use_cross_validation": False,
                    "use_global_disp_range": True,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }
        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_res["output"].update(output_config)
        input_config_sparse_res["applications"] = application_config

        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [2, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["grid_generation"]["left_right"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                -45
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["minimum_disparity"]
                < -15
            )
            assert (
                0
                < out_json["applications"]["disparity_range_computation"][
                    "left_right"
                ]["maximum_disparity"]
                < 24
            )

            # Ref output dir dependent from geometry plugin chosen
            intermediate_output_dir = "intermediate_data"
            ref_output_dir = "ref_output"
            output_dir_res4 = os.path.join(
                input_config_sparse_res["output"]["directory"],
                "intermediate_res/out_res2",
            )

            copy2(
                os.path.join(
                    output_dir_res4,
                    "dsm",
                    "dem_median.tif",
                ),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "dem_median_end2end_ventoux_no_srtm.tif",
                    )
                ),
            )
            copy2(
                os.path.join(output_dir_res4, "dsm", "dem_min.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "dem_min_end2end_ventoux_no_srtm.tif",
                    )
                ),
            )
            copy2(
                os.path.join(output_dir_res4, "dsm", "dem_max.tif"),
                absolute_data_path(
                    os.path.join(
                        intermediate_output_dir,
                        "dem_max_end2end_ventoux_no_srtm.tif",
                    )
                ),
            )

            assert_same_images(
                os.path.join(
                    output_dir_res4,
                    "dsm",
                    "dem_median.tif",
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
                os.path.join(output_dir_res4, "dsm", "dem_min.tif"),
                absolute_data_path(
                    os.path.join(
                        ref_output_dir, "dem_min_end2end_ventoux_no_srtm.tif"
                    )
                ),
                atol=0.0001,
                rtol=1e-6,
            )
            assert_same_images(
                os.path.join(output_dir_res4, "dsm", "dem_max.tif"),
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
                "mode": "multiprocessing",
                "mp_mode": "forkserver",
                "nb_workers": NB_WORKERS,
                "profiling": {
                    "mode": "cars_profiling",
                    "loop_testing": False,
                },
                "max_ram_per_worker": 1000,
                "task_timeout": 600,
                "max_tasks_per_worker": 10,
                "dump_to_disk": True,
                "per_job_timeout": 600,
                "factorize_tasks": True,
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
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["sparse_matching.sift"][
                    "disparity_margin"
                ]
                == 0.25
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )

            # check used_conf sparse_matching configuration
            assert "advanced" in used_conf
            assert "use_epipolar_a_priori" in used_conf["advanced"]

            # use_epipolar_a_priori should be false in used_conf
            assert "epipolar_a_priori" in used_conf["advanced"]
            assert (
                "grid_correction"
                in used_conf["advanced"]["epipolar_a_priori"]["left_right"]
            )
            assert "dem_median" in used_conf["advanced"]["terrain_a_priori"]
            assert "dem_min" in used_conf["advanced"]["terrain_a_priori"]
            assert "dem_max" in used_conf["advanced"]["terrain_a_priori"]

            # check used_conf reentry (without epipolar a priori activated)
            _ = unit.UnitPipeline(used_conf)

        # dense dsm pipeline
        input_config_dense_dsm = used_conf.copy()

        # Set use_epipolar_a_priori to True
        input_config_dense_dsm["advanced"]["use_epipolar_a_priori"] = True

        # update applications
        input_config_dense_dsm["applications"] = input_config_sparse_res[
            "applications"
        ]["resolution_1"]
        dense_dsm_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": True,
                "use_global_disp_range": False,
            },
        }
        input_config_dense_dsm["applications"].update(dense_dsm_applications)
        # product level
        input_config_dense_dsm["output"]["product_level"] = ["dsm"]
        input_config_dense_dsm["output"]["auxiliary"] = {"ambiguity": True}

        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631
        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5
        # Update outdir, write new dir
        input_config_dense_dsm["output"]["directory"] += "dense"

        dense_dsm_pipeline = unit.UnitPipeline(input_config_dense_dsm)

        dense_dsm_pipeline.run(
            use_sift_a_priori=True, first_res_out_dir=output_dir_res4
        )

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
            _ = unit.UnitPipeline(used_conf)

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end" + "_ventoux_no_srtm.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end" + "_ventoux_no_srtm.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dsm", "confidence_from_ambiguity_cars_1.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "confidence_from_ambiguity_end2end_ventoux_no_srtm.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_no_srtm.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
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
                "dsm",
                "confidence_from_ambiguity_cars_1.tif",
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 2000,
            },
        )
        application_config = {
            "resolution_2": {
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "epipolar_error_maximum_bias": 50.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 120.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 100,
                },
            },
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 100},
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True,
                    "denoise_disparity_map": True,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True,
                    "coregistration": False,
                },
            },
        }

        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_res["applications"] = application_config
        input_config_sparse_res["output"].update(output_config)

        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [2, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"][
                "disparity_range_computation"
            ]["left_right"]
            assert out_disp_compute["minimum_disparity"] > -126
            assert out_disp_compute["minimum_disparity"] < -48
            assert out_disp_compute["maximum_disparity"] > -47
            assert out_disp_compute["maximum_disparity"] < 5


@pytest.mark.end2end_tests
def test_end2end_ventoux_full_output_no_elevation():
    """
    End to end processing with all outputs activated, and no input elevation
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # No pairing defined in this input file
        input_json = absolute_data_path(
            "input/phr_ventoux/input_no_elevation.json"
        )

        # Run sensors_to_dense_dsm pipeline
        _, input_config = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )

        application_config = {
            "resolution_2": {
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": 400.0,
                    "elevation_delta_upper_bound": 700.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
            },
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {
                    "method": "bicubic",
                    "strip_height": 80,
                    "save_intermediate_data": True,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": 400.0,
                    "elevation_delta_upper_bound": 700.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dem_generation": {"method": "dichotomic"},
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                },
            },
        }
        advanced_config = {"save_intermediate_data": True}

        out_dir = os.path.join(directory, "output_dsm")
        output_config = {
            "directory": out_dir,
            "product_level": ["depth_map", "point_cloud", "dsm"],
            "auxiliary": {
                "texture": True,
                "weights": True,
                "filling": True,
                "mask": True,
                "classification": True,
                "contributing_pair": True,
            },
        }

        input_config["applications"] = application_config
        input_config["advanced"].update(advanced_config)
        input_config["advanced"]["epipolar_resolutions"] = [2, 1]

        input_config["output"].update(output_config)

        out_dir = os.path.join(directory, "output_dsm")

        pipeline = default.DefaultPipeline(input_config)

        pipeline.run()

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "weights_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "mask_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "contributing_pair_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_color_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "depth_map", "left_right", "classification.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_classification_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_filling_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_mask_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "X.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_X_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "Y.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_Y_end2end_ventoux_no_elevation.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "depth_map", "left_right", "Z.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epi_pc_Z_end2end_ventoux_no_elevation.tif",
                )
            ),
        )

        # DSM
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_ventoux_no_elevation.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "weights_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "mask_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "filling_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "contributing_pair_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

        # Depth map
        depth_map_dir = os.path.join(
            out_dir,
            "depth_map",
            "left_right",
        )
        assert_same_images(
            os.path.join(depth_map_dir, "X.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "epi_pc_X_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "Y.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "epi_pc_Y_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "Z.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "epi_pc_Z_end2end_ventoux_no_elevation.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "mask.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epi_pc_mask_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "filling.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epi_pc_filling_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epi_pc_color_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(depth_map_dir, "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epi_pc_classification_end2end_ventoux_no_elevation.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

        pc_name = "0_0"

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "point_cloud",
                    "left_right",
                    pc_name + ".laz",
                )
            )
            is True
        )

        # Assertions on index files
        depth_map_index_path = os.path.join(out_dir, "depth_map", "index.json")
        dsm_index_path = os.path.join(out_dir, "dsm", "index.json")
        point_cloud_index_path = os.path.join(
            out_dir, "point_cloud", "index.json"
        )

        assert os.path.isfile(depth_map_index_path)
        assert os.path.isfile(dsm_index_path)
        assert os.path.isfile(point_cloud_index_path)

        with open(depth_map_index_path, "r", encoding="utf-8") as json_file:
            depth_map_index = json.load(json_file)
            assert depth_map_index == {
                "left_right": {
                    "x": "left_right/X.tif",
                    "y": "left_right/Y.tif",
                    "z": "left_right/Z.tif",
                    "texture": "left_right/texture.tif",
                    "mask": "left_right/mask.tif",
                    "classification": "left_right/classification.tif",
                    "performance_map": None,
                    "filling": "left_right/filling.tif",
                    "epsg": 4326,
                }
            }

        with open(dsm_index_path, "r", encoding="utf-8") as json_file:
            dsm_index = json.load(json_file)
            assert dsm_index == {
                "dsm": "dsm.tif",
                "texture": "texture.tif",
                "mask": "mask.tif",
                "weights": "weights.tif",
                "classification": "classification.tif",
                "performance_map": None,
                "contributing_pair": "contributing_pair.tif",
                "filling": "filling.tif",
            }

        with open(point_cloud_index_path, "r", encoding="utf-8") as json_file:
            point_cloud_index = json.load(json_file)
            assert point_cloud_index == {
                "left_right": {
                    "0_0": "left_right/0_0.laz",
                    "0_1": "left_right/0_1.laz",
                    "0_2": "left_right/0_2.laz",
                    "1_0": "left_right/1_0.laz",
                    "1_1": "left_right/1_1.laz",
                    "2_1": "left_right/2_1.laz",
                }
            }


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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {
                    "method": "bicubic",
                    "strip_height": 80,
                    "save_intermediate_data": True,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True,
                    "use_cross_validation": False,
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }

        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_res["applications"] = application_config
        input_config_sparse_res["output"].update(output_config)

        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)

        # Add a margin on recitification grid
        # TODO change this when the new API with
        # bicubic interpolator is implemented
        geom_plugin_1 = sparse_res_pipeline.geom_plugin_with_dem_and_geoid
        geom_plugin_2 = sparse_res_pipeline.geom_plugin_without_dem_and_geoid
        geom_plugin_1.rectification_grid_margin = 5
        geom_plugin_2.rectification_grid_margin = 5

        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"][
                "disparity_range_computation"
            ]["left_right"]
            assert out_disp_compute["minimum_disparity"] > -66
            assert out_disp_compute["minimum_disparity"] < -17
            assert out_disp_compute["maximum_disparity"] > 0
            assert out_disp_compute["maximum_disparity"] < 46

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
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
                "save_intermediate_data": True,
            },
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": "accurate",
                "loader": "pandora",
                "save_intermediate_data": True,
                "use_global_disp_range": False,
                "performance_map_method": ["risk", "intervals"],
            },
            "triangulation": {
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "activated": True,
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "activated": True,
                "save_intermediate_data": True,
                "use_median": False,
            },
        }
        input_config_dense_dsm["applications"]["resolution_1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631

        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5

        # update pipeline
        input_config_dense_dsm["output"]["product_level"] = ["dsm"]
        input_config_dense_dsm["output"]["auxiliary"] = {"ambiguity": True}

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)

        # Add a margin on recitification grid
        # TODO change this when the new API with
        # bicubic interpolator is implemented
        geom_plugin_1 = dense_dsm_pipeline.geom_plugin_with_dem_and_geoid
        geom_plugin_2 = dense_dsm_pipeline.geom_plugin_without_dem_and_geoid
        geom_plugin_1.rectification_grid_margin = 5
        geom_plugin_2.rectification_grid_margin = 5

        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dsm",
                    "confidence_from_ambiguity_cars_1.tif",
                )
            )
            is True
        )

        pc1 = "0_0"
        pc2 = "1_0"

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "left_right",
                    "laz",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "left_right",
                    "csv",
                    pc1 + ".csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_1",
                    "left_right",
                    "laz",
                    pc2 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_1",
                    "left_right",
                    "csv",
                    pc2 + ".csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_2",
                    "left_right",
                    "laz",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_2",
                    "left_right",
                    "csv",
                    pc1 + ".csv",
                )
            )
            is True
        )

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_with_color.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_with_color.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dump_dir", "rasterization", "performance_map.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_end2end_ventoux_with_color.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "rasterization", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "weights_end2end_ventoux_with_color.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "triangulation",
                "left_right",
                "performance_map_from_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_from_risk_end2end_ventoux_with_color.tif",
                )
            ),
        )

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
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_with_color.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "rasterization", "performance_map.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "performance_map_end2end_ventoux_with_color.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "rasterization", "weights.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "weights_end2end_ventoux_with_color.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "triangulation",
                "left_right",
                "performance_map_from_risk.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "performance_map_from_risk_end2end_ventoux_with_color.tif",
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        application_config = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {
                    "method": "bicubic",
                    "strip_height": 80,
                    "save_intermediate_data": True,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    # run disp min disp max in the global pipeline
                    "use_global_disp_range": True
                },
                "dem_generation": {
                    # save the dems in the global pipeline
                    "save_intermediate_data": True
                },
            },
        }

        output_config = {
            # reduce computation time by not going further for nothing
            "product_level": ["depth_map"]
        }

        input_config_sparse_res["applications"] = application_config
        input_config_sparse_res["output"].update(output_config)
        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"][
                "disparity_range_computation"
            ]["left_right"]
            assert out_disp_compute["minimum_disparity"] > -66
            assert out_disp_compute["minimum_disparity"] < -17
            assert out_disp_compute["maximum_disparity"] > 0
            assert out_disp_compute["maximum_disparity"] < 46

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
                "sigma": 0.3,
                "dsm_no_data": -999,
                "texture_no_data": 0,
            },
            "dense_matching": {
                "method": "census_sgm_default",
                "use_cross_validation": True,
                "loader": "pandora",
                "save_intermediate_data": True,
                "use_global_disp_range": False,
            },
            "triangulation": {
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "activated": True,
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "activated": True,
                "save_intermediate_data": True,
                "use_median": False,
            },
        }
        input_config_dense_dsm["applications"]["resolution_1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631

        # update epsg
        input_config_dense_dsm["output"]["resolution"] = 0.5

        input_config_dense_dsm["output"]["product_level"] = ["dsm"]

        # Save classif
        input_config_dense_dsm["output"]["auxiliary"] = {"classification": True}
        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])
        pc1 = "0_0"

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "left_right",
                    "laz",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "left_right",
                    "csv",
                    pc1 + ".csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_1",
                    "left_right",
                    "laz",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_1",
                    "left_right",
                    "csv",
                    pc1 + ".csv",
                )
            )
            is True
        )

        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_2",
                    "left_right",
                    "laz",
                    pc1 + ".laz",
                )
            )
            is True
        )
        assert (
            os.path.exists(
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "pc_outlier_removal_2",
                    "left_right",
                    "csv",
                    pc1 + ".csv",
                )
            )
            is True
        )

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_with_classif.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_end2end_ventoux_with_classif.tif",
                )
            ),
        )

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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -1000,  # -20.0,
                    "elevation_delta_upper_bound": 1000,  # 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # resolution
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

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

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end" + "_ventoux_with_roi.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end" + "_ventoux_with_roi.tif",
                )
            ),
        )

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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "triangulation": {
                    "method": "line_of_sight_intersection",
                    "snap_to_img1": True,
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # resolution
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_with_snap_to_img1.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_with_snap_to_img1.tif",
                )
            ),
        )

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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )

        # no srtm
        input_config_dense_dsm["inputs"]["initial_elevation"] = None
        dense_dsm_applications = {
            "resolution_2": {
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                    "nb_points_threshold": 150,
                    "connection_distance": 3.0,
                },
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "decimation_factor": 80,
                },
            },
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": "accurate",
                    "use_global_disp_range": False,
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # resolution
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution

        # Save all intermediate data and add merging
        input_config_dense_dsm["advanced"] = {
            "save_intermediate_data": True,
        }
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [2, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            assert out_disp_compute["global_disp_min"] > -45
            assert out_disp_compute["global_disp_min"] < -15
            assert out_disp_compute["global_disp_max"] > 0
            assert out_disp_compute["global_disp_max"] < 24

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        out_dir_res4 = os.path.join(
            input_config_dense_dsm["output"]["directory"],
            "intermediate_res/out_res2",
        )

        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_median_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_min_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_max_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_mean.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_mean_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_std.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_std_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "rasterization", "dsm_n_pts.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_n_pts_end2end_ventoux_quality_stats.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dump_dir", "rasterization", "dsm_pts_in_cell.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_pts_in_cell_end2end_ventoux_quality_stats.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_median.tif"),
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
            os.path.join(out_dir_res4, "dsm", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dem_min_end2end_ventoux_quality_stats.tif"
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir_res4, "dsm", "dem_max.tif"),
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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "triangulation": {"method": "line_of_sight_intersection"},
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        input_config_dense_dsm["output"]["geoid"] = True

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            assert out_disp_compute["global_disp_min"] > -54
            assert out_disp_compute["global_disp_min"] < 5
            assert out_disp_compute["global_disp_max"] > 0
            assert out_disp_compute["global_disp_max"] < 68

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dsm_end2end_ventoux_egm96.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_ventoux_egm96.tif"
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_egm96.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
    assert os.path.exists(os.path.join(out_dir, "mask.tif")) is False

    # Test that we have the same results without setting the texture1
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_without_color.json"
        )
        # Run dense dsm pipeline
        _, input_config_dense_dsm = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "triangulation": {"method": "line_of_sight_intersection"},
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        # resolution
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution

        input_config_dense_dsm["output"]["geoid"] = True
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_egm96.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "triangulation": {"method": "line_of_sight_intersection"},
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution

        input_config_dense_dsm["output"]["geoid"] = absolute_data_path(
            "input/geoid/egm96_15_modified.tif"
        )

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            assert out_disp_compute["global_disp_min"] > -54
            assert out_disp_compute["global_disp_min"] < 19
            assert out_disp_compute["global_disp_max"] > 0
            assert out_disp_compute["global_disp_max"] < 82

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_egm96_custom_geoid.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_egm96_custom_geoid.tif",
                )
            ),
        )

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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching.sift": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": -20.0,
                    "elevation_delta_upper_bound": 20.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "minimum_nb_matches": 10,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_global_disp_range": False,
                    "use_cross_validation": True,
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                    "nb_points_threshold": 200,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0.0,
                    "mean_factor": 1.0,
                    "activated": True,
                    "use_median": False,
                    "std_dev_factor": 1.0,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                    "msk_no_data": 254,
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                    "activated": True,
                },
                "dsm_filling.2": {"method": "bulldozer", "activated": True},
                "auxiliary_filling": {
                    "save_intermediate_data": True,
                    "mode": "full",
                    "activated": True,
                    "use_mask": True,
                    "texture_interpolator": "linear",
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        input_config_dense_dsm["output"]["auxiliary"] = {
            "mask": True,
            "classification": True,
        }
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline_bulldozer = default.DefaultPipeline(
            input_config_dense_dsm
        )

        # Add a margin on recitification grid
        # TODO change this when the new API with
        # bicubic interpolator is implemented
        geom_plugin_1 = (
            dense_dsm_pipeline_bulldozer.geom_plugin_with_dem_and_geoid
        )
        geom_plugin_2 = (
            dense_dsm_pipeline_bulldozer.geom_plugin_without_dem_and_geoid
        )
        geom_plugin_1.rectification_grid_margin = 5
        geom_plugin_2.rectification_grid_margin = 5

        dense_dsm_pipeline_bulldozer.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dsm_end2end_paca_bulldozer.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_paca_aux_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classification_end2end_paca_aux_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "mask_end2end_paca_bulldozer.tif"
                )
            ),
        )

        # TODO: deal with Bulldozer numerical instability and decrese tolerance
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_paca_bulldozer.tif")
            ),
            rtol=0.1,
            atol=0.1,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "classification_end2end_paca_aux_filling.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_paca_aux_filling.tif"
                )
            ),
            rtol=0.01,
            atol=1,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "mask_end2end_paca_bulldozer.tif")
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )

        # clean out dir for second run
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        input_config_dense_dsm["applications"]["resolution_1"].update(
            {
                "dense_match_filling.2": {
                    "method": "zero_padding",
                    "classification": ["b0", "b2"],
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                    "activated": False,
                },
                "dsm_filling.2": {
                    "method": "bulldozer",
                    "activated": False,
                },
            }
        )

        dense_dsm_pipeline_matches = default.DefaultPipeline(
            input_config_dense_dsm
        )

        # Add a margin on recitification grid
        # TODO change this when the new API with
        # bicubic interpolator is implemented
        geom_plugin_1 = (
            dense_dsm_pipeline_matches.geom_plugin_with_dem_and_geoid
        )
        geom_plugin_2 = (
            dense_dsm_pipeline_matches.geom_plugin_without_dem_and_geoid
        )
        geom_plugin_1.rectification_grid_margin = 5
        geom_plugin_2.rectification_grid_margin = 5

        dense_dsm_pipeline_matches.run()

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_paca_matches_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_paca_matches_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "mask_end2end_paca_matches_filling.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_paca_matches_filling.tif"
                )
            ),
            rtol=1.0e-5,
            atol=2.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_paca_matches_filling.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "mask_end2end_paca_matches_filling.tif"
                )
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )

        # clean out dir for second run
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        input_config_dense_dsm["applications"]["resolution_1"].update(
            {
                "dense_match_filling.2": {
                    "method": "zero_padding",
                    "classification": ["b0", "b2"],
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                    "activated": True,
                    "classification": ["b0", "b2"],
                },
                "dsm_filling.2": {
                    "method": "bulldozer",
                    "activated": False,
                },
                "auxiliary_filling": {"activated": False},
                "dsm_filling.3": {
                    "method": "border_interpolation",
                    "activated": True,
                    "classification": ["b0", "b2"],
                },
            }
        )

        dense_dsm_pipeline_border_interpolation = default.DefaultPipeline(
            input_config_dense_dsm
        )

        dense_dsm_pipeline_border_interpolation.run()

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_paca_border_interpolation.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "dsm_end2end_paca_border_interpolation.tif"
                )
            ),
            rtol=1.0e-5,
            atol=2.0e-7,
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
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 300,
            },
        )
        dense_dsm_applications = {
            "resolution_1": {
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "min_epi_tile_size": 100,
                    "save_intermediate_data": True,
                    "use_global_disp_range": False,
                },
                "dense_match_filling.1": {
                    "method": "plane",
                    "save_intermediate_data": True,
                    "classification": ["b1"],
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "activated": True,
                    "filtering_constant": 0.0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "use_median": False,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "dsm_radius": 3,
                    "sigma": 0.3,
                    "dsm_no_data": -999,
                    "texture_no_data": 0,
                    "msk_no_data": 254,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg

        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        # Save mask and filling
        input_config_dense_dsm["output"]["auxiliary"] = {
            "filling": True,
            "mask": True,
        }

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "dsm_end2end_gizeh_fill.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_gizeh_fill.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "mask_end2end_gizeh_fill.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "filling_end2end_gizeh_fill.tif"
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_gizeh_fill.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
        )
        dense_dsm_applications = {
            "resolution_1": {
                "sparse_matching.sift": {
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "save_intermediate_data": True,
                    "use_global_disp_range": True,
                },
                "dense_match_filling.2": {
                    "method": "zero_padding",
                    "save_intermediate_data": True,
                    "classification": ["b0"],
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "activated": True,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
                    "activated": True,
                    "use_median": False,
                },
            },
        }
        input_config_dense_dsm["applications"] = dense_dsm_applications

        # update epsg
        final_epsg = 32631
        input_config_dense_dsm["output"]["epsg"] = final_epsg
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        # Save mask and filling
        input_config_dense_dsm["output"]["auxiliary"] = {
            "filling": True,
            "mask": True,
        }

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_gizeh_fill_with_zero.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "texture.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_gizeh_fill_with_zero.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "mask.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "mask_end2end_gizeh_fill_with_zero.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_end2end_gizeh_fill_with_zero.tif",
                )
            ),
        )

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
            os.path.join(out_dir, "dsm", "texture.tif"),
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
            "multiprocessing",
        )

        applications = {
            "triangulation": {
                "save_intermediate_data": True,
            }
        }

        sensors_input_config_first_run["applications"] = applications
        sensors_input_config_first_run["advanced"]["epipolar_resolutions"] = [
            4,
            1,
        ]

        sensors_pipeline_first_run = default.DefaultPipeline(
            sensors_input_config_first_run
        )
        sensors_pipeline_first_run.run()
        sensors_out_dir_first_run = os.path.join(
            sensors_input_config_first_run["output"]["directory"]
        )

        # Run sensors pipeline with generated config
        used_conf = os.path.join(sensors_out_dir_first_run, "used_conf.json")
        with open(used_conf, "r", encoding="utf8") as fstream:
            sensors_input_config_second_run = json.load(fstream)

        sensors_input_config_second_run["output"][
            "directory"
        ] += "_from_used_conf"

        first_res_out_dir = os.path.join(
            sensors_input_config_first_run["output"]["directory"],
            "intermediate_res/out_res4",
        )

        sensors_pipeline_second_run = unit.UnitPipeline(
            sensors_input_config_second_run
        )
        sensors_pipeline_second_run.run(
            use_sift_a_priori=True, first_res_out_dir=first_res_out_dir
        )
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
            depth_map_path = os.path.join(
                sensors_out_dir_first_run,
                "dump_dir",
                "triangulation",
                "one_two",
            )
            pc_input_config_first_run = {
                "inputs": {
                    "depth_maps": {
                        "one": {
                            "x": os.path.join(depth_map_path, "X.tif"),
                            "y": os.path.join(depth_map_path, "Y.tif"),
                            "z": os.path.join(depth_map_path, "Z.tif"),
                            "texture": os.path.join(
                                depth_map_path, "texture.tif"
                            ),
                        }
                    }
                },
                "output": {"directory": directory2},
            }

            pc_input_config_first_run["advanced"] = {}
            pc_input_config_first_run["advanced"]["epipolar_resolutions"] = 1

            pc_pipeline_first_run = unit.UnitPipeline(pc_input_config_first_run)
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

            pc_pipeline_second_run = unit.UnitPipeline(
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
