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

import copy
import json
import math
import os
import shutil
import tempfile

# Third party imports
import pyproj
import pytest
import rasterio
from pytest_check import check
from shapely.ops import transform

# CARS imports
from cars.core import inputs, roi_tools
from cars.pipelines.default import default_pipeline as default
from cars.pipelines.unit import unit_pipeline as unit

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
)
from .helpers import cars_copy2 as copy2
from .helpers import (
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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "sparse_matching": {
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
            "performance_map": True,
            "weights": True,
            "filling": True,
            "image": ["b1", "b2", "b3", "b4"],
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            "image": absolute_data_path(
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
            "performance_map": True,
            "weights": True,
            "filling": True,
            "image": ["b1", "b2", "b3", "b4"],
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
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

        with check:
            assert inputs.get_descriptions_bands(
                os.path.join(out_dir, "dsm", "image.tif")
            ) == (None, None, None, None)
        with check:
            assert inputs.get_descriptions_bands(
                os.path.join(out_dir, "dsm", "classification.tif")
            ) == ("1", "2", "3", "4", "5")
        with check:
            assert inputs.get_descriptions_bands(
                os.path.join(out_dir, "dsm", "contributing_pair.tif")
            ) == ("left_right",)
        with check:
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
        "image": absolute_data_path("ref_output/color_end2end_ventoux_lr.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "colorisation_end2end_gizeh_reentrance.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "image.tif"),
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

        # Fill color, and dsm with bulldozer
        dense_dsm_applications = {
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "sparse_matching": {
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
                "auxiliary_filling": {"activated": True},
                "dsm_filling.2": {
                    "method": "bulldozer",
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_gizeh_crop_no_merging.tif",
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
            atol=0.03,  # TODO: analyse
            rtol=1e-2,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_gizeh_crop_no_merging.tif"
                )
            ),
            rtol=0.002,  # TODO: analyse
            atol=2,
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
            "4": {
                "sparse_matching": {
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
            "1": {
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
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    # run disp min disp max in the global pipeline
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

        input_config_sparse_dsm["applications"] = application_config
        input_config_sparse_dsm["output"].update(output_config)

        input_config_sparse_dsm["advanced"]["epipolar_resolutions"] = [4, 1]
        input_config_sparse_dsm["advanced"]["keep_low_res_dir"] = True

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_x"
                    ]
                    == 612
                )
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_y"
                    ]
                    == 612
                )
            with check:
                assert (
                    -85
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["minimum_disparity"]
                    < -75
                )
            with check:
                assert (
                    45
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["maximum_disparity"]
                    < 55
                )

        used_conf_path = os.path.join(out_dir, "current_res_used_conf.json")
        refined_conf_path = os.path.join(out_dir, "refined_conf.json")

        # check used_conf file exists
        with check:
            assert os.path.isfile(used_conf_path)
        with check:
            assert os.path.isfile(refined_conf_path)

        out_dir_res4 = os.path.join(
            input_config_sparse_dsm["output"]["directory"],
            "intermediate_data/out_res4",
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
            "1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
        input_config_sparse_dsm["advanced"]["keep_low_res_dir"] = True

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_dsm)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_dsm["output"]["directory"])
        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_x"
                    ]
                    == 612
                )
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_y"
                    ]
                    == 612
                )
            with check:
                assert (
                    -85
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["minimum_disparity"]
                    < -75
                )
            with check:
                assert (
                    45
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["maximum_disparity"]
                    < 55
                )

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        out_dir_res4 = os.path.join(
            input_config_sparse_dsm["output"]["directory"],
            "intermediate_data/out_res4",
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
                "per_job_timeout": 120,
                "factorize_tasks": True,
            }
        }

        used_conf_path = os.path.join(out_dir, "current_res_used_conf.json")

        # check used_conf file exists
        with check:
            assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            with check:
                assert "inputs" in used_conf
            with check:
                assert "sensors" in used_conf["inputs"]
            # check used_conf sparse_matching configuration
            with check:
                assert (
                    used_conf["applications"]["sparse_matching"][
                        "disparity_margin"
                    ]
                    == 0.25
                )
            # check used_conf orchestrator conf is the same as gt
            with check:
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
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
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
        input_config_dense_dsm["applications"]["1"].update(
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
        used_conf_path = os.path.join(out_dir, "current_res_used_conf.json")

        # check used_conf file exists
        with check:
            assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            with check:
                assert "inputs" in used_conf
            with check:
                assert "sensors" in used_conf["inputs"]
            # check used_conf sparse_matching configuration
            with check:
                assert (
                    used_conf["applications"]["point_cloud_rasterization"][
                        "sigma"
                    ]
                    == 0.3
                )
            # check used_conf orchestrator conf is the same as gt
            with check:
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir, "color_end2end_ventoux.tif"
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ambiguity_end2end_ventoux.tif",
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
                "ambiguity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ambiguity_end2end_ventoux.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
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
            },
        }
        input_config_dense_dsm["applications"]["1"].update(
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
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
            },
        }
        input_config_dense_dsm["applications"]["1"].update(
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux.tif")
            ),
            rtol=0.005,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique_epsg_4326():
    """
    Tes 4326 epsg
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run default pipeline
        _, input_config_dsm = generate_input_json(
            input_json,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 1000,
            },
        )
        input_config_dsm["inputs"]["roi"] = {
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
        input_config_dsm["inputs"]["initial_elevation"] = absolute_data_path(
            "input/phr_ventoux/srtm/N44E005.hgt"
        )
        input_config_dsm["applications"] = {
            "all": {
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                },
                "point_cloud_rasterization": {
                    "method": "simple_gaussian",
                    "save_intermediate_data": True,
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                },
                "dsm_filling.2": {"method": "bulldozer"},
                "auxiliary_filling": {"activated": True},
            }
        }

        input_config_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        sensor_to_dsm_pipeline = default.DefaultPipeline(input_config_dsm)

        input_config_dsm["output"]["epsg"] = 4326

        sensor_to_dsm_pipeline.run()

        out_dir_dsm = os.path.join(input_config_dsm["output"]["directory"])

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"

        copy2(
            os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_end2end_ventoux_4326.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir_dsm, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_4326.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir_dsm,
                "dump_dir",
                "rasterization",
                "contributing_pair.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "contributing_pair_end2end" + "_ventoux_4326.tif",
                )
            ),
        )

        assert_same_images(
            os.path.join(out_dir_dsm, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_end2end_ventoux_4326.tif")
            ),
            atol=0.01,
            rtol=1e-4,
        )
        assert_same_images(
            os.path.join(out_dir_dsm, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_4326.tif")
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(
                out_dir_dsm,
                "dump_dir",
                "rasterization",
                "contributing_pair.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "contributing_pair_end2end_ventoux_4326.tif",
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
            "4": {
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "nb_points_threshold": 150,
                    "connection_distance": 3.0,
                },
                "sparse_matching": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 100,
                },
            },
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 200},
                "sparse_matching": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 100,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
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

        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [4, 1]
        input_config_sparse_res["advanced"]["keep_low_res_dir"] = True

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_x"
                    ]
                    == 612
                )
            with check:
                assert (
                    out_json["applications"]["grid_generation"]["left_right"][
                        "epipolar_size_y"
                    ]
                    == 612
                )
            with check:
                assert (
                    -65
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["minimum_disparity"]
                    < -45
                )
            with check:
                assert (
                    30
                    < out_json["applications"]["disparity_range_computation"][
                        "left_right"
                    ]["maximum_disparity"]
                    < 40
                )

            # Ref output dir dependent from geometry plugin chosen
            intermediate_output_dir = "intermediate_data"
            ref_output_dir = "ref_output"
            output_dir_res4 = os.path.join(
                input_config_sparse_res["output"]["directory"],
                "intermediate_data/out_res4",
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

        refined_conf_path = os.path.join(out_dir, "refined_conf.json")

        # check refined_config_dense_dsm_json file exists
        with check:
            assert os.path.isfile(refined_conf_path)

        with open(refined_conf_path, "r", encoding="utf-8") as json_file:
            refined_conf = json.load(json_file)
            # check refined_conf inputs conf exists
            with check:
                assert "inputs" in refined_conf
            with check:
                assert "sensors" in refined_conf["inputs"]
            with check:
                assert "advanced" in refined_conf

            # use_epipolar_a_priori should be false in refined_conf
            with check:
                assert "epipolar_a_priori" in refined_conf["advanced"]
            with check:
                assert (
                    "grid_correction"
                    in refined_conf["advanced"]["epipolar_a_priori"][
                        "left_right"
                    ]
                )
            with check:
                assert (
                    "dem_median" in refined_conf["advanced"]["terrain_a_priori"]
                )
            with check:
                assert "dem_min" in refined_conf["advanced"]["terrain_a_priori"]
            with check:
                assert "dem_max" in refined_conf["advanced"]["terrain_a_priori"]

            # check refined_conf reentry (without epipolar a priori activated)
            _ = unit.UnitPipeline(refined_conf)

        # dense dsm pipeline
        input_config_dense_dsm = refined_conf.copy()

        # update applications
        input_config_dense_dsm["applications"] = input_config_sparse_res[
            "applications"
        ]["1"]
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

        dense_dsm_pipeline.run()

        out_dir = input_config_dense_dsm["output"]["directory"]

        # Check used_conf for dense_dsm
        used_conf_path = os.path.join(out_dir, "current_res_used_conf.json")

        # check used_conf file exists
        with check:
            assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            with check:
                assert "inputs" in used_conf
            with check:
                assert "sensors" in used_conf["inputs"]
            # check used_conf sparse_matching configuration
            with check:
                assert (
                    used_conf["applications"]["point_cloud_rasterization"][
                        "sigma"
                    ]
                    == 0.3
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end" + "_ventoux_no_srtm.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ambiguity_end2end_ventoux_no_srtm.tif",
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
                "ambiguity.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ambiguity_end2end_ventoux_no_srtm.tif",
                )
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )


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
            "4": {
                "sparse_matching": {
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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 100},
                "dense_matching": {
                    "method": "census_sgm_default",
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

        input_config_sparse_res["advanced"]["epipolar_resolutions"] = [4, 1]

        sparse_res_pipeline = default.DefaultPipeline(input_config_sparse_res)
        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check preproc properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            with check:
                assert out_grid["epipolar_size_x"] == 612
            with check:
                assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"][
                "disparity_range_computation"
            ]["left_right"]
            with check:
                assert out_disp_compute["minimum_disparity"] > -140
            with check:
                assert out_disp_compute["minimum_disparity"] < -120
            with check:
                assert out_disp_compute["maximum_disparity"] > -47
            with check:
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
            "4": {
                "sparse_matching": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "elevation_delta_lower_bound": 400.0,
                    "elevation_delta_upper_bound": 700.0,
                    "disparity_margin": 0.25,
                    "save_intermediate_data": True,
                    "decimation_factor": 80,
                },
            },
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {
                    "method": "bicubic",
                    "strip_height": 80,
                    "save_intermediate_data": True,
                },
                "sparse_matching": {
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
                "image": True,
                "weights": True,
                "filling": True,
                "classification": True,
                "contributing_pair": True,
            },
        }

        input_config["applications"] = application_config
        input_config["advanced"].update(advanced_config)
        input_config["advanced"]["epipolar_resolutions"] = [4, 1]

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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "depth_map", "left_right", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(depth_map_dir, "image.tif"),
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

        with check:
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

        with check:
            assert os.path.isfile(depth_map_index_path)
        with check:
            assert os.path.isfile(dsm_index_path)
        with check:
            assert os.path.isfile(point_cloud_index_path)

        with open(depth_map_index_path, "r", encoding="utf-8") as json_file:
            depth_map_index = json.load(json_file)
            with check:
                assert depth_map_index == {
                    "left_right": {
                        "x": "left_right/X.tif",
                        "y": "left_right/Y.tif",
                        "z": "left_right/Z.tif",
                        "image": "left_right/image.tif",
                        "mask": None,
                        "classification": "left_right/classification.tif",
                        "performance_map": None,
                        "filling": "left_right/filling.tif",
                        "epsg": 4326,
                    }
                }

        with open(dsm_index_path, "r", encoding="utf-8") as json_file:
            dsm_index = json.load(json_file)
            with check:
                assert dsm_index == {
                    "dsm": "dsm.tif",
                    "image": "image.tif",
                    "mask": None,
                    "weights": "weights.tif",
                    "classification": "classification.tif",
                    "performance_map": None,
                    "contributing_pair": "contributing_pair.tif",
                    "filling": "filling.tif",
                }

        with open(point_cloud_index_path, "r", encoding="utf-8") as json_file:
            point_cloud_index = json.load(json_file)
            with check:
                assert point_cloud_index == {
                    "left_right": {
                        "0_0": "left_right/0_0.laz",
                        "1_0": "left_right/1_0.laz",
                        "1_1": "left_right/1_1.laz",
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
            "1": {
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
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
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

        sparse_res_pipeline.run()

        out_dir = os.path.join(input_config_sparse_res["output"]["directory"])

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            with check:
                assert out_grid["epipolar_size_x"] == 612
            with check:
                assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"][
                "disparity_range_computation"
            ]["left_right"]
            with check:
                assert out_disp_compute["minimum_disparity"] > -85
            with check:
                assert out_disp_compute["minimum_disparity"] < -65
            with check:
                assert out_disp_compute["maximum_disparity"] > 45
            with check:
                assert out_disp_compute["maximum_disparity"] < 55

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
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "save_intermediate_data": True,
                "use_median": False,
            },
        }
        input_config_dense_dsm["applications"]["1"].update(
            dense_dsm_applications
        )
        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631

        # resolution
        input_config_dense_dsm["output"]["resolution"] = 0.5

        # update pipeline
        input_config_dense_dsm["output"]["product_level"] = ["dsm"]
        input_config_dense_dsm["output"]["auxiliary"] = {
            "ambiguity": True,
            "image": ["b1", "b2", "b3", "b4"],
            "performance_map": False,
        }

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)

        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        with check:
            assert (
                os.path.exists(
                    os.path.join(
                        out_dir,
                        "dsm",
                        "ambiguity.tif",
                    )
                )
                is True
            )

        pc1 = "0_0"
        pc2 = "1_0"

        with check:
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
        with check:
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
        with check:
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
        with check:
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
        with check:
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
        with check:
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_ventoux_with_color.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dump_dir", "rasterization", "performance_map_raw.tif"
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
                out_dir, "dump_dir", "rasterization", "performance_map_raw.tif"
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
    Test with pandora 3SGM
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_classif.json"
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

        input_config_dense_dsm["output"]["directory"] = directory
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

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
                "loader_conf": {
                    "input": {},
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "census",
                            "window_size": 5,
                            "subpix": 1,
                        },
                        "optimization": {
                            "optimization_method": "3sgm",
                            "overcounting": False,
                            "penalty": {
                                "P1": 8,
                                "P2": 32,
                                "p2_method": "constant",
                                "penalty_method": "sgm_penalty",
                            },
                            "geometric_prior": {
                                "source": "classif",
                                "classes": ["3"],
                            },
                        },
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": "NaN",
                        },
                        "refinement": {"refinement_method": "vfit"},
                        "filter": {
                            "filter_method": "median",
                            "filter_size": 3,
                        },
                    },
                },
            },
            "triangulation": {
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.1": {
                "method": "small_components",
                "save_intermediate_data": True,
            },
            "point_cloud_outlier_removal.2": {
                "method": "statistical",
                "filtering_constant": 0,
                "mean_factor": 1.0,
                "std_dev_factor": 5.0,
                "save_intermediate_data": True,
                "use_median": False,
            },
        }
        input_config_dense_dsm["applications"] = {"1": dense_dsm_applications}

        # update epsg
        input_config_dense_dsm["output"]["epsg"] = 32631

        # update epsg
        input_config_dense_dsm["output"]["resolution"] = 0.5

        input_config_dense_dsm["output"]["product_level"] = ["dsm"]

        # Save classif
        input_config_dense_dsm["output"]["auxiliary"] = {"classification": True}
        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        pc1 = "0_0"

        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with check:
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
        with check:
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
        with check:
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
        with check:
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
        with check:
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
        with check:
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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "use_global_disp_range": False,
                },
                "sparse_matching": {
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

        input_config_dense_dsm["output"]["auxiliary"] = {
            "image": ["b1", "b2", "b3", "b4"]
        }
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_ventoux_with_roi.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

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

        with check:
            assert math.floor(ref_xmin / resolution) * resolution == xmin
        with check:
            assert math.ceil(ref_xmax / resolution) * resolution == xmax
        with check:
            assert math.floor(ref_ymin / resolution) * resolution == ymin
        with check:
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
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "color_end2end_ventoux_with_snap_to_img1.tif",
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )


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
            "4": {
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "nb_points_threshold": 150,
                    "connection_distance": 3.0,
                },
                "sparse_matching": {
                    "method": "sift",
                    "epipolar_error_upper_bound": 43.0,
                    "disparity_margin": 0.25,
                    "decimation_factor": 80,
                },
            },
            "1": {
                "grid_generation": {"method": "epipolar", "epi_step": 30},
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
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

        # Save all intermediate data and
        input_config_dense_dsm["advanced"] = {
            "save_intermediate_data": True,
        }
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline = default.DefaultPipeline(input_config_dense_dsm)
        dense_dsm_pipeline.run()

        out_dir = os.path.join(input_config_dense_dsm["output"]["directory"])
        # Check metadata.json properties
        out_json = os.path.join(out_dir, "metadata.json")
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            with check:
                assert out_grid["epipolar_size_x"] == 612
            with check:
                assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            with check:
                assert out_disp_compute["global_disp_min"] > -65
            with check:
                assert out_disp_compute["global_disp_min"] < -45
            with check:
                assert out_disp_compute["global_disp_max"] > 30
            with check:
                assert out_disp_compute["global_disp_max"] < 40

        # Ref output dir dependent from geometry plugin chosen
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        out_dir_res4 = os.path.join(
            input_config_dense_dsm["output"]["directory"],
            "intermediate_data/out_res4",
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            "1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
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
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            with check:
                assert out_grid["epipolar_size_x"] == 612
            with check:
                assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            with check:
                assert out_disp_compute["global_disp_min"] > -85
            with check:
                assert out_disp_compute["global_disp_min"] < 75
            with check:
                assert out_disp_compute["global_disp_max"] > 45
            with check:
                assert out_disp_compute["global_disp_max"] < 55

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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

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
            "1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "color_end2end_ventoux_egm96.tif")
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

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
            "1": {
                "grid_generation": {
                    "method": "epipolar",
                    "epi_step": 30,
                    "save_intermediate_data": True,
                },
                "resampling": {"method": "bicubic", "strip_height": 80},
                "sparse_matching": {
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
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
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
        with check:
            assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["grid_generation"]["left_right"]
            with check:
                assert out_grid["epipolar_size_x"] == 612
            with check:
                assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["dense_matching"][
                "left_right"
            ]
            with check:
                assert out_disp_compute["global_disp_min"] > -70
            with check:
                assert out_disp_compute["global_disp_min"] < -60
            with check:
                assert out_disp_compute["global_disp_max"] > 0
            with check:
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            "1": {
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
                    "method": "census_sgm_default",
                    "use_global_disp_range": False,
                    "use_cross_validation": True,
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                    "nb_points_threshold": 200,
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0.0,
                    "mean_factor": 1.0,
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
                },
                "dsm_filling.2": {"method": "bulldozer"},
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
            "classification": True,
        }
        resolution = 0.5
        input_config_dense_dsm["output"]["resolution"] = resolution
        input_config_dense_dsm["advanced"]["epipolar_resolutions"] = [4, 1]

        dense_dsm_pipeline_bulldozer = default.DefaultPipeline(
            input_config_dense_dsm
        )

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
            os.path.join(out_dir, "dsm", "image.tif"),
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_paca_aux_filling.tif"
                )
            ),
            rtol=0.01,
            atol=1,
        )

        # clean out dir for second run
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # remove all dsm fillings, add dense match filling
        del input_config_dense_dsm["applications"]["1"]["dsm_filling.1"]
        del input_config_dense_dsm["applications"]["1"]["dsm_filling.2"]

        # Generate new conf with fill with geoid
        input_fill_geoid = copy.deepcopy(input_config_dense_dsm)
        classif_dict = {
            "filling": {
                "fill_with_geoid": 8,
                "interpolate_from_borders": None,
                "fill_with_endogenous_dem": None,
                "fill_with_exogenous_dem": None,
            }
        }
        input_fill_geoid["inputs"]["sensors"]["left"]["classification"].update(
            classif_dict
        )
        input_fill_geoid["inputs"]["sensors"]["right"]["classification"].update(
            classif_dict
        )

        dense_dsm_pipeline_matches = default.DefaultPipeline(input_fill_geoid)

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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_paca_matches_filling.tif",
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_paca_matches_filling.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
        )

        # clean out dir for second run
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # Generate new conf with fill with geoid
        input_fill_border = copy.deepcopy(input_config_dense_dsm)
        classif_dict = {
            "filling": {
                "fill_with_geoid": None,
                "interpolate_from_borders": 8,
                "fill_with_endogenous_dem": None,
                "fill_with_exogenous_dem": None,
            }
        }
        input_fill_border["inputs"]["sensors"]["left"]["classification"].update(
            classif_dict
        )
        input_fill_border["inputs"]["sensors"]["right"][
            "classification"
        ].update(classif_dict)

        dense_dsm_pipeline_border_interpolation = default.DefaultPipeline(
            input_fill_border
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
            "1": {
                "sparse_matching": {
                    "decimation_factor": 80,
                },
                "dense_matching": {
                    "method": "census_sgm_default",
                    "use_cross_validation": True,
                    "save_intermediate_data": True,
                    "use_global_disp_range": True,
                },
                "dense_match_filling": {
                    "method": "zero_padding",
                    "save_intermediate_data": True,
                    "classification": ["1"],
                },
                "point_cloud_outlier_removal.1": {
                    "method": "small_components",
                },
                "point_cloud_outlier_removal.2": {
                    "method": "statistical",
                    "filtering_constant": 0,
                    "mean_factor": 1.0,
                    "std_dev_factor": 5.0,
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "color_end2end_gizeh_fill_with_zero.tif",
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
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir, "color_end2end_gizeh_fill_with_zero.tif"
                )
            ),
            rtol=0.0002,
            atol=1.0e-6,
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
