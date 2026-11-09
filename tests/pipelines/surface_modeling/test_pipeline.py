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
Test pipeline surface modeling
"""
import argparse
import copy
import json
import os
import tempfile

import pytest

from cars.cars import main_cli
from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

from ...helpers import (
    absolute_data_path,
    assert_same_images_list,
    copy_to_intermediate_data,
    temporary_dir,
)

DEFAULT_TOL = 0.1
CARS_GITHUB_ACTIONS = (
    os.getenv("CARS_GITHUB_ACTIONS", "false").lower() == "true"
)
NB_OUTLIERS_ALLOWED_GITHUB = 10


@pytest.mark.end2end_tests
def test_gizeh_with_low_res_dsm():
    """
    End to end pipeline processing
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/crop/img1_crop.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/crop/img2_crop.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
                "low_res_dsm": absolute_data_path(
                    "input/phr_gizeh/low_res_dsm.tif"
                ),
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }

        out_dir_cli = os.path.join(directory, "out_cli")
        out_dir_api = conf["output"]["directory"]

        cli_conf = copy.deepcopy(conf)
        cli_conf["pipeline"] = "surface_modeling"
        cli_conf["output"]["directory"] = out_dir_cli

        cli_conf_path = os.path.join(directory, "surface_modeling_conf.json")
        with open(cli_conf_path, "w", encoding="utf8") as f:
            json.dump(cli_conf, f)

        ref_output_dir = absolute_data_path("ref_output")

        # api run
        api_pipeline = SurfaceModelingPipeline(conf)
        api_pipeline.run()

        # cli run
        main_cli(argparse.Namespace(conf=cli_conf_path))

        # products to check
        to_check_cli = [
            (
                os.path.join(out_dir_cli, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_low_res_dsm.tif",
            ),
            (
                os.path.join(out_dir_cli, "dsm", "image.tif"),
                "image_test_surface_modeling_low_res_dsm.tif",
            ),
        ]

        to_check_api = [
            (
                os.path.join(out_dir_api, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_low_res_dsm.tif",
            ),
            (
                os.path.join(out_dir_api, "dsm", "image.tif"),
                "image_test_surface_modeling_low_res_dsm.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check_cli],
            [file[1] for file in to_check_cli],
        )

        # compare cli and expected output
        assert_same_images_list(
            [file[0] for file in to_check_cli],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check_cli
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )

        # compare api and expected output
        assert_same_images_list(
            [file[0] for file in to_check_api],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check_api
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )


@pytest.mark.end2end_tests
def test_ventoux_full_fail_with_used_band_b1():
    """
    End to end pipeline that failed when used band is equal to b1
    and this band is not available for the right image
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
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
                                },
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
                "auxiliary": {
                    "ambiguity": True,
                    "classification": True,
                    "contributing_pair": True,
                    "image": ["b1", "b2", "b3"],
                    "filling": True,
                    "performance_map": True,
                },
            },
            "surface_modeling": {
                "applications": {"dense_matching": {"used_band": "b1"}}
            },
        }
        with pytest.raises(
            RuntimeError,
            match="Band 'b1' is not available in one of the two datasets.",
        ):
            _ = SurfaceModelingPipeline(conf)


@pytest.mark.end2end_tests
def test_gizeh_res4():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_res4.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_gizeh_res4.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_res16():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res16.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res16.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_res16.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_gizeh_res16.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_res4_with_roi():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        roi_geo_json = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [
                            [
                                [320000, 3317850],
                                [320000, 3318000],
                                [320200, 3318000],
                                [320200, 3317850],
                                [320000, 3317850],
                            ]
                        ],
                        "type": "Polygon",
                    },
                }
            ],
            "crs": {"type": "name", "properties": {"name": "EPSG:32636"}},
            "ignore_roi_during_a_priori": True,
        }

        conf["input"]["roi"] = roi_geo_json
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_res4_with_roi.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_gizeh_res4_with_roi.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_res4_with_a_priori_roi():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        roi_geo_json = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [
                            [
                                [320000, 3317850],
                                [320000, 3318000],
                                [320200, 3318000],
                                [320200, 3317850],
                                [320000, 3317850],
                            ]
                        ],
                        "type": "Polygon",
                    },
                }
            ],
            "crs": {"type": "name", "properties": {"name": "EPSG:32636"}},
        }

        conf["input"]["roi"] = roi_geo_json
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_res4_with_roi.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_gizeh_res4_with_roi.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_res4_with_gt_reprojection():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
                "ground_truth_dsm": {
                    "dsm": absolute_data_path(
                        "input/phr_gizeh/srtm_dir/N29E031_KHEOPS.tif"
                    )
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "surface_modeling": {
                "applications": {
                    "ground_truth_reprojection": {
                        "method": "direct_loc",
                        "target": "all",
                        "save_intermediate_data": True,
                    },
                },
            },
            "output": {"product_level": [], "directory": directory},
        }

        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "ground_truth_reprojection",
                    "image1_image2",
                    "sensor_dsm_ground_truth_left.tif",
                ),
                "sensor_dsm_gt_left_test_gt_reprojection.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "ground_truth_reprojection",
                    "image1_image2",
                    "sensor_dsm_ground_truth_right.tif",
                ),
                "sensor_dsm_gt_right_test_gt_reprojection.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "ground_truth_reprojection",
                    "image1_image2",
                    "epipolar_disp_ground_truth_left.tif",
                ),
                "epipolar_disp_gt_left_test_gt_reprojection.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "ground_truth_reprojection",
                    "image1_image2",
                    "epipolar_disp_ground_truth_right.tif",
                ),
                "epipolar_disp_gt_right_test_gt_reprojection.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_res4_without_tie_points():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img1_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_gizeh/img2_res4.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                    },
                },
            },
            "tie_points": None,
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
        )


@pytest.mark.end2end_tests
def test_ventoux_filling():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        data_path = "input/phr_ventoux/crop"
        conf = {
            "input": {
                "loaders": {"image": "pivot"},
                "sensors": {
                    "image1": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": absolute_data_path(
                                        f"{data_path}/left_image_crop.tif"
                                    ),
                                    "band": 0,
                                },
                                "b1": {
                                    "path": absolute_data_path(
                                        f"{data_path}/color_image_crop.tif"
                                    ),
                                    "band": 0,
                                },
                                "b2": {
                                    "path": absolute_data_path(
                                        f"{data_path}/color_image_crop.tif"
                                    ),
                                    "band": 1,
                                },
                                "b3": {
                                    "path": absolute_data_path(
                                        f"{data_path}/color_image_crop.tif"
                                    ),
                                    "band": 2,
                                },
                                "b4": {
                                    "path": absolute_data_path(
                                        f"{data_path}/color_image_crop.tif"
                                    ),
                                    "band": 3,
                                },
                            },
                        },
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/crop/left_classif_crop.tif"
                        ),
                    },
                    "image2": {
                        "image": {
                            "bands": {
                                "b0": {
                                    "path": absolute_data_path(
                                        f"{data_path}/right_image_crop.tif"
                                    ),
                                    "band": 0,
                                },
                                "b1": {
                                    "path": absolute_data_path(
                                        f"{data_path}/right_image_crop.tif"
                                    ),
                                    "band": 0,
                                },
                            },
                        },
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/crop/right_classif_crop.tif"
                        ),
                    },
                },
                "initial_elevation": absolute_data_path(
                    "input/phr_ventoux/srtm/N44E005.hgt"
                ),
            },
            "surface_modeling": {
                "applications": {
                    "dense_matching": {
                        "filter_incomplete_disparity_range": False
                    },
                    "dense_match_filling": {
                        "method": "zero_padding",
                        "fill_classification": ["3"],
                    },
                    "point_cloud_outlier_removal": None,
                },
                "advanced": {
                    "phasing": {
                        "point": [0.03, 0.03],
                        "unit": "arcsec",
                        "epsg": 4326,
                    }
                },
            },
            "output": {
                "directory": directory,
                "auxiliary": {
                    "ambiguity": True,
                    "classification": True,
                    "filling": True,
                    "contributing_pair": True,
                    "image": ["b1", "b2", "b3"],
                    "performance_map": True,
                },
                "product_level": ["point_cloud", "dsm"],
                "product_format": {
                    "point_cloud": ["laz", "tif"],
                },
                "epsg": 4326,
            },
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "ambiguity.tif"),
                "ambiguity_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "classification.tif"),
                "classification_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "filling.tif"),
                "filling_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "contributing_pair.tif"),
                "contributing_pair_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "performance_map.tif"),
                "performance_map_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "cropped_disp_range.tif"),
                "cropped_disp_range_test_surface_modeling_ventoux_filling.tif",
            ),
            (
                os.path.join(
                    out_dir, "point_cloud", "image1_image2", "tif", "Z.tif"
                ),
                "point_cloud_Z_test_surface_modeling_ventoux_filling.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )

        # assert the laz version of the point cloud is also generated
        assert os.path.exists(
            os.path.join(
                out_dir, "point_cloud", "image1_image2", "laz", "0_0.laz"
            )
        )


@pytest.mark.end2end_tests
def test_gizeh_with_monocular_data():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path("input/phr_gizeh/img1.tif"),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                        "edges": {
                            "edges_mask": absolute_data_path(
                                "input/phr_gizeh/monocular/edges.tif"
                            ),
                            "depth_map": absolute_data_path(
                                "input/phr_gizeh/monocular/depth.tif"
                            ),
                            "normals": absolute_data_path(
                                "input/phr_gizeh/monocular/normals.tif"
                            ),
                            "tile_id": absolute_data_path(
                                "input/phr_gizeh/monocular/tile_id.tif"
                            ),
                        },
                        "mask": absolute_data_path(
                            "input/phr_gizeh/new_mask1.tif"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path("input/phr_gizeh/img2.tif"),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img2.geom"
                        ),
                        "mask": absolute_data_path(
                            "input/phr_gizeh/new_mask2.tif"
                        ),
                    },
                },
                "low_res_dsm": absolute_data_path(
                    "input/phr_gizeh/low_res_dsm.tif"
                ),
            },
            "surface_modeling": {
                "applications": {
                    "dem_generation": {"save_intermediate_data": True},
                    "triangulation": {"save_intermediate_data": True},
                    "depth_to_z_fusion": {"activated": False},
                },
                "advanced": {
                    "use_sensor_disp": True,
                },
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {
                "directory": directory,
                "auxiliary": {
                    "image": True,
                },
            },
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        ref_output_dir = "ref_output"

        to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_surface_modeling_gizeh_monocular.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "common_sensor",
                    "tif",
                    "Z.tif",
                ),
                "Z_test_surface_modeling_gizeh_monocular.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    "dump_dir",
                    "triangulation",
                    "common_sensor",
                    "tif",
                    "intersections_residues.tif",
                ),
                "intersections_residues_test_"
                "surface_modeling_gizeh_monocular.tif",
            ),
            (
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_median.tif"
                ),
                "dem_median_test_surface_modeling_gizeh_monocular.tif",
            ),
            (
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_min.tif"
                ),
                "dem_min_test_surface_modeling_gizeh_monocular.tif",
            ),
            (
                os.path.join(
                    out_dir, "dump_dir", "dem_generation", "dem_max.tif"
                ),
                "dem_max_test_surface_modeling_gizeh_monocular.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in to_check],
            [file[1] for file in to_check],
        )

        assert_same_images_list(
            [file[0] for file in to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in to_check
            ],
            atol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )
