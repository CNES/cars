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
import os
import tempfile

import pytest

from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

from ...helpers import absolute_data_path, assert_same_images
from ...helpers import cars_copy2 as copy2
from ...helpers import temporary_dir


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
                        "image": absolute_data_path("input/phr_gizeh/img1.tif"),
                        "geomodel": absolute_data_path(
                            "input/phr_gizeh/img1.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path("input/phr_gizeh/img2.tif"),
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
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_low_res_dsm.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_low_res_dsm.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_low_res_dsm.tif",
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
                    "image_test_surface_modeling_low_res_dsm.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_ventoux_full():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/left_classif.tif"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
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
                    "image": True,
                    "performance_map": True,
                },
            },
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "ambiguity_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classif_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "cp_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "pm_test_surface_modeling_ventoux.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_ventoux.tif",
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
                    "image_test_surface_modeling_ventoux.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "ambiguity.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "ambiguity_test_surface_modeling_ventoux.tif",
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
                    "classif_test_surface_modeling_ventoux.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "contributing_pair.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "cp_test_surface_modeling_ventoux.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )

        assert_same_images(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "pm_test_surface_modeling_ventoux.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_ventoux_without_filter_incomplete_disparity_range():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
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
                    }
                },
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_ventoux_wo_fidr.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_ventoux_wo_fidr.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_ventoux_wo_fidr.tif",
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
                    "image_test_surface_modeling_ventoux_wo_fidr.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_ventoux_depth_maps_point_clouds():
    """
    End to end pipeline processing
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/left_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/right_image.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/right_image.geom"
                        ),
                    },
                },
                "initial_elevation": absolute_data_path(
                    "input/phr_ventoux/srtm/N44E005.hgt"
                ),
            },
            "output": {
                "directory": directory,
                "product_level": ["depth_map", "point_cloud", "dsm"],
            },
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "depth_map", "image1_image2", "Z.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dm_Z_test_surface_modeling_ventoux_dm_pc.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "depth_map", "image1_image2", "Z.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dm_Z_test_surface_modeling_ventoux_dm_pc.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert os.path.exists(
            os.path.join(out_dir, "point_cloud", "image1_image2", "2_1.laz")
        )


@pytest.mark.end2end_tests
def test_gizeh_dem_min_max_median():
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
                    },
                    "image2": {
                        "image": absolute_data_path("input/phr_gizeh/img2.tif"),
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
            "surface_modeling": {
                "applications": {
                    "dem_generation": {"save_intermediate_data": True}
                }
            },
            "output": {"directory": directory, "product_level": []},
        }
        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_min_test_surface_modeling_dem_min_max_median.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_max_test_surface_modeling_dem_min_max_max.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_median_test_surface_modeling_dem_min_max_median.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_min.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_min_test_surface_modeling_dem_min_max_median.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_max_test_surface_modeling_dem_min_max_max.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_median_test_surface_modeling_dem_min_max_median.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
