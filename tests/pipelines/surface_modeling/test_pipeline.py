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
                "auxiliary": {
                    "ambiguity": True,
                    "classification": True,
                    "contributing_pair": True,
                    "image": ["b1", "b2", "b3"],
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
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_gizeh_res4.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_gizeh_res4.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_gizeh_res4.tif",
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
                    "image_test_surface_modeling_gizeh_res4.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
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
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_gizeh_res16.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_gizeh_res16.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_gizeh_res16.tif",
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
                    "image_test_surface_modeling_gizeh_res16.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
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
        }

        conf["input"]["roi"] = roi_geo_json
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
                    "dsm_test_surface_modeling_gizeh_res4_with_roi.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_gizeh_res4_with_roi.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_gizeh_res4_with_roi.tif",
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
                    "image_test_surface_modeling_gizeh_res4_with_roi.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
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
                "advanced": {
                    "ground_truth_dsm": {
                        "dsm": absolute_data_path(
                            "input/phr_gizeh/srtm_dir/N29E031_KHEOPS.tif"
                        )
                    }
                },
            },
            "output": {"product_level": [], "directory": directory},
        }

        out_dir = conf["output"]["directory"]
        surface_modeling_pipeline = SurfaceModelingPipeline(conf)
        surface_modeling_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "sensor_dsm_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "sensor_dsm_gt_left_test_gt_reprojection.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "sensor_dsm_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "sensor_dsm_gt_right_test_gt_reprojection.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "epipolar_disp_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epipolar_disp_gt_left_test_gt_reprojection.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "epipolar_disp_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "epipolar_disp_gt_right_test_gt_reprojection.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "sensor_dsm_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "sensor_dsm_gt_left_test_gt_reprojection.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "sensor_dsm_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "sensor_dsm_gt_right_test_gt_reprojection.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "epipolar_disp_ground_truth_left.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epipolar_disp_gt_left_test_gt_reprojection.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "dump_dir",
                "ground_truth_reprojection",
                "image1_image2",
                "epipolar_disp_ground_truth_right.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "epipolar_disp_gt_right_test_gt_reprojection.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_gizeh_with_mask():
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
                    "dsm_test_surface_modeling_with_mask.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_with_mask.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_with_mask.tif",
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
                    "image_test_surface_modeling_with_mask.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
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
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
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
                    "image_test_surface_modeling_gizeh_res4_wo_tie_points.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
