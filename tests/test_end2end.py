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

import os
import tempfile

# import pytest
import pytest
import rasterio as rio
import yaml

# CARS imports
from cars.pipelines.default import default_pipeline as default

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images_list,
    copy_to_intermediate_data,
    temporary_dir,
)

NB_WORKERS = 2

CARS_GITHUB_ACTIONS = (
    os.getenv("CARS_GITHUB_ACTIONS", "false").lower() == "true"
)
DEFAULT_TOL_GITHUB = 0.1
NB_OUTLIERS_ALLOWED_GITHUB = 10


@pytest.mark.end2end_tests
def test_end2end_ventoux_with_filling():
    """
    End to end processing with color
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {
            "input": {
                "sensors": {
                    "image1": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/crop/left_image_crop.tif"
                        ),
                        "geomodel": absolute_data_path(
                            "input/phr_ventoux/left_image.geom"
                        ),
                        "classification": absolute_data_path(
                            "input/phr_ventoux/crop/left_classif_crop.tif"
                        ),
                    },
                    "image2": {
                        "image": absolute_data_path(
                            "input/phr_ventoux/crop/right_image_crop.tif"
                        ),
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
                "filling": {"fill_with_geoid": 3},
                "scaling_coeff": 1.0,
            },
            "subsampling": {
                "advanced": {
                    "resolutions": [4, 2, 1],
                    "min_image_size": 50,
                }
            },
            "tie_points": {
                "applications": {
                    "all": {
                        "sparse_matching": {
                            "epipolar_error_maximum_bias": 50.0,
                            "epipolar_error_upper_bound": 10.0,
                        }
                    },
                    "2": {
                        "sparse_matching": {
                            "method": "pandora2d",
                            "step": [10, 10],
                            "elevation_delta_lower_bound": -1000,
                            "elevation_delta_upper_bound": 3000,
                        }
                    },
                    "1": {
                        "sparse_matching": {
                            "method": "pandora2d",
                            "step": [10, 10],
                            "elevation_delta_lower_bound": -1000,
                            "elevation_delta_upper_bound": 3000,
                        }
                    },
                }
            },
            "surface_modeling": {
                "advanced": {
                    "all": {
                        "save_intermediate_data": True,
                    },
                    "2": {
                        "use_endogenous_dem": True,
                    },
                    "1": {
                        "use_endogenous_dem": True,
                    },
                },
                "applications": {
                    "all": {
                        "dense_matching": {
                            "filter_incomplete_disparity_range": False,
                            "use_cross_validation": "accurate",
                        },
                        "point_cloud_rasterization": {
                            "fill_nodata": False,
                        },
                    },
                    "2": {
                        "dem_generation": {
                            "save_intermediate_data": True,
                        },
                    },
                    "1": {
                        "dem_generation": {
                            "save_intermediate_data": True,
                        },
                    },
                },
            },
            "pipeline": {
                "subsampling": True,
                "surface_modeling": True,
                "tie_points": True,
                "filling": True,
                "merging": False,
                "formatting": True,
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {
                "directory": directory,
                "auxiliary": {
                    "filling": True,
                    "performance_map": True,
                },
                "product_level": ["dtm"],
                "epsg": 4326,
                "resolution": [0.0000127, 0.0000045],
            },
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data/surface_modeling"
        ref_output_dir = "ref_output"

        files_to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "dtm.tif"),
                "dtm_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "performance_map.tif"),
                "performance_map_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "filling.tif"),
                "filling_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "classification.tif"),
                "classif_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    f"{intermediate_output_dir}/res1/dump_dir/dem_generation",
                    "dem_median.tif",
                ),
                "dem_median_res1_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    f"{intermediate_output_dir}/res1/dump_dir/dem_generation",
                    "dem_max.tif",
                ),
                "dem_max_res1_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    f"{intermediate_output_dir}/res2/dump_dir/dem_generation",
                    "dem_median.tif",
                ),
                "dem_median_res2_test_ventoux_with_filling.tif",
            ),
            (
                os.path.join(
                    out_dir,
                    f"{intermediate_output_dir}/res2/dump_dir/dem_generation",
                    "dem_max.tif",
                ),
                "dem_max_res2_test_ventoux_with_filling.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in files_to_check],
            [file[1] for file in files_to_check],
        )
        assert_same_images_list(
            [file[0] for file in files_to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in files_to_check
            ],
            atol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )

        with rio.open(os.path.join(out_dir, "dsm", "filling.tif")) as src:
            epsg = src.crs.to_epsg()
        assert epsg == 9707  # vertical crs

        with rio.open(os.path.join(out_dir, "dsm", "dsm.tif")) as dsm:
            assert dsm.res == (0.0000127, 0.0000045)

        # Check metadata file
        metadata_path = os.path.join(out_dir, "metadata.yaml")
        assert os.path.isfile(metadata_path)
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata_dict = yaml.safe_load(metadata_file)
            assert "surface_modeling" in metadata_dict
            assert "filling" in metadata_dict
            assert "formatting" in metadata_dict
            assert "subsampling" in metadata_dict
            assert (
                "before_correction_epi_error_mean"
                in metadata_dict["surface_modeling"]["1"]["applications"][
                    "match_filtering"
                ]["image1_image2"]
            )
            assert (
                "disp_to_alt_ratio"
                in metadata_dict["surface_modeling"]["1"]["applications"][
                    "grid_generation"
                ]["image1_image2"]
            )


@pytest.mark.end2end_tests
def test_end2end_gizeh_merging():
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
            "pipeline": {
                "subsampling": False,
                "surface_modeling": False,
                "tie_points": False,
                "filling": False,
                "merging": True,
                "formatting": True,
            },
            "merging": {
                "applications": {"dsm_merging": {"method": "weighted_fusion"}},
                "advanced": {"save_intermediate_data": True},
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        ref_output_dir = "ref_output"

        files_to_check = [
            (os.path.join(out_dir, "dsm", "dsm.tif"), "dsm_test_merging.tif"),
        ]

        copy_to_intermediate_data(
            [file[0] for file in files_to_check],
            [file[1] for file in files_to_check],
        )
        assert_same_images_list(
            [file[0] for file in files_to_check],
            [
                absolute_data_path(os.path.join(ref_output_dir, file[1]))
                for file in files_to_check
            ],
            atol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )


@pytest.mark.end2end_tests
def test_run_with_used_conf():
    """
    Default pipeline initialized with generated used conf
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
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "monocular": {
                "advanced": {
                    "activated": False,
                }
            },
            "output": {
                "directory": directory,
                "epsg": 32636,
                "resolution": 1.0,
            },
        }
        out_dir = conf["output"]["directory"]
        _ = default.DefaultPipeline(conf)
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        with open(
            os.path.join(out_dir, "global_used_conf.yaml"), encoding="utf-8"
        ) as used_conf_file:
            used_conf_1 = yaml.safe_load(used_conf_file)
        meta_pipeline_2 = default.DefaultPipeline(used_conf_1)
        meta_pipeline_2.run()
        with open(
            os.path.join(out_dir, "global_used_conf.yaml"), encoding="utf-8"
        ) as used_conf_file:
            used_conf_2 = yaml.safe_load(used_conf_file)
        assert used_conf_1 == used_conf_2

        files_to_check = [
            (
                os.path.join(out_dir, "dsm", "dsm.tif"),
                "dsm_test_gizeh_run_with_used_conf.tif",
            ),
            (
                os.path.join(out_dir, "dsm", "image.tif"),
                "image_test_gizeh_run_with_used_conf.tif",
            ),
        ]

        copy_to_intermediate_data(
            [file[0] for file in files_to_check],
            [file[1] for file in files_to_check],
        )
        assert_same_images_list(
            [file[0] for file in files_to_check],
            [
                absolute_data_path(os.path.join("ref_output", file[1]))
                for file in files_to_check
            ],
            atol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 0.0001,
            rtol=DEFAULT_TOL_GITHUB if CARS_GITHUB_ACTIONS else 1e-6,
            nb_outliers_allowed=(
                NB_OUTLIERS_ALLOWED_GITHUB if CARS_GITHUB_ACTIONS else 0
            ),
        )

        with rio.open(os.path.join(out_dir, "dsm", "dsm.tif")) as dsm:
            assert dsm.res == (1.0, 1.0)

        with rio.open(os.path.join(out_dir, "dsm", "dsm.tif")) as src:
            crs = src.crs

        # compound CRS: EPSG:32636 (from config) + geoid (EGM96) (elevation CRS)
        assert "WGS 84 / UTM zone 36N + EGM96 height" in crs.to_wkt()
