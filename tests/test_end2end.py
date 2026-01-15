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

# CARS imports
from cars.pipelines.default import default_pipeline as default

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
)
from .helpers import cars_copy2 as copy2
from .helpers import (
    temporary_dir,
)

NB_WORKERS = 2


@pytest.mark.end2end_tests
def test_end2end_gizeh_meta_pipeline():
    """
    End to end processing with color
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
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_gizeh_meta_pipeline.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_gizeh_meta_pipeline.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_gizeh_meta_pipeline.tif",
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
                    "image_test_gizeh_meta_pipeline.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_ventoux_meta_pipeline():
    """
    End to end processing with color
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
                    "all": {
                        "dense_matching": {
                            "filter_incomplete_disparity_range": False,
                        }
                    }
                }
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_ventoux_meta_pipeline.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "image.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "image_test_ventoux_meta_pipeline.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_ventoux_meta_pipeline.tif",
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
                    "image_test_ventoux_meta_pipeline.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )


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
                "filling": {"fill_with_geoid": 3},
            },
            "surface_modeling": {
                "applications": {
                    "all": {
                        "dense_matching": {
                            "filter_incomplete_disparity_range": False,
                        }
                    }
                }
            },
            "orchestrator": {
                "mode": "multiprocessing",
                "nb_workers": 4,
                "max_ram_per_worker": 1000,
            },
            "output": {"directory": directory, "auxiliary": {"filling": True}},
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_ventoux_with_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "classification.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "classif_test_ventoux_with_filling.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "filling_test_ventoux_with_filling.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_ventoux_with_filling.tif",
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
                    "classif_test_ventoux_with_filling.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "filling.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "filling_test_ventoux_with_filling.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
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
            "merging": {
                "applications": {"dsm_merging": {"method": "weighted_fusion"}},
                "advanced": {"save_intermediate_data": True},
            },
            "output": {"directory": directory},
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(intermediate_output_dir, "dsm_test_merging.tif")
            ),
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(ref_output_dir, "dsm_test_merging.tif")
            ),
            atol=0.0001,
            rtol=1e-6,
        )


@pytest.mark.end2end_tests
def test_end2end_gizeh_use_endogenous_dem():
    """
    End to end processing with color
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
                "initial_elevation": absolute_data_path(
                    "input/phr_gizeh/srtm_dir/N29E031_KHEOPS.tif"
                ),
            },
            "subsampling": {"advanced": {"resolutions": [4, 2, 1]}},
            "surface_modeling": {
                "advanced": {
                    "2": {"use_endogenous_dem": True},
                    "1": {"use_endogenous_dem": True},
                },
                "applications": {
                    "2": {"dem_generation": {"save_intermediate_data": True}},
                    "1": {"dem_generation": {"save_intermediate_data": True}},
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
                    "performance_map": True,
                },
            },
        }
        out_dir = conf["output"]["directory"]
        meta_pipeline = default.DefaultPipeline(conf)
        meta_pipeline.run()
        intermediate_output_dir = "intermediate_data"
        ref_output_dir = "ref_output"
        copy2(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_median_res1_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dump_dir", "dem_generation", "dem_max.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_max_res1_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "intermediate_data",
                "surface_modeling",
                "res2",
                "dump_dir",
                "dem_generation",
                "dem_median.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_median_res2_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        copy2(
            os.path.join(
                out_dir,
                "intermediate_data",
                "surface_modeling",
                "res2",
                "dump_dir",
                "dem_generation",
                "dem_max.tif",
            ),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dem_max_res2_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "dsm_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        copy2(
            os.path.join(out_dir, "dsm", "performance_map.tif"),
            absolute_data_path(
                os.path.join(
                    intermediate_output_dir,
                    "performance_map_test_gizeh_use_endogenous_dem.tif",
                )
            ),
        )
        assert_same_images(
            os.path.join(
                out_dir, "dump_dir", "dem_generation", "dem_median.tif"
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_median_res1_test_gizeh_use_endogenous_dem.tif",
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
                    "dem_max_res1_test_gizeh_use_endogenous_dem.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "surface_modeling",
                "res2",
                "dump_dir",
                "dem_generation",
                "dem_median.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_median_res2_test_gizeh_use_endogenous_dem.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(
                out_dir,
                "intermediate_data",
                "surface_modeling",
                "res2",
                "dump_dir",
                "dem_generation",
                "dem_max.tif",
            ),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dem_max_res2_test_gizeh_use_endogenous_dem.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm", "dsm.tif"),
            absolute_data_path(
                os.path.join(
                    ref_output_dir,
                    "dsm_test_gizeh_use_endogenous_dem.tif",
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
                    "performance_map_test_gizeh_use_endogenous_dem.tif",
                )
            ),
            atol=0.0001,
            rtol=1e-6,
        )
