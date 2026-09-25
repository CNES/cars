#!/usr/bin/env python
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
Test module for cars/core/preprocessing.py
"""

# Standard imports

import numpy as np

# Third party imports
import pytest

# CARS import
from cars.core import preprocessing
from cars.pipelines import pipeline_constants as pipeline_cst
from cars.pipelines.pipeline_constants import ADVANCED, APPLICATIONS


@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    vals = [
        (1.442299, 43.600764, 32631),
        (None, 43.600764, 32632),
        (1.442299, None, 32632),
        (None, None, 32632),
        (np.nan, 43.600764, 32632),
        (1.442299, np.nan, 32632),
        (np.nan, np.nan, 32632),
    ]

    for lon, lat, gt in vals:
        epsg = preprocessing.get_utm_zone_as_epsg_code(lon, lat)
        assert epsg == gt


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "corresponding_conf_name,expected_resolution,expected_edges,expected_depth",
    [
        ("census_sgm_urban", 1, True, True),
        ("census_sgm_default", 4, False, False),
        (None, 4, False, False),
    ],
)
def test_get_land_cover_auto_configuration(
    corresponding_conf_name,
    expected_resolution,
    expected_edges,
    expected_depth,
):
    """Test the auto-configuration derived from world classification."""

    auto_conf = preprocessing.get_land_cover_auto_configuration(
        corresponding_conf_name
    )

    assert auto_conf["use_monocular"] is True
    assert auto_conf["monocular_resolution"] == expected_resolution
    assert auto_conf["dense_matching_edges_3sgm"] is expected_edges
    assert auto_conf["point_cloud_refinement_activated"] is True
    assert auto_conf["depth_to_z_fusion_activated"] is expected_depth


@pytest.mark.unit_tests
def test_apply_land_cover_auto_configuration_preserves_user_values():
    """
    Test that auto land-cover config fills
    defaults without overriding user conf.
    """

    conf = {
        pipeline_cst.SURFACE_MODELING: {
            APPLICATIONS: {
                "1": {
                    "dense_matching": {"edges_3sgm": True},
                    "depth_to_z_fusion": {"activated": True},
                }
            }
        },
        pipeline_cst.MONOCULAR: {ADVANCED: {"resolution": 2}},
    }

    preprocessing.apply_land_cover_auto_configuration(
        conf,
        "census_sgm_default",
        use_monocular=True,
    )

    apps_conf = conf[pipeline_cst.SURFACE_MODELING][APPLICATIONS]["1"]

    assert apps_conf["dense_matching"]["edges_3sgm"] is True
    assert apps_conf["point_cloud_refinement"]["activated"] is True
    assert apps_conf["depth_to_z_fusion"]["activated"] is True
    assert conf[pipeline_cst.MONOCULAR][ADVANCED]["resolution"] == 2


@pytest.mark.unit_tests
def test_apply_land_cover_auto_configuration_sets_other_defaults():
    """Test that non-urban land cover gets the expected default policy."""

    conf = {
        pipeline_cst.SURFACE_MODELING: {APPLICATIONS: {}},
        pipeline_cst.MONOCULAR: {ADVANCED: {}},
    }

    preprocessing.apply_land_cover_auto_configuration(
        conf,
        "census_sgm_default",
        use_monocular=True,
    )

    apps_conf = conf[pipeline_cst.SURFACE_MODELING][APPLICATIONS]["1"]

    assert apps_conf["dense_matching"]["edges_3sgm"] is False
    assert apps_conf["point_cloud_refinement"]["activated"] is True
    assert apps_conf["depth_to_z_fusion"]["activated"] is False
    assert conf[pipeline_cst.MONOCULAR][ADVANCED]["resolution"] == 4
