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
Test module for cars/parameters/advanced_parameters.py
"""
import json_checker
import pytest
import rasterio as rio

from cars.pipelines.parameters import advanced_parameters


@pytest.mark.unit_tests
def test_advanced_parameters_full_config():
    """
    Test configuration check for advanced parameters
    """

    config = {
        "debug_with_roi": True,
        "use_epipolar_a_priori": True,
        "epipolar_a_priori": {
            "left_right": {
                "grid_correction": [
                    4.288258198116454,
                    -5.126994797702842e-05,
                    -0.0001868504592756687,
                    0.7633338209640762,
                    0.00017551039816977979,
                    7.416961236000771e-05,
                ],
                "disparity_range": [-26.157764557028997, 26.277429517242638],
            }
        },
        "terrain_a_priori": {
            "dem_median": "dem_median.tif",
            "dem_min": "dem_min.tif",
            "dem_max": "dem_max.tif",
        },
        "ground_truth_dsm": {
            "dsm": "tests/data/input/phr_gizeh/img1.tif",
            "geoid": True,
        },
    }

    advanced_parameters.check_advanced_parameters(config)


@pytest.mark.unit_tests
def test_advanced_parameters_minimal():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    advanced_parameters.check_advanced_parameters(config)


@pytest.mark.unit_tests
def test_advanced_parameters_update_conf():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    # First config check without epipolar a priori
    updated_config = advanced_parameters.check_advanced_parameters(
        config, check_epipolar_a_priori=False
    )

    # TODO: maybe move this inside update conf
    updated_config["epipolar_a_priori"] = {}
    updated_config["terrain_a_priori"] = {}
    updated_config["use_epipolar_a_priori"] = True
    updated_config["ground_truth_dsm"] = {}

    # Cars level conf
    full_config = {"advanced": updated_config}

    # Update config
    advanced_parameters.update_conf(
        full_config,
        grid_correction_coef=[1, 2, 3, 4, 5, 6],
        dmin=-10,
        dmax=10,
        pair_key="pair_key",
        dem_median="dem_median.tif",
        dem_min="dem_min.tif",
        dem_max="dem_max.tif",
    )

    # First config check without epipolar a priori
    _ = advanced_parameters.check_advanced_parameters(
        full_config["advanced"], check_epipolar_a_priori=True
    )


def test_check_ground_truth_dsm_data():
    """
    Test check_ground_truth_dsm_data function
    """

    ground_truth_dsm_conf = {"dsm": "tests/data/input/phr_gizeh/img1.tif"}

    # Should pass
    advanced_parameters.check_ground_truth_dsm_data(ground_truth_dsm_conf)

    # Should raise an error because of wrong dsm file is used
    ground_truth_dsm_conf["dsm"] = "wrong_file.tif"
    with pytest.raises(rio.errors.RasterioIOError):
        advanced_parameters.check_ground_truth_dsm_data(ground_truth_dsm_conf)

    # Should raise an error because of wrong dsm type is used
    ground_truth_dsm_conf["dsm"] = True
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        advanced_parameters.check_ground_truth_dsm_data(ground_truth_dsm_conf)
