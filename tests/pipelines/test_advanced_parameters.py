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

from cars.pipelines.parameters import advanced_parameters, sensor_inputs

from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_advanced_parameters_full_config():
    """
    Test configuration check for advanced parameters
    """

    config = {
        "debug_with_roi": True,
        "ground_truth_dsm": {
            "dsm": "tests/data/input/phr_gizeh/img1.tif",
            "geoid": True,
        },
    }

    inputs_config = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
        }
    }
    inputs_config = sensor_inputs.sensors_check_inputs(
        inputs_config,
        config_dir=absolute_data_path("input/data_gizeh_crop/"),
    )

    advanced_parameters.check_advanced_parameters(inputs_config, config)


@pytest.mark.unit_tests
def test_advanced_parameters_minimal():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    inputs_config = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
        }
    }
    inputs_config = sensor_inputs.sensors_check_inputs(
        inputs_config,
        config_dir=absolute_data_path("input/data_gizeh_crop/"),
    )

    advanced_parameters.check_advanced_parameters(inputs_config, config)


@pytest.mark.unit_tests
def test_advanced_parameters_update_conf():
    """
    Test configuration check for advanced parameters
    """

    config = {"debug_with_roi": True}

    inputs_config = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
        }
    }
    inputs_config = sensor_inputs.sensors_check_inputs(
        inputs_config,
        config_dir=absolute_data_path("input/data_gizeh_crop/"),
    )

    # First config check without epipolar a priori
    _, updated_config, _, _, _, _, _, _, _, _ = (
        advanced_parameters.check_advanced_parameters(inputs_config, config)
    )

    # TODO: maybe move this inside update conf
    updated_config["ground_truth_dsm"] = {}

    # Cars level conf
    full_config = {"advanced": updated_config}

    # First config check without epipolar a priori
    _ = advanced_parameters.check_advanced_parameters(
        inputs_config, full_config["advanced"]
    )


@pytest.mark.unit_tests
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


@pytest.mark.unit_tests
def test_check_phasing_none():
    """
    Test phasing with None value.
    """

    assert advanced_parameters.check_phasing(None) is None


@pytest.mark.unit_tests
@pytest.mark.parametrize("epsg", [4326, "4326"])
def test_check_phasing_geographic_degree(epsg):
    """
    Test phasing with geographic CRS and degree unit.
    """

    phasing = {
        "point": [31.132, 29.978],
        "epsg": epsg,
        "unit": "degree",
    }

    checked_phasing = advanced_parameters.check_phasing(phasing)

    assert checked_phasing["point"] == phasing["point"]
    assert checked_phasing["epsg"] == 4326
    assert checked_phasing["unit"] == "degree"


@pytest.mark.unit_tests
@pytest.mark.parametrize("epsg", [4326, "4326"])
def test_check_phasing_geographic_arcsec(epsg):
    """
    Test phasing with geographic CRS and arcsec unit.
    """

    phasing = {
        "point": [0.0085, 0.0085],
        "epsg": epsg,
        "unit": "arcsec",
    }

    checked_phasing = advanced_parameters.check_phasing(phasing)

    assert checked_phasing["epsg"] == 4326
    assert checked_phasing["unit"] == "degree"
    assert checked_phasing["point"][0] == pytest.approx(0.0085 / 3600.0)
    assert checked_phasing["point"][1] == pytest.approx(0.0085 / 3600.0)


@pytest.mark.unit_tests
@pytest.mark.parametrize("epsg", [32636, "32636"])
def test_check_phasing_projected_meter(epsg):
    """
    Test phasing with projected CRS and meter unit.
    """

    phasing = {
        "point": [500000.0, 3300000.0],
        "epsg": epsg,
        "unit": "meter",
    }

    checked_phasing = advanced_parameters.check_phasing(phasing)

    assert checked_phasing["point"] == phasing["point"]
    assert checked_phasing["epsg"] == 32636
    assert checked_phasing["unit"] == "meter"


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "phasing, expected_message",
    [
        (
            {
                "point": [31.132, 29.978],
                "epsg": 4326,
                "unit": "meter",
            },
            "Phasing unit meter is incompatible with EPSG 4326.",
        ),
        (
            {
                "point": [500000.0, 3300000.0],
                "epsg": 32636,
                "unit": "degree",
            },
            "Phasing unit degree is incompatible with EPSG 32636.",
        ),
        (
            {
                "point": [500000.0, 3300000.0],
                "epsg": 32636,
                "unit": "arcsec",
            },
            "Phasing unit arcsec is incompatible with EPSG 32636.",
        ),
    ],
)
def test_check_phasing_incompatible_unit(phasing, expected_message):
    """
    Test phasing with units incompatible with the selected CRS.
    """

    with pytest.raises(RuntimeError, match=expected_message):
        advanced_parameters.check_phasing(phasing)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "phasing",
    [
        {
            "epsg": 4326,
            "unit": "degree",
        },
        {
            "point": [31.132, 29.978],
            "unit": "degree",
        },
        {
            "point": [31.132, 29.978],
            "epsg": 4326,
        },
        {
            "point": [31.132],
            "epsg": 4326,
            "unit": "degree",
        },
        {
            "point": ["31.132", 29.978],
            "epsg": 4326,
            "unit": "degree",
        },
        {
            "point": [31.132, 29.978],
            "epsg": 4326,
            "unit": "radian",
        },
    ],
)
def test_check_phasing_invalid_configuration(phasing):
    """
    Test invalid phasing configurations.
    """

    with pytest.raises(json_checker.core.exceptions.CheckerError):
        advanced_parameters.check_phasing(phasing)


@pytest.mark.unit_tests
def test_advanced_parameters_phasing_arcsec():
    """
    Test phasing arcsec conversion through advanced parameters check.
    """

    config = {
        "phasing": {
            "point": [0.0085, 0.0085],
            "epsg": 4326,
            "unit": "arcsec",
        }
    }

    inputs_config = {
        "sensors": {
            "one": {"image": "img1_crop.tif", "geomodel": "img1_crop.geom"},
            "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
        }
    }
    inputs_config = sensor_inputs.sensors_check_inputs(
        inputs_config,
        config_dir=absolute_data_path("input/data_gizeh_crop/"),
    )

    _, updated_config, _, _, _, _, _, _, _, _ = (
        advanced_parameters.check_advanced_parameters(
            inputs_config,
            config,
        )
    )

    phasing = updated_config["phasing"]

    assert phasing["epsg"] == 4326
    assert phasing["unit"] == "degree"
    assert phasing["point"][0] == pytest.approx(0.0085 / 3600.0)
    assert phasing["point"][1] == pytest.approx(0.0085 / 3600.0)


@pytest.mark.unit_tests
def test_check_phasing_compound_crs():
    """
    Test that compound CRS cannot be used for phasing.
    """

    phasing = {
        "point": [31.132, 29.978],
        "epsg": "4326+5773",
        "unit": "degree",
    }

    with pytest.raises(
        RuntimeError,
        match="A compound CRS cannot be used for phasing.",
    ):
        advanced_parameters.check_phasing(phasing)


@pytest.mark.unit_tests
def test_check_phasing_invalid_epsg():
    """
    Test phasing with an invalid EPSG code.
    """

    phasing = {
        "point": [31.132, 29.978],
        "epsg": "invalid",
        "unit": "degree",
    }

    with pytest.raises(RuntimeError, match="Invalid phasing EPSG invalid."):
        advanced_parameters.check_phasing(phasing)
