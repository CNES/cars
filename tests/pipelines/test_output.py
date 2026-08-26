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
Test pipeline output
"""

import os
import tempfile

import pytest
from json_checker.core.exceptions import DictCheckerError
from pyproj.exceptions import CRSError

from cars.pipelines.parameters import output_parameters, sensor_inputs

from ..helpers import absolute_data_path, temporary_dir


@pytest.mark.unit_tests
def test_output_full():
    """
    Test output
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "product_level": "dsm",
            "auxiliary": {
                "performance_map": False,
                "image": True,
                "classification": False,
                "contributing_pair": False,
            },
            "epsg": 4326,
            "resolution": 0.5,
            "geoid": "path/to/geoid",
            "save_by_pair": False,
        }

        inputs = {
            "sensors": {
                "one": {
                    "image": {
                        "loader": "pivot_image",
                        "main_file": "img1_crop.tif",
                        "bands": {
                            "b0": {"path": "img1_crop.tif", "band": 0},
                            "b1": {"path": "color1.tif", "band": 1},
                            "b2": {"path": "color1.tif", "band": 2},
                            "b3": {"path": "color1.tif", "band": 2},
                        },
                    },
                    "geomodel": "img1_crop.geom",
                },
                "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
            }
        }

        print(f"config {config}")
        output_parameters.check_output_parameters(inputs, config, 1)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "case",
    [
        {"epsg": None, "expected": "valid"},
        {"epsg": 4326, "expected": "valid"},
        {"epsg": "4326", "expected": "valid"},
        {"epsg": "4326+5773", "expected": "valid"},
        {"epsg": "3857", "expected": "valid"},
        {"epsg": 999999, "expected": "invalid"},
        {"epsg": 0, "expected": "invalid"},
        {"epsg": "not_a_code", "expected": "invalid"},
    ],
)
def test_output_epsg(case):
    """
    Test output_parameters.check_output_parameters with different EPSG inputs
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "product_level": "dsm",
            "auxiliary": {
                "performance_map": False,
                "image": True,
                "classification": False,
                "contributing_pair": False,
            },
            "epsg": case["epsg"],
            "resolution": 0.5,
            "geoid": "path/to/geoid",
            "save_by_pair": False,
        }

        inputs = {
            "sensors": {
                "one": {
                    "image": {
                        "loader": "pivot_image",
                        "main_file": "img1_crop.tif",
                        "bands": {
                            "b0": {"path": "img1_crop.tif", "band": 0},
                            "b1": {"path": "color1.tif", "band": 1},
                            "b2": {"path": "color1.tif", "band": 2},
                            "b3": {"path": "color1.tif", "band": 2},
                        },
                    },
                    "geomodel": "img1_crop.geom",
                },
                "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
            }
        }

        if case["expected"] == "valid":
            # Should succeed without raising
            output_parameters.check_output_parameters(inputs, config, 1)
        else:
            with pytest.raises((DictCheckerError, CRSError)):
                # Expecting some sort of failure
                output_parameters.check_output_parameters(inputs, config, 1)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "resolution, epsg, expected",
    [
        # Resolution can be omitted.
        (None, 4326, None),
        # Legacy float syntax must still be supported without requiring an EPSG.
        (0.6, None, 0.6),
        # Projected CRS: meters are accepted and returned unchanged.
        (
            {"value": 0.6, "unit": "meter"},
            32631,
            0.6,
        ),
        # Geographic CRS: degrees are accepted and returned unchanged.
        (
            {"value": 0.00001, "unit": "degree"},
            4326,
            0.00001,
        ),
        # Geographic CRS: arcseconds are converted to degrees.
        (
            {"value": 1, "unit": "arcsec"},
            4326,
            1 / 3600,
        ),
    ],
)
def test_resolution_valid(resolution, epsg, expected):
    """
    Test valid resolution configurations.

    Tested cases:
    - legacy float syntax
    - meters with a projected CRS
    - degrees with a geographic CRS
    - arcseconds with a geographic CRS
    """

    assert output_parameters.check_resolution(
        resolution, epsg
    ) == pytest.approx(expected)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "resolution, epsg",
    [
        # Geographic CRS: meters are not allowed.
        (
            {"value": 0.6, "unit": "meter"},
            4326,
        ),
        # Projected CRS: degrees are not allowed.
        (
            {"value": 0.00001, "unit": "degree"},
            32631,
        ),
        # Projected CRS: arcseconds are not allowed.
        (
            {"value": 1, "unit": "arcsec"},
            32631,
        ),
        # A resolution unit requires an EPSG.
        (
            {"value": 1, "unit": "arcsec"},
            None,
        ),
    ],
)
def test_resolution_invalid(resolution, epsg):
    """
    Test invalid resolution and CRS combinations.
    """

    with pytest.raises(RuntimeError):
        output_parameters.check_resolution(resolution, epsg)


@pytest.mark.unit_tests
def test_output_parameters_with_resolution_in_arcsec():
    """
    Test resolution conversion from arcseconds to degrees.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "epsg": 4326,
            "resolution": {
                "value": 1,
                "unit": "arcsec",
            },
        }

        inputs = {
            "sensors": {
                "one": {
                    "image": {
                        "loader": "pivot_image",
                        "main_file": "img1_crop.tif",
                        "bands": {
                            "b0": {"path": "img1_crop.tif", "band": 0},
                            "b1": {"path": "color1.tif", "band": 1},
                            "b2": {"path": "color1.tif", "band": 2},
                            "b3": {"path": "color1.tif", "band": 2},
                        },
                    },
                    "geomodel": "img1_crop.geom",
                },
                "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
            }
        }

        overloaded_conf, _ = output_parameters.check_output_parameters(
            inputs,
            config,
            1,
        )

        assert overloaded_conf["resolution"] == pytest.approx(
            [1 / 3600, 1 / 3600]
        )


@pytest.mark.unit_tests
def test_output_minimal():
    """
    Test output
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {"directory": os.path.join(directory, "outdir")}

        inputs = {
            "sensors": {
                "one": {
                    "image": {
                        "loader": "pivot_image",
                        "main_file": "img1_crop.tif",
                        "bands": {
                            "b0": {"path": "img1_crop.tif", "band": 0},
                            "b1": {"path": "color1.tif", "band": 1},
                            "b2": {"path": "color1.tif", "band": 2},
                            "b3": {"path": "color1.tif", "band": 2},
                        },
                    },
                    "geomodel": "img1_crop.geom",
                },
                "two": {"image": "img2_crop.tif", "geomodel": "img2_crop.geom"},
            }
        }

        print(f"config {config}")
        overload = output_parameters.check_output_parameters(inputs, config, 1)
        print(overload)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "case",
    [
        {"classification": True, "expected": {1: 1, 2: 3, 3: 2}},
        {"classification": False, "expected": False},
        {"classification": [3, 2, 1], "expected": {3: 3, 2: 2, 1: 1}},
        {"classification": [52, 3, 1], "expected": "invalid"},
        {
            "classification": {"1": 1, "52": 2, "7": 3},
            "expected": {"1": 1, "52": 2, "7": 3},
        },
    ],
)
def test_classification_parameter(case):
    """
    Test output_parameters.check_output_parameters
    with different classif auxiliary parameter
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        config = {
            "directory": os.path.join(directory, "outdir"),
            "product_level": "dsm",
            "auxiliary": {
                "performance_map": False,
                "image": True,
                "classification": case["classification"],
                "contributing_pair": False,
            },
            "resolution": 0.5,
            "geoid": "path/to/geoid",
            "save_by_pair": False,
        }

        inputs = {
            "sensors": {
                "one": {
                    "image": absolute_data_path("input/phr_gizeh/img1.tif"),
                    "geomodel": absolute_data_path("input/phr_gizeh/img1.geom"),
                    "classification": absolute_data_path(
                        "input/phr_gizeh/classif1.tif"
                    ),
                },
                "two": {
                    "image": absolute_data_path("input/phr_gizeh/img2.tif"),
                    "geomodel": absolute_data_path("input/phr_gizeh/img2.geom"),
                },
            }
        }

        inputs = sensor_inputs.sensors_check_inputs(inputs, directory)

        if case["expected"] == "invalid":
            with pytest.raises(RuntimeError):
                # Expecting some sort of failure
                output_parameters.check_output_parameters(inputs, config, 1)
        else:
            # Should succeed without raising
            output_parameters.check_output_parameters(inputs, config, 1)
