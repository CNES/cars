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
Test module for surface modeling pipeline configuration.
"""

import tempfile

import pytest

from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

from ...helpers import absolute_data_path, temporary_dir


def _make_gizeh_config(directory, output_epsg, resolution, phasing):
    """
    Create a minimal Gizeh surface modeling configuration.

    :param directory: output directory
    :type directory: str
    :param output_epsg: output EPSG
    :type output_epsg: int, str, or None
    :param resolution: output resolution
    :type resolution: float or dict
    :param phasing: phasing configuration
    :type phasing: dict or None

    :return: surface modeling configuration
    :rtype: dict
    """

    return {
        "input": {
            "sensors": {
                "image1": {
                    "image": absolute_data_path("input/phr_gizeh/img1.tif"),
                    "geomodel": absolute_data_path("input/phr_gizeh/img1.geom"),
                },
                "image2": {
                    "image": absolute_data_path("input/phr_gizeh/img2.tif"),
                    "geomodel": absolute_data_path("input/phr_gizeh/img2.geom"),
                },
            },
        },
        "surface_modeling": {
            "advanced": {
                "phasing": phasing,
            }
        },
        "output": {
            "directory": directory,
            "product_level": [],
            "epsg": output_epsg,
            "resolution": resolution,
        },
    }


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "output_epsg, resolution, phasing, expected_epsg",
    [
        # Geographic CRS with resolution and phasing in arcseconds.
        (
            4326,
            {"value": 1, "unit": "arcsec"},
            {
                "point": [1, 2],
                "epsg": 4326,
                "unit": "arcsec",
            },
            4326,
        ),
        # Geographic CRS provided as strings.
        (
            "4326",
            {"value": 1, "unit": "arcsec"},
            {
                "point": [1, 2],
                "epsg": "4326",
                "unit": "arcsec",
            },
            4326,
        ),
        # Geographic CRS with resolution and phasing in degrees.
        (
            4326,
            {"value": 0.00001, "unit": "degree"},
            {
                "point": [31.132, 29.978],
                "epsg": 4326,
                "unit": "degree",
            },
            4326,
        ),
        # Compound geographic CRS: phasing must match the horizontal CRS.
        (
            "4326+5773",
            {"value": 1, "unit": "arcsec"},
            {
                "point": [1, 2],
                "epsg": 4326,
                "unit": "arcsec",
            },
            4326,
        ),
        # Projected CRS with resolution and phasing in meters.
        (
            32636,
            {"value": 0.6, "unit": "meter"},
            {
                "point": [319000.0, 3317000.0],
                "epsg": 32636,
                "unit": "meter",
            },
            32636,
        ),
        # Without phasing, output EPSG is used directly.
        (
            32636,
            {"value": 0.6, "unit": "meter"},
            None,
            32636,
        ),
        # Without phasing or output EPSG, EPSG remains undefined
        # until it is computed during processing.
        (
            None,
            0.6,
            None,
            None,
        ),
    ],
)
def test_phasing_output_consistency_valid(
    output_epsg,
    resolution,
    phasing,
    expected_epsg,
):
    """
    Test valid resolution, output EPSG and phasing combinations.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = _make_gizeh_config(
            directory,
            output_epsg,
            resolution,
            phasing,
        )

        pipeline = SurfaceModelingPipeline(conf)

        assert pipeline.epsg == expected_epsg


@pytest.mark.unit_tests
def test_phasing_output_epsg_mismatch():
    """
    Test that phasing and output EPSG must match.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = _make_gizeh_config(
            directory,
            32636,
            {"value": 0.6, "unit": "meter"},
            {
                "point": [31.132, 29.978],
                "epsg": 4326,
                "unit": "degree",
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Phasing EPSG 4326 does not match "
            "output horizontal EPSG 32636.",
        ):
            SurfaceModelingPipeline(conf)


@pytest.mark.unit_tests
def test_phasing_requires_output_epsg():
    """
    Test that an output EPSG must be specified when phasing is used.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = _make_gizeh_config(
            directory,
            None,
            0.6,
            {
                "point": [31.132, 29.978],
                "epsg": 4326,
                "unit": "degree",
            },
        )

        with pytest.raises(
            RuntimeError,
            match="An output EPSG must be specified when phasing is used.",
        ):
            SurfaceModelingPipeline(conf)


@pytest.mark.unit_tests
def test_surface_modeling_arcsec_configuration():
    """
    Test arcsecond resolution and phasing normalization.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = _make_gizeh_config(
            directory,
            4326,
            {"value": 1, "unit": "arcsec"},
            {
                "point": [1, 2],
                "epsg": 4326,
                "unit": "arcsec",
            },
        )

        pipeline = SurfaceModelingPipeline(conf)

        assert pipeline.epsg == 4326

        assert pipeline.used_conf["output"]["resolution"] == pytest.approx(
            [
                1 / 3600,
                1 / 3600,
            ]
        )

        assert pipeline.phasing["point"] == pytest.approx([1 / 3600, 2 / 3600])
        assert pipeline.phasing["unit"] == "degree"
        assert pipeline.phasing["epsg"] == 4326
