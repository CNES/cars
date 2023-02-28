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
Test module for cars/core/roi_tools.py
"""


# Third party imports
import pytest
from shapely.geometry import mapping

# CARS imports
from cars.core import roi_tools

# CARS Tests imports
from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_parse_roi_file():
    """
    Test parse_roi_file

    """

    path_roi_file = absolute_data_path("input/phr_gizeh/roi/roi_poly_gizeh.shp")

    # Transform to shapely polygon, epsg
    roi_poly, roi_epsg = roi_tools.parse_roi_file(path_roi_file)

    assert roi_epsg == 4326
    assert len(mapping(roi_poly)["coordinates"][0]) == 6


@pytest.mark.unit_tests
def test_geojson_to_shapely():
    """
    Test geojson_to_shapely

    """

    # test default epsg
    geojson_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [
                        [
                            [1.4431120297833502, 43.67951364710356],
                            [1.4431120297833502, 43.557235281894805],
                            [1.5887192444447464, 43.557235281894805],
                            [1.5887192444447464, 43.67951364710356],
                            [1.4431120297833502, 43.67951364710356],
                        ]
                    ],
                    "type": "Polygon",
                },
            }
        ],
    }

    # Transform to shapely polygon, epsg

    roi_poly, roi_epsg = roi_tools.geojson_to_shapely(geojson_dict)

    assert roi_epsg == 4326
    assert len(mapping(roi_poly)["coordinates"][0]) == 5

    # test with new epsg
    geojson_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:5349"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [
                        [
                            [1.4431120297833502, 43.67951364710356],
                            [1.4431120297833502, 43.557235281894805],
                            [1.5887192444447464, 43.557235281894805],
                            [1.5887192444447464, 43.67951364710356],
                            [1.4431120297833502, 43.67951364710356],
                        ]
                    ],
                    "type": "Polygon",
                },
            }
        ],
    }

    # Transform to shapely polygon, epsg

    roi_poly, roi_epsg = roi_tools.geojson_to_shapely(geojson_dict)

    assert roi_epsg == 5349
    assert len(mapping(roi_poly)["coordinates"][0]) == 5


@pytest.mark.unit_tests
def test_resample_polygon():
    """
    Test resample_polygon

    """

    path_roi_file = absolute_data_path("input/phr_gizeh/roi/roi_poly_gizeh.shp")

    # Transform to shapely polygon, epsg
    roi_poly, roi_epsg = roi_tools.parse_roi_file(path_roi_file)

    # Resample polygon

    new_roi_poly = roi_tools.resample_polygon(roi_poly, roi_epsg, resolution=20)

    assert len(mapping(new_roi_poly)["coordinates"][0]) == 42
