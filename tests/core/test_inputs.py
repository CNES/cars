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
Test module for cars/core/inputs.py
"""

# Third party imports
import pytest
from shapely.geometry import Polygon

# CARS imports
from cars.core import inputs

# CARS Tests imports
from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_ncdf_can_open():
    fake_path = "/here/im/not.nc"
    assert not inputs.ncdf_can_open(fake_path)


@pytest.mark.unit_tests
def test_rasterio_can_open():
    """
    Test rasterio_can_open function with existing and fake raster
    """
    existing = absolute_data_path("input/phr_ventoux/left_image.tif")
    not_existing = "/stuff/dummy_file.doe"
    assert inputs.rasterio_can_open(existing)
    assert not inputs.rasterio_can_open(not_existing)


@pytest.mark.unit_tests
def test_otb_can_open():
    """
    Test otb_can_open() with different geom configurations
    """
    # existing
    existing_with_geom = absolute_data_path("input/phr_ventoux/left_image.tif")
    # existing with no geom file
    existing_no_geom = absolute_data_path("input/utils_input/im1.tif")
    not_existing = "/stuff/dummy_file.doe"

    assert inputs.otb_can_open(existing_with_geom)
    assert not inputs.otb_can_open(existing_no_geom)
    assert not inputs.otb_can_open(not_existing)


@pytest.mark.unit_tests
def test_fix_shapely():
    """
    Test read_vector fix shapely with poly.gpkg example
    """
    poly, _ = inputs.read_vector(
        absolute_data_path("input/utils_input/poly.gpkg")
    )
    assert poly.is_valid is False
    poly = poly.buffer(0)
    assert poly.is_valid is True


@pytest.mark.unit_tests
def test_read_vector():
    """
    Test if read_vector function works with an input example
    """
    path_to_shapefile = absolute_data_path(
        "input/utils_input/left_envelope.shp"
    )

    poly, epsg = inputs.read_vector(path_to_shapefile)

    assert epsg == 4326
    assert isinstance(poly, Polygon)
    assert list(poly.exterior.coords) == [
        (5.193406138843349, 44.20805805252155),
        (5.1965650939582435, 44.20809526197842),
        (5.196654349708835, 44.205901416036546),
        (5.193485218293437, 44.205842790578764),
        (5.193406138843349, 44.20805805252155),
    ]

    # test exception
    with pytest.raises(Exception) as read_error:
        inputs.read_vector("test.shp")
    assert str(read_error.value) == "Impossible to read test.shp file"
