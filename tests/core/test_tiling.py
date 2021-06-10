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
Test module for cars/core/tiling.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import pytest

# CARS imports
from cars.core import tiling


@pytest.mark.unit_tests
def test_grid():

    grid = tiling.grid(0, 0, 500, 400, 90, 90)
    assert grid.shape == (6, 7, 2)


@pytest.mark.unit_tests
def test_split():
    """
    Test split terrain method
    """
    splits = tiling.split(0, 0, 500, 500, 100, 100)
    print(splits)
    assert len(splits) == 25


@pytest.mark.unit_tests
def test_crop():
    """
    Test crop function
    """
    region1 = [0, 0, 100, 100]
    region2 = [50, 0, 120, 80]

    cropped = tiling.crop(region1, region2)

    assert cropped == [50, 0, 100, 80]


@pytest.mark.unit_tests
def test_pad():
    """
    Test pad function
    """

    region = [1, 2, 3, 4]
    margin = [5, 6, 7, 8]

    assert tiling.pad(region, margin) == [-4, -4, 10, 12]


@pytest.mark.unit_tests
def test_empty():
    """
    Test empty function
    """
    assert tiling.empty([0, 0, 0, 10])
    assert tiling.empty([0, 0, 10, 0])
    assert tiling.empty([10, 10, 0, 0])
    assert not tiling.empty([0, 0, 10, 10])


@pytest.mark.unit_tests
def test_union():
    """
    Test union function
    """
    assert tiling.union([[0, 0, 5, 6], [2, 3, 10, 11]]) == (0, 0, 10, 11)


@pytest.mark.unit_tests
def test_list_tiles():
    """
    Test list_tiles function
    """
    region = [45, 65, 55, 75]
    largest_region = [0, 0, 100, 100]
    tile_size = 10

    tiles = tiling.list_tiles(region, largest_region, tile_size, margin=0)

    assert tiles == [
        [40, 60, 50, 70],
        [40, 70, 50, 80],
        [50, 60, 60, 70],
        [50, 70, 60, 80],
    ]

    tiles = tiling.list_tiles(region, largest_region, tile_size, margin=1)

    assert tiles == [
        [30, 50, 40, 60],
        [30, 60, 40, 70],
        [30, 70, 40, 80],
        [30, 80, 40, 90],
        [40, 50, 50, 60],
        [40, 60, 50, 70],
        [40, 70, 50, 80],
        [40, 80, 50, 90],
        [50, 50, 60, 60],
        [50, 60, 60, 70],
        [50, 70, 60, 80],
        [50, 80, 60, 90],
        [60, 50, 70, 60],
        [60, 60, 70, 70],
        [60, 70, 70, 80],
        [60, 80, 70, 90],
    ]


@pytest.mark.unit_tests
def test_roi_to_start_and_size():
    """
    Test roi_to_start_and_size function
    """
    res = tiling.roi_to_start_and_size([0, 0, 10, 10], 10)

    assert res == (0, 10, 1, 1)


@pytest.mark.unit_tests
def test_snap_to_grid():
    """
    Test snap_to_grid function
    """
    assert (0, 0, 11, 11) == tiling.snap_to_grid(0.1, 0.2, 10.1, 10.2, 1.0)


# function parameters are fixtures set in conftest.py
@pytest.mark.unit_tests
def test_terrain_region_to_epipolar(
    images_and_grids_conf,  # pylint: disable=redefined-outer-name
    disparities_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test transform to epipolar method
    """
    configuration = images_and_grids_conf
    configuration["preprocessing"]["output"].update(
        disparities_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )

    region = [5.1952, 44.205, 5.2, 44.208]
    out_region = tiling.terrain_region_to_epipolar(region, configuration)
    assert out_region == [0.0, 0.0, 612.0, 400.0]
