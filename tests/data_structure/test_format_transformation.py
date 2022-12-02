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
Test module for cars/data_structure/cars_dataset.py
"""

# Standard imports
from __future__ import absolute_import

import numpy as np

# Third party imports
import pytest
import xarray as xr

# CARS imports
from cars.data_structures import format_transformation

# CARS Tests import


@pytest.mark.unit_tests
def test_grid_margins_2_overlaps():
    """
    Test grid_margins_2_overlaps
    """

    # Create grid
    grid_col_min = np.expand_dims(np.array([[0, 100], [0, 100]]), axis=2)
    grid_col_max = np.expand_dims(np.array([[100, 150], [100, 150]]), axis=2)
    grid_row_min = np.expand_dims(np.array([[0, 0], [110, 110]]), axis=2)
    grid_row_max = np.expand_dims(np.array([[110, 110], [170, 170]]), axis=2)

    grid = np.squeeze(
        np.stack(
            [grid_row_min, grid_row_max, grid_col_min, grid_col_max], axis=2
        )
    )

    # margins  pandora convention : ['left','up', 'right', 'down']
    margins = [3.2, 5, 2, 6.2]

    out_overlaps = format_transformation.grid_margins_2_overlaps(grid, margins)

    # expected
    expected_overlap_col_min = np.expand_dims(
        np.array([[0, 3], [0, 3]]), axis=2
    )
    expected_overlap_col_max = np.expand_dims(
        np.array([[2, 0], [2, 0]]), axis=2
    )
    expected_overlap_row_min = np.expand_dims(
        np.array([[0, 0], [5, 5]]), axis=2
    )
    expected_overlap_row_max = np.expand_dims(
        np.array([[7, 7], [0, 0]]), axis=2
    )

    expected_overlap = np.squeeze(
        np.stack(
            [
                expected_overlap_row_min,
                expected_overlap_row_max,
                expected_overlap_col_min,
                expected_overlap_col_max,
            ],
            axis=2,
        )
    )

    np.testing.assert_allclose(out_overlaps, expected_overlap)


@pytest.mark.unit_tests
def test_region_margins_from_window():
    """
    Test region_margins_from_window
    """

    window = {"col_min": 10, "col_max": 120, "row_min": 30, "row_max": 80}

    left_overlap = {"left": 2, "right": -3, "up": 4, "down": 5}

    right_overlap = {"left": -7, "right": 8, "up": 9, "down": 10}

    # Create initial margin
    corner = ["left", "up", "right", "down"]
    data = np.zeros(len(corner))
    col = np.arange(len(corner))
    initial_margin = xr.Dataset(
        {"left_margin": (["col"], data)}, coords={"col": col}
    )
    initial_margin["right_margin"] = xr.DataArray(data, dims=["col"])

    expected_region = [10, 30, 120, 80]
    expected_left_margin = np.array([2, 4, 3, 5])
    expected_right_margin = np.array([7, 9, 8, 10])

    out_region, out_margin = format_transformation.region_margins_from_window(
        initial_margin, window, left_overlap, right_overlap
    )

    assert out_region == expected_region

    np.testing.assert_allclose(
        out_margin["left_margin"].data, expected_left_margin
    )

    np.testing.assert_allclose(
        out_margin["right_margin"].data, expected_right_margin
    )


@pytest.mark.unit_tests
def test_get_corresponding_indexes():
    """
    Test get_corresponding_indexes
    """

    row, col = 0, 1

    pc_row, pc_col = format_transformation.get_corresponding_indexes(row, col)

    assert pc_row == 1
    assert pc_col == 0
