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
Test module for cars/data_structure/corresponding_tiles_tools.py
"""

# Standard imports
from __future__ import absolute_import

import numpy as np

# Third party imports
import pytest
import xarray as xr

# CARS imports
from cars.data_structures import corresponding_tiles_tools

# CARS Tests import


def generate_dataset(window, overlap):
    """
    Generate dataset
    """
    nb_rows = int(window[1] - window[0] + overlap[0] + overlap[1])
    nb_cols = int(window[3] - window[2] + overlap[2] + overlap[3])

    real_row_min = window[0] - overlap[0]
    real_col_min = window[2] - overlap[2]

    coords = {}
    coords["row"] = range(nb_rows)
    coords["col"] = range(nb_cols)

    new_dataset = xr.Dataset(data_vars={}, coords=coords)

    data = np.zeros((nb_rows, nb_cols))
    for row in range(nb_rows):
        for col in range(nb_cols):
            data[row, col] = (row + real_row_min) * (col + real_col_min)

    new_dataset["disp"] = xr.DataArray(
        data,
        dims=["row", "col"],
    )
    return new_dataset


@pytest.mark.unit_tests
def test_reconstruct_data():
    """
    Test reconstruct_data
    """

    window1 = [0, 40, 0, 50]
    overlap1 = [3, 4, 5, 6]
    tile1 = generate_dataset(window1, overlap1)

    window2 = [0, 40, 50, 100]
    overlap2 = [3, 4, 5, 6]
    tile2 = generate_dataset(window2, overlap2)

    window3 = [40, 80, 0, 50]
    overlap3 = [3, 4, 5, 6]
    tile3 = generate_dataset(window3, overlap3)

    window4 = [40, 80, 50, 100]
    overlap4 = [3, 4, 5, 6]
    tile4 = generate_dataset(window4, overlap4)

    corresponding_tiles = [
        (window1, overlap1, tile1),
        (window2, overlap2, tile2),
        (window3, overlap3, tile3),
        (window4, overlap4, tile4),
    ]

    new_dataset, row_min, col_min = corresponding_tiles_tools.reconstruct_data(
        corresponding_tiles, window1, overlap1
    )

    expected_dataset = generate_dataset([0, 80, 0, 100], [3, 4, 5, 6])

    # assert
    np.testing.assert_allclose(
        new_dataset["disp"].values, expected_dataset["disp"].values
    )

    assert row_min == -3
    assert col_min == -5
