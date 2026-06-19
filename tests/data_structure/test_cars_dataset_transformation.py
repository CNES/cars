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
Test module cars dataset transformation
"""

# __future__ import
from __future__ import absolute_import

# Third-party imports
import pytest

# CARS imports
from cars.data_structures import cars_dataset, cars_dataset_transformations


@pytest.mark.unit_tests
def test_extract_cars_dataset():
    """
    Test extract_cars_dataset function
    """

    # Generate fake CarsDataset
    height_src, width_src = 1000, 3000
    tile_size = 40
    disp_cars_ds = cars_dataset.CarsDataset("arrays", name="disp_test")

    disp_cars_ds.create_grid(width_src, height_src, tile_size, tile_size, 0, 0)

    assert disp_cars_ds.shape == (25, 75)

    # fill for coordinates testing
    for row in range(disp_cars_ds.shape[0]):
        for col in range(disp_cars_ds.shape[1]):
            disp_cars_ds[row, col] = (row, col)

    # Define region to extract
    region = [100, 200, 500, 1000]

    cropped_cars_ds = cars_dataset_transformations.extract_cars_dataset(
        disp_cars_ds, region
    )

    assert cropped_cars_ds.shape == (4, 14)
    assert cropped_cars_ds[0, 0] == (2, 12)
    assert cropped_cars_ds[2, 12] == (4, 24)
