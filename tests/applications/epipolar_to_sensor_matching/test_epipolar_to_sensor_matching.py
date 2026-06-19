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
Test module for cars/applications/epipolar_to_sensor_matching
"""

# __future__ import
from __future__ import absolute_import

# Third-party imports
import os
import tempfile

# Standard library
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

import numpy as np
import pytest

# CARS imports
from cars.applications.epipolar_to_sensor_matching import (
    epipolar_to_sensor_matching_app,
)
from cars.applications.grid_generation import grid_generation_algo
from cars.data_structures import cars_dataset

# CARS test utilities
from tests.helpers import (
    temporary_dir,
)


@pytest.mark.unit_tests
def test_get_cropped_disparity_map_tiles():
    """
    Call right grid correction method and check outputs properties
    """

    # create fake CarsDataset
    height_src, width_src = 300, 200
    grid_tile_size = 10
    disp_cars_ds = cars_dataset.CarsDataset("arrays", name="disp_test")
    disp_cars_ds.create_grid(
        width_src, height_src, grid_tile_size, grid_tile_size, 0, 0
    )

    assert disp_cars_ds.shape == (30, 20)

    # create epipolar grid left
    step = 30
    height_tgt, width_tgt = 500, 600
    grid_height = int(height_src / step)
    grid_width = int(width_src / step)
    row_off, col_off = 400, 500
    row_coords = np.linspace(row_off, row_off + height_tgt - 1, grid_height)
    col_coords = np.linspace(col_off, col_off + width_tgt - 1, grid_width)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    mapping_grid = np.stack([row_grid, col_grid], axis=-1)

    grid_origin = (0, 0)
    grid_spacing = (30, 30)

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        left_grid_path = os.path.join(directory, "grid_left.tif")
        grid_generation_algo.write_grid(
            mapping_grid, left_grid_path, grid_origin, grid_spacing
        )

        grid_left = {
            "grid_spacing": grid_spacing,
            "grid_origin": grid_origin,
            "epipolar_size_x": 300,
            "epipolar_size_y": 200,
            "epipolar_origin_x": 0,
            "epipolar_origin_y": 0,
            "epipolar_spacing_x": 30,
            "epipolar_spacing": step,
            "disp_to_alt_ratio": 0.3,
            "epipolar_step": step,
            "path": left_grid_path,
        }
        sensor_window = [650, 750, 560, 740]

        cropped_disp_cars_ds = (
            epipolar_to_sensor_matching_app.get_cropped_disparity_map_tiles(
                disp_cars_ds, grid_left, sensor_window
            )
        )

        assert isinstance(cropped_disp_cars_ds, cars_dataset.CarsDataset)

        assert cropped_disp_cars_ds.shape == (14, 13)
