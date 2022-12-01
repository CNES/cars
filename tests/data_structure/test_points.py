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
Test module for cars/data_structure/cars_dataset.py used with points
"""

# Standard imports
from __future__ import absolute_import

import os
import tempfile

import numpy as np
import pandas

# Third party imports
import pytest

# CARS imports
from cars.data_structures import cars_dataset

# CARS Tests import
from tests.helpers import assert_same_dataframes, temporary_dir


def create_points_object(grid=(4, 3), nb_elements=5):
    """
    Create points object
    """

    points_manager = cars_dataset.CarsDataset("points")
    # Artificial grid
    overlap = (0, 0)
    tile_size = (100, 100)
    width = tile_size[1] * grid[1] - 1
    height = tile_size[0] * grid[0] - 1

    points_manager.create_grid(
        width,
        height,
        tile_size[0],
        tile_size[1],
        overlap[0],
        overlap[1],
    )

    def create_dataframe(start=0, end=30):
        """
        Create dataframe
        """

        values = range(start, end)

        dataframe = pandas.DataFrame(values, columns=["x"])

        return dataframe

    points_manager.generate_none_tiles()
    tmp_value = 0
    for i in range(points_manager.tiling_grid.shape[0]):
        for j in range(points_manager.tiling_grid.shape[1]):
            # Create dataframe
            sparse_dataframe = create_dataframe(
                start=tmp_value, end=tmp_value + nb_elements
            )
            tmp_value += nb_elements

            attributes = {"test": 1}
            # add attributes
            cars_dataset.fill_dataframe(
                sparse_dataframe, saving_info=None, attributes=attributes
            )

            points_manager.tiles[i][j] = sparse_dataframe

    return points_manager


@pytest.mark.unit_tests
def test_save_pandas_and_constructor():
    """
    Test tile_manager creation, saving to disk  with merge
    """

    def artificial_manager_save(manager, save_file):
        """
        Save all tiles to manager
        """
        for i in range(manager.tiling_grid.shape[0]):
            for j in range(manager.tiling_grid.shape[1]):
                overwrite = bool(i + j == 0)
                cars_dataset.CarsDataset("points").run_save(
                    manager.tiles[i][j],
                    save_file,
                    overwrite=overwrite,
                )

    # create object
    points_object = create_points_object(grid=(4, 3), nb_elements=5)

    # reconstruct dataframe and save it
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_df = os.path.join(directory, "df.csv")

        artificial_manager_save(points_object, out_df)

        # compare files
        out_dataframe = pandas.read_csv(out_df)

        # Assert that the sums of dataframes are equals

        sum_from_manager = 0
        for i in range(points_object.tiling_grid.shape[0]):
            for j in range(points_object.tiling_grid.shape[1]):
                sum_from_manager += np.sum(points_object.tiles[i][j].to_numpy())

        sum_merged_object = np.sum(out_dataframe.to_numpy())

        assert sum_merged_object == sum_from_manager


@pytest.mark.unit_tests
def test_save_to_disk_and_load():
    """
    Test save_to_disk and load from path functions.
    """

    # create object
    grid_shape = (4, 3)
    points_object = create_points_object(grid=grid_shape, nb_elements=5)

    # reconstruct dataframe and save it
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        # save tiled object

        left_pc_folder = os.path.join(directory, "left_pc_object")
        points_object.save_cars_dataset(left_pc_folder)

        # Create new object and load previous object

        new_pc_object = cars_dataset.CarsDataset(
            "points", load_from_disk=left_pc_folder
        )

        # Assert grids and overlaps  are the same
        np.testing.assert_allclose(
            points_object.tiling_grid, new_pc_object.tiling_grid
        )
        np.testing.assert_allclose(
            points_object.overlaps, new_pc_object.overlaps
        )

        # Assert dataframes are the same
        for col in range(grid_shape[1]):
            for row in range(grid_shape[0]):
                assert_same_dataframes(
                    points_object.tiles[row][col],
                    new_pc_object.tiles[row][col],
                )
