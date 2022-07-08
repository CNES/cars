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
Test module for cars/data_structure/cars_dataset.py used with arrays
"""

# Standard imports
from __future__ import absolute_import

import os
import tempfile

import numpy as np

# Third party imports
import pytest
import xarray as xr

import cars.core.constants as cst
from cars.core import inputs

# CARS imports
from cars.data_structures import cars_dataset

# CARS Tests import
from tests.helpers import (
    absolute_data_path,
    assert_same_datasets,
    assert_same_images,
    temporary_dir,
)


def create_cars_dataset_from_path(
    image, tag, tile_size=(40, 40), overlap=(0, 0)
):
    """
    Create a array tile_manager from image on disk

    """

    # Create manager
    array_cars_ds = cars_dataset.CarsDataset("arrays")
    # Create grid
    width, height = inputs.rasterio_get_size(image)
    array_cars_ds.create_grid(
        width,
        height,
        tile_size[0],
        tile_size[1],
        overlap[0],
        overlap[1],
    )

    def generate_dataset(image_path, tag, window):
        """
        Generate xarray dataset
        """

        rio_window = cars_dataset.generate_rasterio_window(window)
        array, profile = inputs.rasterio_read_as_array(
            image_path, window=rio_window
        )

        # create dataset
        if len(array.shape) == 2:
            dataset = xr.Dataset(
                {tag: ([cst.ROW, cst.COL], array)},
                coords={
                    "row": np.arange(array.shape[0]),
                    "col": np.arange(array.shape[1]),
                },
            )
        else:
            dataset = xr.Dataset(
                {tag: ([cst.BAND, cst.ROW, cst.COL], array)},
                coords={
                    cst.BAND: np.arange(array.shape[0]),
                    cst.ROW: np.arange(array.shape[1]),
                    cst.COL: np.arange(array.shape[2]),
                },
            )

        return dataset, profile

    # Fill datasets
    array_cars_ds.tiles = []
    for i in range(array_cars_ds.tiling_grid.shape[0]):
        object_row = []
        for j in range(array_cars_ds.tiling_grid.shape[1]):

            # Create window
            dto_window = cars_dataset.window_aray_to_dict(
                array_cars_ds.tiling_grid[i, j, :],
                overlap=array_cars_ds.overlaps[i, j, :],
            )

            array_overlap = cars_dataset.overlap_aray_to_dict(
                array_cars_ds.overlaps[i, j, :]
            )

            # Create object
            dense_dataset, profile = generate_dataset(image, tag, dto_window)

            cars_dataset.fill_dataset(
                dense_dataset,
                saving_info=None,
                window=dto_window,
                profile=profile,
                attributes=None,
                overlaps=array_overlap,
            )

            object_row.append(dense_dataset)

        array_cars_ds.tiles.append(object_row)

    return array_cars_ds


def artificial_manager_save(manager, save_file, tag):
    """
    Save all tiles of manager to file
    """

    dsc = manager.generate_descriptor(
        manager.tiles[0][0], save_file, tag, dtype="float32", nodata=None
    )

    for i in range(manager.tiling_grid.shape[0]):
        for j in range(manager.tiling_grid.shape[1]):
            cars_dataset.CarsDataset("arrays").run_save(
                manager.tiles[i][j], save_file, tag=tag, descriptor=dsc
            )

    dsc.close()


@pytest.mark.unit_tests
def test_save_tif_and_constructor():
    """
    Test tile_manager creation from tif, saving to disk as tif and compare files
    Tests Arrays init from tif, and save functions
    """

    # read imput
    in_file = absolute_data_path("../data/input/phr_paca/left_image.tif")

    # create object
    sensor_image = create_cars_dataset_from_path(
        in_file, "im", tile_size=(40, 40), overlap=(0, 0)
    )

    # reconstruct and save tif
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_left_image = os.path.join(directory, "left_image_recreated.tif")

        artificial_manager_save(sensor_image, out_left_image, "im")

        # compare files
        assert_same_images(out_left_image, in_file)


@pytest.mark.unit_tests
def test_save_to_disk_and_load():
    """
    Test save_to_disk and load from path functions.
    """

    # read imput
    in_file = absolute_data_path("../data/input/phr_paca/left_image.tif")

    # create object
    sensor_image = create_cars_dataset_from_path(
        in_file, "im", tile_size=(40, 40), overlap=(0, 0)
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        # save tiled object

        left_image_folder = os.path.join(directory, "left_image_object")
        sensor_image.save_cars_dataset(left_image_folder)

        # Create new object and load previous object

        new_sensor_image_object = cars_dataset.CarsDataset(
            "arrays", load_from_disk=left_image_folder
        )

        # Assert they are the same
        previous_tif = os.path.join(directory, "left_image_recreated.tif")
        new_tif = os.path.join(directory, "left_image_recreated.tif")
        artificial_manager_save(sensor_image, previous_tif, "im")
        artificial_manager_save(new_sensor_image_object, new_tif, "im")
        assert_same_images(previous_tif, new_tif)

        # Assert grids and overlaps  are the same
        np.testing.assert_allclose(
            sensor_image.tiling_grid, new_sensor_image_object.tiling_grid
        )
        np.testing.assert_allclose(
            sensor_image.overlaps, new_sensor_image_object.overlaps
        )

        # Assert datasets are the same
        assert_same_datasets(
            sensor_image.tiles[0][0],
            new_sensor_image_object.tiles[0][0],
        )
