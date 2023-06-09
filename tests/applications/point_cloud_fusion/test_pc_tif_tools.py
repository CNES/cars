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
Test module for cars/steps/pc_tif_tools.py
"""

import os

# Third party imports
import pytest
import rasterio as rio
from shapely.geometry import mapping

from cars.applications.point_cloud_fusion import pc_tif_tools
from cars.core import constants as cst
from cars.core import tiling

from ...helpers import absolute_data_path


def generate_test_inputs():
    """
    Helper function for generating input
    """

    path_pc = absolute_data_path("input/epi_pc_gizeh")

    data = {
        cst.X: os.path.join(path_pc, "epi_pc_X.tif"),
        cst.Y: os.path.join(path_pc, "epi_pc_Y.tif"),
        cst.Z: os.path.join(path_pc, "epi_pc_Z.tif"),
        cst.POINTS_CLOUD_CLR_KEY_ROOT: os.path.join(
            path_pc, "epi_pc_color.tif"
        ),
        cst.POINTS_CLOUD_CLASSIF_KEY_ROOT: os.path.join(
            path_pc, "epi_classification.tif"
        ),
        cst.POINTS_CLOUD_CONFIDENCE: {
            "confidence1": os.path.join(path_pc, "epi_confidence1.tif"),
            "confidence2": os.path.join(path_pc, "epi_confidence2.tif"),
        },
        cst.POINTS_CLOUD_VALID_DATA: None,
        cst.PC_EPSG: 4326,
    }

    return data


@pytest.mark.end2end_tests
def test_create_polygon_from_list_points():
    """
    test create_polygon_from_list_points
    """

    list_points = [(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]
    poly = pc_tif_tools.create_polygon_from_list_points(list_points)

    assert len(mapping(poly)["coordinates"][0]) == 5


@pytest.mark.end2end_tests
def test_compute_epsg_from_point_cloud():
    """
    test compute_epsg_from_point_cloud
    """

    data_tif = generate_test_inputs()

    list_epi_pc = {"pc_0": data_tif}

    epsg = pc_tif_tools.compute_epsg_from_point_cloud(list_epi_pc)

    assert epsg == 32636


@pytest.mark.end2end_tests
def test_get_min_max_band():
    """
    test get_min_max_band
    """

    data_tif = generate_test_inputs()

    x_y_min_max = pc_tif_tools.get_min_max_band(
        data_tif[cst.X],
        data_tif[cst.Y],
        data_tif[cst.Z],
        4326,
        32636,
        window=rio.windows.Window.from_slices((200, 300), (200, 300)),
    )

    assert x_y_min_max == [
        319922.73987110204,
        320000.461985868,
        3317814.1553263,
        3317873.37849824,
    ]


@pytest.mark.end2end_tests
def test_transform_input_pc_and_metrics():
    """
    test transform_input_pc
    test compute_max_nb_point_clouds
    test compute_average_distance
    """

    # test transform_input_pc

    data_tif = generate_test_inputs()
    list_epi_pc = {"pc_0": data_tif, "pc_1": data_tif}

    (
        terrain_bbox,
        list_epipolar_points_cloud_by_tiles,
    ) = pc_tif_tools.transform_input_pc(
        list_epi_pc, 32636, epipolar_tile_size=200
    )

    assert terrain_bbox == [
        319796.7957302701,
        3317678.622369081,
        320316.8316215427,
        3318157.288182468,
    ]

    assert len(list_epipolar_points_cloud_by_tiles) == 2

    assert list_epipolar_points_cloud_by_tiles[0].shape == (5, 5)

    # tes compute_max_nb_point_clouds

    nb_max_nb_pc = pc_tif_tools.compute_max_nb_point_clouds(
        list_epipolar_points_cloud_by_tiles
    )

    assert nb_max_nb_pc == 2

    # test compute_average_distance
    average_dist = pc_tif_tools.compute_average_distance(
        list_epipolar_points_cloud_by_tiles
    )

    assert int(100 * average_dist) / 100 == 0.46


@pytest.mark.end2end_tests
def test_transform_input_pc_and_correspondance():
    """
    test transform_input_pc
    test get_tiles_corresponding_tiles_tif
    """

    # test transform_input_pc

    data_tif = generate_test_inputs()
    list_epi_pc = {"pc_0": data_tif, "pc_1": data_tif}

    (
        terrain_bbox,
        list_epipolar_points_cloud_by_tiles,
    ) = pc_tif_tools.transform_input_pc(
        list_epi_pc, 32636, epipolar_tile_size=200
    )

    # Test correspondances
    # Compute bounds and terrain grid
    [xmin, ymin, xmax, ymax] = terrain_bbox

    # terrain tile size
    optimal_terrain_tile_width = 200

    # Split terrain bounding box in pieces
    terrain_tiling_grid = tiling.generate_tiling_grid(
        xmin,
        ymin,
        xmax,
        ymax,
        optimal_terrain_tile_width,
        optimal_terrain_tile_width,
    )

    corresponding_tiles = pc_tif_tools.get_tiles_corresponding_tiles_tif(
        terrain_tiling_grid, list_epipolar_points_cloud_by_tiles, margins=0
    )

    assert len(corresponding_tiles[0, 0]["required_point_clouds"]) == 8
    assert len(corresponding_tiles[1, 0]["required_point_clouds"]) == 14
    assert len(corresponding_tiles[2, 2]["required_point_clouds"]) == 8
    assert len(corresponding_tiles[1, 2]["required_point_clouds"]) == 10
