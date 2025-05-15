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
import numpy as np
import pytest
import rasterio as rio
from shapely.geometry import mapping

from cars.applications.point_cloud_fusion import (
    point_cloud_algo,
    point_cloud_wrappers,
)
from cars.core import constants as cst
from cars.core import tiling
from cars.orchestrator import orchestrator

from ...helpers import absolute_data_path


def generate_test_inputs():
    """
    Helper function for generating input
    """

    path_pc = absolute_data_path("input/depth_map_gizeh")

    data = {
        cst.X: os.path.join(path_pc, "X.tif"),
        cst.Y: os.path.join(path_pc, "Y.tif"),
        cst.Z: os.path.join(path_pc, "Z.tif"),
        cst.POINT_CLOUD_CLR_KEY_ROOT: os.path.join(path_pc, "color.tif"),
        cst.POINT_CLOUD_CLASSIF_KEY_ROOT: os.path.join(
            path_pc, "classification.tif"
        ),
        cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT: {
            "confidence1": os.path.join(path_pc, "performance_map.tif")
        },
        cst.POINT_CLOUD_MSK: None,
        cst.PC_EPSG: 4326,
    }

    return data


@pytest.mark.unit_tests
def test_create_polygon_from_list_points():
    """
    test create_polygon_from_list_points
    """

    list_points = [(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]
    poly = point_cloud_wrappers.create_polygon_from_list_points(list_points)

    assert len(mapping(poly)["coordinates"][0]) == 6


@pytest.mark.unit_tests
def test_compute_epsg_from_point_cloud():
    """
    test compute_epsg_from_point_cloud
    """

    data_tif = generate_test_inputs()

    list_epi_pc = {"pc_0": data_tif}

    epsg = point_cloud_wrappers.compute_epsg_from_point_cloud(list_epi_pc)

    assert epsg == 32636


@pytest.mark.unit_tests
def test_generate_point_clouds():
    """
    test compute_epsg_from_point_cloud
    """

    data_tif = generate_test_inputs()

    list_epi_pc = {"pc_0": data_tif}

    with orchestrator.Orchestrator(
        orchestrator_conf={"mode": "sequential"}
    ) as cars_orchestrator:
        point_cloud_algo.generate_point_clouds(
            list_epi_pc, cars_orchestrator, tile_size=1000
        )


@pytest.mark.unit_tests
def test_get_min_max_band():
    """
    test get_min_max_band
    """

    data_tif = generate_test_inputs()

    x_y_min_max = point_cloud_wrappers.get_min_max_band(
        data_tif[cst.X],
        data_tif[cst.Y],
        data_tif[cst.Z],
        4326,
        32636,
        window=rio.windows.Window.from_slices((200, 300), (200, 300)),
    )
    assert np.allclose(
        x_y_min_max,
        [
            319922.5918467394,
            320001.11206965527,
            3317814.25094218,
            3317873.400102443,
        ],
    )


@pytest.mark.unit_tests
def test_transform_input_pc_and_metrics():
    """
    test transform_input_pc
    test compute_max_nb_point_clouds
    test compute_average_distance
    """

    # test transform_input_pc

    data_tif = generate_test_inputs()
    list_epi_pc = {"pc_0": data_tif, "pc_1": data_tif}

    with orchestrator.Orchestrator(
        orchestrator_conf={"mode": "sequential"}
    ) as cars_orchestrator:
        (
            terrain_bbox,
            list_epipolar_point_clouds_by_tiles,
        ) = point_cloud_algo.transform_input_pc(
            list_epi_pc,
            32636,
            epipolar_tile_size=200,
            orchestrator=cars_orchestrator,
        )

    assert terrain_bbox == [
        319796.54507901485,
        3317678.368442808,
        320316.9220161168,
        3318157.187469204,
    ]

    assert len(list_epipolar_point_clouds_by_tiles) == 2

    assert list_epipolar_point_clouds_by_tiles[0].shape == (5, 5)

    # tes compute_max_nb_point_clouds

    nb_max_nb_pc = point_cloud_wrappers.compute_max_nb_point_clouds(
        list_epipolar_point_clouds_by_tiles
    )

    assert nb_max_nb_pc == 2

    # test compute_average_distance
    average_dist = point_cloud_wrappers.compute_average_distance(
        list_epipolar_point_clouds_by_tiles
    )

    assert int(100 * average_dist) / 100 == 0.46


@pytest.mark.unit_tests
def test_transform_input_pc_and_correspondance():
    """
    test transform_input_pc
    test get_tiles_corresponding_tiles_tif
    """

    # test transform_input_pc

    data_tif = generate_test_inputs()
    list_epi_pc = {"pc_0": data_tif, "pc_1": data_tif}

    with orchestrator.Orchestrator(
        orchestrator_conf={"mode": "sequential"}
    ) as cars_orchestrator:
        (
            terrain_bbox,
            list_epipolar_point_clouds_by_tiles,
        ) = point_cloud_algo.transform_input_pc(
            list_epi_pc,
            32636,
            epipolar_tile_size=200,
            orchestrator=cars_orchestrator,
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

    with orchestrator.Orchestrator(
        orchestrator_conf={"mode": "sequential"}
    ) as cars_orchestrator:
        corresponding_tiles = point_cloud_algo.get_corresponding_tiles_tif(
            terrain_tiling_grid,
            list_epipolar_point_clouds_by_tiles,
            margins=0,
            orchestrator=cars_orchestrator,
        )

    assert len(corresponding_tiles[0, 0]["required_point_clouds"]) == 8
    assert len(corresponding_tiles[1, 0]["required_point_clouds"]) == 14
    assert len(corresponding_tiles[2, 2]["required_point_clouds"]) == 8
    assert len(corresponding_tiles[1, 2]["required_point_clouds"]) == 12
