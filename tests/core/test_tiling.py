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

import json
import os
import tempfile

# Third party imports
import fiona
import numpy as np
import pytest
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from scipy.spatial import tsearch  # pylint: disable=no-name-in-module

from cars.applications.grid_generation import grids

# CARS imports
from cars.core import tiling
from cars.data_structures import cars_dataset, format_transformation

# CARS Tests import
from ..helpers import get_geometry_loader, temporary_dir


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

    def remove_tile(tiles):
        """
        Remove tile key in dict
        """
        for tile in tiles:
            del tile["tile"]

    tiles = tiling.list_tiles(region, largest_region, tile_size, margin=0)
    remove_tile(tiles)

    assert tiles == [
        {"idx": 4, "idy": 6},
        {"idx": 4, "idy": 7},
        {"idx": 5, "idy": 6},
        {"idx": 5, "idy": 7},
    ]

    tiles = tiling.list_tiles(region, largest_region, tile_size, margin=1)
    remove_tile(tiles)

    assert tiles == [
        {"idx": 3, "idy": 5},
        {"idx": 3, "idy": 6},
        {"idx": 3, "idy": 7},
        {"idx": 3, "idy": 8},
        {"idx": 4, "idy": 5},
        {"idx": 4, "idy": 6},
        {"idx": 4, "idy": 7},
        {"idx": 4, "idy": 8},
        {"idx": 5, "idy": 5},
        {"idx": 5, "idy": 6},
        {"idx": 5, "idy": 7},
        {"idx": 5, "idy": 8},
        {"idx": 6, "idy": 5},
        {"idx": 6, "idy": 6},
        {"idx": 6, "idy": 7},
        {"idx": 6, "idy": 8},
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
    out_region = tiling.terrain_region_to_epipolar(
        get_geometry_loader(), region, configuration
    )
    assert out_region == [0.0, 0.0, 612.0, 400.0]


# function parameters are fixtures set in conftest.py
@pytest.mark.unit_tests
@pytest.mark.parametrize(
    ",".join(["terrain_tile_size", "epipolar_tile_size", "nb_corresp_tiles"]),
    [[500, 612, 1], [45, 70, 15]],
)
def test_tiles_pairing(
    terrain_tile_size,
    epipolar_tile_size,
    nb_corresp_tiles,
    images_and_grids_conf,  # pylint: disable=redefined-outer-name
    disparities_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test terrain_grid_to_epipolar + get_corresponding_tiles
    """
    configuration = images_and_grids_conf
    configuration["preprocessing"]["output"].update(
        disparities_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )

    # fill constants with final dsm footprint
    terrain_region = [675248, 4897075, 675460.5, 4897173]
    largest_epipolar_region = [0, 0, 612, 612]
    disp_min, disp_max = -20, 15
    epsg = 32631
    terrain_grid = tiling.grid(
        *terrain_region, terrain_tile_size, terrain_tile_size
    )
    epipolar_regions_params = [
        *largest_epipolar_region,
        epipolar_tile_size,
        epipolar_tile_size,
    ]
    epipolar_regions_grid = tiling.grid(*epipolar_regions_params)

    # compute epipolar grid min max
    epipolar_grid_min, epipolar_grid_max = grids.compute_epipolar_grid_min_max(
        get_geometry_loader(),
        epipolar_regions_grid,
        epsg,
        configuration,
        disp_min,
        disp_max,
    )

    # compute points min/max epipolar corresponding to terrain grid
    points_min, points_max = tiling.terrain_grid_to_epipolar(
        terrain_grid,
        epipolar_regions_grid,
        epipolar_grid_min,
        epipolar_grid_max,
        epsg,
    )

    # Create empty manager with needed epipolar image information
    pc_left = cars_dataset.CarsDataset("arrays")
    pc_right = cars_dataset.CarsDataset("arrays")
    pc_left.tiling_grid = (
        format_transformation.tilling_grid_2_cars_dataset_grid(
            epipolar_regions_grid
        )
    )
    pc_left.generate_none_tiles()
    pc_right.tiling_grid = (
        format_transformation.tilling_grid_2_cars_dataset_grid(
            epipolar_regions_grid
        )
    )
    pc_right.generate_none_tiles()
    pc_left.attributes["largest_epipolar_region"] = largest_epipolar_region
    pc_left.attributes["opt_epipolar_tile_size"] = epipolar_tile_size

    list_points_clouds_left = [pc_left]
    list_points_clouds_right = [pc_right]
    list_epipolar_points_min = [points_min]
    list_epipolar_points_max = [points_max]

    # get epipolar tiles corresponding to the terrain grid for tile [0,0]
    (
        terrain_region,
        corresp_tiles_left,
        __,
        __,
        list_indexes,
    ) = tiling.get_corresponding_tiles_row_col(
        terrain_grid,
        0,
        0,
        list_points_clouds_left,
        list_points_clouds_right,
        list_epipolar_points_min,
        list_epipolar_points_max,
    )

    def create_region_from_grid(id_x, id_y, epi_grid):
        "create region"
        pos_1 = epi_grid[id_x, id_y]
        pos_2 = epi_grid[id_x + 1, id_y + 1]
        return [pos_1[0], pos_1[1], pos_2[0], pos_2[1]]

    epi_tiles = [
        create_region_from_grid(id_x, id_y, epipolar_regions_grid)
        for id_x, id_y in list_indexes
    ]

    # count the number of epipolar tiles for the first terrain tile
    assert len(corresp_tiles_left) == nb_corresp_tiles

    terrain_regions = [terrain_region]
    corresp_tiles = [epi_tiles]

    ter_geodict, epi_geodict = tiling.get_paired_regions_as_geodict(
        terrain_regions, corresp_tiles
    )

    # check geodict writing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        ter_filename = f"terrain_tiles_{nb_corresp_tiles}.geojson"
        epi_filename = f"epipolar_tiles_{nb_corresp_tiles}.geojson"
        # CRS for all GeoJSON is epsg:4326: to convert for QGIS:
        # > ogr2ogr -f "GeoJSON" out.geojson in.geojson \
        # > -s_srs EPSG:32631 -t_srs EPSG:4326
        with open(
            os.path.join(tmp_dir, ter_filename), "w", encoding="utf-8"
        ) as writer:
            writer.write(json.dumps(ter_geodict))
        with open(
            os.path.join(tmp_dir, epi_filename), "w", encoding="utf-8"
        ) as writer:
            writer.write(json.dumps(epi_geodict))
        for tmp_filename in [ter_filename, epi_filename]:
            with fiona.open(os.path.join(tmp_dir, tmp_filename)):
                pass


@pytest.mark.unit_tests
def test_filter_simplices_on_the_edges():
    """
    Test filter simplices on the edges
    """
    epipolar_grid = tiling.grid(0, 0, 2, 2, 1, 1)

    # shift one point to obtain a concave hull
    epipolar_grid[0, 1, 1] = 0.5
    epipolar_grid_shape = epipolar_grid.shape[:2]
    projected_epipolar = epipolar_grid.reshape(-1, 2)

    terrain_grid = np.array(
        [
            [0.25, 0.25],  # in a triangle
            [0.75, 1.25],  # in a triangle
            [0.25, 1.75],  # in a triangle
            [2.05, 1.00],  # not in a triangle
            [1.00, 0.25],  # in a "edges" triangle
        ]
    )

    tri = Delaunay(projected_epipolar)
    simplices = tsearch(tri, terrain_grid)
    original_simplices = simplices.copy()

    tiling.filter_simplices_on_the_edges(epipolar_grid_shape, tri, simplices)

    # only the last point must be filtered
    (diff_indexes,) = np.where(original_simplices != simplices)
    assert diff_indexes.tolist() == [4]
    assert simplices[4] == -1
