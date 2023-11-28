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
Tiling module:
contains functions related to regions and tiles management
"""
# pylint: disable=too-many-lines

import logging

# Standard imports
import math
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
from pyproj import CRS
from scipy.ndimage import generic_filter
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial import tsearch  # pylint: disable=no-name-in-module
from shapely.geometry import box, mapping
from shapely.geometry.multipolygon import MultiPolygon


def grid(
    xmin: float, ymin: float, xmax: float, ymax: float, xsplit: int, ysplit: int
) -> np.ndarray:
    """
    Generate grid of positions by splitting [xmin, xmax]x[ymin, ymax]
        in splits of xsplit x ysplit size

    :param xmin : xmin of the bounding box of the region to split
    :param ymin : ymin of the bounding box of the region to split
    :param xmax : xmax of the bounding box of the region to split
    :param ymax : ymax of the bounding box of the region to split
    :param xsplit: width of splits
    :param ysplit: height of splits

    :return: The output ndarray grid with nb_ysplits splits in first direction
             and nb_xsplits in second direction for 2 dimensions 0:x, 1:y
    :rtype: numpy array
    """
    nb_xsplits = math.ceil((xmax - xmin) / xsplit)
    nb_ysplits = math.ceil((ymax - ymin) / ysplit)

    out_grid = np.ndarray(
        shape=(nb_ysplits + 1, nb_xsplits + 1, 2), dtype=float
    )

    for i in range(0, nb_xsplits + 1):
        for j in range(0, nb_ysplits + 1):
            out_grid[j, i, 0] = min(xmax, xmin + i * xsplit)
            out_grid[j, i, 1] = min(ymax, ymin + j * ysplit)

    return out_grid


def transform_four_layers_to_two_layers_grid(tiling_grid, terrain=False):
    """
    Transform a 4 layer grid: (N, M, 4) containing
        [rowmin, rowmax, colmin, colmax] when epipolar
        and [xmin, xmax, ymin, ymax]
        with x = col and y = row
        into a 2 layer grid: (N+1, M+1, 2) containing x and y
        defined like : grid[j, i, 0] = min(xmax, xmin + i * xsplit)
        and grid[j, i, 1] = min(ymax, ymin + j * ysplit)

    :param tiling_grid: tiling grid
    :type tiling_grid: np.ndarray

    :return: 2D grid
    :rtype: np.ndarray
    """

    if terrain is False:
        tiling_grid_ = tiling_grid.copy()
        tiling_grid_[:, :, [0, 1, 2, 3]] = tiling_grid_[:, :, [2, 3, 0, 1]]
    else:
        tiling_grid_ = tiling_grid.transpose(1, 0, 2).copy()

    arr = np.ndarray(
        shape=(tiling_grid_.shape[0] + 1, tiling_grid_.shape[1] + 1, 2),
        dtype=float,
    )

    # Fill x
    arr[0:-1, 0:-1, 0] = tiling_grid_[:, :, 0]
    arr[0:-1, -1, 0] = tiling_grid_[:, -1, 1]
    arr[-1, :, 0] = arr[0, :, 0]  # All rows are identical

    # Fill y
    arr[0:-1, 0:-1, 1] = tiling_grid_[:, :, 2]
    arr[-1, 0:-1, 1] = tiling_grid_[-1, :, 3]
    arr[:, -1, 1] = arr[:, 0, 1]  # All cols are identical

    return arr


def transform_disp_range_grid_to_two_layers(disp_min_grid, disp_max_grid):
    """
    Transform tiling disp min and max to N+1 M+1 array corresponding
        to N+1, M+1 , 2  tiling grid

    :param disp_min_grid: disp min tiling
    :type disp_min_grid: np ndarray
    :param disp_max_grid: disp max tiling
    :type disp_max_grid: np ndarray

    :return: disp_min_grid, disp_max_grid
    :rtype: np ndarray , np ndarray
    """

    # Create a 2xN+1, 2xM+1 matrix to apply filter on it
    nb_row = 2 * disp_min_grid.shape[0] + 1
    nb_col = 2 * disp_min_grid.shape[1] + 1

    disp_min = np.full((nb_row, nb_col), np.nan)
    disp_max = np.full((nb_row, nb_col), np.nan)

    disp_min[1::2, 1::2] = disp_min_grid
    disp_max[1::2, 1::2] = disp_max_grid

    # Apply filter min and max:
    # as each cell represent a node of 4 tiles from a regular grid
    # we want for each node to
    # represent the min and max of 4 cells

    disp_min = generic_filter(disp_min, np.nanmin, [3, 3])
    disp_max = generic_filter(disp_max, np.nanmax, [3, 3])

    # eliminate odd indexes
    disp_min = disp_min[::2, ::2]
    disp_max = disp_max[::2, ::2]

    return disp_min, disp_max


def generate_tiling_grid(
    row_min: float,
    col_min: float,
    row_max: float,
    col_max: float,
    row_split: int,
    col_split: int,
) -> np.ndarray:
    """
    Generate grid of positions by splitting [row_min, row_max] x
     [col_min, col_max]
     in splits of row_split x col_split size

    :param row_min : row_min of the bounding box of the region to split
    :param col_min : col_min of the bounding box of the region to split
    :param row_max : row_max of the bounding box of the region to split
    :param col_max : col_max of the bounding box of the region to split
    :param row_split: height of splits
    :param col_split: width of splits

    :return: The output ndarray grid with nb_row_split splits in first direction
             and nb_col_split in second direction for 2 dimensions 0:y, 1:x
             [row, col, :] containing [row_min, row_max, col_min, col_max]
    :rtype: numpy array
    """

    nb_col_split = math.ceil((col_max - col_min) / col_split)
    nb_row_split = math.ceil((row_max - row_min) / row_split)

    out_grid = np.ndarray(shape=(nb_row_split, nb_col_split, 4), dtype=float)

    for row in range(0, nb_row_split):
        for col in range(0, nb_col_split):
            out_grid[row, col, 0] = min(row_max, row_min + row * row_split)
            out_grid[row, col, 1] = min(
                row_max, row_min + (row + 1) * row_split
            )
            out_grid[row, col, 2] = min(col_max, col_min + col * col_split)
            out_grid[row, col, 3] = min(
                col_max, col_min + (col + 1) * col_split
            )

    return out_grid


def split(xmin, ymin, xmax, ymax, xsplit, ysplit):
    """
    Split a region defined by [xmin, xmax] x [ymin, ymax]
        in splits of xsplit x ysplit size

    :param xmin : xmin of the bounding box of the region to split
    :type xmin: float
    :param ymin : ymin of the bounding box of the region to split
    :type ymin: float
    :param xmax : xmax of the bounding box of the region to split
    :type xmax: float
    :param ymax : ymax of the bounding box of the region to split
    :type ymax: float
    :param xsplit: width of splits
    :type xsplit: int
    :param ysplit: height of splits
    :type ysplit: int

    :return: A list of splits represented
             by arrays of 4 elements [xmin, ymin, xmax, ymax]
    :rtype: list of 4 float
    """
    nb_xsplits = math.ceil((xmax - xmin) / xsplit)
    nb_ysplits = math.ceil((ymax - ymin) / ysplit)

    terrain_regions = []

    for i in range(0, nb_xsplits):
        for j in range(0, nb_ysplits):
            region = [
                xmin + i * xsplit,
                ymin + j * ysplit,
                xmin + (i + 1) * xsplit,
                ymin + (j + 1) * ysplit,
            ]

            # Crop to largest region
            region = crop(region, [xmin, ymin, xmax, ymax])

            terrain_regions.append(region)

    return terrain_regions


def crop(region1, region2):
    """
    Crop a region by another one

    :param region1: The region to crop as an array [xmin, ymin, xmax, ymax]
    :type region1: list of four float
    :param region2: The region used for cropping
           as an array [xmin, ymin, xmax, ymax]
    :type region2: list of four float

    :return: The cropped regiona as an array [xmin, ymin, xmax, ymax].
             If region1 is outside region2, might result in inconsistent region
    :rtype: list of four float
    """
    out = region1[:]
    out[0] = min(region2[2], max(region2[0], region1[0]))
    out[2] = min(region2[2], max(region2[0], region1[2]))
    out[1] = min(region2[3], max(region2[1], region1[1]))
    out[3] = min(region2[3], max(region2[1], region1[3]))

    return out


def pad(region, margins):
    """
    Pad region according to a margin

    :param region: The region to pad
    :type region: list of four floats
    :param margins: Margin to add
    :type margins: list of four floats
    :return: padded region
    :rtype: list of four float
    """
    out = region[:]
    out[0] -= margins[0]
    out[1] -= margins[1]
    out[2] += margins[2]
    out[3] += margins[3]

    return out


def empty(region):
    """
    Check if a region is empty or inconsistent

    :param region: region as an array [xmin, ymin, xmax, ymax]
    :type region: list of four float
    :return: True if the region is considered empty (no pixels inside),
             False otherwise
    :rtype: bool"""
    return region[0] >= region[2] or region[1] >= region[3]


def union(regions):
    """
    Returns union of all regions

    :param regions: list of region as an array [xmin, ymin, xmax, ymax]
    :type regions: list of list of four float
    :return: xmin, ymin, xmax, ymax
    :rtype: list of 4 float
    """

    xmin = min((r[0] for r in regions))
    xmax = max((r[2] for r in regions))
    ymin = min((r[1] for r in regions))
    ymax = max((r[3] for r in regions))

    return xmin, ymin, xmax, ymax


def list_tiles(region, largest_region, tile_size, margin=1):
    """
    Given a region, cut largest_region into tiles of size tile_size
    and return tiles that intersect region within margin pixels.

    :param region: The region to list intersecting tiles
    :type region: list of four float
    :param largest_region: The region to split
    :type largest_region: list of four float
    :param tile_size: Width of tiles for splitting (squared tiles)
    :type tile_size: int
    :param margin: Also include margin neighboring tiles
    :type margin: int
    :return: A list of tiles as dicts containing idx and idy
    :rtype: list of dict
    """
    # Find tile indices covered by region
    min_tile_idx_x = int(math.floor(region[0] / tile_size))
    max_tile_idx_x = int(math.ceil(region[2] / tile_size))
    min_tile_idx_y = int(math.floor(region[1] / tile_size))
    max_tile_idx_y = int(math.ceil(region[3] / tile_size))

    # Include additional tiles
    min_tile_idx_x -= margin
    min_tile_idx_y -= margin
    max_tile_idx_x += margin
    max_tile_idx_y += margin

    out = []

    # Loop on tile idx
    for tile_idx_x in range(min_tile_idx_x, max_tile_idx_x):
        for tile_idx_y in range(min_tile_idx_y, max_tile_idx_y):
            # Derive tile coordinates
            tile = [
                tile_idx_x * tile_size,
                tile_idx_y * tile_size,
                (tile_idx_x + 1) * tile_size,
                (tile_idx_y + 1) * tile_size,
            ]

            # Crop to largest region
            tile = crop(tile, largest_region)

            # Check if tile is empty
            if not empty(tile):
                out.append({"idx": tile_idx_x, "idy": tile_idx_y, "tile": tile})

    return out


def roi_to_start_and_size(region, resolution):
    """
    Convert roi as array of [xmin, ymin, xmax, ymax]
    to xmin, ymin, xsize, ysize given a resolution

    Beware that a negative spacing is considered for y axis,
    and thus returned ystart is in fact ymax

    :param region: The region to convert
    :type region: list of four float
    :param resolution: The resolution to use to determine sizes
    :type resolution: float
    :return: xstart, ystart, xsize, ysize tuple
    :rtype: list of two float + two int
    """
    xstart = region[0]
    ystart = region[3]
    xsize = int(np.round((region[2] - region[0]) / resolution))
    ysize = int(np.round((region[3] - region[1]) / resolution))

    return xstart, ystart, xsize, ysize


def snap_to_grid(xmin, ymin, xmax, ymax, resolution):
    """
    Given a roi as xmin, ymin, xmax, ymax, snap values to entire step
    of resolution

    :param xmin: xmin of the roi
    :type xmin: float
    :param ymin: ymin of the roi
    :type ymin: float
    :param xmax: xmax of the roi
    :type xmax: float
    :param ymax: ymax of the roi
    :type ymax: float
    :param resolution: size of cells for snapping
    :type resolution: float
    :return: xmin, ymin, xmax, ymax snapped tuple
    :rtype: list of four float
    """
    xmin = math.floor(xmin / resolution) * resolution
    xmax = math.ceil(xmax / resolution) * resolution
    ymin = math.floor(ymin / resolution) * resolution
    ymax = math.ceil(ymax / resolution) * resolution

    return xmin, ymin, xmax, ymax


def filter_simplices_on_the_edges(
    original_grid_shape: Tuple, tri: Delaunay, simplices: np.ndarray
):
    """
    Filter simplices on the edges which allows to cut triangles out of the
    concave Delaunay triangulation.

    :param original_grid_shape: shape of the original grid (almost regular) used
           to create delaunay triangulation
    :param tri: Delaunay triangulation
    :param simplices: Selected simplices to filter: set -1 if selected simplex
           is on the edges
    """

    # Filter simplices on the edges
    edges = np.zeros((4, *original_grid_shape))

    # left, bottom, right, top
    edges[0, :, 0] = 1
    edges[1, -1, :] = 1
    edges[2, :, -1] = 1
    edges[3, 0, :] = 1

    for idx in range(edges.shape[0]):
        edges_ravel = np.ravel(edges[idx, :, :])
        # simplices filtered if all points are on an edge
        edges_simplices = np.sum(edges_ravel[tri.simplices], axis=1) == 3
        simplices[edges_simplices[simplices]] = -1


def terrain_grid_to_epipolar(
    terrain_tiling_grid,
    epipolar_tiling_grid,
    epipolar_grid_min,
    epipolar_grid_max,
    epsg,
):
    """
    Transform terrain grid to epipolar region
    """

    terrain_regions_grid = transform_four_layers_to_two_layers_grid(
        terrain_tiling_grid, terrain=True
    )
    epipolar_regions_grid = transform_four_layers_to_two_layers_grid(
        epipolar_tiling_grid
    )

    epipolar_regions_grid_shape = np.shape(epipolar_regions_grid)[:2]
    epipolar_regions_grid_flat = epipolar_regions_grid.reshape(
        -1, epipolar_regions_grid.shape[-1]
    )

    # in the following code a factor is used to increase the precision
    spatial_ref = CRS.from_epsg(epsg)
    if spatial_ref.is_geographic:
        precision_factor = 1000.0
    else:
        precision_factor = 1.0

    # Build delaunay_triangulation
    tri_min = Delaunay(epipolar_grid_min * precision_factor)
    tri_max = Delaunay(epipolar_grid_max * precision_factor)

    # Build kdtrees
    tree_min = cKDTree(epipolar_grid_min * precision_factor)
    tree_max = cKDTree(epipolar_grid_max * precision_factor)

    # Look-up terrain_regions_grid with Delaunay
    s_min = tsearch(tri_min, terrain_regions_grid * precision_factor)
    s_max = tsearch(tri_max, terrain_regions_grid * precision_factor)

    # Filter simplices on the edges
    filter_simplices_on_the_edges(epipolar_regions_grid_shape, tri_min, s_min)
    filter_simplices_on_the_edges(epipolar_regions_grid_shape, tri_max, s_max)

    points_disp_min = epipolar_regions_grid_flat[tri_min.simplices[s_min]]

    points_disp_max = epipolar_regions_grid_flat[tri_max.simplices[s_max]]

    nn_disp_min = epipolar_regions_grid_flat[
        tree_min.query(terrain_regions_grid * precision_factor)[1]
    ]

    nn_disp_max = epipolar_regions_grid_flat[
        tree_max.query(terrain_regions_grid * precision_factor)[1]
    ]

    points_disp_min_min = np.min(points_disp_min, axis=2)
    points_disp_min_max = np.max(points_disp_min, axis=2)
    points_disp_max_min = np.min(points_disp_max, axis=2)
    points_disp_max_max = np.max(points_disp_max, axis=2)

    # Use either Delaunay search or NN search
    # if delaunay search fails (point outside triangles)
    points_disp_min_min = np.where(
        np.stack((s_min, s_min), axis=-1) != -1,
        points_disp_min_min,
        nn_disp_min,
    )

    points_disp_min_max = np.where(
        np.stack((s_min, s_min), axis=-1) != -1,
        points_disp_min_max,
        nn_disp_min,
    )

    points_disp_max_min = np.where(
        np.stack((s_max, s_max), axis=-1) != -1,
        points_disp_max_min,
        nn_disp_max,
    )

    points_disp_max_max = np.where(
        np.stack((s_max, s_max), axis=-1) != -1,
        points_disp_max_max,
        nn_disp_max,
    )

    points = np.stack(
        (
            points_disp_min_min,
            points_disp_min_max,
            points_disp_max_min,
            points_disp_max_max,
        ),
        axis=0,
    )

    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)

    return points_min, points_max


def region_hash_string(region: Tuple):
    """
    This lambda will allow to derive a key
    to index region in the previous dictionary

    :param region: region to hash
    """
    return "{}_{}_{}_{}".format(region[0], region[1], region[2], region[3])


def get_corresponding_tiles_row_col(
    terrain_tiling_grid: np.ndarray,
    row: int,
    col: int,
    list_points_clouds: list,
    list_epipolar_points_min: list,
    list_epipolar_points_max: list,
) -> Tuple[List, List, List]:
    """
    This function allows to get required points cloud for each
    terrain region.

    :param terrain_tiling_grid: terrain grid positions
    :param row: row
    :param col: column
           epipolar input tiles where keys are image pairs index and values are
           epipolar_points_min, epipolar_points_max, largest_epipolar_region,
           opt_epipolar_tile_size

    :return: Terrain regions
             Corresponding tiles selected from delayed_point_clouds with
             associated id
             Terrain regions "rank" allowing to sorting tiles for dask
             processing
    """

    logging.debug(
        "Processing tile located at {},{} in tile grid".format(row, col)
    )

    # Terrain grid [row, j, :] = [xmin, xmax, ymin, ymax]
    # terrain region = [xmin, ymin, xmax, ymax]
    terrain_region = [
        terrain_tiling_grid[row, col, 0],
        terrain_tiling_grid[row, col, 2],
        terrain_tiling_grid[row, col, 1],
        terrain_tiling_grid[row, col, 3],
    ]

    # reverse convention as row and col correspond to new format
    # Former format is transposed
    row, col = col, row

    logging.debug("Corresponding terrain region: {}".format(terrain_region))

    # This list will hold the required points clouds for this terrain tile
    required_point_clouds = []

    # This list contains indexes of tiles (debug purpose)
    list_indexes = []

    # For each stereo configuration
    for pc_id, (
        points_cloud,
        epipolar_points_min,
        epipolar_points_max,
    ) in enumerate(
        zip(  # noqa: B905
            list_points_clouds,
            list_epipolar_points_min,
            list_epipolar_points_max,
        )
    ):
        largest_epipolar_region = points_cloud.attributes[
            "largest_epipolar_region"
        ]
        opt_epipolar_tile_size = points_cloud.attributes[
            "opt_epipolar_tile_size"
        ]

        tile_min = np.minimum(
            np.minimum(
                np.minimum(
                    epipolar_points_min[row, col],
                    epipolar_points_min[row + 1, col],
                ),
                np.minimum(
                    epipolar_points_min[row + 1, col + 1],
                    epipolar_points_min[row, col + 1],
                ),
            ),
            np.minimum(
                np.minimum(
                    epipolar_points_max[row, col],
                    epipolar_points_max[row + 1, col],
                ),
                np.minimum(
                    epipolar_points_max[row + 1, col + 1],
                    epipolar_points_max[row, col + 1],
                ),
            ),
        )

        tile_max = np.maximum(
            np.maximum(
                np.maximum(
                    epipolar_points_min[row, col],
                    epipolar_points_min[row + 1, col],
                ),
                np.maximum(
                    epipolar_points_min[row + 1, col + 1],
                    epipolar_points_min[row, col + 1],
                ),
            ),
            np.maximum(
                np.maximum(
                    epipolar_points_max[row, col],
                    epipolar_points_max[row + 1, col],
                ),
                np.maximum(
                    epipolar_points_max[row + 1, col + 1],
                    epipolar_points_max[row, col + 1],
                ),
            ),
        )

        # Bounding region of corresponding cell
        epipolar_region_minx = tile_min[0]
        epipolar_region_miny = tile_min[1]
        epipolar_region_maxx = tile_max[0]
        epipolar_region_maxy = tile_max[1]

        epipolar_region = [
            epipolar_region_minx,
            epipolar_region_miny,
            epipolar_region_maxx,
            epipolar_region_maxy,
        ]

        # Crop epipolar region to largest region
        epipolar_region = crop(epipolar_region, largest_epipolar_region)

        logging.debug(
            "Corresponding epipolar region: {}".format(epipolar_region)
        )

        # Check if the epipolar region contains any pixels to process
        if empty(epipolar_region):
            logging.debug(
                "Skipping terrain region "
                "because corresponding epipolar region is empty"
            )
        else:
            # Loop on all epipolar tiles covered by epipolar region
            for epipolar_tile in list_tiles(
                epipolar_region,
                largest_epipolar_region,
                opt_epipolar_tile_size,
            ):
                id_x = epipolar_tile["idx"]
                id_y = epipolar_tile["idy"]

                epi_grid_shape = points_cloud.tiling_grid.shape

                if (
                    0 <= id_x < epi_grid_shape[1]
                    and 0 <= id_y < epi_grid_shape[0]
                ):
                    required_point_clouds.append(
                        (points_cloud[id_y, id_x], pc_id)
                    )
                    list_indexes.append([id_y, id_x])

    rank = col * col + row * row

    return (
        terrain_region,
        required_point_clouds,
        rank,
        list_indexes,
    )


def get_paired_regions_as_geodict(
    terrain_regions: List, epipolar_regions: List
) -> Tuple[Dict, Dict]:
    """
    Get paired regions (terrain/epipolar) as "geodictionnaries": these
    objects can be dumped into geojson files to be visualized.

    :param terrain_regions: terrain region respecting cars tiling
    :param epipolar_regions: corresponding epipolar regions

    :return: Terrain dictionary and Epipolar dictionary containing
             respectively Terrain tiles in terrain projection and Epipolar tiles
             in epipolar projection
    """

    ter_geodict = {"type": "FeatureCollection", "features": []}
    epi_geodict = {"type": "FeatureCollection", "features": []}

    for idx, (ter, epi_list) in enumerate(
        zip(terrain_regions, epipolar_regions)  # noqa: B905
    ):
        feature = {}
        feature["type"] = "Feature"
        feature["properties"] = {"id": idx, "nb_epi": len(epi_list)}
        feature["geometry"] = mapping(box(*ter))
        ter_geodict["features"].append(feature.copy())
        feature["geometry"] = mapping(MultiPolygon(box(*x) for x in epi_list))

        epi_geodict["features"].append(feature.copy())

    return ter_geodict, epi_geodict
