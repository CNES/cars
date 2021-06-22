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

import logging

# Standard imports
import math

# Third party imports
import numpy as np
from osgeo import osr
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial import tsearch  # pylint: disable=no-name-in-module
from tqdm import tqdm

# CARS imports
from cars.conf import output_prepare
from cars.steps.epi_rectif.grids import compute_epipolar_grid_min_max


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
    :returns: The output ndarray grid with nb_ysplits splits in first direction
        and nb_xsplits in second direction for 2 dimensions 0:x, 1:y
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
    :returns: A list of splits represented
        by arrays of 4 elements [xmin, ymin, xmax, ymax]
    :type list of 4 float
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
    :returns: The cropped regiona as an array [xmin, ymin, xmax, ymax].
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
    :returns: padded region
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
    :returns: True if the region is considered empty (no pixels inside),
        False otherwise
    :rtype: bool"""
    return region[0] >= region[2] or region[1] >= region[3]


def union(regions):
    """
    Returns union of all regions

    :param regions: list of region as an array [xmin, ymin, xmax, ymax]
    :type regions: list of list of four float
    :returns: xmin, ymin, xmax, ymax
    :rtype: list of 4 float
    """

    xmin = min([r[0] for r in regions])
    xmax = max([r[2] for r in regions])
    ymin = min([r[1] for r in regions])
    ymax = max([r[3] for r in regions])

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
    :returns: A list of tiles as arrays of [xmin, ymin, xmax, ymax]
    :rtype: list of 4 float
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

            # Check if tile is emtpy
            if not empty(tile):
                out.append(tile)

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
    :returns: xstart, ystart, xsize, ysize tuple
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
    :returns: xmin, ymin, xmax, ymax snapped tuple
    :type: list of four float
    """
    xmin = math.floor(xmin / resolution) * resolution
    xmax = math.ceil(xmax / resolution) * resolution
    ymin = math.floor(ymin / resolution) * resolution
    ymax = math.ceil(ymax / resolution) * resolution

    return xmin, ymin, xmax, ymax


def terrain_region_to_epipolar(
    region, conf, epsg=4326, disp_min=None, disp_max=None, step=100
):
    """
    Transform terrain region to epipolar region
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_conf = conf[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ]
    minimum_disparity = preprocessing_output_conf[
        output_prepare.MINIMUM_DISPARITY_TAG
    ]
    maximum_disparity = preprocessing_output_conf[
        output_prepare.MAXIMUM_DISPARITY_TAG
    ]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    region_grid = np.array(
        [
            [region[0], region[1]],
            [region[2], region[1]],
            [region[2], region[3]],
            [region[0], region[3]],
        ]
    )

    epipolar_grid = grid(
        0,
        0,
        preprocessing_output_conf[output_prepare.EPIPOLAR_SIZE_X_TAG],
        preprocessing_output_conf[output_prepare.EPIPOLAR_SIZE_Y_TAG],
        step,
        step,
    )

    epi_grid_flat = epipolar_grid.reshape(-1, epipolar_grid.shape[-1])

    epipolar_grid_min, epipolar_grid_max = compute_epipolar_grid_min_max(
        epipolar_grid, epsg, conf, disp_min, disp_max
    )

    # Build Delaunay triangulations
    delaunay_min = Delaunay(epipolar_grid_min)
    delaunay_max = Delaunay(epipolar_grid_max)

    # Build kdtrees
    tree_min = cKDTree(epipolar_grid_min)
    tree_max = cKDTree(epipolar_grid_max)

    # Look-up terrain grid with Delaunay
    s_min = tsearch(delaunay_min, region_grid)
    s_max = tsearch(delaunay_max, region_grid)

    points_list = []
    # For each corner
    for i in range(0, 4):
        # If we are inside triangulation of s_min
        if s_min[i] != -1:
            # Add points from surrounding triangle
            for point in epi_grid_flat[delaunay_min.simplices[s_min[i]]]:
                points_list.append(point)
        else:
            # else add nearest neighbor
            __, point_idx = tree_min.query(region_grid[i, :])
            points_list.append(epi_grid_flat[point_idx])
            # If we are inside triangulation of s_min
            if s_max[i] != -1:
                # Add points from surrounding triangle
                for point in epi_grid_flat[delaunay_max.simplices[s_max[i]]]:
                    points_list.append(point)
            else:
                # else add nearest neighbor
                __, point_nn_idx = tree_max.query(region_grid[i, :])
                points_list.append(epi_grid_flat[point_nn_idx])

    points_min = np.min(points_list, axis=0)
    points_max = np.max(points_list, axis=0)

    # Bouding region of corresponding cell
    epipolar_region_minx = points_min[0]
    epipolar_region_miny = points_min[1]
    epipolar_region_maxx = points_max[0]
    epipolar_region_maxy = points_max[1]

    # This mimics the previous code that was using
    # terrain_region_to_epipolar
    epipolar_region = [
        epipolar_region_minx,
        epipolar_region_miny,
        epipolar_region_maxx,
        epipolar_region_maxy,
    ]
    return epipolar_region


def terrain_grid_to_epipolar(terrain_grid, conf, epsg):
    """
    Transform terrain grid to epipolar region
    """
    # Compute disp_min and disp_max location for epipolar grid
    (epipolar_grid_min, epipolar_grid_max,) = compute_epipolar_grid_min_max(
        conf["epipolar_regions_grid"],
        epsg,
        conf["configuration"],
        conf["disp_min"],
        conf["disp_max"],
    )

    epipolar_regions_grid_size = np.shape(conf["epipolar_regions_grid"])[:2]
    epipolar_regions_grid_flat = conf["epipolar_regions_grid"].reshape(
        -1, conf["epipolar_regions_grid"].shape[-1]
    )

    # in the following code a factor is used to increase the precision
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)
    if spatial_ref.IsGeographic():
        precision_factor = 1000.0
    else:
        precision_factor = 1.0

    # Build delaunay_triangulation
    delaunay_min = Delaunay(epipolar_grid_min * precision_factor)
    delaunay_max = Delaunay(epipolar_grid_max * precision_factor)

    # Build kdtrees
    tree_min = cKDTree(epipolar_grid_min * precision_factor)
    tree_max = cKDTree(epipolar_grid_max * precision_factor)

    # Look-up terrain_grid with Delaunay
    s_min = tsearch(delaunay_min, terrain_grid * precision_factor)
    s_max = tsearch(delaunay_max, terrain_grid * precision_factor)

    # Filter simplices on the edges
    edges = np.ones(epipolar_regions_grid_size)
    edges[1:-1, 1:-1] = 0
    edges_ravel = np.ravel(edges)
    s_min_edges = np.sum(edges_ravel[delaunay_min.simplices], axis=1) == 3
    s_max_edges = np.sum(edges_ravel[delaunay_max.simplices], axis=1) == 3
    s_min[s_min_edges[s_min]] = -1
    s_max[s_max_edges[s_max]] = -1

    points_disp_min = epipolar_regions_grid_flat[delaunay_min.simplices[s_min]]

    points_disp_max = epipolar_regions_grid_flat[delaunay_max.simplices[s_max]]

    nn_disp_min = epipolar_regions_grid_flat[
        tree_min.query(terrain_grid * precision_factor)[1]
    ]

    nn_disp_max = epipolar_regions_grid_flat[
        tree_max.query(terrain_grid * precision_factor)[1]
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


def region_hash_string(region):
    """
    This lambda will allow to derive a key
    to index region in the previous dictionnary
    """
    return "{}_{}_{}_{}".format(region[0], region[1], region[2], region[3])


def get_corresponding_tiles(terrain_grid, configurations_data):
    """
    This function allows to get required points cloud for each
    terrain region.
    """
    terrain_regions = []
    corresponding_tiles = []
    rank = []

    number_of_terrain_splits = (terrain_grid.shape[0] - 1) * (
        terrain_grid.shape[1] - 1
    )

    logging.info(
        "Terrain bounding box will be processed in {} splits".format(
            number_of_terrain_splits
        )
    )

    # Loop on terrain regions and derive dependency to epipolar regions
    for terrain_region_dix in tqdm(
        range(number_of_terrain_splits),
        total=number_of_terrain_splits,
        desc="Delaunay look-up",
    ):

        j = int(terrain_region_dix / (terrain_grid.shape[1] - 1))
        i = terrain_region_dix % (terrain_grid.shape[1] - 1)

        logging.debug(
            "Processing tile located at {},{} in tile grid".format(i, j)
        )

        terrain_region = [
            terrain_grid[j, i, 0],
            terrain_grid[j, i, 1],
            terrain_grid[j + 1, i + 1, 0],
            terrain_grid[j + 1, i + 1, 1],
        ]

        terrain_regions.append(terrain_region)

        logging.debug("Corresponding terrain region: {}".format(terrain_region))

        # This list will hold the required points clouds for this terrain tile
        required_point_clouds = []

        # For each stereo configuration
        for _, conf in configurations_data.items():

            epipolar_points_min = conf["epipolar_points_min"]
            epipolar_points_max = conf["epipolar_points_max"]

            tile_min = np.minimum(
                np.minimum(
                    np.minimum(
                        epipolar_points_min[j, i], epipolar_points_min[j + 1, i]
                    ),
                    np.minimum(
                        epipolar_points_min[j + 1, i + 1],
                        epipolar_points_min[j, i + 1],
                    ),
                ),
                np.minimum(
                    np.minimum(
                        epipolar_points_max[j, i], epipolar_points_max[j + 1, i]
                    ),
                    np.minimum(
                        epipolar_points_max[j + 1, i + 1],
                        epipolar_points_max[j, i + 1],
                    ),
                ),
            )

            tile_max = np.maximum(
                np.maximum(
                    np.maximum(
                        epipolar_points_min[j, i], epipolar_points_min[j + 1, i]
                    ),
                    np.maximum(
                        epipolar_points_min[j + 1, i + 1],
                        epipolar_points_min[j, i + 1],
                    ),
                ),
                np.maximum(
                    np.maximum(
                        epipolar_points_max[j, i], epipolar_points_max[j + 1, i]
                    ),
                    np.maximum(
                        epipolar_points_max[j + 1, i + 1],
                        epipolar_points_max[j, i + 1],
                    ),
                ),
            )

            # Bouding region of corresponding cell
            epipolar_region_minx = tile_min[0]
            epipolar_region_miny = tile_min[1]
            epipolar_region_maxx = tile_max[0]
            epipolar_region_maxy = tile_max[1]

            # This mimics the previous code that was using
            # terrain_region_to_epipolar
            epipolar_region = [
                epipolar_region_minx,
                epipolar_region_miny,
                epipolar_region_maxx,
                epipolar_region_maxy,
            ]

            # Crop epipolar region to largest region
            epipolar_region = crop(
                epipolar_region, conf["largest_epipolar_region"]
            )

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
                    conf["largest_epipolar_region"],
                    conf["opt_epipolar_tile_size"],
                ):

                    cur_hash = region_hash_string(epipolar_tile)

                    # Look for corresponding hash in delayed point clouds
                    # dictionnary
                    if cur_hash in conf["epipolar_regions_hash"]:

                        # If hash can be found, append it to the required
                        # clouds to compute for this terrain tile
                        pos = conf["epipolar_regions_hash"].index(cur_hash)
                        required_point_clouds.append(
                            conf["delayed_point_clouds"][pos]
                        )

        corresponding_tiles.append(required_point_clouds)
        rank.append(i * i + j * j)

    return terrain_regions, corresponding_tiles, rank
