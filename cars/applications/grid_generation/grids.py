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
Grids module:
contains functions used for epipolar grid creation and correction
"""

# Standard imports
from __future__ import absolute_import

import logging
import math
import os
from typing import Union

# Third party imports
import numpy as np
import pandas
import rasterio as rio
import xarray as xr
from affine import Affine

# TODO depends on another step (and a later one) : make it independent
from cars.applications.triangulation.triangulation_tools import (
    triangulate_matches,
)
from cars.core import constants as cst
from cars.core import former_confs_utils, projection, tiling

# CARS imports
from cars.core.geometry import AbstractGeometry


def get_new_path(path):
    """
    Check path, if exists, creates new one

    :param path: path to check
    :type path: str

    :return : new path
    :rtype: str
    """

    current_increment = 0
    head, tail = os.path.splitext(path)

    current_path = path
    while os.path.exists(current_path):
        current_increment += 1
        current_path = head + "_" + repr(current_increment) + tail

    return current_path


def write_grid(grid, fname, origin, spacing):
    """
    Write an epipolar rectification grid to file

    :param grid: the grid to write
    :type grid: 3D numpy array
    :param fname: the filename to which the grid will be written
    :type fname: string
    :param origin: origin of the grid
    :type origin: (float, float)
    :param spacing: spacing of the grid
    :type spacing: (float, float)
    """

    geotransform = (
        origin[0] - 0.5 * spacing[0],
        spacing[0],
        0.0,
        origin[1] - 0.5 * spacing[1],
        0.0,
        spacing[1],
    )

    transform = Affine.from_gdal(*geotransform)

    with rio.open(
        fname,
        "w",
        height=grid.shape[0],
        width=grid.shape[1],
        count=2,
        driver="GTiff",
        dtype=grid.dtype,
        transform=transform,
    ) as dst:
        dst.write_band(1, grid[:, :, 0])
        dst.write_band(2, grid[:, :, 1])


def generate_epipolar_grids(
    conf,
    geometry_loader_to_use,
    dem: Union[None, str],
    default_alt: Union[None, float] = None,
    epipolar_step: int = 30,
    geoid: Union[str, None] = None,
):
    """
    Computes the left and right epipolar grids

    :param conf: input configuration dictionary
    :param geometry_loader_to_use: geometry loader to use
    :type geometry_loader_to_use: str
    :param dem: path to the dem folder
    :param default_alt: default altitude to use in the missing dem regions
    :param epipolar_step: step to use to construct the epipolar grids
    :param geoid: path to the geoid file
    :return: Tuple composed of :

        - the left epipolar grid as a numpy array
        - the right epipolar grid as a numpy array
        - the left grid origin as a list of float
        - the left grid spacing as a list of float
        - the epipolar image size as a list of int\
        (x-axis size is given with the index 0, y-axis size with index 1)
        - the disparity to altitude ratio as a float
    """
    logging.info("Generating epipolar rectification grid ...")

    geometry_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            geometry_loader_to_use
        )
    )

    return geometry_loader.generate_epipolar_grids(
        conf,
        dem=dem,
        geoid=geoid,
        default_alt=default_alt,
        epipolar_step=epipolar_step,
    )


def compute_epipolar_grid_min_max(
    geometry_loader_to_use, grid, epsg, conf, disp_min=None, disp_max=None
):
    """
    Compute ground terrain location of epipolar grids at disp_min and disp_max

    :param geometry_loader_to_use: geometry loader to use
    :type geometry_loader_to_use: str
    :param grid: The epipolar grid to project
    :type grid: np.ndarray of shape (N,M,2)
    :param epsg: EPSG code of the terrain projection
    :type epsg: Int
    :param conf: Configuration dictionary from prepare step
    :type conf: Dict
    :param disp_min: Minimum disparity
                     (if None, read from configuration dictionary)
    :type disp_min: Float or None
    :param disp_max: Maximum disparity
                     (if None, read from configuration dictionary)
    :type disp_max: Float or None
    :return: a tuple of location grid at disp_min and disp_max
    :rtype: Tuple(np.ndarray, np.ndarray) same shape as grid param
    """

    # Retrieve disp min and disp max if needed
    (
        minimum_disparity,
        maximum_disparity,
    ) = former_confs_utils.get_disp_min_max(conf)

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Generate disp_min and disp_max matches
    matches_min = np.stack(
        (
            grid[:, :, 0].flatten(),
            grid[:, :, 1].flatten(),
            grid[:, :, 0].flatten() + disp_min,
            grid[:, :, 1].flatten(),
        ),
        axis=1,
    )
    matches_max = np.stack(
        (
            grid[:, :, 0].flatten(),
            grid[:, :, 1].flatten(),
            grid[:, :, 0].flatten() + disp_max,
            grid[:, :, 1].flatten(),
        ),
        axis=1,
    )

    # Generate corresponding points clouds
    pc_min = triangulate_matches(geometry_loader_to_use, conf, matches_min)
    pc_max = triangulate_matches(geometry_loader_to_use, conf, matches_max)

    # Convert to correct EPSG
    projection.points_cloud_conversion_dataset(pc_min, epsg)
    projection.points_cloud_conversion_dataset(pc_max, epsg)

    # Form grid_min and grid_max
    grid_min = None
    grid_max = None
    if isinstance(pc_min, xr.Dataset):
        grid_min = np.concatenate(
            (pc_min[cst.X].values, pc_min[cst.Y].values), axis=1
        )
        grid_max = np.concatenate(
            (pc_max[cst.X].values, pc_max[cst.Y].values), axis=1
        )
    elif isinstance(pc_min, pandas.DataFrame):
        grid_min = np.stack(
            (pc_min[cst.X].to_numpy(), pc_min[cst.Y].to_numpy()), axis=-1
        )
        grid_max = np.stack(
            (pc_max[cst.X].to_numpy(), pc_max[cst.Y].to_numpy()), axis=-1
        )
    else:
        logging.error("pc min/max error: point cloud is unknown")

    return grid_min, grid_max


def terrain_region_to_epipolar(
    region,
    conf,
    geometry_loader,
    epsg=4326,
    disp_min=0,
    disp_max=0,
    tile_size=100,
    epipolar_size_x=None,
    epipolar_size_y=None,
):
    """
    Transform terrain region to epipolar region

    :param region: terrain region to use
    :param conf: config with epipolar grids infos
    :param geometry_loader: geometry loader to use
    :param epsg: epsg
    :param disp_min: minimum disparity
    :param disp_max: maximum disparity
    :param tile_size: tile size for grid
    :param epipolar_size_x: epipolar_size_x
    :param epipolar_size_y: epipolar_size_y

    :return: epipolar region to use, with tile_size a sample
    """

    disp_min = int(disp_min)
    disp_max = int(disp_max)

    # Generate terrain grid only on roi
    xmin = region[0]
    xmax = region[2]
    ymin = region[1]
    ymax = region[3]
    opt_terrain_size = max((xmax - xmin), (ymax - ymin))
    region_grid = tiling.generate_tiling_grid(
        xmin,
        ymin,
        xmax,
        ymax,
        opt_terrain_size,
        opt_terrain_size,
    )

    # Generate fake epipolar grid
    epipolar_grid = tiling.generate_tiling_grid(
        0,
        0,
        epipolar_size_y,
        epipolar_size_x,
        tile_size,
        tile_size,
    )

    # Compute disp_min and disp_max location for epipolar grid
    (
        epipolar_grid_min,
        epipolar_grid_max,
    ) = compute_epipolar_grid_min_max(
        geometry_loader,
        tiling.transform_four_layers_to_two_layers_grid(epipolar_grid),
        epsg,
        conf,
        disp_min,
        disp_max,
    )

    # Compute epipolar points min and max on terrain region
    points_min, points_max = tiling.terrain_grid_to_epipolar(
        region_grid, epipolar_grid, epipolar_grid_min, epipolar_grid_max, epsg
    )

    # Bouding region of corresponding cell
    epipolar_region_minx = np.min(points_min[:, :, 0])
    epipolar_region_miny = np.min(points_min[:, :, 1])
    epipolar_region_maxx = np.max(points_max[:, :, 0])
    epipolar_region_maxy = np.max(points_max[:, :, 1])

    # Generate epipolar region
    epipolar_region = [
        epipolar_region_miny,
        epipolar_region_maxy,
        epipolar_region_minx,
        epipolar_region_maxx,
    ]

    return epipolar_region
