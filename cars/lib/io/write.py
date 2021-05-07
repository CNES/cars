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
io.write module:
contains write grid functions used during CARS prepare pipeline step
"""
# TODO : to move with epipolar_steps ?

# Standard imports
from __future__ import absolute_import

# Third party imports
import rasterio as rio
from affine import Affine


def write_grid(grid, fname, origin, spacing):
    """
    Write an epipolar resampling grid to file

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
        spacing[1])

    transform = Affine.from_gdal(*geotransform)

    with rio.open(fname, 'w', height=grid.shape[0],
                  width=grid.shape[1], count=2, driver='GTiff',
                  dtype=grid.dtype, transform=transform)\
        as dst:
        dst.write_band(1, grid[:, :, 0])
        dst.write_band(2, grid[:, :, 1])
