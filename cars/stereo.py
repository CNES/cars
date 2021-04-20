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
Stereo module:
contains stereo-rectification, disparity map estimation
"""

# Standard imports
from __future__ import absolute_import
from typing import Dict, Tuple
import warnings
import math
import logging

# Third party imports
import numpy as np

from scipy.spatial import Delaunay #pylint: disable=no-name-in-module
from scipy.spatial import tsearch #pylint: disable=no-name-in-module
from scipy.spatial import cKDTree #pylint: disable=no-name-in-module

import rasterio as rio
import xarray as xr
from dask import sizeof

# Cars imports
from cars.conf import input_parameters as in_params
from cars import projection
from cars import tiling
from cars.conf import output_prepare
from cars import constants as cst
from cars import matching_regularisation
from cars.lib.steps.epi_rectif.grids import compute_epipolar_grid_min_max
from cars.lib.steps.epi_rectif.resampling import epipolar_rectify_images
from cars.lib.steps import triangulation
from cars.lib.steps.matching import dense_matching

# Register sizeof for xarray
@sizeof.sizeof.register_lazy("xarray")
def register_xarray():
    """
    Add hook to dask so it correctly estimates memory used by xarray
    """
    @sizeof.sizeof.register(xr.DataArray)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataarray(xarr):
        """
        Inner function for total size of xarray_dataarray
        """
        total_size = sizeof.sizeof(xarr.values)
        for __, carray in xarr.coords.items():
            total_size += sizeof.sizeof(carray.values)
        total_size += sizeof.sizeof(xarr.attrs)
        return total_size
    @sizeof.sizeof.register(xr.Dataset)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataset(xdat):
        """
        Inner function for total size of xarray_dataset
        """
        total_size = 0
        for __, varray in xdat.data_vars.items():
            total_size += sizeof.sizeof(varray.values)
        for __, carray in xdat.coords.items():
            total_size += sizeof.sizeof(carray)
        total_size += sizeof.sizeof(xdat.attrs)
        return total_size

# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def images_pair_to_3d_points(input_stereo_cfg,
                             region,
                             corr_cfg,
                             epsg=None,
                             disp_min=None,
                             disp_max=None,
                             out_epsg=None,
                             geoid_data=None,
                             use_sec_disp=False,
                             snap_to_img1=False,
                             align=False,
                             add_msk_info=False) -> Dict[str,
                                                         Tuple[xr.Dataset,
                                                               xr.Dataset]]:
    # Retrieve disp min and disp max if needed
    """
    This function will produce a 3D points cloud as an xarray.Dataset from the
    given stereo configuration (from both left to right disparity map and right
    to left disparity map if the latter is computed by Pandora).
    Clouds will be produced over the region with the specified EPSG, using
    disp_min and disp_max
    :param input_stereo_cfg: Configuration for stereo processing
    :type StereoConfiguration
    :param region: Array defining region.

    * For espg region as [lat_min, lon_min, lat_max, lon_max]
    * For epipolar region as [xmin, ymin, xmax, ymax]

    :type region: numpy.array
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param epsg: EPSG code for the region,
                 if None then epipolar geometry is considered
    :type epsg: int
    :param disp_min: Minimum disparity value
    :type disp_min: int
    :param disp_max: Maximum disparity value
    :type disp_max: int
    :param geoid_data: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_data: xarray.Dataset
    :param use_sec_disp: Boolean activating the use of the secondary
                         disparity map
    :type use_sec_disp: bool
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so
                         as to cross those of img1
    :param snap_to_img1: bool
    :param align: If True, apply correction to point after triangulation to
                  align with lowres DEM (if available. If not, no correction
                  is applied)
    :param align: bool
    :param add_msk_info: boolean enabling the addition of the masks'
                         information in the point clouds final dataset
    :returns: Dictionary of tuple. The tuple are constructed with the dataset
              containing the 3D points +
    A dataset containing color of left image, or None

    The dictionary keys are :
        * 'ref' to retrieve the dataset built from the left to right
          disparity map
        * 'sec' to retrieve the dataset built from the right to left
          disparity map (if computed in Pandora)
    """


    # Retrieve disp min and disp max if needed
    preprocessing_output_cfg = input_stereo_cfg\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_cfg[
        output_prepare.MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_cfg[
        output_prepare.MAXIMUM_DISPARITY_TAG]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Compute margins for the correlator
    margins = dense_matching.get_margins(
        disp_min, disp_max, corr_cfg)

    # Reproject region to epipolar geometry if necessary
    if epsg is not None:
        region = transform_terrain_region_to_epipolar(
            region, input_stereo_cfg, epsg,  disp_min, disp_max)

    # Rectify images
    left, right, color = epipolar_rectify_images(input_stereo_cfg,
                                                 region,
                                                 margins)
    # Compute disparity
    # TODO: c'est pas bizarre que la conf de pandora soit donnee à
    #       dense_matching ? peut être que dense_matching
    #       sachant qu'il va faire du pandora pourrait recuperer
    #       la conf tout seul ?
    disp = dense_matching.compute_disparity(
        left, right, input_stereo_cfg, corr_cfg, disp_min, disp_max,
        use_sec_disp=use_sec_disp)

    # If necessary, set disparity to 0 for classes to be set to input dem
    mask1_classes = input_stereo_cfg \
        [in_params.INPUT_SECTION_TAG].get(in_params.MASK1_CLASSES_TAG, None)
    mask2_classes = input_stereo_cfg \
        [in_params.INPUT_SECTION_TAG].get(in_params.MASK2_CLASSES_TAG, None)
    matching_regularisation.update_disp_to_0(
        disp, left, right, mask1_classes, mask2_classes)

    colors = dict()
    colors[cst.STEREO_REF] = color
    if cst.STEREO_SEC in disp:
        # compute right color image from right-left disparity map
        colors[cst.STEREO_SEC] = dense_matching.estimate_color_from_disparity(
            disp[cst.STEREO_SEC], left, color)

    im_ref_msk = None
    im_sec_msk = None
    if add_msk_info:
        ref_values_list = [key for key, _ in left.items()]
        if cst.EPI_MSK in ref_values_list:
            im_ref_msk = left
        else:
            worker_logger = logging.getLogger('distributed.worker')
            worker_logger.warning("Left image does not have a "
                                  "mask to rasterize")
        if cst.STEREO_SEC in disp:
            sec_values_list = [key for key, _ in right.items()]
            if cst.EPI_MSK in sec_values_list:
                im_sec_msk = right
            else:
                worker_logger = logging.getLogger('distributed.worker')
                worker_logger.warning("Right image does not have a "
                                      "mask to rasterize")

    # Triangulate
    if cst.STEREO_SEC in disp:
        points = triangulation.triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF], disp[cst.STEREO_SEC],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)
    else:
        points = triangulation.triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key in points:
            points[key] = triangulation.geoid_offset(points[key], geoid_data)

    if out_epsg is not None:
        for key in points:
            projection.points_cloud_conversion_dataset(points[key], out_epsg)

    return points, colors


def transform_terrain_region_to_epipolar(
        region, conf,
        epsg = 4326,
        disp_min = None,
        disp_max = None,
        step = 100):
    """
    Transform terrain region to epipolar region according to ground_positions

    :param region: The terrain region to transform to epipolar region
                   ([lat_min, lon_min, lat_max, lon_max])
    :type region: list of four float
    :param ground_positions: Grid of ground positions for epipolar geometry
    :type ground_positions: numpy array # TODO pas a jour
    :param origin: origin of the grid # TODO pas a jour
    :type origin: list of two float # TODO pas a jour
    :param spacing: spacing of the grid # TODO pas a jour
    :type spacing: list of two float # TODO pas a jour
    :returns: The epipolar region as [xmin, ymin, xmax, ymax]
    :rtype: list of four float
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_conf = conf\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_conf[
        output_prepare.MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_conf[
        output_prepare.MAXIMUM_DISPARITY_TAG]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    region_grid = np.array([[region[0],region[1]],
                            [region[2],region[1]],
                            [region[2],region[3]],
                            [region[0],region[3]]])

    epipolar_grid = tiling.grid(0, 0,
                         preprocessing_output_conf[
                             output_prepare.EPIPOLAR_SIZE_X_TAG],
                         preprocessing_output_conf[
                             output_prepare.EPIPOLAR_SIZE_Y_TAG],
                         step,
                         step)

    epi_grid_flat = epipolar_grid.reshape(-1, epipolar_grid.shape[-1])

    epipolar_grid_min, epipolar_grid_max = compute_epipolar_grid_min_max(
        epipolar_grid, epsg, conf,disp_min, disp_max)

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
    for i in range(0,4):
        # If we are inside triangulation of s_min
        if s_min[i] != -1:
            # Add points from surrounding triangle
            for point in epi_grid_flat[delaunay_min.simplices[s_min[i]]]:
                points_list.append(point)
        else:
            # else add nearest neighbor
            __, point_idx = tree_min.query(region_grid[i,:])
            points_list.append(epi_grid_flat[point_idx])
        # If we are inside triangulation of s_min
            if s_max[i] != -1:
                # Add points from surrounding triangle
                for point in epi_grid_flat[delaunay_max.simplices[s_max[i]]]:
                    points_list.append(point)
            else:
                # else add nearest neighbor
                __, point_nn_idx = tree_max.query(region_grid[i,:])
                points_list.append(epi_grid_flat[point_nn_idx])

    points_min = np.min(points_list, axis=0)
    points_max = np.max(points_list, axis=0)

    # Bouding region of corresponding cell
    epipolar_region_minx = points_min[0]
    epipolar_region_miny = points_min[1]
    epipolar_region_maxx = points_max[0]
    epipolar_region_maxy = points_max[1]

    # This mimics the previous code that was using
    # transform_terrain_region_to_epipolar
    epipolar_region = [
        epipolar_region_minx,
        epipolar_region_miny,
        epipolar_region_maxx,
        epipolar_region_maxy]
    return epipolar_region
