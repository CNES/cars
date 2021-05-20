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
CARS pipelines/wrapper module:
"""

# Standard imports
import logging
import math
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
import xarray as xr

# CARS imports
from cars.conf import (
    input_parameters,
    mask_classes,
    output_prepare,
    static_conf,
)
from cars.core import constants as cst
from cars.core import projection, tiling
from cars.steps import triangulation
from cars.steps.epi_rectif import resampling
from cars.steps.matching import dense_matching, regularisation
from cars.steps.sparse_matching import sift


def matching_wrapper(
    left_region: List[float],
    right_region: List[float],
    img1: str,
    img2: str,
    grid1: str,
    grid2: str,
    mask1: str,
    mask2: str,
    mask1_classes: str,
    mask2_classes: str,
    nodata1: float,
    nodata2: float,
    epipolar_size_x: int,
    epipolar_size_y: int,
) -> np.ndarray:
    """
    Wrapper for matching step in prepare

    It performs epipolar resampling of both images and returns matches

    :param left_region: Region of img1 to process
    :param right_region: Region of img2 to process
    :param img1: path to first image
    :param img2: path to second image
    :param grid1: path to epipolar resampling grid for first image
    :param grid2: path to epipolar resampling grid for second image
    :param mask1: path to mask for first image, or None
    :param mask2: path to mask for second image, or None
    :param mask1_classes: path to the mask1's classes usage json file
    :param mask2_classes: path to the mask2's classes usage json file
    :param nodata1: nodata value for first image
    :param nodata2: nodata value for second image
    :param epipolar_size_x: size of epipolar images in x dimension
    :param epipolar_size_y: size of epipolar images in x dimension
    :rtype: matches as a np.array of shape (nb_matches,4)
    """
    worker_logger = logging.getLogger("distributed.worker")
    worker_logger.debug("Matching keypoints on region {}".format(left_region))

    largest_size = [epipolar_size_x, epipolar_size_y]

    # Resample left dataset
    left_ds = resampling.resample_image(
        img1,
        grid1,
        largest_size,
        region=left_region,
        nodata=nodata1,
        mask=mask1,
    )

    # handle multi classes mask if necessary
    if mask1_classes is not None:
        left_ds[cst.EPI_MSK].values = mask_classes.create_msk_from_tag(
            left_ds[cst.EPI_MSK].values,
            mask1_classes,
            mask_classes.ignored_by_sift_matching_tag,
            mask_intern_no_data_val=True,
        )

    # Resample right dataset
    right_ds = resampling.resample_image(
        img2,
        grid2,
        largest_size,
        region=right_region,
        nodata=nodata2,
        mask=mask2,
    )

    # handle multi classes mask if necessary
    if mask2_classes is not None:
        right_ds[cst.EPI_MSK].values = mask_classes.create_msk_from_tag(
            right_ds[cst.EPI_MSK].values,
            mask2_classes,
            mask_classes.ignored_by_sift_matching_tag,
            mask_intern_no_data_val=True,
        )

    # Perform matching
    sift_params = static_conf.get_sift_params()
    matches = sift.dataset_matching(
        left_ds,
        right_ds,
        matching_threshold=getattr(
            sift_params, static_conf.sift_matching_threshold_tag
        ),
        n_octave=getattr(sift_params, static_conf.sift_n_octave_tag),
        n_scale_per_octave=getattr(
            sift_params, static_conf.sift_n_scale_per_octave_tag
        ),
        dog_threshold=getattr(sift_params, static_conf.sift_dog_threshold_tag),
        edge_threshold=getattr(
            sift_params, static_conf.sift_edge_threshold_tag
        ),
        magnification=getattr(sift_params, static_conf.sift_magnification_tag),
        backmatching=getattr(sift_params, static_conf.sift_back_matching_tag),
    )

    return matches


def images_pair_to_3d_points(
    input_stereo_cfg,
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
    add_msk_info=False,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
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
    preprocessing_output_cfg = input_stereo_cfg[
        output_prepare.PREPROCESSING_SECTION_TAG
    ][output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_cfg[
        output_prepare.MINIMUM_DISPARITY_TAG
    ]
    maximum_disparity = preprocessing_output_cfg[
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

    # Compute margins for the correlator
    margins = dense_matching.get_margins(disp_min, disp_max, corr_cfg)

    # Reproject region to epipolar geometry if necessary
    if epsg is not None:
        region = tiling.transform_terrain_region_to_epipolar(
            region, input_stereo_cfg, epsg, disp_min, disp_max
        )

    # Rectify images
    left, right, color = resampling.epipolar_rectify_images(
        input_stereo_cfg, region, margins
    )
    # Compute disparity
    disp = dense_matching.compute_disparity(
        left,
        right,
        input_stereo_cfg,
        corr_cfg,
        disp_min,
        disp_max,
        use_sec_disp=use_sec_disp,
    )

    # If necessary, set disparity to 0 for classes to be set to input dem
    mask1_classes = input_stereo_cfg[input_parameters.INPUT_SECTION_TAG].get(
        input_parameters.MASK1_CLASSES_TAG, None
    )
    mask2_classes = input_stereo_cfg[input_parameters.INPUT_SECTION_TAG].get(
        input_parameters.MASK2_CLASSES_TAG, None
    )
    regularisation.update_disp_to_0(
        disp, left, right, mask1_classes, mask2_classes
    )

    colors = {}
    colors[cst.STEREO_REF] = color
    if cst.STEREO_SEC in disp:
        # compute right color image from right-left disparity map
        colors[cst.STEREO_SEC] = dense_matching.estimate_color_from_disparity(
            disp[cst.STEREO_SEC], left, color
        )

    im_ref_msk = None
    im_sec_msk = None
    if add_msk_info:
        ref_values_list = [key for key, _ in left.items()]
        if cst.EPI_MSK in ref_values_list:
            im_ref_msk = left
        else:
            worker_logger = logging.getLogger("distributed.worker")
            worker_logger.warning(
                "Left image does not have a " "mask to rasterize"
            )
        if cst.STEREO_SEC in disp:
            sec_values_list = [key for key, _ in right.items()]
            if cst.EPI_MSK in sec_values_list:
                im_sec_msk = right
            else:
                worker_logger = logging.getLogger("distributed.worker")
                worker_logger.warning(
                    "Right image does not have a " "mask to rasterize"
                )

    # Triangulate
    if cst.STEREO_SEC in disp:
        points = triangulation.triangulate(
            input_stereo_cfg,
            disp[cst.STEREO_REF],
            disp[cst.STEREO_SEC],
            snap_to_img1=snap_to_img1,
            align=align,
            im_ref_msk_ds=im_ref_msk,
            im_sec_msk_ds=im_sec_msk,
        )
    else:
        points = triangulation.triangulate(
            input_stereo_cfg,
            disp[cst.STEREO_REF],
            snap_to_img1=snap_to_img1,
            align=align,
            im_ref_msk_ds=im_ref_msk,
            im_sec_msk_ds=im_sec_msk,
        )

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key in points:
            points[key] = triangulation.geoid_offset(points[key], geoid_data)

    if out_epsg is not None:
        for key in points:
            projection.points_cloud_conversion_dataset(points[key], out_epsg)

    return points, colors
