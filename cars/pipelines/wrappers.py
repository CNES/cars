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
#TODO to be removed
#TODO what should we be looking for here ?
#     what does "wrappers" stands for ?
#     what is it that is not a wrapper of something inside cars ?

import logging
from typing import List

import numpy as np

from cars.conf import static_conf
from cars.conf import mask_classes
from cars.lib.steps.sparse_matching import sift
from cars import constants as cst
from cars.lib.steps.epi_rectif import resampling


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
        epipolar_size_y: int) -> np.ndarray :
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
    worker_logger = logging.getLogger('distributed.worker')
    worker_logger.debug("Matching keypoints on region {}".format(left_region))

    largest_size = [epipolar_size_x, epipolar_size_y]

    # Resample left dataset
    left_ds = resampling.resample_image(
        img1, grid1, largest_size,
        region=left_region, nodata=nodata1, mask=mask1)

    # handle multi classes mask if necessary
    if mask1_classes is not None:
        left_ds[cst.EPI_MSK].values =\
            mask_classes.create_msk_from_tag(
                left_ds[cst.EPI_MSK].values,
                mask1_classes,
                mask_classes.ignored_by_sift_matching_tag,
                mask_intern_no_data_val=True
            )

    # Resample right dataset
    right_ds = resampling.resample_image(
        img2,
        grid2,
        largest_size,
        region=right_region,
        nodata=nodata2,
        mask=mask2)

    # handle multi classes mask if necessary
    if mask2_classes is not None:
        right_ds[cst.EPI_MSK].values =\
            mask_classes.create_msk_from_tag(
                right_ds[cst.EPI_MSK].values,
                mask2_classes,
                mask_classes.ignored_by_sift_matching_tag,
                mask_intern_no_data_val=True
            )

    # Perform matching
    sift_params = static_conf.get_sift_params()
    matches = \
        sift.dataset_matching(left_ds, right_ds,
            matching_threshold =\
                getattr(sift_params, static_conf.sift_matching_threshold_tag),
            n_octave =\
                getattr(sift_params, static_conf.sift_n_octave_tag),
            n_scale_per_octave =\
                getattr(sift_params, static_conf.sift_n_scale_per_octave_tag),
            dog_threshold =\
                getattr(sift_params, static_conf.sift_dog_threshold_tag),
            edge_threshold =\
                getattr(sift_params, static_conf.sift_edge_threshold_tag),
            magnification =\
                getattr(sift_params, static_conf.sift_magnification_tag),
            backmatching =\
                getattr(sift_params, static_conf.sift_back_matching_tag))

    return matches
