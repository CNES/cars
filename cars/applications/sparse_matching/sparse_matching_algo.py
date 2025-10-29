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
# pylint: disable=too-many-lines

"""
Sparse matching Sift module:
contains sift sparse matching method
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
from vlsift.sift.sift import sift

# CARS imports
from cars.applications.sparse_matching.sparse_matching_wrappers import (
    euclidean_matrix_distance,
)


def compute_matches(  # pylint: disable=too-many-positional-arguments
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray = None,
    right_mask: np.ndarray = None,
    left_origin: [float, float] = None,
    right_origin: [float, float] = None,
    matching_threshold: float = 0.7,
    n_octave: int = 8,
    n_scale_per_octave: int = 3,
    peak_threshold: float = 4.0,
    edge_threshold: float = 10.0,
    magnification: float = 7.0,
    window_size: int = 2,
    backmatching: bool = True,
    disp_lower_bound=None,
    disp_upper_bound=None,
):
    """
    Compute matches between left and right
    Convention for masks: True is a valid pixel

    :param left: left image as numpy array
    :type left: np.ndarray
    :param right: right image as numpy array
    :type right: np.ndarray
    :param left_mask: left mask as numpy array
    :type left_mask: np.ndarray
    :param right_mask: right mask as numpy array
    :type right_mask: np.ndarray
    :param left_origin: left image origin in the full image
    :type left_origin: [float, float]
    :param right_origin: right image origin in the full image
    :type right_origin: [float, float]
    :param matching_threshold: threshold for the ratio to nearest second match
    :type matching_threshold: float
    :param n_octave: the number of octaves of the DoG scale space
    :type n_octave: int
    :param n_scale_per_octave: the nb of levels / octave of the DoG scale space
    :type n_scale_per_octave: int
    :param peak_threshold: the peak selection threshold
    :type peak_threshold: float
    :param edge_threshold: the edge selection threshold
    :type edge_threshold: float
    :param magnification: set the descriptor magnification factor
    :type magnification: float
    :param window_size: size of the window
    :type window_size: int
    :param backmatching: also check that right vs. left gives same match
    :type backmatching: bool

    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)

    """
    left_origin = [0, 0] if left_origin is None else left_origin
    right_origin = [0, 0] if right_origin is None else right_origin

    # compute keypoints + descriptors
    left_frames, left_descr = sift(
        left,
        n_octaves=n_octave,
        n_levels=n_scale_per_octave,
        first_octave=-1,
        peak_thresh=peak_threshold,
        edge_thresh=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        float_descriptors=True,
        compute_descriptor=True,
        verbose=False,
    )

    right_frames, right_descr = sift(
        right,
        n_octaves=n_octave,
        n_levels=n_scale_per_octave,
        first_octave=-1,
        peak_thresh=peak_threshold,
        edge_thresh=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        float_descriptors=True,
        compute_descriptor=True,
        verbose=False,
    )

    # Filter keypoints that falls out of the validity mask (0=valid)
    if left_mask is not None:
        pixel_indices = np.floor(left_frames[:, 0:2]).astype(int)
        valid_left_frames_mask = left_mask[
            pixel_indices[:, 0], pixel_indices[:, 1]
        ]
        left_frames = left_frames[valid_left_frames_mask]
        left_descr = left_descr[valid_left_frames_mask]

    if right_mask is not None:
        pixel_indices = np.floor(right_frames[:, 0:2]).astype(int)
        valid_right_frames_mask = right_mask[
            pixel_indices[:, 0], pixel_indices[:, 1]
        ]
        right_frames = right_frames[valid_right_frames_mask]
        right_descr = right_descr[valid_right_frames_mask]

    # Early return for empty frames
    # also if there are points to match
    # need minimum two right points to find the second nearest neighbor
    # (and two left points for backmatching)
    if left_frames.shape[0] < 2 or right_frames.shape[0] < 2:
        return np.empty((0, 4))

    # translate matches according image origin
    # revert origin due to frame convention: [Y, X, S, TH] X: 1, Y: 0)
    left_frames[:, 0:2] += left_origin[::-1]
    right_frames[:, 0:2] += right_origin[::-1]

    # sort frames (and descriptors) along X axis
    order = np.argsort(left_frames[:, 1])
    left_frames = left_frames[order]
    left_descr = left_descr[order]

    order = np.argsort(right_frames[:, 1])
    right_frames = right_frames[order]
    right_descr = right_descr[order]

    # compute best matches by blocks
    splits = np.arange(500, len(left_frames), 500)
    left_frames_splitted = np.split(left_frames, splits)
    left_descr_splitted = np.split(left_descr, splits)
    splits = np.insert(splits, 0, 0)

    matches_id = []

    for (
        left_id_offset,
        left_frames_block,
        left_descr_block,
    ) in zip(  # noqa: B905
        splits, left_frames_splitted, left_descr_splitted
    ):
        if disp_lower_bound is not None and disp_upper_bound is not None:
            # Find right block extremas
            right_x_min = np.min(left_frames_block[:, 1]) + disp_lower_bound
            right_x_max = np.max(left_frames_block[:, 1]) + disp_upper_bound
            if (
                np.max(right_frames[:, 1]) > right_x_min
                and np.min(right_frames[:, 1]) < right_x_max
            ):
                left_id = np.min(np.where(right_frames[:, 1] > right_x_min))
                right_id = np.max(np.where(right_frames[:, 1] < right_x_max))
                right_descr_block = right_descr[left_id:right_id]
                right_id_offset = left_id
            else:
                right_descr_block = []
                right_id_offset = 0
        else:
            right_descr_block = right_descr
            right_id_offset = 0

        if len(left_descr_block) >= 2 and len(right_descr_block) >= 2:
            # compute euclidean matrix distance
            emd = euclidean_matrix_distance(left_descr_block, right_descr_block)

            # get nearest sift (regarding descriptors)
            id_nearest_dlr = (
                np.arange(np.shape(emd)[0]),
                np.nanargmin(emd, axis=1),
            )
            id_nearest_drl = (
                np.nanargmin(emd, axis=0),
                np.arange(np.shape(emd)[1]),
            )

            # get absolute distances
            dist_dlr = emd[id_nearest_dlr]
            dist_drl = emd[id_nearest_drl]

            # get relative distance (ratio to second nearest distance)
            second_dist_dlr = np.partition(emd, 1, axis=1)[:, 1]
            dist_dlr /= second_dist_dlr
            second_dist_drl = np.partition(emd, 1, axis=0)[1, :]
            dist_drl /= second_dist_drl

            # stack matches which its distance
            id_matches_dlr = np.column_stack((*id_nearest_dlr, dist_dlr))
            id_matches_drl = np.column_stack((*id_nearest_drl, dist_drl))

            # check backmatching
            if backmatching is True:
                back = (
                    id_matches_dlr[:, 0]
                    == id_matches_drl[id_matches_dlr[:, 1].astype(int)][:, 0]
                )
                id_matches_dlr = id_matches_dlr[back]

            # threshold matches
            id_matches_dlr = id_matches_dlr[
                id_matches_dlr[:, -1] < matching_threshold, :
            ][:, :-1]

            id_matches_dlr += (left_id_offset, right_id_offset)

            matches_id.append(id_matches_dlr)

    if matches_id:
        matches_id = np.concatenate(matches_id)
    else:
        matches_id = np.empty((0, 4))

    # retrieve points: [Y, X, S, TH] X: 1, Y: 0
    # fyi: ``S`` is the scale and ``TH`` is the orientation (in radians)
    left_points = left_frames[matches_id[:, 0].astype(int), 1::-1]
    right_points = right_frames[matches_id[:, 1].astype(int), 1::-1]
    matches = np.concatenate((left_points, right_points), axis=1)
    return matches


def dataset_matching(  # pylint: disable=too-many-positional-arguments
    ds1,
    ds2,
    used_band,
    matching_threshold=0.7,
    n_octave=8,
    n_scale_per_octave=3,
    peak_threshold=4.0,
    edge_threshold=10.0,
    magnification=7.0,
    window_size=2,
    backmatching=True,
    disp_lower_bound=None,
    disp_upper_bound=None,
    classif_bands_to_mask=None,
):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param matching_threshold: threshold for the ratio to nearest second match
    :type matching_threshold: float
    :param n_octave: the number of octaves of the DoG scale space
    :type n_octave: int
    :param n_scale_per_octave: the nb of levels / octave of the DoG scale space
    :type n_scale_per_octave: int
    :param peak_threshold: the peak selection threshold
    :type peak_threshold: int
    :param edge_threshold: the edge selection threshold.
    :param magnification: set the descriptor magnification factor
    :type magnification: float
    :param window_size: size of the window
    :type window_size: int
    :param backmatching: also check that right vs. left gives same match
    :type backmatching: bool
    :param classif_bands_to_mask: bands from classif to mask
    :type classif_bands_to_mask: list of str / int

    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    # get input data from dataset
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]

    left = ds1.im.loc[used_band].values
    right = ds2.im.loc[used_band].values
    # Generate validity masks
    left_mask = ds1.msk.loc[used_band].values == 0
    right_mask = ds2.msk.loc[used_band].values == 0

    # Update validity masks: all classes (used in filling) in
    # classification should be 0
    if "classification" in ds1:
        if classif_bands_to_mask is not None:
            classif_values = (
                ds1["classification"].loc[classif_bands_to_mask].values
            )
        else:
            classif_values = (
                ds1["classification"].loc[classif_bands_to_mask].values
            )
        left_mask = np.logical_and(
            left_mask, ~np.any(classif_values > 0, axis=0)
        )
    if "classification" in ds2:
        if classif_bands_to_mask is not None:
            classif_values = (
                ds2["classification"].loc[classif_bands_to_mask].values
            )
        else:
            classif_values = (
                ds1["classification"].loc[classif_bands_to_mask].values
            )
        right_mask = np.logical_and(
            right_mask, ~np.any(classif_values > 0, axis=0)
        )

    matches = compute_matches(
        left,
        right,
        left_mask=left_mask,
        right_mask=right_mask,
        left_origin=origin1,
        right_origin=origin2,
        matching_threshold=matching_threshold,
        n_octave=n_octave,
        n_scale_per_octave=n_scale_per_octave,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        backmatching=backmatching,
        disp_lower_bound=disp_lower_bound,
        disp_upper_bound=disp_upper_bound,
    )

    return matches
