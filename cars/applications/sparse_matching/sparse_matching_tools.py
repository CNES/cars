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
Sparse matching Sift module:
contains sift sparse matching method
"""

# Standard imports
from __future__ import absolute_import

import logging

# Third party imports
import numpy as np
from vlsift.sift.sift import sift

# CARS imports
import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars.applications import application_constants
from cars.applications.point_cloud_outliers_removing import (
    outlier_removing_tools,
)


def euclidean_matrix_distance(descr1: np.array, descr2: np.array):
    """Compute a matrix containing cross euclidean distance
    :param descr1: first keypoints descriptor
    :type descr1: numpy.ndarray
    :param descr2: second keypoints descriptor
    :type descr2: numpy.ndarray
    :return euclidean matrix distance
    :rtype: float
    """
    sq_descr1 = np.sum(descr1**2, axis=1)[:, np.newaxis]
    sq_descr2 = np.sum(descr2**2, axis=1)
    dot_descr12 = np.dot(descr1, descr2.T)
    return np.sqrt(sq_descr1 + sq_descr2 - 2 * dot_descr12)


def compute_matches(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray = None,
    right_mask: np.ndarray = None,
    left_origin: [float, float] = None,
    right_origin: [float, float] = None,
    matching_threshold: float = 0.6,
    n_octave: int = 8,
    n_scale_per_octave: int = 3,
    peak_threshold: float = 20.0,
    edge_threshold: float = 5.0,
    magnification: float = 2.0,
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


def dataset_matching(
    ds1,
    ds2,
    matching_threshold=0.6,
    n_octave=8,
    n_scale_per_octave=3,
    peak_threshold=20.0,
    edge_threshold=5.0,
    magnification=2.0,
    backmatching=True,
    disp_lower_bound=None,
    disp_upper_bound=None,
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
    :param backmatching: also check that right vs. left gives same match
    :type backmatching: bool
    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    # get input data from dataset
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]
    left = ds1.im.values
    right = ds2.im.values
    left_mask = ds1.msk.values == 0
    right_mask = ds2.msk.values == 0

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
        backmatching=backmatching,
        disp_lower_bound=disp_lower_bound,
        disp_upper_bound=disp_upper_bound,
    )

    return matches


def remove_epipolar_outliers(matches, percent=0.1):
    # TODO used only in test functions to test compute_disparity_range
    # Refactor with sparse_matching
    """
    This function will filter the match vector
    according to a quantile of epipolar error
    used for testing compute_disparity_range sparse method

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema
    :type percent: float
    :return: the filtered match array
    :rtype: numpy array
    """
    epipolar_error_min = np.percentile(matches[:, 1] - matches[:, 3], percent)
    epipolar_error_max = np.percentile(
        matches[:, 1] - matches[:, 3], 100 - percent
    )
    logging.info(
        "Epipolar error range after outlier rejection: [{},{}]".format(
            epipolar_error_min, epipolar_error_max
        )
    )
    out = matches[(matches[:, 1] - matches[:, 3]) < epipolar_error_max]
    out = out[(out[:, 1] - out[:, 3]) > epipolar_error_min]

    return out


def compute_disparity_range(matches, percent=0.1):
    # TODO: Refactor with dense_matching to have only one API ?
    """
    This function will compute the disparity range
    from matches by filtering percent outliers

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema (in %)
    :type percent: float
    :return: the disparity range
    :rtype: float, float
    """
    disparity = matches[:, 2] - matches[:, 0]

    mindisp = np.percentile(disparity, percent)
    maxdisp = np.percentile(disparity, 100 - percent)

    return mindisp, maxdisp


def filter_point_cloud_matches(
    pd_cloud,
    matches_filter_knn=25,
    matches_filter_dev_factor=3,
):
    """
    Filter triangulated  matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param matches_filter_knn: number of neighboors used to measure
                               isolation of matches
    :type matches_filter_knn: int
    :param matches_filter_dev_factor: factor of deviation in the
                                      formula to compute threshold of outliers
    :type matches_filter_dev_factor: float

    :return: disp min and disp max
    :rtype: float, float
    """

    # Statistical filtering
    filter_cloud, _ = outlier_removing_tools.statistical_outliers_filtering(
        pd_cloud,
        k=matches_filter_knn,
        dev_factor=matches_filter_dev_factor,
    )

    # filter nans
    filter_cloud.dropna(axis=0, inplace=True)

    return filter_cloud


def compute_disp_min_disp_max(
    pd_cloud,
    orchestrator,
    disp_margin=0.1,
    pair_key=None,
    disp_to_alt_ratio=None,
):
    """
    Compute disp min and disp max from triangulated and filtered matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param orchestrator: orchestrator used
    :type orchestrator: Orchestrator
    :param disp_margin: disparity margin
    :type disp_margin: float
    :param disp_to_alt_ratio: used for logging info
    :type disp_to_alt_ratio: float

    :return: disp min and disp max
    :rtype: float, float
    """

    # Obtain dmin dmax
    filt_disparity = np.array(pd_cloud.iloc[:, 3])
    dmax = np.nanmax(filt_disparity)
    dmin = np.nanmin(filt_disparity)

    margin = abs(dmax - dmin) * disp_margin
    dmin -= margin
    dmax += margin

    logging.info(
        "Disparity range with margin: [{:.3f} pix., {:.3f} pix.] "
        "(margin = {:.3f} pix.)".format(dmin, dmax, margin)
    )

    if disp_to_alt_ratio is not None:
        logging.info(
            "Equivalent range in meters: [{:.3f} m, {:.3f} m] "
            "(margin = {:.3f} m)".format(
                dmin * disp_to_alt_ratio,
                dmax * disp_to_alt_ratio,
                margin * disp_to_alt_ratio,
            )
        )

    # update orchestrator_out_json
    updating_infos = {
        application_constants.APPLICATION_TAG: {
            pair_key: {
                sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                    sm_cst.DISPARITY_MARGIN_PARAM_TAG: disp_margin,
                    sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                    sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                }
            }
        }
    }
    orchestrator.update_out_info(updating_infos)

    return dmin, dmax
