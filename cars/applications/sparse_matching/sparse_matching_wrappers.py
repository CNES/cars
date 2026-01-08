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

import logging

# Third party imports
import numpy as np
import pandas
import xarray as xr

# CARS imports
import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars.applications import application_constants
from cars.applications.point_cloud_outlier_removal import outlier_removal_algo
from cars.orchestrator.cluster.log_wrapper import cars_profile


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
            sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                pair_key: {
                    sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                    sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                }
            }
        }
    }
    orchestrator.update_out_info(updating_infos)

    return dmin, dmax


@cars_profile(name="Filter_point_cloud_matches")
def filter_point_cloud_matches(
    pd_cloud,
    match_filter_knn=25,
    match_filter_constant=0,
    match_filter_mean_factor=1,
    match_filter_dev_factor=3,
):
    """
    Filter triangulated  matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param match_filter_knn: number of neighboors used to measure
                               isolation of matches
    :type match_filter_knn: int
    :param match_filter_dev_factor: factor of deviation in the
                                      formula to compute threshold of outliers
    :type match_filter_dev_factor: float

    :return: disp min and disp max
    :rtype: float, float
    """

    # Statistical filtering
    filter_cloud, _ = outlier_removal_algo.statistical_outlier_filtering(
        pd_cloud,
        k=match_filter_knn,
        filtering_constant=match_filter_constant,
        mean_factor=match_filter_mean_factor,
        dev_factor=match_filter_dev_factor,
    )

    # filter nans
    filter_cloud.dropna(axis=0, inplace=True)

    return filter_cloud


def transform_triangulated_matches_to_dataframe(triangulated_matches):
    """

    :param triangulated_matches: triangulated matches
    :type: cars_dataset
    """
    # Concatenated matches
    list_matches = []
    attrs = None
    for row in range(triangulated_matches.shape[0]):
        for col in range(triangulated_matches.shape[1]):
            # CarsDataset containing Pandas DataFrame, not Delayed anymore
            if triangulated_matches[row, col] is not None:
                epipolar_matches = triangulated_matches[row, col]

                if attrs is None:
                    attrs = epipolar_matches.attrs

                list_matches.append(epipolar_matches)

    if list_matches:
        triangulated_matches_df = pandas.concat(list_matches, ignore_index=True)
        triangulated_matches_df.attrs = attrs
    else:
        raise RuntimeError("No match have been found in sparse matching")

    return triangulated_matches_df


def get_margins(margin, disp_min, disp_max):
    """
    Get margins for the dense matching steps

    :param margin: margins object
    :type margin: Margins
    :param disp_min: Minimum disparity
    :type disp_min: int
    :param disp_max: Maximum disparity
    :type disp_max: int
    :return: margins of the matching algorithm used
    """
    col = np.arange(4)

    left_margins = [
        margin + disp_max,
        margin,
        margin - disp_min,
        margin,
    ]
    right_margins = [
        margin - disp_min,
        margin,
        margin + disp_max,
        margin,
    ]
    same_margins = [
        max(left, right)
        for left, right in zip(left_margins, right_margins)  # noqa: B905
    ]

    margins = xr.Dataset(
        {
            "left_margin": (
                ["col"],
                same_margins,
            )
        },
        coords={"col": col},
    )
    margins["right_margin"] = xr.DataArray(same_margins, dims=["col"])

    margins.attrs["disp_min"] = disp_min
    margins.attrs["disp_max"] = disp_max

    return margins
