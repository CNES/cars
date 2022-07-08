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
import os

# Third party imports
import numpy as np

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.core.utils import safe_makedirs

# CARS imports
from cars.externals import otb_pipelines


def dataset_matching(
    ds1,
    ds2,
    matching_threshold=0.6,
    n_octave=8,
    n_scale_per_octave=3,
    dog_threshold=20,
    edge_threshold=5,
    magnification=2.0,
    backmatching=True,
):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param threshold: Threshold for matches
    :type threshold: float
    :param backmatching: Also check that right vs. left gives same match
    :type backmatching: bool
    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    size1 = [
        int(ds1.attrs["region"][2] - ds1.attrs["region"][0]),
        int(ds1.attrs["region"][3] - ds1.attrs["region"][1]),
    ]
    roi1 = [0, 0, size1[0], size1[1]]
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]

    size2 = [
        int(ds2.attrs["region"][2] - ds2.attrs["region"][0]),
        int(ds2.attrs["region"][3] - ds2.attrs["region"][1]),
    ]
    roi2 = [0, 0, size2[0], size2[1]]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]

    matches = otb_pipelines.epipolar_sparse_matching(
        ds1,
        roi1,
        size1,
        origin1,
        ds2,
        roi2,
        size2,
        origin2,
        matching_threshold,
        n_octave,
        n_scale_per_octave,
        dog_threshold,
        edge_threshold,
        magnification,
        backmatching,
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


def derive_disparity_range_from_matches(
    corrected_matches,
    orchestrator=None,
    disparity_margin=0.1,
    pair_key="PAIR_0",
    pair_folder=None,
    disp_to_alt_ratio=None,
    disparity_outliers_rejection_percent=0.1,
    save_matches=False,
):
    """
    Compute disp min and disp max from matches

    :param cars_orchestrator: orchestrator : used for info writting
    :param corrected_matches: matches
    :type corrected_matches: np.ndarray
    :param disparity_margin: disparity margin
    :type disparity_margin: float
    :param pair_key: id of pair : only used for info writting
    :type pair_key: int
    :param disp_to_alt_ratio: used for logging info
    :type disp_to_alt_ratio: float
    :param disparity_outliers_rejection_percent: percentage of
            outliers to reject
    :type disparity_outliers_rejection_percent: float
    :param save_matches: true is matches needs to be saved
    :type save_matches: bool

    :return: disp min and disp max
    :rtype: float, float

    """

    # Default orchestrator
    if orchestrator is None:
        # Create defaut sequential orchestrator for current application
        # be awere, no out_json will be shared between orchestrators
        # No files saved
        cars_orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )
    else:
        cars_orchestrator = orchestrator

    if pair_folder is None:
        pair_folder = os.path.join(cars_orchestrator.out_dir, "tmp")
        safe_makedirs(pair_folder)

    nb_matches = corrected_matches.shape[0]

    corrected_epipolar_error = corrected_matches[:, 1] - corrected_matches[:, 3]

    # Compute the disparity range (we filter matches that are too off epipolar
    # lins after correction)
    corrected_std = np.std(corrected_epipolar_error)

    corrected_matches = corrected_matches[
        np.fabs(corrected_epipolar_error) < 3 * corrected_std
    ]
    logging.info(
        "{} matches discarded because "
        "their epipolar error is greater than 3*stdev of epipolar error "
        "after correction (3*stddev = {:.3f} pix.)".format(
            nb_matches - corrected_matches.shape[0], 3 * corrected_std
        )
    )

    logging.info(
        "Number of matches kept "
        "for disparity range estimation: {} matches".format(
            corrected_matches.shape[0]
        )
    )

    # Compute disparity range
    dmin, dmax = compute_disparity_range(
        corrected_matches,
        disparity_outliers_rejection_percent,
    )

    margin = abs(dmax - dmin) * disparity_margin
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

    # Export matches
    matches_array_path = None
    if save_matches:
        logging.info("Writing matches file")
        if pair_folder is None:
            current_out_dir = cars_orchestrator.out_dir
        else:
            current_out_dir = pair_folder
        matches_array_path = os.path.join(current_out_dir, "matches.npy")
        np.save(matches_array_path, corrected_matches)

    # update orchestrator_out_json
    updating_infos = {
        application_constants.APPLICATION_TAG: {
            pair_key: {
                sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                    sm_cst.DISPARITY_MARGIN_PARAM_TAG: disparity_margin,
                    sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                    sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                    sm_cst.MATCHES_TAG: matches_array_path,
                }
            }
        }
    }
    cars_orchestrator.update_out_info(updating_infos)

    return dmin, dmax
