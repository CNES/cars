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
