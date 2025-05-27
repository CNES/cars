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
Test module for cars/steps/matching/sparse_matching_algo.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pytest

# CARS imports
from cars.applications.resampling import resampling_algo
from cars.applications.sparse_matching import (
    sparse_matching_algo,
    sparse_matching_wrappers,
)

# CARS Tests imports
from tests.helpers import absolute_data_path


@pytest.mark.unit_tests
def test_dataset_matching():
    """
    Test dataset_matching method
    """
    region = [200, 250, 320, 400]
    img1 = absolute_data_path("input/phr_reunion/left_image.tif")
    img2 = absolute_data_path("input/phr_reunion/right_image.tif")
    mask1 = absolute_data_path("input/phr_reunion/left_mask.tif")
    mask2 = absolute_data_path("input/phr_reunion/right_mask.tif")
    nodata1 = 0
    nodata2 = 0
    grid1 = absolute_data_path(
        "input/preprocessing_input/left_epipolar_grid_reunion.tif"
    )
    grid2 = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_reunion.tif"
    )

    epipolar_size_x = 596
    epipolar_size_y = 596

    left = resampling_algo.resample_image(
        img1,
        grid1,
        [epipolar_size_x, epipolar_size_y],
        region=region,
        nodata=nodata1,
        mask=mask1,
    )
    right = resampling_algo.resample_image(
        img2,
        grid2,
        [epipolar_size_x, epipolar_size_y],
        region=region,
        nodata=nodata2,
        mask=mask2,
    )

    matches = sparse_matching_algo.dataset_matching(left, right)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output_application/sparse_matching"
    # "/matches.npy"), matches)

    matches_ref = np.load(
        absolute_data_path("ref_output_application/sparse_matching/matches.npy")
    )
    np.testing.assert_allclose(matches, matches_ref)

    # Case with no matches
    region = [0, 0, 2, 2]

    left = resampling_algo.resample_image(
        img1,
        grid1,
        [epipolar_size_x, epipolar_size_y],
        region=region,
        nodata=nodata1,
        mask=mask1,
    )
    right = resampling_algo.resample_image(
        img1,
        grid1,
        [epipolar_size_x, epipolar_size_y],
        region=region,
        nodata=nodata1,
        mask=mask1,
    )

    matches = sparse_matching_algo.dataset_matching(left, right)

    assert matches.shape == (0, 4)


@pytest.mark.unit_tests
def test_remove_epipolar_outliers():
    """
    Test remove epipolar outliers function
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_reunion.npy"
    )

    matches = np.load(matches_file)

    matches_filtered = sparse_matching_wrappers.remove_epipolar_outliers(
        matches
    )

    nb_filtered_points = matches.shape[0] - matches_filtered.shape[0]
    assert nb_filtered_points == 2
