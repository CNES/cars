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
Test module for cars/lib/steps/sparse_matching.sift.py
"""

from __future__ import absolute_import

import pytest
import numpy as np

from cars import stereo
from .utils import absolute_data_path, temporary_dir, assert_same_datasets
from cars.lib.steps.sparse_matching import sift


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
        "input/preprocessing_input/left_epipolar_grid_reunion.tif")
    grid2 = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_reunion.tif")

    epipolar_size_x = 596
    epipolar_size_y = 596

    left = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)
    right = stereo.resample_image(
        img2, grid2, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata2, mask=mask2)

    matches = sift.dataset_matching(left, right)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/matches.npy"), matches)

    matches_ref = np.load(absolute_data_path("ref_output/matches.npy"))
    np.testing.assert_allclose(matches, matches_ref)

    # Case with no matches
    region = [0, 0, 2, 2]

    left = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)
    right = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)

    matches = sift.dataset_matching(left, right)

    assert matches.shape == (0, 4)
