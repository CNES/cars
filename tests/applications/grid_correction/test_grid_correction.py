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
Test module for cars/steps/epi_rectif/test_grid_correction.py
"""

# __future__ import
from __future__ import absolute_import

# Standard library
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

# Third-party imports
import numpy as np
import pytest
import rasterio as rio

# CARS imports
from cars.applications.application import Application
from cars.applications.sparse_matching import sparse_matching_wrappers

# CARS test utilities
from tests.helpers import absolute_data_path


@pytest.mark.unit_tests
def test_correct_right_grid():
    """
    Call right grid correction method and check outputs properties
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_ventoux.npy"
    )
    grid_file = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_uncorrected_ventoux.tif"
    )
    origin = [0, 0]
    spacing = [30, 30]

    matches = np.load(matches_file)
    matches = np.array(matches)

    matches_filtered = sparse_matching_wrappers.remove_epipolar_outliers(
        matches
    )

    grid = rio.open(grid_file).read()
    grid = np.moveaxis(grid, 0, -1)

    grid_right = {
        "grid_origin": origin,
        "grid_spacing": spacing,
        "path": grid_file,
    }

    grid_correction_app = Application("grid_correction")

    (corrected_grid_dict, corrected_matches, in_stats, out_stats) = (
        grid_correction_app.run(matches_filtered, grid_right)
    )

    corrected_grid = rio.open(corrected_grid_dict["path"]).read()
    corrected_grid = np.moveaxis(corrected_grid, 0, -1)

    # Uncomment to update ref
    # np.save(absolute_data_path("ref_output_application/grid_generation"
    # "/corrected_right_grid.npy"),
    #  corrected_grid)
    corrected_grid_ref = np.load(
        absolute_data_path(
            "ref_output_application/grid_generation/corrected_right_grid.npy"
        )
    )
    np.testing.assert_allclose(
        corrected_grid, corrected_grid_ref, atol=0.05, rtol=1.0e-6
    )

    assert corrected_grid.shape == grid.shape

    # Assert that we improved all stats
    assert abs(out_stats["mean_epipolar_error"][0]) < abs(
        in_stats["mean_epipolar_error"][0]
    )
    assert abs(out_stats["mean_epipolar_error"][1]) < abs(
        in_stats["mean_epipolar_error"][1]
    )
    assert abs(out_stats["median_epipolar_error"][0]) < abs(
        in_stats["median_epipolar_error"][0]
    )
    assert abs(out_stats["median_epipolar_error"][1]) < abs(
        in_stats["median_epipolar_error"][1]
    )
    assert (
        out_stats["std_epipolar_error"][0] < in_stats["std_epipolar_error"][0]
    )
    assert (
        out_stats["std_epipolar_error"][1] < in_stats["std_epipolar_error"][1]
    )
    assert out_stats["rms_epipolar_error"] < in_stats["rms_epipolar_error"]
    assert out_stats["rmsd_epipolar_error"] < in_stats["rmsd_epipolar_error"]

    # Assert absolute performances

    assert abs(out_stats["median_epipolar_error"][0]) < 0.1
    assert abs(out_stats["median_epipolar_error"][1]) < 0.1

    assert abs(out_stats["mean_epipolar_error"][0]) < 0.1
    assert abs(out_stats["mean_epipolar_error"][1]) < 0.1
    assert out_stats["rms_epipolar_error"] < 0.5

    # Assert corrected matches are corrected
    assert (
        np.fabs(np.mean(corrected_matches[:, 1] - corrected_matches[:, 3]))
        < 0.1
    )
