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
Test module for cars/otb_pipelines.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pytest

# CARS imports
from cars import otb_pipelines

# CARS Tests imports
from .utils import absolute_data_path, otb_geoid_file_set, otb_geoid_file_unset


@pytest.mark.unit_tests
def test_build_stereorectification_grid_pipeline():
    """
    Test if the pipeline is correctly built and produces consistent grids
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45
    # Set the geoid file from code source
    otb_geoid_file_set()

    # Launch otb stereorectification grid pipeline
    (
        left_grid_np,
        right_grid_np,
        left_grid_origin,
        left_grid_spacing,
        epipolar_size_x,
        epipolar_size_y,
        disp_to_alt_ratio,
    ) = otb_pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem, epi_step=step
    )

    assert epipolar_size_x == 612
    assert epipolar_size_y == 612
    assert left_grid_np.shape == (15, 15, 2)
    assert left_grid_origin[0] == 0
    assert left_grid_origin[1] == 0
    assert left_grid_spacing[0] == step
    assert left_grid_spacing[1] == step
    assert np.isclose(disp_to_alt_ratio, 0.7, 0.01)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/left_grid.npy"), left_grid_np)

    left_grid_np_reference = np.load(
        absolute_data_path("ref_output/left_grid.npy")
    )
    np.testing.assert_allclose(left_grid_np, left_grid_np_reference)

    assert right_grid_np.shape == (15, 15, 2)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/right_grid.npy"), right_grid_np)

    right_grid_np_reference = np.load(
        absolute_data_path("ref_output/right_grid.npy")
    )
    np.testing.assert_allclose(right_grid_np, right_grid_np_reference)

    # unset otb geoid file
    otb_geoid_file_unset()


@pytest.mark.unit_tests
def test_build_extract_roi_application():
    """
    Test that input region is correctly use to build the roi extraction
    application
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    region = [100, 200, 300, 400]

    app = otb_pipelines.build_extract_roi_application(img, region)

    assert app.GetParameterInt("startx") == region[0]
    assert app.GetParameterInt("starty") == region[1]
    assert app.GetParameterInt("sizex") == region[2] - region[0]
    assert app.GetParameterInt("sizey") == region[3] - region[1]


@pytest.mark.unit_tests
def test_build_mask_pipeline():
    """
    Test that the pipeline is correctly built
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    nodata = 0
    mask = absolute_data_path("input/phr_reunion/left_mask.tif")
    roi = [100, 200, 300, 400]
    out_np = otb_pipelines.build_mask_pipeline(
        img, grid, nodata, mask, 2387, 2387, roi
    )

    assert out_np.shape == (200, 200)


@pytest.mark.unit_tests
def test_build_image_resampling_pipeline():
    """
    Test that the pipeline is correctly built
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    roi = [100, 200, 300, 400]
    out_np = otb_pipelines.build_image_resampling_pipeline(
        img, grid, 2387, 2387, roi
    )

    assert out_np.shape == (200, 200, 1)
