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

import pytest
import numpy as np
from cars import pipelines
from utils import absolute_data_path


@pytest.mark.unit_tests
def test_build_stereorectification_grid_pipeline():
    """
    Test if the pipeline is correctly built and produces consistent grids
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45
    left_grid, right_grid, epipolar_size_x, \
        epipolar_size_y, disp_to_alt_ratio, pipeline \
            = pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem, epi_step=step)
    assert epipolar_size_x == 612
    assert epipolar_size_y == 612
    assert "stereo_app" in pipeline

    left_grid_np = pipeline["stereo_app"].GetVectorImageAsNumpyArray(
        "io.outleft")
    assert left_grid_np.shape == (15, 15, 2)
    left_grid_origin = pipeline["stereo_app"].GetImageOrigin("io.outleft")
    assert left_grid_origin[0] == 0
    assert left_grid_origin[1] == 0
    left_grid_spacing = pipeline["stereo_app"].GetImageSpacing("io.outleft")
    assert left_grid_spacing[0] == step
    assert left_grid_spacing[1] == step

    # Uncomment to update baseline
    #np.save(absolute_data_path("ref_output/left_grid.npy"), left_grid_np)

    left_grid_np_reference = np.load(
        absolute_data_path("ref_output/left_grid.npy"))
    np.testing.assert_allclose(left_grid_np, left_grid_np_reference)

    right_grid_np = pipeline["stereo_app"].GetVectorImageAsNumpyArray(
        "io.outright")
    assert right_grid_np.shape == (15, 15, 2)
    right_grid_origin = pipeline["stereo_app"].GetImageOrigin("io.outright")
    assert right_grid_origin[0] == 0
    assert right_grid_origin[1] == 0
    right_grid_spacing = pipeline["stereo_app"].GetImageSpacing("io.outright")
    assert right_grid_spacing[0] == step
    assert right_grid_spacing[1] == step

    # Uncomment to update baseline
    #np.save(absolute_data_path("ref_output/right_grid.npy"), right_grid_np)

    right_grid_np_reference = np.load(
        absolute_data_path("ref_output/right_grid.npy"))
    np.testing.assert_allclose(right_grid_np, right_grid_np_reference)


@pytest.mark.unit_tests
def test_build_extract_roi_application():
    """
    Test that input region is correctly use to build the roi extraction
    application
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    region = [100, 200, 300, 400]

    img_otb, app = pipelines.build_extract_roi_application(img, region)

    assert app.GetParameterInt("startx") == region[0]
    assert app.GetParameterInt("starty") == region[1]
    assert app.GetParameterInt("sizex") == region[2] - region[0]
    assert app.GetParameterInt("sizey") == region[3] - region[1]


@pytest.mark.unit_tests
def test_build_mask_pipeline():
    """
    Test that the pipeline is correctly built (case with no input roi)
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    nodata = 0
    mask = absolute_data_path("input/phr_reunion/left_mask.tif")
    out_ptr, pipeline = pipelines.build_mask_pipeline(
        img, grid, nodata, mask, 2387, 2387)

    assert "mask_app" in pipeline
    assert "resampling_app" in pipeline

    out_np = pipeline["resampling_app"].GetVectorImageAsNumpyArray("io.out")
    assert out_np.shape == (2387, 2387, 1)
    origin = pipeline["resampling_app"].GetImageOrigin("io.out")
    assert origin[0] == 0
    assert origin[1] == 0
    spacing = pipeline["resampling_app"].GetImageSpacing("io.out")
    assert spacing[0] == 1
    assert spacing[1] == 1

    assert out_ptr == pipeline["resampling_app"].GetParameterOutputImage(
        "io.out")


@pytest.mark.unit_tests
def test_build_mask_pipeline_with_roi():
    """
    Test that the pipeline is correctly built (case with input roi)
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    nodata = 0
    mask = absolute_data_path("input/phr_reunion/left_mask.tif")
    roi = [100, 200, 300, 400]
    out_ptr, pipeline = pipelines.build_mask_pipeline(
        img, grid, nodata, mask, 2387, 2387, roi)
    assert "mask_app" in pipeline
    assert "resampling_app" in pipeline
    assert "extract_app" in pipeline

    out_np = pipeline["extract_app"].GetVectorImageAsNumpyArray("out")
    assert out_np.shape == (200, 200, 1)
    origin = pipeline["extract_app"].GetImageOrigin("out")
    assert origin[0] == 100
    assert origin[1] == 200
    spacing = pipeline["extract_app"].GetImageSpacing("out")
    assert spacing[0] == 1
    assert spacing[1] == 1

    assert out_ptr == pipeline["extract_app"].GetParameterOutputImage("out")


@pytest.mark.unit_tests
def test_build_image_resampling_pipeline():
    """
    Test that the image resampling pipeline is correctly built
    (case with no input roi)
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    out_ptr, pipeline = pipelines.build_image_resampling_pipeline(
        img, grid, 2387, 2387)

    assert "resampling_app" in pipeline

    out_np = pipeline["resampling_app"].GetVectorImageAsNumpyArray("io.out")
    assert out_np.shape == (2387, 2387, 1)
    origin = pipeline["resampling_app"].GetImageOrigin("io.out")
    assert origin[0] == 0
    assert origin[1] == 0
    spacing = pipeline["resampling_app"].GetImageSpacing("io.out")
    assert spacing[0] == 1
    assert spacing[1] == 1

    assert out_ptr == pipeline["resampling_app"].GetParameterOutputImage(
        "io.out")


@pytest.mark.unit_tests
def test_build_image_resampling_pipeline_with_roi():
    """
    Test that the pipeline is correctly built (case with input roi)
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    roi = [100, 200, 300, 400]
    out_ptr, pipeline = pipelines.build_image_resampling_pipeline(
        img, grid, 2387, 2387, roi)
    assert "resampling_app" in pipeline
    assert "extract_app" in pipeline

    out_np = pipeline["extract_app"].GetVectorImageAsNumpyArray("out")
    assert out_np.shape == (200, 200, 1)
    origin = pipeline["extract_app"].GetImageOrigin("out")
    assert origin[0] == 100
    assert origin[1] == 200
    spacing = pipeline["extract_app"].GetImageSpacing("out")
    assert spacing[0] == 1
    assert spacing[1] == 1

    assert out_ptr == pipeline["extract_app"].GetParameterOutputImage("out")
