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
Test module for cars/steps/matching/dense_matching_tools.py
Important : Uses conftest.py for shared pytest fixtures
"""

# Third party imports
import numpy as np
import pytest
import xarray as xr

from cars.applications.dense_matching import dense_matching_tools

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import inputs

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_datasets,
    corr_conf_defaut,
    corr_conf_with_confidence,
    create_corr_conf,
)


@pytest.mark.unit_tests
def test_optimal_tile_size():
    """
    Test optimal_tile_size function
    """
    disp = 61
    mem = 313

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=1000, max_ram_per_worker=mem
    )

    assert res == 350

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=300, max_ram_per_worker=mem
    )

    assert res == 300

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=500, max_tile_size=1000, max_ram_per_worker=mem
    )

    assert res == 500

    # Test case where default tile size is returned
    assert (
        dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
            -1000,
            1000,
            min_tile_size=0,
            max_tile_size=1000,
            max_ram_per_worker=100,
            tile_size_rounding=33,
        )
        == 33
    )


@pytest.mark.unit_tests
def test_get_max_disp_from_opt_tile_size():
    """
    Test get_max_disp_from_opt_tile_size function
    """

    max_ram_per_worker = 313

    max_range = dense_matching_tools.get_max_disp_from_opt_tile_size(
        300, max_ram_per_worker, margin=20, used_disparity_range=0
    )

    assert max_range == 76

    max_range = dense_matching_tools.get_max_disp_from_opt_tile_size(
        300, max_ram_per_worker, margin=20, used_disparity_range=100
    )

    assert max_range == 58


@pytest.mark.unit_tests
def test_compute_disparity_1():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )

    # Pandora configuration
    corr_cfg = corr_conf_defaut()
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    disp_min_grid = -13 * np.ones(left_input["im"].values.shape)
    disp_max_grid = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(absolute_data_path("ref_output/disp1_ref_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp1_ref_pandora.nc"))
    assert_same_datasets(output, ref, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_3():
    """
    Test compute_disparity on paca dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_right.nc")
    )

    # Pandora configuration
    corr_cfg = corr_conf_defaut()
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    disp_min_grid = -43 * np.ones(left_input["im"].values.shape)
    disp_max_grid = 41 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
    )

    # TODO add validity mask input

    assert output[cst_disp.MAP].shape == (90, 90)
    assert output[cst_disp.VALID].shape == (90, 90)

    np.testing.assert_allclose(
        output.attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )

    # Uncomment to update baseline
    # output.to_netcdf(absolute_data_path("ref_output/disp3_ref_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp3_ref_pandora.nc"))
    assert_same_datasets(output, ref, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_with_all_confidences():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )

    # Pandora configuration
    corr_cfg = corr_conf_with_confidence()
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    disp_min_grid = -13 * np.ones(left_input["im"].values.shape)
    disp_max_grid = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(
    #     absolute_data_path("ref_output/disp_with_confidences_ref_pandora.nc")
    # )
    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp_with_confidences_ref_pandora.nc")
    )
    assert_same_datasets(output, ref, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_ref():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_left_masked.nc"
        )
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )

    # Pandora configuration
    corr_cfg = corr_conf_defaut()
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    disp_min_grid = -13 * np.ones(left_input["im"].values.shape)
    disp_max_grid = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        compute_disparity_masks=True,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(
    #     absolute_data_path("ref_output/disp1_ref_pandora_mask_ref.nc")
    # )

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_mask_ref.nc")
    )
    assert_same_datasets(output, ref, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_sec():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_right_masked.nc"
        )
    )

    # Pandora configuration
    corr_cfg = corr_conf_defaut()
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    disp_min_grid = -13 * np.ones(left_input["im"].values.shape)
    disp_max_grid = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        compute_disparity_masks=True,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(
    #     absolute_data_path("ref_output/disp1_ref_pandora_mask_sec.nc")
    # )

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_mask_sec.nc")
    )
    assert_same_datasets(output, ref, atol=5.0e-6)


@pytest.mark.unit_tests
def test_estimate_right_grid_disp():
    """
    Test estimate_right_grid_disp
    """

    left_grid_min = np.array([[-2, -1, 0, 1, 2, 3], [3, 2, 1, 0, -1, -2]])

    left_grid_max = left_grid_min + 2

    # compute
    (
        disp_min_right_grid,
        disp_max_right_grid,
    ) = dense_matching_tools.estimate_right_grid_disp(
        left_grid_min, left_grid_max
    )

    # assert
    ref_right_min = np.array([[0, 0, -1, -1, -2, -2], [-5, -5, -5, -3, -4, -5]])
    ref_right_max = np.array([[1, 0, 0, 0, 0, 0], [2, 2, 2, 2, 1, 0]])

    assert (disp_min_right_grid == ref_right_min).all()
    assert (disp_max_right_grid == ref_right_max).all()


@pytest.mark.unit_tests
def test_estimate_right_classif_on_left():
    """
    Test estimate_right_classif_on_left
    """

    right_classif, _ = inputs.rasterio_read_as_array(
        absolute_data_path(
            "input/test_classification/epi_img_right_classif.tif"
        )
    )

    disp_map, _ = inputs.rasterio_read_as_array(
        absolute_data_path("input/test_classification/epi_disp.tif")
    )
    # add nan
    disp_map[2::20, 4::25] = np.nan

    # Test1: one band

    left_from_right_classif = (
        dense_matching_tools.estimate_right_classif_on_left(
            np.expand_dims(right_classif, axis=0), disp_map, None, -10, 10
        )
    )

    # Uncomment to update baseline
    # np.save(
    #     absolute_data_path(
    #         "ref_output/dense_matching_classif_right1.npy"
    #     ),
    #     left_from_right_classif
    # )

    ref_array = np.load(
        absolute_data_path("ref_output/dense_matching_classif_right1.npy")
    )

    # assert
    np.testing.assert_allclose(left_from_right_classif, ref_array)

    # Test 2: 2 bands
    left_from_right_classif = (
        dense_matching_tools.estimate_right_classif_on_left(
            np.stack([right_classif, right_classif], axis=0),
            disp_map,
            None,
            -10,
            10,
        )
    )

    # Uncomment to update baseline
    # np.save(
    #     absolute_data_path(
    #         "ref_output/dense_matching_classif_right2.npy"
    #     ),
    #     left_from_right_classif
    # )

    ref_array = np.load(
        absolute_data_path("ref_output/dense_matching_classif_right2.npy")
    )

    # assert
    np.testing.assert_allclose(left_from_right_classif, ref_array)


@pytest.mark.unit_tests
def test_merge_classif_left_right():
    """
    Test merge_classif_left_right
    """

    left_classif, _ = inputs.rasterio_read_as_array(
        absolute_data_path("input/test_classification/epi_img_left_classif.tif")
    )
    right_classif, _ = inputs.rasterio_read_as_array(
        absolute_data_path(
            "input/test_classification/epi_img_right_classif.tif"
        )
    )
    disp_map, _ = inputs.rasterio_read_as_array(
        absolute_data_path("input/test_classification/epi_disp.tif")
    )

    left_classif = np.stack([left_classif, left_classif], axis=0)
    left_from_right_classif = (
        dense_matching_tools.estimate_right_classif_on_left(
            np.stack([right_classif, right_classif], axis=0),
            disp_map,
            None,
            -10,
            10,
        )
    )

    # Test 1: same keys

    merged_clasif, band_list = dense_matching_tools.merge_classif_left_right(
        left_classif,
        ["cloud", "building"],
        left_from_right_classif,
        ["cloud", "building"],
    )

    # Uncomment to update baseline
    # np.save(
    #     absolute_data_path(
    #         "ref_output/dense_matching_merged_classif1.npy"
    #     ),
    #     merged_clasif
    # )

    ref_array = np.load(
        absolute_data_path("ref_output/dense_matching_merged_classif1.npy")
    )

    # assert
    np.testing.assert_allclose(merged_clasif, ref_array)
    assert band_list == ["cloud", "building"]

    # Test 2: different keys
    merged_clasif, band_list = dense_matching_tools.merge_classif_left_right(
        left_classif,
        ["cloud", "building"],
        left_from_right_classif,
        ["cloud", "forest"],
    )

    # Uncomment to update baseline
    # np.save(
    #     absolute_data_path(
    #         "ref_output/dense_matching_merged_classif2.npy"
    #     ),
    #     merged_clasif
    # )

    ref_array = np.load(
        absolute_data_path("ref_output/dense_matching_merged_classif2.npy")
    )

    # assert
    np.testing.assert_allclose(merged_clasif, ref_array)
    assert merged_clasif.shape[0] == 3
    assert band_list == ["cloud", "building", "forest"]
