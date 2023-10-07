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

    assert res == 400

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
    corr_cfg = create_corr_conf(corr_cfg)

    disp_min_right = -13 * np.ones(left_input["im"].values.shape)
    disp_max_right = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_right=disp_min_right,
        disp_max_right=disp_max_right,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)
    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(absolute_data_path(
    #     "ref_output/disp1_ref_pandora.nc"))

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
    corr_cfg = create_corr_conf(corr_cfg)

    disp_min_right = -43 * np.ones(left_input["im"].values.shape)
    disp_max_right = 41 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_right=disp_min_right,
        disp_max_right=disp_max_right,
    )

    # TODO add validity mask input

    assert output[cst_disp.MAP].shape == (90, 90)
    assert output[cst_disp.VALID].shape == (90, 90)
    assert output[cst_disp.MAP].shape == (90, 90)
    assert output[cst_disp.VALID].shape == (90, 90)

    np.testing.assert_allclose(
        output.attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )
    np.testing.assert_allclose(
        output.attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )

    # Uncomment to update baseline
    # output.to_netcdf(absolute_data_path(
    #     "ref_output/disp3_ref_pandora.nc"))

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
    corr_cfg = create_corr_conf(corr_cfg)

    disp_min_right = -13 * np.ones(left_input["im"].values.shape)
    disp_max_right = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_right=disp_min_right,
        disp_max_right=disp_max_right,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)
    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
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
    corr_cfg = create_corr_conf(corr_cfg)

    disp_min_right = -13 * np.ones(left_input["im"].values.shape)
    disp_max_right = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_right=disp_min_right,
        disp_max_right=disp_max_right,
        compute_disparity_masks=True,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora_msk_ref.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_ref.nc")
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
    corr_cfg = create_corr_conf(corr_cfg)

    disp_min_right = -13 * np.ones(left_input["im"].values.shape)
    disp_max_right = 14 * np.ones(left_input["im"].values.shape)

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min_right=disp_min_right,
        disp_max_right=disp_max_right,
        compute_disparity_masks=True,
    )

    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)
    assert output[cst_disp.MAP].shape == (120, 110)
    assert output[cst_disp.VALID].shape == (120, 110)

    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output.attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output.to_netcdf(
    #     absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc")
    # )

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc")
    )
    assert_same_datasets(output, ref, atol=5.0e-6)
