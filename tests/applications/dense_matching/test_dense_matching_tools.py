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

# Standard imports
import math
from copy import deepcopy

# Third party imports
import numpy as np
import pytest
import xarray as xr

from cars.applications.dense_matching import dense_matching_tools

# CARS imports
from cars.conf import input_parameters as in_params
from cars.conf import mask_classes
from cars.core import constants as cst

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_datasets,
    create_corr_conf,
    read_mask_classes,
)


@pytest.mark.unit_tests
def test_optimal_tile_size():
    """
    Test optimal_tile_size function
    """
    disp = 61
    mem = 313

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=1000, otb_max_ram_hint=mem
    )

    assert res == 400

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=300, otb_max_ram_hint=mem
    )

    assert res == 300

    res = dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=500, max_tile_size=1000, otb_max_ram_hint=mem
    )

    assert res == 500

    # Test case where default tile size is returned
    assert (
        dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
            -1000,
            1000,
            min_tile_size=0,
            max_tile_size=1000,
            otb_max_ram_hint=100,
            tile_size_rounding=33,
        )
        == 33
    )


@pytest.mark.unit_tests
def test_compute_disparity_1(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )

    # Get mask values
    mask1_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK1_CLASSES_TAG, None
    )
    mask2_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK2_CLASSES_TAG, None
    )

    if mask1_classes is not None:
        mask1_classes_dict = read_mask_classes(mask1_classes)
        mask1_ignored_by_corr = mask1_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask1_ignored_by_corr = None
    if mask2_classes is not None:
        mask2_classes_dict = read_mask_classes(mask2_classes)
        mask2_ignored_by_corr = mask2_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask2_ignored_by_corr = None

    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min,
        disp_max,
        mask1_ignored_by_corr=mask1_ignored_by_corr,
        mask2_ignored_by_corr=mask2_ignored_by_corr,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([387, 180, 564, 340]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp1_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    sec = xr.open_dataset(absolute_data_path("ref_output/disp1_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_3(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on paca dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_right.nc")
    )

    # Get mask values
    mask1_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK1_CLASSES_TAG, None
    )
    mask2_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK2_CLASSES_TAG, None
    )

    if mask1_classes is not None:
        mask1_classes_dict = read_mask_classes(mask1_classes)
        mask1_ignored_by_corr = mask1_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask1_ignored_by_corr = None
    if mask2_classes is not None:
        mask2_classes_dict = read_mask_classes(mask2_classes)
        mask2_ignored_by_corr = mask2_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask2_ignored_by_corr = None

    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -43
    disp_max = 41

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min,
        disp_max,
        mask1_ignored_by_corr=mask1_ignored_by_corr,
        mask2_ignored_by_corr=mask2_ignored_by_corr,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (90, 90)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (90, 90)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (170, 254)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (170, 254)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([16417, 23120, 16671, 23290]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp3_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp3_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp3_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    sec = xr.open_dataset(absolute_data_path("ref_output/disp3_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_ref(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
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

    # Get mask values
    mask1_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK1_CLASSES_TAG, None
    )
    mask2_classes = images_and_grids_conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK2_CLASSES_TAG, None
    )

    if mask1_classes is not None:
        mask1_classes_dict = read_mask_classes(mask1_classes)
        mask1_ignored_by_corr = mask1_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask1_ignored_by_corr = None
    if mask2_classes is not None:
        mask2_classes_dict = read_mask_classes(mask2_classes)
        mask2_ignored_by_corr = mask2_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask2_ignored_by_corr = None

    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min,
        disp_max,
        mask1_ignored_by_corr=mask1_ignored_by_corr,
        mask2_ignored_by_corr=mask2_ignored_by_corr,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora_msk_ref.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora_msk_ref.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_ref.nc")
    )
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)

    sec = xr.open_dataset(
        absolute_data_path("ref_output/disp1_sec_pandora_msk_ref.nc")
    )
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)

    # test multi-classes left mask
    left_input[cst.EPI_MSK].values[10, 10] = 1  # valid class
    left_input[cst.EPI_MSK].values[10, 140] = 2  # nonvalid class
    conf = deepcopy(images_and_grids_conf)
    conf["input"]["mask1_classes"] = absolute_data_path(
        "input/intermediate_results/data1_ref_left_mask_classes.json"
    )

    # Get mask values
    mask1_classes = conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK1_CLASSES_TAG, None
    )
    mask2_classes = conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK2_CLASSES_TAG, None
    )

    if mask1_classes is not None:
        mask1_classes_dict = read_mask_classes(mask1_classes)
        mask1_ignored_by_corr = mask1_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask1_ignored_by_corr = None
    if mask2_classes is not None:
        mask2_classes_dict = read_mask_classes(mask2_classes)
        mask2_ignored_by_corr = mask2_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask2_ignored_by_corr = None

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min,
        disp_max,
        mask1_ignored_by_corr=mask1_ignored_by_corr,
        mask2_ignored_by_corr=mask2_ignored_by_corr,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_sec(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
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
    conf = deepcopy(images_and_grids_conf)
    conf["input"]["mask2_classes"] = absolute_data_path(
        "input/intermediate_results/data1_ref_right_mask_classes.json"
    )

    # Get mask values
    mask1_classes = conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK1_CLASSES_TAG, None
    )
    mask2_classes = conf[in_params.INPUT_SECTION_TAG].get(
        in_params.MASK2_CLASSES_TAG, None
    )

    if mask1_classes is not None:
        mask1_classes_dict = read_mask_classes(mask1_classes)
        mask1_ignored_by_corr = mask1_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask1_ignored_by_corr = None
    if mask2_classes is not None:
        mask2_classes_dict = read_mask_classes(mask2_classes)
        mask2_ignored_by_corr = mask2_classes_dict.get(
            mask_classes.ignored_by_corr_tag, None
        )
    else:
        mask2_ignored_by_corr = None

    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching_tools.compute_disparity(
        left_input,
        right_input,
        corr_cfg,
        disp_min,
        disp_max,
        mask1_ignored_by_corr=mask1_ignored_by_corr,
        mask2_ignored_by_corr=mask2_ignored_by_corr,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([387, 180, 564, 340]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora_msk_sec.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora_msk_sec.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc")
    )
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)

    sec = xr.open_dataset(
        absolute_data_path("ref_output/disp1_sec_pandora_msk_sec.nc")
    )
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_mask_to_use_in_pandora():
    """
    Test compute_mask_to_use_in_pandora with a cloud "data1_ref_right_masked.nc"
    """

    right_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_right_masked.nc"
        )
    )

    test_mask = dense_matching_tools.compute_mask_to_use_in_pandora(
        right_input, cst.EPI_MSK, [100]
    )

    ref_msk = np.copy(right_input[cst.EPI_MSK].values)
    ref_msk.astype(np.int16)
    ref_msk[np.where(right_input[cst.EPI_MSK].values == 100, True, False)] = 1

    assert np.allclose(test_mask, ref_msk)


@pytest.mark.unit_tests
def test_create_inside_sec_roi_mask():
    """
    Test create_inside_sec_roi_mask function
    """
    # create fake dataset with margins values and image dimension
    dataset_left_margin = 1
    dataset_right_margin = 2
    dataset_top_margin = 1
    dataset_bottom_margin = 1
    dataset_nb_col = 6
    dataset_nb_row = 5
    test_dataset = xr.Dataset(
        {
            cst.EPI_IMAGE: (
                [cst.ROW, cst.COL],
                np.ones((dataset_nb_row, dataset_nb_col), dtype=bool),
            )
        }
    )
    test_dataset.attrs[cst.EPI_MARGINS] = np.array(
        [
            dataset_left_margin,
            dataset_top_margin,
            dataset_right_margin,
            dataset_bottom_margin,
        ]
    )

    # create fake disp map and mask
    disp_nb_col = 7
    disp_nb_row = 6
    disp_value = 0.5
    disp = np.full((disp_nb_row, disp_nb_col), disp_value)
    mask = np.full((disp_nb_row, disp_nb_col), 255, dtype=np.int16)

    # add an invalid pixel in the useful zone
    mask[1, 1] = 0

    msk = dense_matching_tools.create_inside_sec_roi_mask(
        disp, mask, test_dataset
    )

    # create reference data
    ref_msk = np.zeros((disp_nb_row, disp_nb_col), dtype=np.int16)
    ref_msk[
        math.ceil(disp_value) : disp_nb_row - dataset_bottom_margin,
        math.ceil(disp_value) : disp_nb_col
        - dataset_right_margin
        - math.floor(disp_value),
    ] = 255
    ref_msk[1, 1] = 0

    assert np.allclose(msk, ref_msk)


@pytest.mark.unit_tests
def test_estimate_color_from_disparity():
    """
    Test estimate_color_from_disparity function
    """
    # create fake disp map and mask
    margins = [1, 1, 1, 1]
    disp_nb_col = 12
    disp_nb_row = 10
    disp_value = 0.75
    disp_row = np.array(range(disp_nb_row))
    disp_col = np.array(range(disp_nb_col))

    disp = np.full((disp_nb_row, disp_nb_col), disp_value)

    mask = np.full((disp_nb_row - 2, disp_nb_col - 3), 255, dtype=np.int16)
    mask = np.pad(mask, ((1, 1), (1, 2)), constant_values=0)
    mask[2, 2] = 0

    disp_dataset = xr.Dataset(
        {
            cst.DISP_MAP: ([cst.ROW, cst.COL], np.copy(disp)),
            cst.DISP_MSK: ([cst.ROW, cst.COL], np.copy(mask)),
        },
        coords={cst.ROW: disp_row, cst.COL: disp_col},
    )

    disp_dataset.attrs[cst.ROI] = [
        margins[0],
        margins[1],
        disp_nb_col - margins[2],
        disp_nb_row - margins[3],
    ]
    disp_dataset.attrs[cst.ROI_WITH_MARGINS] = [0, 0, disp_nb_col, disp_nb_row]
    disp_dataset.attrs[cst.EPI_FULL_SIZE] = [100, 100]

    # create fake color image
    clr_nb_col = 10
    clr_nb_row = 8
    clr_nb_band = 3
    clr_row = np.array(range(clr_nb_row))
    clr_col = np.array(range(clr_nb_col))

    clr = np.ones((clr_nb_band, clr_nb_row, clr_nb_col), dtype=np.float64)
    val = np.arange(clr_nb_row * clr_nb_col)
    val = val.reshape(clr_nb_row, clr_nb_col)
    for band in range(clr_nb_band):
        clr[band, :, :] = val

    clr_mask = np.full((clr_nb_row, clr_nb_col), 255, dtype=np.int16)
    clr_mask[4, 4] = 0
    clr_dataset = xr.Dataset(
        {
            cst.EPI_COLOR: ([cst.BAND, cst.ROW, cst.COL], np.copy(clr)),
            cst.EPI_COLOR_MSK: ([cst.ROW, cst.COL], np.copy(clr_mask)),
        },
        coords={
            cst.BAND: range(clr_nb_band),
            cst.ROW: clr_row,
            cst.COL: clr_col,
        },
    )

    # create fake secondary dataset
    sec_margins = [1, 1, 1, 1]
    sec_dataset = xr.Dataset()
    sec_dataset.attrs[cst.EPI_MARGINS] = np.array(sec_margins)

    # interpolate color
    interp_clr_dataset = dense_matching_tools.estimate_color_from_disparity(
        disp_dataset, sec_dataset, clr_dataset
    )

    # reference
    ref_mask = mask.astype(bool)
    ref_mask[margins[0] + 4, margins[2] + 4 - math.ceil(disp_value)] = False
    ref_mask = ~ref_mask

    assert np.allclose(
        ref_mask, np.isnan(interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :])
    )

    ref_data = np.ceil(disp)
    ref_data[margins[1] : -margins[3], margins[0] : -margins[2]] += val
    ref_data[ref_mask] = 0

    interp_clr_msk = np.isnan(interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :])
    interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :][interp_clr_msk] = 0

    assert np.allclose(
        ref_data, interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :]
    )
