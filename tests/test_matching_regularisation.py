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
Test module for cars/matching_regularisation.py
"""

# Standard imports
from __future__ import absolute_import

import os
from copy import deepcopy

# Third party imports
import numpy as np
import pytest
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.steps.matching import regularisation

# CARS Tests imports
from .utils import absolute_data_path, assert_same_datasets


@pytest.fixture(scope="module")
def ref_row():
    return 3


@pytest.fixture(scope="module")
def ref_col():
    return 3


@pytest.fixture(scope="module")
def sec_row():
    return 4


@pytest.fixture(scope="module")
def sec_col():
    return 4


@pytest.fixture(scope="module")
def margins():
    return 1, 2, 3, 4


@pytest.fixture(scope="module")
def ref_ds(ref_row, ref_col, margins):  # pylint: disable=redefined-outer-name
    """Returns a small reference dataset"""
    ref_mc_msk = np.array([[1, 2, 0], [0, 0, 3], [0, 0, 0]], dtype=np.uint16)

    ref_mc_msk = np.pad(
        ref_mc_msk, [(margins[1], margins[3]), (margins[0], margins[2])]
    )

    out_ds = xr.Dataset(
        {cst.EPI_MSK: ([cst.ROW, cst.COL], ref_mc_msk)},
        coords={
            cst.ROW: np.array(range(ref_row + margins[1] + margins[3])),
            cst.COL: np.array(range(ref_col + margins[0] + margins[2])),
        },
    )

    out_ds.attrs[cst.EPI_MARGINS] = np.array(
        (-margins[0], -margins[1], margins[2], margins[3])
    )

    return out_ds


@pytest.fixture(scope="module")
def sec_ds(sec_row, sec_col):  # pylint: disable=redefined-outer-name
    """Returns a secondary dataset"""
    sec_mc_msk = np.array(
        [[0, 0, 0, 0], [3, 3, 0, 0], [0, 4, 4, 0], [0, 0, 5, 5]],
        dtype=np.uint16,
    )

    out_ds = xr.Dataset(
        {cst.EPI_MSK: ([cst.ROW, cst.COL], sec_mc_msk)},
        coords={
            cst.ROW: np.array(range(sec_row)),
            cst.COL: np.array(range(sec_col)),
        },
    )

    out_ds.attrs[cst.EPI_MARGINS] = np.array((0, 0, 0, 0))

    return out_ds


@pytest.fixture(scope="module")
def ref_disp(ref_row, ref_col):  # pylint: disable=redefined-outer-name
    """Returns a reference disparity map"""
    np_ref_disp = np.arange(ref_row * ref_col, dtype=np.int16)
    np_ref_disp = np_ref_disp.reshape((ref_row, ref_col))

    np_ref_disp_msk = np.full(
        (ref_row, ref_col), fill_value=255, dtype=np.uint16
    )
    np_ref_disp_msk[2, :] = 0  # last column set to unvalid data

    disp_ds = xr.Dataset(
        {
            cst.DISP_MAP: ([cst.ROW, cst.COL], np_ref_disp),
            cst.DISP_MSK: ([cst.ROW, cst.COL], np_ref_disp_msk),
        },
        coords={
            cst.ROW: np.array(range(ref_row)),
            cst.COL: np.array(range(ref_col)),
        },
    )

    return disp_ds


@pytest.fixture(scope="module")
def sec_disp(sec_row, sec_col):  # pylint: disable=redefined-outer-name
    """Returns a secondary disparity map"""
    np_sec_disp = np.arange(sec_row * sec_col, dtype=np.int16)
    np_sec_disp = np_sec_disp.reshape((sec_row, sec_col))

    np_sec_disp_msk = np.full(
        (sec_row, sec_col), fill_value=255, dtype=np.uint16
    )
    np_sec_disp_msk[3, :] = 0  # last line set to unvalid data

    disp_ds = xr.Dataset(
        {
            cst.DISP_MAP: ([cst.ROW, cst.COL], np_sec_disp),
            cst.DISP_MSK: ([cst.ROW, cst.COL], np_sec_disp_msk),
        },
        coords={
            cst.ROW: np.array(range(sec_row)),
            cst.COL: np.array(range(sec_col)),
        },
    )

    return disp_ds


@pytest.mark.unit_tests
def test_update_disp_to_0_no_tags_in_jsons(
    ref_ds, sec_ds, ref_disp, sec_disp
):  # pylint: disable=redefined-outer-name
    """Tests update disp to 0 with an empty dict"""
    # disp dictionary
    disp = {cst.STEREO_REF: ref_disp, cst.STEREO_SEC: sec_disp}

    # prepare mask path
    mask_path_list = [
        "input",
        "matching_regularisation_input",
        "mask_no_set_to_ref_alt_classes.json",
    ]

    # test
    disp_no_tags_in_json = deepcopy(disp)
    regularisation.update_disp_to_0(
        disp_no_tags_in_json,
        ref_ds,
        sec_ds,
        absolute_data_path(os.path.join(*mask_path_list)),
        absolute_data_path(os.path.join(*mask_path_list)),
    )

    assert_same_datasets(
        disp_no_tags_in_json[cst.STEREO_REF], disp[cst.STEREO_REF]
    )
    assert_same_datasets(
        disp_no_tags_in_json[cst.STEREO_SEC], disp[cst.STEREO_SEC]
    )


@pytest.mark.unit_tests
def test_update_disp_0(
    ref_ds, sec_ds, ref_disp, sec_disp
):  # pylint: disable=redefined-outer-name
    """Tests update disp to 0 with a filled dict"""
    # disp dictionary
    disp = {
        cst.STEREO_REF: deepcopy(ref_disp),
        cst.STEREO_SEC: deepcopy(sec_disp),
    }

    # prepare masks paths
    mask1_path_list = [
        "input",
        "matching_regularisation_input",
        "mask1_set_to_ref_alt_classes.json",
    ]
    mask2_path_list = [
        "input",
        "matching_regularisation_input",
        "mask2_set_to_ref_alt_classes.json",
    ]

    # test
    regularisation.update_disp_to_0(
        disp,
        ref_ds,
        sec_ds,
        absolute_data_path(os.path.join(*mask1_path_list)),
        absolute_data_path(os.path.join(*mask2_path_list)),
    )

    # crop mask to ROI
    ref_roi = [
        int(-ref_ds.attrs[cst.EPI_MARGINS][0]),
        int(-ref_ds.attrs[cst.EPI_MARGINS][1]),
        int(ref_ds.dims[cst.COL] - ref_ds.attrs[cst.EPI_MARGINS][2]),
        int(ref_ds.dims[cst.ROW] - ref_ds.attrs[cst.EPI_MARGINS][3]),
    ]
    c_ref_msk = ref_ds[cst.EPI_MSK].values[
        ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
    ]

    # create ref
    up_ref_disp = deepcopy(ref_disp[cst.DISP_MAP].values)
    up_ref_disp_msk = deepcopy(ref_disp[cst.DISP_MSK].values)

    ref_msk_set_to_ref_alt = np.logical_or(
        np.where(c_ref_msk == 1, True, False),
        np.where(c_ref_msk == 3, True, False),
    )
    up_ref_disp[ref_msk_set_to_ref_alt] = 0
    up_ref_disp_msk[np.where(c_ref_msk == 3, True, False)] = 255

    assert np.allclose(disp[cst.STEREO_REF][cst.DISP_MAP].values, up_ref_disp)
    assert np.allclose(
        disp[cst.STEREO_REF][cst.DISP_MSK].values, up_ref_disp_msk
    )
    assert np.allclose(
        disp[cst.STEREO_REF][cst.DISP_MSK_DISP_TO_0].values,
        ref_msk_set_to_ref_alt,
    )

    up_sec_disp = deepcopy(sec_disp[cst.DISP_MAP].values)
    up_sec_disp_msk = deepcopy(sec_disp[cst.DISP_MSK].values)

    sec_msk_set_to_ref_alt = np.logical_or(
        np.where(sec_ds[cst.EPI_MSK].values == 4, True, False),
        np.where(sec_ds[cst.EPI_MSK].values == 5, True, False),
    )
    up_sec_disp[sec_msk_set_to_ref_alt] = 0
    up_sec_disp_msk[
        np.where(sec_ds[cst.EPI_MSK].values == 5, True, False)
    ] = 255

    assert np.allclose(disp[cst.STEREO_SEC][cst.DISP_MAP].values, up_sec_disp)
    assert np.allclose(
        disp[cst.STEREO_SEC][cst.DISP_MSK].values, up_sec_disp_msk
    )
    assert np.allclose(
        disp[cst.STEREO_SEC][cst.DISP_MSK_DISP_TO_0].values,
        sec_msk_set_to_ref_alt,
    )
