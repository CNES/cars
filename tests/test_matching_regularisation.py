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
import xarray as xr
from copy import deepcopy

from utils import absolute_data_path, assert_same_datasets

from cars import constants as cst
from cars import parameters as params
from cars import matching_regularisation


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
def ref_ds(ref_row, ref_col):

    ref_mc_msk = np.array([[1, 2, 0],
                           [0, 0, 3],
                           [0, 0, 0]], dtype=np.uint16)

    ds = xr.Dataset({
        cst.EPI_MSK: ([cst.ROW, cst.COL], ref_mc_msk)
    }, coords={cst.ROW: np.array(range(ref_row)), cst.COL: np.array(range(ref_col))})

    return ds


@pytest.fixture(scope="module")
def sec_ds(sec_row, sec_col):
    sec_mc_msk = np.array([[0, 0, 0, 0],
                           [3, 3, 0, 0],
                           [0, 4, 4, 0],
                           [0, 0, 5, 5]], dtype=np.uint16)

    ds = xr.Dataset({
        cst.EPI_MSK: ([cst.ROW, cst.COL], sec_mc_msk)
    }, coords={cst.ROW: np.array(range(sec_row)), cst.COL: np.array(range(sec_col))})

    return ds


@pytest.fixture(scope="module")
def ref_disp(ref_row, ref_col):
    ref_disp = np.arange(ref_row * ref_col, dtype=np.int16)
    ref_disp = ref_disp.reshape((ref_row, ref_col))

    ref_disp_msk = np.full((ref_row, ref_col), fill_value=255, dtype=np.uint16)
    ref_disp_msk[2, :] = 0  # last column set to unvalid data

    disp_ds = xr.Dataset({
        cst.DISP_MAP: ([cst.ROW, cst.COL], ref_disp),
        cst.DISP_MSK: ([cst.ROW, cst.COL], ref_disp_msk)
    }, coords={cst.ROW: np.array(range(ref_row)), cst.COL: np.array(range(ref_col))})

    return disp_ds


@pytest.fixture(scope="module")
def sec_disp(sec_row, sec_col):
    sec_disp = np.arange(sec_row * sec_col, dtype=np.int16)
    sec_disp = sec_disp.reshape((sec_row, sec_col))

    sec_disp_msk = np.full((sec_row, sec_col), fill_value=255, dtype=np.uint16)
    sec_disp_msk[3, :] = 0  # last line set to unvalid data

    disp_ds = xr.Dataset({
        cst.DISP_MAP: ([cst.ROW, cst.COL], sec_disp),
        cst.DISP_MSK: ([cst.ROW, cst.COL], sec_disp_msk)
    }, coords={cst.ROW: np.array(range(sec_row)), cst.COL: np.array(range(sec_col))})

    return disp_ds


@pytest.mark.unit_tests
def test_update_disp_to_0_no_tags_in_jsons(ref_ds, sec_ds, ref_disp, sec_disp):

    # disp dictionary
    disp = {
        cst.STEREO_REF: ref_disp,
        cst.STEREO_SEC: sec_disp
    }

    # test
    disp_no_tags_in_json = deepcopy(disp)
    matching_regularisation.update_disp_to_0(disp_no_tags_in_json, ref_ds, sec_ds,
                         absolute_data_path("input/matching_regularisation_input/mask_no_set_to_ref_alt_classes.json"),
                         absolute_data_path("input/matching_regularisation_input/mask_no_set_to_ref_alt_classes.json"))

    assert_same_datasets(disp_no_tags_in_json[cst.STEREO_REF], disp[cst.STEREO_REF])
    assert_same_datasets(disp_no_tags_in_json[cst.STEREO_SEC], disp[cst.STEREO_SEC])


@pytest.mark.unit_tests
def test_update_disp_0(ref_ds, sec_ds, ref_disp, sec_disp):

    # disp dictionary
    disp = {
        cst.STEREO_REF: deepcopy(ref_disp),
        cst.STEREO_SEC: deepcopy(sec_disp)
    }

    # test
    matching_regularisation.\
        update_disp_to_0(disp, ref_ds, sec_ds,
                         absolute_data_path("input/matching_regularisation_input/mask1_set_to_ref_alt_classes.json"),
                         absolute_data_path("input/matching_regularisation_input/mask2_set_to_ref_alt_classes.json"))

    up_ref_disp = deepcopy(ref_disp[cst.DISP_MAP].values)
    up_ref_disp_msk = deepcopy(ref_disp[cst.DISP_MSK].values)

    ref_msk_set_to_ref_alt = np.logical_or(np.where(ref_ds[cst.EPI_MSK].values == 1, True, False),
                                             np.where(ref_ds[cst.EPI_MSK].values == 3, True, False))
    up_ref_disp[ref_msk_set_to_ref_alt] = 0
    up_ref_disp_msk[np.where(ref_ds[cst.EPI_MSK].values == 3, True, False)] = 255

    assert np.allclose(disp[cst.STEREO_REF][cst.DISP_MAP].values, up_ref_disp)
    assert np.allclose(disp[cst.STEREO_REF][cst.DISP_MSK].values, up_ref_disp_msk)
    assert np.allclose(disp[cst.STEREO_REF][cst.DISP_MSK_DISP_TO_0].values,
                       ref_msk_set_to_ref_alt)

    up_sec_disp = deepcopy(sec_disp[cst.DISP_MAP].values)
    up_sec_disp_msk = deepcopy(sec_disp[cst.DISP_MSK].values)

    sec_msk_set_to_ref_alt = np.logical_or(np.where(sec_ds[cst.EPI_MSK].values == 4, True, False),
                                             np.where(sec_ds[cst.EPI_MSK].values == 5, True, False))
    up_sec_disp[sec_msk_set_to_ref_alt] = 0
    up_sec_disp_msk[np.where(sec_ds[cst.EPI_MSK].values == 5, True, False)] = 255

    assert np.allclose(disp[cst.STEREO_SEC][cst.DISP_MAP].values, up_sec_disp)
    assert np.allclose(disp[cst.STEREO_SEC][cst.DISP_MSK].values, up_sec_disp_msk)
    assert np.allclose(disp[cst.STEREO_SEC][cst.DISP_MSK_DISP_TO_0].values,
                       sec_msk_set_to_ref_alt)
