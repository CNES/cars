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
This module contains functions related to the regularisation of the matching outputs
"""

# Standard imports
import logging

# Third party imports
import numpy as np
import xarray as xr

# Cars imports
from cars import parameters as params
from cars import constants as cst
from cars import mask_classes


def update_disp_to_0(disp, ref_ds, sec_ds, input_stereo_cfg):
    """
    Inplace function
    Updates the disparity maps in order to set the final raster's output altitudes to the ones of the input dem ones.
    The updated pixels belong to the classes specified by the mask_classes.set_to_ref_alt_tag of the mask classes
    json files of the stereo input configuration. Their disparities will be set to 0.

    :param disp: disparity dictionary with the reference disparity map (cst.STEREO_REF key) and eventually the
    secondary disparity map (cst.STEREO_SEC key)
    :param ref_ds: reference image dataset containing and eventual multi-classes mask (cst.EPI_MSK key)
    :param sec_ds: secondary image dataset containing and eventual multi-classes mask (cst.EPI_MSK key)
    :param input_stereo_cfg: the input stereo images configuration dictionary
    """
    mask1_classes = input_stereo_cfg[params.input_section_tag].get(params.mask1_classes_tag, None)
    mask2_classes = input_stereo_cfg[params.input_section_tag].get(params.mask2_classes_tag, None)

    mask_ref = None
    if mask1_classes is not None:
        mask_ref = mask_classes.create_msk_from_tag(ref_ds[cst.EPI_MSK].values, mask1_classes,
                                                    mask_classes.set_to_ref_alt_tag, out_msk_dtype=np.bool)

    mask_sec = None
    if mask2_classes is not None and cst.STEREO_SEC in disp:
        mask_sec = mask_classes.create_msk_from_tag(sec_ds[cst.EPI_MSK].values, mask2_classes,
                                                    mask_classes.set_to_ref_alt_tag, out_msk_dtype=np.bool)

    if mask_ref is not None:
        update_disp_ds_from_msk(disp[cst.STEREO_REF], mask_ref)
    if mask_sec is not None and cst.STEREO_SEC in disp:
        update_disp_ds_from_msk(disp[cst.STEREO_SEC], mask_sec)


def update_disp_ds_from_msk(disp, mask):
    """
    Update a disparity dataset to set the pixels indicated to the mask to 0.
    The corresponding pixels are passed to valid ones in the disparity mask.
    A cst.DISP_MSK_SET_TO_INPUT_DEM mask is also added to the dataset with the mask used here.

    :param disp: disparity dataset to update
    :param mask: mask identifying the pixels for which the disparity has to be set to 0
    """
    disp[cst.DISP_MAP].values[mask] = 0
    disp[cst.DISP_MSK].values[mask] = 255
    disp[cst.DISP_MSK_DISP_TO_0] = xr.DataArray(mask, dims=[cst.ROW, cst.COL])
