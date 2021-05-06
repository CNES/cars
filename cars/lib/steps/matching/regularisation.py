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
Matching regularisation module:
contains matching outputs regularisation functions
"""

# Standard imports
from typing import Union

# Third party imports
import numpy as np
import xarray as xr

# Cars imports
from cars.core import constants as cst
from cars.conf import mask_classes


def update_disp_to_0(
        disp,
        ref_ds,
        sec_ds,
        mask_ref_classes: Union[str, None]=None,
        mask_sec_classes: Union[str, None]=None):
    """
    Inplace function
    Updates the disparity maps in order to set the indicated pixels to 0.
    The updated pixels belong to the classes specified
    by the mask_classes.set_to_ref_alt_tag of the mask classes json files.

    :param disp: disparity dictionary
        with the reference disparity map (cst.STEREO_REF key) and
        eventually the secondary disparity map (cst.STEREO_SEC key)
    :param ref_ds: reference image dataset
        containing and eventual multi-classes mask (cst.EPI_MSK key)
    :param sec_ds: secondary image dataset
        containing and eventual multi-classes mask (cst.EPI_MSK key)
    :param mask_ref_classes: path to the json file
        describing mask classes usage of the reference image mask
    :param mask_sec_classes: path to the json file
        describing mask classes usage of the secondary image mask
    """
    mask_ref = None
    if mask_ref_classes is not None:
        mask_ref = mask_classes.create_msk_from_tag(
                    ref_ds[cst.EPI_MSK].values, mask_ref_classes,
                    mask_classes.set_to_ref_alt_tag, out_msk_dtype=np.bool)

        # crop mask to ROI
        ref_roi = [int(-ref_ds.attrs[cst.EPI_MARGINS][0]),
                   int(-ref_ds.attrs[cst.EPI_MARGINS][1]),
                   int(ref_ds.dims[cst.COL] - \
                       ref_ds.attrs[cst.EPI_MARGINS][2]),
                   int(ref_ds.dims[cst.ROW] - \
                       ref_ds.attrs[cst.EPI_MARGINS][3])]
        mask_ref = mask_ref[ref_roi[1]:ref_roi[3], ref_roi[0]:ref_roi[2]]

    mask_sec = None
    if mask_sec_classes is not None and cst.STEREO_SEC in disp:
        mask_sec = mask_classes.create_msk_from_tag(
                    sec_ds[cst.EPI_MSK].values, mask_sec_classes,
                    mask_classes.set_to_ref_alt_tag, out_msk_dtype=np.bool)

        # crop mask to ROI
        sec_roi = [int(-sec_ds.attrs[cst.EPI_MARGINS][0]),
                   int(-sec_ds.attrs[cst.EPI_MARGINS][1]),
                   int(sec_ds.dims[cst.COL] - \
                       sec_ds.attrs[cst.EPI_MARGINS][2]),
                   int(sec_ds.dims[cst.ROW] - \
                       sec_ds.attrs[cst.EPI_MARGINS][3])]
        mask_sec = mask_sec[sec_roi[1]:sec_roi[3], sec_roi[0]:sec_roi[2]]

    if mask_ref is not None:
        update_disp_ds_from_msk(disp[cst.STEREO_REF], mask_ref)
    if mask_sec is not None and cst.STEREO_SEC in disp:
        update_disp_ds_from_msk(disp[cst.STEREO_SEC], mask_sec)


def update_disp_ds_from_msk(disp, mask):
    """
    Update a disparity dataset to set the indicated pixels to the mask to 0.
    The corresponding pixels are passed to valid ones in the disparity mask.

    A cst.DISP_MSK_SET_TO_INPUT_DEM mask is also added
    to the dataset with the mask used here.

    :param disp: disparity dataset to update
    :param mask: mask identifying the pixels
        for which the disparity has to be set to 0
    """
    disp[cst.DISP_MAP].values[mask] = 0
    disp[cst.DISP_MSK].values[mask] = 255
    disp[cst.DISP_MSK_DISP_TO_0] = xr.DataArray(mask, dims=[cst.ROW, cst.COL])
