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
# pylint: disable=too-many-lines
"""
This module is responsible for the filling disparity algorithms:
thus it fills the disparity map with values estimated according to
their neighbourhood.
"""

# Third party imports
import numpy as np
import xarray as xr

# Cars import
from cars.core import constants as cst


def add_empty_filling_band(
    output_dataset: xr.Dataset,
    filling_types: list,
):
    """
    Add filling attribute to dataset or band to filling attribute
    if it already exists

    :param output_dataset: output dataset
    :param filling: input mask of filled pixels
    :param band_filling: type of filling (zero padding or plane)

    """
    nb_band = len(filling_types)
    nb_row = len(output_dataset.coords[cst.ROW])
    nb_col = len(output_dataset.coords[cst.COL])
    filling = np.zeros((nb_band, nb_row, nb_col), dtype=bool)
    filling = xr.Dataset(
        data_vars={
            cst.EPI_FILLING: ([cst.BAND_FILLING, cst.ROW, cst.COL], filling)
        },
        coords={
            cst.BAND_FILLING: filling_types,
            cst.ROW: output_dataset.coords[cst.ROW],
            cst.COL: output_dataset.coords[cst.COL],
        },
    )
    # Add band to EPI_FILLING attribute or create the attribute
    return xr.merge([output_dataset, filling])


def add_empty_crop_disp_range_band(
    output_dataset: xr.Dataset,
    crop_disp_range_types: list,
):
    """
    Add filling attribute to dataset or band to filling attribute
    if it already exists

    :param output_dataset: output dataset
    :param filling: input mask of filled pixels
    :param band_filling: type of filling (zero padding or plane)

    """
    nb_band = len(crop_disp_range_types)
    nb_row = len(output_dataset.coords[cst.ROW])
    nb_col = len(output_dataset.coords[cst.COL])
    crop_disp_range_data = np.zeros((nb_band, nb_row, nb_col), dtype=bool)
    crop_disp_range = xr.Dataset(
        data_vars={
            cst.CROPPED_DISPARITY_RANGE: (
                [cst.BAND_CROP_DISP_RANGE, cst.ROW, cst.COL],
                crop_disp_range_data,
            )
        },
        coords={
            cst.BAND_CROP_DISP_RANGE: crop_disp_range_types,
            cst.ROW: output_dataset.coords[cst.ROW],
            cst.COL: output_dataset.coords[cst.COL],
        },
    )

    # Add band to EPI_FILLING attribute or create the attribute
    return xr.merge([output_dataset, crop_disp_range])


def update_filling(
    output_dataset: xr.Dataset,
    filling: np.ndarray = None,
    filling_type: str = None,
):
    """
    Update filling attribute of dataset with an additional mask

    :param output_dataset: output dataset
    :param filling: input mask of filled pixels
    :param band_filling: type of filling (zero padding or plane)

    """
    # Select accurate band of output according to the type of filling
    filling_type = {cst.BAND_FILLING: filling_type}
    # Add True values from inputmask to output accurate band
    filling = filling.astype(bool)
    output_dataset[cst.EPI_FILLING].sel(**filling_type).values[filling] = True


def update_crop_disp_range(
    output_dataset: xr.Dataset,
    crop_disp_range: np.ndarray = None,
    crop_disp_range_type: str = None,
):
    """
    Update filling attribute of dataset with an additional mask

    :param output_dataset: output dataset
    :param crop_disp_range: input mask of filled pixels
    :param crop_disp_range_type: the type of crop_disp_range

    """
    # Select accurate band of output according to the type of filling
    crop_disp_range_type = {cst.BAND_CROP_DISP_RANGE: crop_disp_range_type}
    # Add True values from inputmask to output accurate band
    crop_disp_range = crop_disp_range.astype(bool)
    output_dataset[cst.CROPPED_DISPARITY_RANGE].sel(
        **crop_disp_range_type
    ).values[crop_disp_range] = True
