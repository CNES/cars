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
