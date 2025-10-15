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
"""
This module is responsible for the dense matching algorithms:
- thus it creates a disparity map from a pair of images
"""
# pylint: disable=too-many-lines

# Standard imports
import logging
from typing import Dict

import numpy as np
import pandora
import xarray as xr

# Third party imports
from pandora.check_configuration import check_datasets
from pandora.state_machine import PandoraMachine
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from cars.applications.dense_matching import dense_matching_wrappers as dm_wrap

# CARS imports
from cars.core import constants as cst
from cars.core import inputs


def compute_disparity_grid(
    disp_range_grid,
    left_image_object,
    right_image_object,
    used_band,
    threshold_disp_range_to_borders,
):
    """
    Compute dense disparity grids min and max for pandora
    superposable to left image

    :param disp_range_grid: disp range grid with min and max grids
    :type disp_range_grid: CarsDataset
    :param left_image_object: left image
    :type left_image_object: xr.Dataset

    :return disp min map, disp_max_map
    :rtype np.ndarray, np.ndarray
    """

    disp_min_grid_arr, _ = inputs.rasterio_read_as_array(
        disp_range_grid["grid_min_path"]
    )
    disp_max_grid_arr, _ = inputs.rasterio_read_as_array(
        disp_range_grid["grid_max_path"]
    )
    row_range = disp_range_grid["row_range"]
    col_range = disp_range_grid["col_range"]

    # Create interpolators
    interp_min = RegularGridInterpolator(
        (
            row_range,
            col_range,
        ),
        disp_min_grid_arr,
    )
    interp_max = RegularGridInterpolator(
        (
            row_range,
            col_range,
        ),
        disp_max_grid_arr,
    )

    # Interpolate disp on grid
    roi_with_margins = left_image_object.attrs["roi_with_margins"]

    row_range = np.arange(roi_with_margins[1], roi_with_margins[3])
    col_range = np.arange(roi_with_margins[0], roi_with_margins[2])

    row_grid, col_grid = np.meshgrid(row_range, col_range, indexing="ij")

    disp_min_grid = interp_min((row_grid, col_grid)).astype("float32")
    disp_max_grid = interp_max((row_grid, col_grid)).astype("float32")

    # Compute extremums of disparity considering left image borders
    if threshold_disp_range_to_borders:
        disp_min_from_borders = np.zeros_like(disp_min_grid)
        disp_max_from_borders = np.zeros_like(disp_max_grid)
        right_msk = (
            np.array(right_image_object[cst.EPI_MSK].loc[used_band]) == 0
        )
        index_of_first_valid_pixel = np.argmax(right_msk, axis=1)
        index_of_last_valid_pixel = np.argmax(
            np.flip(right_msk, axis=1), axis=1
        )
        index_of_last_valid_pixel = (
            right_msk.shape[1] - index_of_last_valid_pixel
        )
        any_valid_pixel_exists = np.any(right_msk, axis=1)
        right_msk_indices = zip(  # noqa: B905
            index_of_first_valid_pixel,
            index_of_last_valid_pixel,
            any_valid_pixel_exists,
        )
        for row_id, (first, last, exists) in enumerate(right_msk_indices):
            if exists:
                disp_min_from_borders[row_id, first:last] = np.flip(
                    np.arange(first - last, 0)
                )
                disp_max_from_borders[row_id, first:last] = np.flip(
                    np.arange(0, last - first)
                )
        disp_min_from_borders = np.minimum(disp_max_grid, disp_min_from_borders)
        disp_max_from_borders = np.maximum(disp_min_grid, disp_max_from_borders)
        disp_min_grid = np.maximum(disp_min_from_borders, disp_min_grid)
        disp_max_grid = np.minimum(disp_max_from_borders, disp_max_grid)

    # Interpolation might create min > max
    disp_min_grid, disp_max_grid = dm_wrap.to_safe_disp_grid(
        disp_min_grid, disp_max_grid
    )

    return disp_min_grid, disp_max_grid


def compute_disparity(  # pylint: disable=too-many-positional-arguments
    left_dataset,
    right_dataset,
    corr_cfg,
    used_band=None,
    disp_min_grid=None,
    disp_max_grid=None,
    compute_disparity_masks=False,
    cropped_range=None,
    margins_to_keep=0,
    classification_fusion_margin=-1,
    texture_bands=None,
) -> Dict[str, xr.Dataset]:
    """
    This function will compute disparity.

    :param left_dataset: Dataset containing left image and mask
    :type left_dataset: xarray.Dataset
    :param right_dataset: Dataset containing right image and mask
    :type right_dataset: xarray.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param used_band: name of band used for correlation
    :type used_band: str
    :param disp_min_grid: Minimum disparity grid
                     (if None, value is taken from left dataset)
    :type disp_min_grid: np ndarray
    :param disp_max_grid: Maximum disparity grid
                     (if None, value is taken from left dataset)
    :type disp_max_grid: np ndarray
    :param compute_disparity_masks: Activation of compute_disparity_masks mode
    :type compute_disparity_masks: Boolean
    :param cropped_range: true if disparity range was cropped
    :type cropped_range: numpy array
    :param margins_to_keep: margin to keep after dense matching
    :type margins_to_keep: int
    :param classification_fusion_margin: the margin to add for the fusion
    :type classification_fusion_margin: int


    :return: Disparity dataset
    """

    # Check disp min and max bounds with respect to margin used for
    # rectification

    # Get tile global disp
    disp_min = np.floor(np.min(disp_min_grid))
    disp_max = np.ceil(np.max(disp_max_grid))

    if disp_min < left_dataset.attrs[cst.EPI_DISP_MIN]:
        logging.error(
            "disp_min ({}) is lower than disp_min used to determine "
            "margin during rectification ({})".format(
                disp_min, left_dataset.attrs["disp_min"]
            )
        )

    if disp_max > left_dataset.attrs[cst.EPI_DISP_MAX]:
        logging.error(
            "disp_max ({}) is greater than disp_max used to determine "
            "margin during rectification ({})".format(
                disp_max, left_dataset.attrs["disp_max"]
            )
        )

    # Load pandora plugin
    pandora.import_plugin()

    # Update nodata values
    left_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_left"]
    right_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_right"]

    # Put disparity in datasets
    left_disparity = xr.DataArray(
        data=np.array(
            [disp_min_grid, disp_max_grid],
        ),
        dims=["band_disp", "row", "col"],
        coords={"band_disp": ["min", "max"]},
    )

    (disp_min_right_grid, disp_max_right_grid) = (
        dm_wrap.estimate_right_grid_disp(disp_min_grid, disp_max_grid)
    )
    # estimation might create max < min
    disp_min_right_grid, disp_max_right_grid = dm_wrap.to_safe_disp_grid(
        disp_min_right_grid, disp_max_right_grid
    )

    right_disparity = xr.DataArray(
        data=np.array(
            [disp_min_right_grid, disp_max_right_grid],
        ),
        dims=["band_disp", "row", "col"],
        coords={"band_disp": ["min", "max"]},
    )

    left_dataset["disparity"] = left_disparity
    right_dataset["disparity"] = right_disparity

    if used_band is not None:
        # Remove band_im dimension from mask
        left_msk = left_dataset[cst.EPI_MSK]
        left_msk = left_msk.loc[used_band]
        left_dataset = left_dataset.drop_vars([cst.EPI_MSK])
        left_dataset[cst.EPI_MSK] = left_msk

        right_msk = right_dataset[cst.EPI_MSK]
        right_msk = right_msk.loc[used_band]
        right_dataset = right_dataset.drop_vars([cst.EPI_MSK])
        right_dataset[cst.EPI_MSK] = right_msk

    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()

    # check datasets
    check_datasets(left_dataset, right_dataset)

    # Run the Pandora pipeline
    ref, _ = pandora.run(
        pandora_machine,
        left_dataset,
        right_dataset,
        corr_cfg,
    )

    disp_dataset = dm_wrap.create_disp_dataset(
        ref,
        left_dataset,
        right_dataset,
        compute_disparity_masks=compute_disparity_masks,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        cropped_range=cropped_range,
        margins_to_keep=margins_to_keep,
        classification_fusion_margin=classification_fusion_margin,
        texture_bands=texture_bands,
    )

    return disp_dataset


class LinearInterpNearestExtrap:  # pylint: disable=too-few-public-methods
    """
    Linear interpolation and nearest neighbour extrapolation
    """

    def __init__(self, points, values):
        self.interp = LinearNDInterpolator(points, values)
        self.extrap = NearestNDInterpolator(points, values)

    def __call__(self, *args):
        z_values = self.interp(*args)
        nan_mask = np.isnan(z_values)
        if nan_mask.any():
            return np.where(nan_mask, self.extrap(*args), z_values)
        return z_values
