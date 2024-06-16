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
import math
from importlib import metadata
from typing import Dict, List

import numpy as np
import pandora
import pandora.marge
import xarray as xr

# Third party imports
from numba import njit, prange
from pandora import constants as p_cst
from pandora.check_configuration import check_datasets
from pandora.state_machine import PandoraMachine
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from cars.applications.dense_matches_filling import fill_disp_tools

# CARS imports
from cars.applications.dense_matching import (
    dense_matching_constants as dense_match_cst,
)
from cars.conf import mask_cst as msk_cst
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp


def get_margins(disp_min, disp_max, corr_cfg):
    """
    Get margins for the dense matching steps

    :param disp_min: Minimum disparity
    :type disp_min: int
    :param disp_max: Maximum disparity
    :type disp_max: int
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :return: margins of the matching algorithm used
    """
    return pandora.marge.get_margins(disp_min, disp_max, corr_cfg["pipeline"])


def get_masks_from_pandora(
    disp: xr.Dataset, compute_disparity_masks: bool
) -> Dict[str, np.ndarray]:
    """
    Get masks dictionary from the disparity map in output of pandora.

    :param disp: disparity map (pandora output)
    :param compute_disparity_masks: compute_disparity_masks activation status
    :return: masks dictionary
    """
    masks = {}

    # Retrieve validity mask from pandora
    # Invalid pixels in validity mask are:
    #  * Bit 0: Edge of the reference image or nodata in reference image
    #  * Bit 1: Disparity interval to explore is missing or nodata in the
    #           secondary image
    #  * Bit 6: Pixel is masked on the mask of the reference image
    #  * Bit 7: Disparity to explore is masked on the mask of the secondary
    #           image
    #  * Bit 8: Pixel located in an occlusion region
    #  * Bit 9: Fake match
    validity_mask_cropped = disp.validity_mask.values
    # Mask initialization to false (all is invalid)
    masks[cst_disp.VALID] = np.full(validity_mask_cropped.shape, False)
    # Identify valid points
    masks[cst_disp.VALID][
        np.where((validity_mask_cropped & p_cst.PANDORA_MSK_PIXEL_INVALID) == 0)
    ] = True

    # With compute_disparity_masks, produce one mask for each invalid flag in
    if compute_disparity_masks:
        msk_table = dense_match_cst.MASK_HASH_TABLE
        for key, val in msk_table.items():
            masks[key] = np.full(validity_mask_cropped.shape, False)
            masks[key][np.where((validity_mask_cropped & val) == 0)] = True

    # Build final mask with 255 for valid points and 0 for invalid points
    # The mask is used by rasterize method (non zero are valid points)
    for key, mask in masks.items():
        final_msk = np.ndarray(mask.shape, dtype=np.int16)
        final_msk[mask] = 255
        final_msk[np.equal(mask, False)] = 0
        masks[key] = final_msk

    return masks


def add_disparity_grids(
    output_dataset: xr.Dataset,
    disp_min_grid: np.ndarray = None,
    disp_max_grid: np.ndarray = None,
):
    """
    Add  disparity min and max grids to dataset

    :param output_dataset: output dataset
    :param disp_min_grid: dense disparity map grid min
    :param disp_max_grid: dense disparity map grid max

    """
    if disp_min_grid is not None:
        output_dataset[cst_disp.EPI_DISP_MIN_GRID] = xr.DataArray(
            disp_min_grid,
            dims=[cst.ROW, cst.COL],
        )

    # Add color mask
    if disp_max_grid is not None:
        output_dataset[cst_disp.EPI_DISP_MAX_GRID] = xr.DataArray(
            disp_max_grid,
            dims=[cst.ROW, cst.COL],
        )


def add_color(
    output_dataset: xr.Dataset,
    color: np.ndarray = None,
    color_type=None,
    band_im: list = None,
):
    """
    Add color and color mask to dataset

    :param output_dataset: output dataset
    :param color: color array
    :param color_type: data type of pixels
    :param band_im: list of band names

    """
    if color is not None:
        if band_im is not None and cst.BAND_IM not in output_dataset.dims:
            output_dataset.coords[cst.BAND_IM] = band_im
        output_dataset[cst.EPI_COLOR] = xr.DataArray(
            color,
            dims=[cst.BAND_IM, cst.ROW, cst.COL],
        )

        if color_type is not None:
            # Add input color type
            output_dataset[cst.EPI_COLOR].attrs["color_type"] = color_type


def add_classification(
    output_dataset: xr.Dataset,
    classif: np.ndarray = None,
    band_classif: list = None,
):
    """
    Add classification to dataset

    :param output_dataset: output dataset
    :param classif: classif array
    :param band_im: list of band names

    """
    if classif is not None:
        if (
            band_classif is not None
            and cst.BAND_CLASSIF not in output_dataset.dims
        ):
            output_dataset.coords[cst.BAND_CLASSIF] = band_classif
        output_dataset[cst.EPI_CLASSIFICATION] = xr.DataArray(
            classif,
            dims=[cst.BAND_CLASSIF, cst.ROW, cst.COL],
        )


def create_disp_dataset(  # noqa: C901
    disp: xr.Dataset,
    ref_dataset: xr.Dataset,
    secondary_dataset: xr.Dataset,
    compute_disparity_masks: bool = False,
    generate_performance_map=False,
    perf_ambiguity_threshold=0.6,
    disp_to_alt_ratio=None,
    disp_min_grid=None,
    disp_max_grid=None,
    cropped_range=None,
) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param secondary_dataset: secondary dataset for the considered
        disparity map
    :param compute_disparity_masks: compute_disparity_masks activation status
    :param generate_performance_map: True if generate performance map
    :type generate_performance_map: bool
    :param perf_ambiguity_threshold: ambiguity threshold used for
        performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float
    :param disp_min_grid: disparity min grid
    :param disp_max_grid: disparity max grid
    :param cropped_range: true if disparity range was cropped
    :type cropped_range: numpy array

    :return: disparity dataset as used in cars
    """

    # Crop disparity to ROI
    ref_roi = [
        int(-ref_dataset.attrs[cst.EPI_MARGINS][0]),
        int(-ref_dataset.attrs[cst.EPI_MARGINS][1]),
        int(ref_dataset.dims[cst.COL] - ref_dataset.attrs[cst.EPI_MARGINS][2]),
        int(ref_dataset.dims[cst.ROW] - ref_dataset.attrs[cst.EPI_MARGINS][3]),
    ]

    # Retrieve disparity values
    disp_map = disp.disparity_map.values

    # Retrive left panchromatic image
    epi_image = ref_dataset[cst.EPI_IMAGE].values

    # Retrieve original mask of panchromatic image
    epi_msk = None
    if cst.EPI_MSK in ref_dataset:
        epi_msk = ref_dataset[cst.EPI_MSK].values

    epi_msk_right = None
    if cst.EPI_MSK in secondary_dataset:
        epi_msk_right = secondary_dataset[cst.EPI_MSK].values

    # Retrieve masks from pandora
    pandora_masks = get_masks_from_pandora(disp, compute_disparity_masks)

    # Retrieve colors
    color = None
    color_type = None
    band_im = None
    if cst.EPI_COLOR in ref_dataset:
        color = ref_dataset[cst.EPI_COLOR].values
        color_type = ref_dataset[cst.EPI_COLOR].attrs["color_type"]
        if ref_dataset[cst.EPI_COLOR].values.shape[0] > 1:
            band_im = ref_dataset.coords[cst.BAND_IM]
        else:
            band_im = ["Gray"]

    # retrieve classif
    left_classif = None
    left_band_classif = None
    if cst.EPI_CLASSIFICATION in ref_dataset:
        left_classif = ref_dataset[cst.EPI_CLASSIFICATION].values
        left_band_classif = ref_dataset.coords[cst.BAND_CLASSIF].values

        # mask left classif outside right sensor
        if epi_msk_right is not None:
            left_classif = mask_left_classif_from_right_mask(
                left_classif,
                epi_msk_right == msk_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION,
                np.floor(disp_min_grid).astype(np.int16),
                np.ceil(disp_max_grid).astype(np.int16),
            )

    left_from_right_classif = None
    right_band_classif = None
    if cst.EPI_CLASSIFICATION in secondary_dataset:
        right_classif = secondary_dataset[cst.EPI_CLASSIFICATION].values
        right_band_classif = secondary_dataset.coords[cst.BAND_CLASSIF].values

        left_from_right_classif = estimate_right_classif_on_left(
            right_classif,
            disp_map,
            pandora_masks[cst_disp.VALID],
            int(np.floor(np.min(disp_min_grid))),
            int(np.ceil(np.max(disp_max_grid))),
        )
        # mask outside left sensor
        left_mask = (
            ref_dataset[cst.EPI_MSK].values
            == msk_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
        )
        left_mask_stacked = np.repeat(
            np.expand_dims(left_mask, axis=0),
            left_from_right_classif.shape[0],
            axis=0,
        )
        left_from_right_classif[left_mask_stacked] = 0

    # Merge right classif
    classif, band_classif = merge_classif_left_right(
        left_classif,
        left_band_classif,
        left_from_right_classif,
        right_band_classif,
    )

    # Crop disparity map
    disp_map = disp_map[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop left panchromatic image
    epi_image = epi_image[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop original mask
    if epi_msk is not None:
        epi_msk = epi_msk[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop color
    if color is not None:
        color = color[:, ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop masks
    for key in pandora_masks.copy():
        pandora_masks[key] = pandora_masks[key][
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Crop classif
    if classif is not None:
        classif = classif[:, ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop disparity min max grids
    if disp_min_grid is not None:
        disp_min_grid = disp_min_grid[
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]
    if disp_max_grid is not None:
        disp_max_grid = disp_max_grid[
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]
    if cropped_range is not None:
        cropped_range = cropped_range[
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Fill disparity array with 0 value for invalid points
    disp_map[pandora_masks[cst_disp.VALID] == 0] = 0

    # Build output dataset
    row = np.array(
        range(ref_dataset.attrs[cst.ROI][1], ref_dataset.attrs[cst.ROI][3])
    )
    col = np.array(
        range(ref_dataset.attrs[cst.ROI][0], ref_dataset.attrs[cst.ROI][2])
    )

    disp_ds = xr.Dataset(
        {
            cst_disp.MAP: ([cst.ROW, cst.COL], np.copy(disp_map)),
            cst_disp.VALID: (
                [cst.ROW, cst.COL],
                pandora_masks[cst_disp.VALID].astype("uint8"),
            ),
        },
        coords={cst.ROW: row, cst.COL: col},
    )

    # add left panchromatic image
    disp_ds[cst.EPI_IMAGE] = xr.DataArray(epi_image, dims=[cst.ROW, cst.COL])

    # add original mask
    if epi_msk is not None:
        disp_ds[cst.EPI_MSK] = xr.DataArray(
            epi_msk.astype("uint8"), dims=[cst.ROW, cst.COL]
        )

    # add color
    add_color(
        disp_ds,
        color=color,
        color_type=color_type,
        band_im=band_im,
    )

    # add confidence
    add_confidence(disp_ds, disp, ref_roi)

    # add performance map
    if generate_performance_map:
        add_performance_map(
            disp_ds, disp, ref_roi, perf_ambiguity_threshold, disp_to_alt_ratio
        )

    # add classif
    add_classification(disp_ds, classif=classif, band_classif=band_classif)

    # Add disparity grids
    add_disparity_grids(
        disp_ds, disp_min_grid=disp_min_grid, disp_max_grid=disp_max_grid
    )

    # Add filling infos
    if cropped_range is not None:
        disp_ds = add_crop_info(disp_ds, cropped_range)

    if compute_disparity_masks:
        for key, val in pandora_masks.items():
            disp_ds[key] = xr.DataArray(np.copy(val), dims=[cst.ROW, cst.COL])

    disp_ds.attrs[cst.ROI] = ref_dataset.attrs[cst.ROI]

    disp_ds.attrs[cst.EPI_FULL_SIZE] = ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


def add_crop_info(disp_ds, cropped_range):
    """
    Add crop info

    :param disp: disp xarray
    :param cropped_range: was cropped range, bool

    :return updated dataset
    """

    disp_ds = fill_disp_tools.add_empty_filling_band(
        disp_ds, ["cropped_disp_range"]
    )
    fill_disp_tools.update_filling(disp_ds, cropped_range, "cropped_disp_range")
    return disp_ds


@njit
def estimate_right_classif_on_left(
    right_classif, disp_map, disp_mask, disp_min, disp_max
):
    """
    Estimate right classif on left image

    :param right_classif: right classification
    :type right_classif: np ndarray
    :param disp_map: disparity map
    :type disp_map: np ndarray
    :param disp_mask: disparity mask
    :type disp_mask: np ndarray
    :param disp_min: disparity min
    :type disp_min: int
    :param disp_max: disparity max
    :type disp_max: int

    :return: right classif on left image
    :rtype: np nadarray
    """

    left_from_right_classif = np.empty(right_classif.shape)

    data_shape = left_from_right_classif.shape
    for row in prange(data_shape[1]):  # pylint: disable=E1133
        for col in prange(data_shape[2]):  # pylint: disable=E1133
            # find classif
            disp = disp_map[row, col]
            valid = not np.isnan(disp)
            if disp_mask is not None:
                valid = disp_mask[row, col]
            if valid:
                # direct value
                disp = int(math.floor(disp))
                left_from_right_classif[:, row, col] = right_classif[
                    :, row, col + disp
                ]
            else:
                # estimate with global range
                classif_in_range = np.full(
                    (left_from_right_classif.shape[0]), False
                )

                for classif_c in prange(  # pylint: disable=E1133
                    classif_in_range.shape[0]
                ):
                    for col_classif in prange(  # pylint: disable=E1133
                        max(0, col + disp_min),
                        min(data_shape[1], col + disp_max),
                    ):
                        if right_classif[classif_c, row, col_classif]:
                            classif_in_range[classif_c] = True

                left_from_right_classif[:, row, col] = classif_in_range

    return left_from_right_classif


@njit
def mask_left_classif_from_right_mask(
    left_classif, right_mask, disp_min, disp_max
):
    """
    Mask left classif with right mask.

    :param left_classif: right classification
    :type right_left_classifclassif: np ndarray
    :param right_mask: right mask
    :type right_mask: np ndarray
    :param disp_min: disparity min
    :type disp_min: np.array type int
    :param disp_max: disparity max
    :type disp_max: np.array type int

    :return: masked left classif
    :rtype: np nadarray
    """

    data_shape = left_classif.shape
    for row in prange(data_shape[1]):  # pylint: disable=E1133
        for col in prange(data_shape[2]):  # pylint: disable=E1133
            # estimate with global range
            all_masked = True
            for col_classif in prange(  # pylint: disable=E1133
                max(0, col + disp_min[row, col]),
                min(data_shape[1], col + disp_max[row, col]),
            ):
                if not right_mask[row, col_classif]:
                    all_masked = False

            if all_masked:
                # Remove classif
                left_classif[:, row, col] = False

    return left_classif


def merge_classif_left_right(
    left_classif, left_band_classif, right_classif, right_band_classif
):
    """
    Merge left and right classif

    :param left_classif: left classif
    :type left_classif: np nadarray
    :param left_band_classif: list of tag
    :type left_band_classif: list
    :param right_classif: left classif
    :type right_classif: np nadarray
    :param right_band_classif: list of tag
    :type right_band_classif: list

    :return: merged classif, merged tag list
    :rtype: np ndarray, list

    """

    classif = None
    band_classif = None

    if left_classif is None and right_classif is not None:
        classif = right_classif
        band_classif = right_band_classif

    elif left_classif is not None and right_classif is None:
        classif = left_classif
        band_classif = left_band_classif

    elif left_classif is not None and right_classif is not None:
        list_classif = []
        band_classif = []
        seen_tag = []

        # search in left
        for ind_left, tag_left in enumerate(left_band_classif):
            seen_tag.append(tag_left)
            band_classif.append(tag_left)
            if tag_left in right_band_classif:
                # merge left and right
                ind_right = list(right_band_classif).index(tag_left)
                list_classif.append(
                    np.logical_or(
                        left_classif[ind_left, :, :],
                        right_classif[ind_right, :, :],
                    )
                )
            else:
                list_classif.append(left_classif[ind_left, :, :])

        # search in right not already seen
        for ind_right, tag_right in enumerate(right_band_classif):
            if tag_right not in seen_tag:
                band_classif.append(tag_right)
                list_classif.append(right_classif[ind_right, :, :])

        # Stack classif
        classif = np.stack(list_classif, axis=0)

    return classif, band_classif


def add_confidence(
    output_dataset: xr.Dataset,
    disp: xr.Dataset,
    ref_roi: List[int],
):
    """
    Add confidences to dataset

    :param output_dataset: output dataset
    :param disp: disp xarray

    """
    confidence_measure_indicator_list = np.array(
        disp.confidence_measure.indicator
    )
    for key in confidence_measure_indicator_list:
        confidence_idx = list(disp.confidence_measure.indicator).index(key)
        output_dataset[key] = xr.DataArray(
            np.copy(
                disp.confidence_measure.data[
                    ref_roi[1] : ref_roi[3],
                    ref_roi[0] : ref_roi[2],
                    confidence_idx,
                ]
            ),
            dims=[cst.ROW, cst.COL],
        )


def add_performance_map(
    output_dataset: xr.Dataset,
    disp: xr.Dataset,
    ref_roi: List[int],
    perf_ambiguity_threshold: float,
    disp_to_alt_ratio: float,
):
    """
    Add performance map to dataset

    :param output_dataset: output dataset
    :param disp: disp xarray
    :param perf_ambiguity_threshold: ambiguity threshold used for
        performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float

    """
    confidence_measure_indicator_list = np.array(
        disp.confidence_measure.indicator
    )

    ambiguity_indicator = "confidence_from_ambiguity.cars_1"
    risk_mask_indicator = "confidence_from_risk_max.cars_2"

    if ambiguity_indicator not in confidence_measure_indicator_list or (
        risk_mask_indicator not in confidence_measure_indicator_list
    ):
        raise RuntimeError(
            "{} or {} not generated by pandora".format(
                ambiguity_indicator, risk_mask_indicator
            )
        )

    # Get confidences map
    ambi_idx = list(disp.confidence_measure.indicator).index(
        ambiguity_indicator
    )
    ambiguity_map = disp.confidence_measure.data[
        ref_roi[1] : ref_roi[3],
        ref_roi[0] : ref_roi[2],
        ambi_idx,
    ]

    risk_max_idx = list(disp.confidence_measure.indicator).index(
        risk_mask_indicator
    )
    risk_max_map = disp.confidence_measure.data[
        ref_roi[1] : ref_roi[3],
        ref_roi[0] : ref_roi[2],
        risk_max_idx,
    ]

    mask_ambi = ambiguity_map > perf_ambiguity_threshold
    w_ambi = ambiguity_map / perf_ambiguity_threshold
    w_ambi[mask_ambi] = 1

    # Compute performance map
    performance_map = w_ambi * risk_max_map * disp_to_alt_ratio

    # Set performance map in dataset
    performance_map_key = cst_disp.CONFIDENCE + "_performance_map"

    output_dataset[performance_map_key] = xr.DataArray(
        performance_map,
        dims=[cst.ROW, cst.COL],
    )


def compute_disparity_grid(disp_range_grid, left_image_object):
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
    # Create interpolators
    interp_min = RegularGridInterpolator(
        (
            disp_range_grid.attributes["row_range"],
            disp_range_grid.attributes["col_range"],
        ),
        disp_range_grid[0, 0][dense_match_cst.DISP_MIN_GRID].values,
    )
    interp_max = RegularGridInterpolator(
        (
            disp_range_grid.attributes["row_range"],
            disp_range_grid.attributes["col_range"],
        ),
        disp_range_grid[0, 0][dense_match_cst.DISP_MAX_GRID].values,
    )

    # Interpolate disp on grid
    roi_with_margins = left_image_object.attrs["roi_with_margins"]

    row_range = np.arange(roi_with_margins[1], roi_with_margins[3])
    col_range = np.arange(roi_with_margins[0], roi_with_margins[2])

    row_grid, col_grid = np.meshgrid(row_range, col_range, indexing="ij")

    disp_min_grid = interp_min((row_grid, col_grid)).astype("float32")
    disp_max_grid = interp_max((row_grid, col_grid)).astype("float32")

    return disp_min_grid, disp_max_grid


def compute_disparity(
    left_dataset,
    right_dataset,
    corr_cfg,
    disp_min_grid=None,
    disp_max_grid=None,
    compute_disparity_masks=False,
    generate_performance_map=False,
    perf_ambiguity_threshold=0.6,
    disp_to_alt_ratio=None,
    cropped_range=None,
) -> Dict[str, xr.Dataset]:
    """
    This function will compute disparity.

    :param left_dataset: Dataset containing left image and mask
    :type left_dataset: xarray.Dataset
    :param right_dataset: Dataset containing right image and mask
    :type right_dataset: xarray.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min_grid: Minimum disparity grid
                     (if None, value is taken from left dataset)
    :type disp_min_grid: np ndarray
    :param disp_max_grid: Maximum disparity grid
                     (if None, value is taken from left dataset)
    :type disp_max_grid: np ndarray
    :param compute_disparity_masks: Activation of compute_disparity_masks mode
    :type compute_disparity_masks: Boolean
    :param generate_performance_map: True if generate performance map
    :type generate_performance_map: bool
    :param perf_ambiguity_threshold: ambiguity threshold used for
        performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float
    :param cropped_range: true if disparity range was cropped
    :type cropped_range: numpy array
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
    if "pandora.plugin" in metadata.entry_points():
        for entry_point in metadata.entry_points()["pandora.plugin"]:
            entry_point.load()
    else:
        raise ImportError(
            "Pandora plugin is not correctly installed or missing."
        )

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

    (disp_min_right_grid, disp_max_right_grid) = estimate_right_grid_disp(
        disp_min_grid, disp_max_grid
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

    disp_dataset = create_disp_dataset(
        ref,
        left_dataset,
        right_dataset,
        compute_disparity_masks=compute_disparity_masks,
        generate_performance_map=generate_performance_map,
        perf_ambiguity_threshold=perf_ambiguity_threshold,
        disp_to_alt_ratio=disp_to_alt_ratio,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        cropped_range=cropped_range,
    )

    return disp_dataset


@njit()
def estimate_right_grid_disp(disp_min_grid, disp_max_grid):
    """
    Estimate right grid min and max.
    Correspond to the range of pixels that can be correlated
    from left -> right.
    If no left pixels can be associated to right, use global values

    :param disp_min_grid: left disp min grid
    :type disp_min_grid: numpy ndarray
    :param disp_max_grid: left disp max grid
    :type disp_max_grid: numpy ndarray

    :return: disp_min_right_grid, disp_max_right_grid
    :rtype: numpy ndarray, numpy ndarray
    """

    global_left_min = np.min(disp_min_grid)
    global_left_max = np.max(disp_max_grid)

    d_shp = disp_min_grid.shape

    disp_min_right_grid = np.empty(d_shp)
    disp_max_right_grid = np.empty(d_shp)

    for row in prange(d_shp[0]):  # pylint: disable=not-an-iterable
        for col in prange(d_shp[1]):  # pylint: disable=not-an-iterable
            min_right = d_shp[1]
            max_right = 0
            is_correlated_left = False
            for left_col in prange(d_shp[1]):  # pylint: disable=not-an-iterable
                left_min = disp_min_grid[row, left_col] + left_col
                left_max = disp_max_grid[row, left_col] + left_col
                if left_min <= col <= left_max:
                    is_correlated_left = True
                    # can be found, is candidate to min and max
                    min_right = min(min_right, left_col - col)
                    max_right = max(max_right, left_col - col)

            if is_correlated_left:
                disp_min_right_grid[row, col] = min_right
                disp_max_right_grid[row, col] = max_right
            else:
                disp_min_right_grid[row, col] = -global_left_max
                disp_max_right_grid[row, col] = -global_left_min

    return disp_min_right_grid, disp_max_right_grid


def optimal_tile_size_pandora_plugin_libsgm(
    disp_min: int,
    disp_max: int,
    min_tile_size: int,
    max_tile_size: int,
    max_ram_per_worker: int,
    tile_size_rounding: int = 50,
    margin: int = 0,
) -> int:
    """
    Compute optimal tile size according to estimated memory usage
    (pandora_plugin_libsgm)
    Returned optimal tile size will be at least equal to tile_size_rounding.

    :param disp_min: Minimum disparity to explore
    :param disp_max: Maximum disparity to explore
    :param min_tile_size: Minimal tile size
    :param max_tile_size: Maximal tile size
    :param max_ram_per_worker: amount of RAM allocated per worker
    :param tile_size_rounding: Optimal tile size will be aligned to multiples\
                               of tile_size_rounding
    :param margin: margin to remove to the computed tile size
                   (as a percent of the computed tile size)
    :returns: Optimal tile size according to benchmarked memory usage
    """

    memory = max_ram_per_worker
    disp = disp_max - disp_min

    image = 32 * 2
    disp_ref = 32
    validity_mask_ref = 16
    confidence = 32
    cv_ = disp * 32
    nan_ = disp * 8
    cv_uint = disp * 8
    penal = 8 * 32 * 2
    img_crop = 32 * 2

    tot = image + disp_ref + validity_mask_ref
    tot += confidence + 2 * cv_ + nan_ + cv_uint + penal + img_crop
    import_ = 200  # MiB

    row_or_col = float(((memory - import_) * 2**23)) / tot

    if row_or_col <= 0:
        logging.warning(
            "Optimal tile size is null, "
            "forcing it to {} pixels".format(tile_size_rounding)
        )
        tile_size = tile_size_rounding
    else:
        # nb_pixels = tile_size x (tile_size + disp)
        # hence tile_size not equal to sqrt(nb_pixels)
        # but sqrt(nb_pixels + (disp/2)**2) - disp/2
        tile_size = np.sqrt(row_or_col + (disp / 2) ** 2) - disp / 2
        tile_size = (1.0 - margin / 100.0) * tile_size
        tile_size = tile_size_rounding * int(tile_size / tile_size_rounding)

    if tile_size > max_tile_size:
        tile_size = max_tile_size
    elif tile_size < min_tile_size:
        tile_size = min_tile_size

    return tile_size


def get_max_disp_from_opt_tile_size(
    opt_epipolar_tile_size, max_ram_per_worker, margin=0, used_disparity_range=0
):
    """
    Compute max range possible depending on max ram per worker
    Return max range usable

    :param opt_epipolar_tile_size: used tile size
    :param max_ram_per_worker: amount of RAM allocated per worker
    :param tile_size_rounding: Optimal tile size will be aligned to multiples\
                               of tile_size_rounding
    :param margin: margin to remove to the computed tile size
                   (as a percent of the computed tile size)
    :returns: max disp range to use
    """

    import_ = 200
    memory = max_ram_per_worker

    # not depending on disp
    image = 32 * 2
    disp_ref = 32
    validity_mask_ref = 16
    confidence = 32
    penal = 8 * 32 * 2
    img_crop = 32 * 2
    # depending on disp  : data * disp
    cv_ = 32
    nan_ = 8
    cv_uint = 8

    row = opt_epipolar_tile_size / (1.0 - margin / 100.0)
    col = row + used_disparity_range
    row_or_col = row * col
    tot = float(((memory - import_) * 2**23)) / row_or_col

    disp_tot = tot - (
        image + disp_ref + validity_mask_ref + confidence + penal + img_crop
    )
    # disp tot = disp x (2 x cv_ + nan_ + cv uint)
    max_range = int(disp_tot / (2 * cv_ + nan_ + cv_uint) + 1)

    return max_range


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
