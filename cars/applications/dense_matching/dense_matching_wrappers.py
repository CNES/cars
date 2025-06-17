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
import warnings
from typing import Dict

import numpy as np
import xarray as xr

# Third party imports
from pandora import constants as p_cst
from scipy.ndimage import generic_filter

from cars.applications.dense_match_filling import fill_disp_wrappers

# CARS imports
from cars.applications.dense_matching import (
    dense_matching_constants as dense_match_cst,
)
from cars.conf import mask_cst as msk_cst
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp

from .cpp import dense_matching_cpp


def get_margins(margin, disp_min, disp_max):
    """
    Get margins for the dense matching steps

    :param margin: margins object
    :type margin: Margins
    :param disp_min: Minimum disparity
    :type disp_min: int
    :param disp_max: Maximum disparity
    :type disp_max: int
    :return: margins of the matching algorithm used
    """

    corner = ["left", "up", "right", "down"]
    col = np.arange(len(corner))

    left_margins = [
        margin.left + disp_max,
        margin.up,
        margin.right - disp_min,
        margin.down,
    ]
    right_margins = [
        margin.left - disp_min,
        margin.up,
        margin.right + disp_max,
        margin.down,
    ]
    same_margins = [
        max(left, right)
        for left, right in zip(left_margins, right_margins)  # noqa: B905
    ]

    margins = xr.Dataset(
        {
            "left_margin": (
                ["col"],
                same_margins,
            )
        },
        coords={"col": col},
    )
    margins["right_margin"] = xr.DataArray(same_margins, dims=["col"])

    margins.attrs["disp_min"] = disp_min
    margins.attrs["disp_max"] = disp_max

    return margins


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


def add_texture(
    output_dataset: xr.Dataset,
    texture: np.ndarray = None,
    color_type=None,
    band_im: list = None,
    texture_bands: list = None,
):
    """
    Add image and image mask to dataset

    :param output_dataset: output dataset
    :param image: image array
    :param image_type: data type of pixels
    :param band_im: list of band names

    """
    if texture_bands:
        output_dataset.coords[cst.BAND_IM] = band_im[texture_bands]
        output_dataset[cst.EPI_TEXTURE] = xr.DataArray(
            texture[texture_bands],
            dims=[cst.BAND_IM, cst.ROW, cst.COL],
        )

        if color_type is not None:
            # Add input image type
            output_dataset[cst.EPI_TEXTURE].attrs["color_type"] = color_type


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


def compute_cropped_roi(
    current_margins, margins_to_keep, tile_roi, nb_rows, nb_cols
):
    """
    Compute cropped roi, with associated margins and

    :param current_margins: current dataset margins
    :type current_margins: list[int]
    :param margins_to_keep: margin to keep
    :type margins_to_keep: int
    :param nb_rows: number of current rows
    :type margins_to_keep: int
    :param tile_roi: roi without margin of tile
    :type tile_roi: list
    :param nb_cols: number of current cols
    :type nb_cols: int

    :return: (borders to use as roi, new dataset roi with margin,
        margin associated to roi)
    :rtype: tuple(list, list, list)
    """

    def cast_int_list(current_list):
        """
        Apply int cast to list
        """
        new_list = []
        for obj in current_list:
            new_list.append(int(obj))
        return new_list

    new_margin_neg = list(
        np.clip(np.array(current_margins[0:2]), -margins_to_keep, 0)
    )
    new_margin_pos = list(
        np.clip(np.array(current_margins[2:4]), 0, margins_to_keep)
    )
    new_margin = list(np.array(new_margin_neg + new_margin_pos).astype(int))

    new_roi = list(np.array(tile_roi) + np.array(new_margin))

    ref_roi = [
        int(-current_margins[0] + new_margin[0]),
        int(-current_margins[1] + new_margin[1]),
        int(nb_cols - current_margins[2] + new_margin[2]),
        int(nb_rows - current_margins[3] + new_margin[3]),
    ]

    return (
        cast_int_list(ref_roi),
        cast_int_list(new_roi),
        cast_int_list(new_margin),
    )


def create_disp_dataset(  # noqa: C901
    disp: xr.Dataset,
    ref_dataset: xr.Dataset,
    secondary_dataset: xr.Dataset,
    compute_disparity_masks: bool = False,
    disp_min_grid=None,
    disp_max_grid=None,
    cropped_range=None,
    margins_to_keep=0,
    texture_bands=None,
) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param secondary_dataset: secondary dataset for the considered
        disparity map
    :param compute_disparity_masks: compute_disparity_masks activation status
    :param disp_min_grid: disparity min grid
    :param disp_max_grid: disparity max grid
    :param cropped_range: true if disparity range was cropped
    :type cropped_range: numpy array
    :param margins_to_keep: margin to keep after dense matching
    :type margins_to_keep: int

    :return: disparity dataset as used in cars
    """

    # Crop disparity to ROI
    ref_roi, new_roi, new_margin = compute_cropped_roi(
        ref_dataset.attrs[cst.EPI_MARGINS],
        margins_to_keep,
        ref_dataset.attrs[cst.ROI],
        ref_dataset.sizes[cst.ROW],
        ref_dataset.sizes[cst.COL],
    )

    # Crop datasets
    disp = disp.isel(
        row=slice(ref_roi[1], ref_roi[3]), col=slice(ref_roi[0], ref_roi[2])
    )
    ref_dataset = ref_dataset.isel(
        row=slice(ref_roi[1], ref_roi[3]), col=slice(ref_roi[0], ref_roi[2])
    )
    secondary_dataset = secondary_dataset.isel(
        row=slice(ref_roi[1], ref_roi[3]), col=slice(ref_roi[0], ref_roi[2])
    )
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

    # Retrieve disparity values
    disp_map = disp.disparity_map.values

    # Transform image to texture
    epi_image = ref_dataset[cst.EPI_IMAGE].values
    band_im = ref_dataset.coords[cst.BAND_IM]
    image_type = ref_dataset.attrs.get("image_type", "float32")
    if isinstance(image_type, list):
        if texture_bands is not None:
            image_type = image_type[texture_bands[0]]
        else:
            image_type = image_type[0]
    # Cast image
    if np.issubdtype(image_type, np.floating):
        min_value_clr = np.finfo(image_type).min
        max_value_clr = np.finfo(image_type).max
    else:
        min_value_clr = np.iinfo(image_type).min
        max_value_clr = np.iinfo(image_type).max
    epi_image = np.clip(epi_image, min_value_clr, max_value_clr).astype(
        image_type
    )

    # Retrieve original mask of panchromatic image
    epi_msk = None
    if cst.EPI_MSK in ref_dataset:
        epi_msk = ref_dataset[cst.EPI_MSK].values

    epi_msk_right = None
    if cst.EPI_MSK in secondary_dataset:
        epi_msk_right = secondary_dataset[cst.EPI_MSK].values

    # Retrieve masks from pandora
    pandora_masks = get_masks_from_pandora(disp, compute_disparity_masks)
    pandora_masks[cst_disp.VALID][np.isnan(disp_map)] = 0

    # retrieve classif
    left_classif = None
    left_band_classif = None
    if cst.EPI_CLASSIFICATION in ref_dataset:
        left_classif = ref_dataset[cst.EPI_CLASSIFICATION].values
        left_band_classif = ref_dataset.coords[cst.BAND_CLASSIF].values

        # mask left classif outside right sensor
        if epi_msk_right is not None:
            left_classif = mask_left_classif_from_right_mask(  # here
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
        # pylint: disable=unsupported-assignment-operation
        left_from_right_classif[left_mask_stacked] = 0
    # Merge right classif
    classif, band_classif = merge_classif_left_right(
        left_classif,
        left_band_classif,
        left_from_right_classif,
        right_band_classif,
    )

    # Fill disparity array with 0 value for invalid points
    disp_map[pandora_masks[cst_disp.VALID] == 0] = 0

    # Build output dataset
    row = np.array(range(new_roi[1], new_roi[3]))
    col = np.array(range(new_roi[0], new_roi[2]))

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

    # add left texture
    add_texture(
        disp_ds,
        texture=epi_image,
        color_type=image_type,
        band_im=band_im,
        texture_bands=texture_bands,
    )

    # add original mask
    if epi_msk is not None:
        disp_ds[cst.EPI_MSK] = xr.DataArray(
            epi_msk.astype("uint8"), dims=[cst.ROW, cst.COL]
        )

    # add confidence
    if "confidence_measure" in disp:
        add_confidence(disp_ds, disp)

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
    disp_ds.attrs[cst.ROI_WITH_MARGINS] = new_roi
    disp_ds.attrs[cst.EPI_MARGINS] = new_margin

    disp_ds.attrs[cst.EPI_FULL_SIZE] = ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


def add_crop_info(disp_ds, cropped_range):
    """
    Add crop info

    :param disp: disp xarray
    :param cropped_range: was cropped range, bool

    :return updated dataset
    """

    disp_ds = fill_disp_wrappers.add_empty_filling_band(
        disp_ds, ["cropped_disp_range"]
    )
    fill_disp_wrappers.update_filling(
        disp_ds, cropped_range, "cropped_disp_range"
    )
    return disp_ds


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
    return dense_matching_cpp.estimate_right_classif_on_left(
        right_classif, disp_map, disp_mask, disp_min, disp_max
    )


def mask_left_classif_from_right_mask(
    left_classif, right_mask, disp_min, disp_max
):
    """
    Mask left classif with right mask.

    :param left_classif: right classification
    :type left_classif: np ndarray
    :param right_mask: right mask
    :type right_mask: np ndarray
    :param disp_min: disparity min
    :type disp_min: np.array type int
    :param disp_max: disparity max
    :type disp_max: np.array type int

    :return: masked left classif
    :rtype: np nadarray
    """
    return dense_matching_cpp.mask_left_classif_from_right_mask(
        left_classif, right_mask, disp_min, disp_max
    )


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
                    :,
                    :,
                    confidence_idx,
                ]
            ),
            dims=[cst.ROW, cst.COL],
        )


def to_safe_disp_grid(grid_disp_min, grid_disp_max):
    """
    Generate safe grids, with min < max for each point

    :param grid_disp_min: min disp grid
    :param grid_disp_max: max disp grid

    :return: grid_disp_min, grid_disp_max
    """

    stacked_disp_range = np.dstack([grid_disp_min, grid_disp_max])
    grid_disp_min = np.nanmin(stacked_disp_range, axis=2)
    grid_disp_max = np.nanmax(stacked_disp_range, axis=2)

    # convert nan
    grid_disp_min[np.isnan(grid_disp_min)] = 0
    grid_disp_max[np.isnan(grid_disp_max)] = 0

    if (grid_disp_min > grid_disp_max).any():
        raise RuntimeError("grid min > max")

    return grid_disp_min, grid_disp_max


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
    float_types = [np.float16, np.float32, np.float64, np.float128]
    int_types = [
        int,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]
    if disp_min_grid.dtype in float_types:
        return dense_matching_cpp.estimate_right_grid_disp_float(
            disp_min_grid, disp_max_grid
        )
    if disp_min_grid.dtype in int_types:
        return dense_matching_cpp.estimate_right_grid_disp_int(
            disp_min_grid, disp_max_grid
        )

    raise TypeError(
        "estimate_right_grid_disp does not support"
        f"{disp_min_grid.dtype} as an input type"
    )


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

def nan_ratio_func(window):
    """ "
    Calculate the number of nan in the window

    :param window: the window in the image
    """

    total_pixels = window.size
    nan_count = np.isnan(window).sum()
    return nan_count / total_pixels


def confidence_filtering(
    dataset,
    disp_map,
    requested_confidence,
    conf_filtering,
):
    """
    Filter the disparity map by using the confidence

    :param dataset: the epipolar disparity map dataset
    :type dataset: cars dataset
    :param disp_map: the disparity map
    :type disp_map: numpy darray
    :param requested_confidence: the confidence to use
    :type requested_confidence: list
    :param conf_filtering: the confidence_filtering parameters
    :type conf_filtering: dict
    """

    data_risk = dataset[requested_confidence[0]].values
    data_bounds_sup = dataset[requested_confidence[1]].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nan_ratio = generic_filter(
            disp_map, nan_ratio_func, size=conf_filtering["win_nanratio"]
        )
        var_map = generic_filter(
            data_risk, np.nanmean, size=conf_filtering["win_mean_risk_max"]
        )

    mask = (
        (data_bounds_sup > conf_filtering["upper_bound"])
        | (data_bounds_sup <= conf_filtering["lower_bound"])
    ) | (
        (var_map > conf_filtering["risk_max"])
        & (nan_ratio > conf_filtering["nan_threshold"])
    )
    disp_map[mask] = np.nan

    dims = list(dataset.sizes.keys())[:2]

    var_mean_risk = xr.DataArray(var_map, dims=dims)
    var_nan_ratio = xr.DataArray(nan_ratio, dims=dims)

    # We add the new variables to the dataset
    dataset["confidence_from_mean_risk_max"] = var_mean_risk
    dataset["confidence_from_nanratio"] = var_nan_ratio
