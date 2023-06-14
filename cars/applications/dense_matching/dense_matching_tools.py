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
from typing import Dict, List

# Third party imports
import numpy as np
import pandora
import pandora.marge
import xarray as xr
from pandora import constants as p_cst
from pandora.img_tools import check_dataset
from pandora.state_machine import PandoraMachine
from pkg_resources import iter_entry_points

# CARS imports
from cars.applications.dense_matching import (
    dense_matching_constants as dense_match_cst,
)
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


def add_color(
    output_dataset: xr.Dataset,
    color: np.ndarray = None,
    color_mask: np.ndarray = None,
    band_im: list = None,
):
    """
    Add color and color mask to dataset

    :param output_dataset: output dataset
    :param color: color array
    :param color_mask: color mask array
    :param band_im: list of band names

    """
    if color is not None:
        if band_im is not None and cst.BAND_IM not in output_dataset.dims:
            output_dataset.coords[cst.BAND_IM] = band_im
        output_dataset[cst.EPI_COLOR] = xr.DataArray(
            color,
            dims=[cst.BAND_IM, cst.ROW, cst.COL],
        )

    # Add color mask
    if color_mask is not None:
        output_dataset[cst.EPI_COLOR_MSK] = xr.DataArray(
            color_mask,
            dims=[cst.ROW, cst.COL],
        )


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


def create_disp_dataset(
    disp: xr.Dataset,
    ref_dataset: xr.Dataset,
    compute_disparity_masks: bool = False,
    generate_performance_map=False,
    perf_ambiguity_threshold=0.6,
    disp_to_alt_ratio=None,
) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param compute_disparity_masks: compute_disparity_masks activation status
    :param generate_performance_map: True if generate performance map
    :type generate_performance_map: bool
    :param perf_ambiguity_threshold: ambiguity threshold used for
        performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float

    :return: disparity dataset as used in cars
    """
    # Retrieve disparity values
    disp_map = disp.disparity_map.values

    # retrieve masks
    masks = get_masks_from_pandora(disp, compute_disparity_masks)

    # retrieve colors
    color = None
    band_im = None
    if cst.EPI_COLOR in ref_dataset:
        color = ref_dataset[cst.EPI_COLOR].values
        if ref_dataset[cst.EPI_COLOR].values.shape[0] > 1:
            band_im = ref_dataset.coords[cst.BAND_IM]
        else:
            band_im = ["Gray"]

    color_mask = None
    if cst.EPI_COLOR_MSK in ref_dataset:
        color_mask = ref_dataset[cst.EPI_COLOR_MSK].values

    # retrieve classif
    classif = None
    band_classif = None
    if cst.EPI_CLASSIFICATION in ref_dataset:
        classif = ref_dataset[cst.EPI_CLASSIFICATION].values
        band_classif = ref_dataset.coords[cst.BAND_CLASSIF]

    # Crop disparity to ROI
    ref_roi = [
        int(-ref_dataset.attrs[cst.EPI_MARGINS][0]),
        int(-ref_dataset.attrs[cst.EPI_MARGINS][1]),
        int(ref_dataset.dims[cst.COL] - ref_dataset.attrs[cst.EPI_MARGINS][2]),
        int(ref_dataset.dims[cst.ROW] - ref_dataset.attrs[cst.EPI_MARGINS][3]),
    ]

    # Crop disparity map
    disp_map = disp_map[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop color
    if color is not None:
        color = color[:, ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop color mask
    if color_mask is not None:
        color_mask = color_mask[
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Crop masks
    for key in masks.copy():
        masks[key] = masks[key][
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Crop classif
    if classif is not None:
        classif = classif[:, ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Fill disparity array with 0 value for invalid points
    disp_map[masks[cst_disp.VALID] == 0] = 0

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
                np.copy(masks[cst_disp.VALID]),
            ),
        },
        coords={cst.ROW: row, cst.COL: col},
    )

    # add color
    add_color(disp_ds, color=color, color_mask=color_mask, band_im=band_im)

    # add confidence
    add_confidence(disp_ds, disp, ref_roi)

    # add performance map
    if generate_performance_map:
        add_performance_map(
            disp_ds, disp, ref_roi, perf_ambiguity_threshold, disp_to_alt_ratio
        )
    # add classif
    add_classification(disp_ds, classif=classif, band_classif=band_classif)

    if compute_disparity_masks:
        for key, val in masks.items():
            disp_ds[key] = xr.DataArray(np.copy(val), dims=[cst.ROW, cst.COL])

    disp_ds.attrs = disp.attrs.copy()
    disp_ds.attrs[cst.ROI] = ref_dataset.attrs[cst.ROI]

    disp_ds.attrs[cst.EPI_FULL_SIZE] = ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


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


def compute_disparity(
    left_dataset,
    right_dataset,
    corr_cfg,
    disp_min=None,
    disp_max=None,
    compute_disparity_masks=False,
    generate_performance_map=False,
    perf_ambiguity_threshold=0.6,
    disp_to_alt_ratio=None,
) -> Dict[str, xr.Dataset]:
    """
    This function will compute disparity.

    :param left_dataset: Dataset containing left image and mask
    :type left_dataset: xarray.Dataset
    :param right_dataset: Dataset containing right image and mask
    :type right_dataset: xarray.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min: Minimum disparity
                     (if None, value is taken from left dataset)
    :type disp_min: int
    :param disp_max: Maximum disparity
                     (if None, value is taken from left dataset)
    :type disp_max: int
    :param compute_disparity_masks: Activation of compute_disparity_masks mode
    :type compute_disparity_masks: Boolean
    :param generate_performance_map: True if generate performance map
    :type generate_performance_map: bool
    :param perf_ambiguity_threshold: ambiguity threshold used for
        performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float
    :return: Disparity dataset
    """

    # Check disp min and max bounds with respect to margin used for
    # rectification

    if disp_min is None:
        disp_min = left_dataset.attrs[cst.EPI_DISP_MIN]
    else:
        if disp_min < left_dataset.attrs[cst.EPI_DISP_MIN]:
            raise ValueError(
                "disp_min ({}) is lower than disp_min used to determine "
                "margin during rectification ({})".format(
                    disp_min, left_dataset["disp_min"]
                )
            )

    if disp_max is None:
        disp_max = left_dataset.attrs[cst.EPI_DISP_MAX]
    else:
        if disp_max > left_dataset.attrs[cst.EPI_DISP_MAX]:
            raise ValueError(
                "disp_max ({}) is greater than disp_max used to determine "
                "margin during rectification ({})".format(
                    disp_max, left_dataset["disp_max"]
                )
            )

    # Load pandora plugin
    for entry_point in iter_entry_points(group="pandora.plugin"):
        entry_point.load()

    # Update nodata values
    left_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_left"]
    right_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_right"]

    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()

    # check datasets
    check_dataset(left_dataset)
    check_dataset(right_dataset)

    # Run the Pandora pipeline
    ref, _ = pandora.run(
        pandora_machine,
        left_dataset,
        right_dataset,
        int(disp_min),
        int(disp_max),
        corr_cfg["pipeline"],
    )

    disp_dataset = create_disp_dataset(
        ref,
        left_dataset,
        compute_disparity_masks=compute_disparity_masks,
        generate_performance_map=generate_performance_map,
        perf_ambiguity_threshold=perf_ambiguity_threshold,
        disp_to_alt_ratio=disp_to_alt_ratio,
    )

    return disp_dataset


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
        tile_size = (1.0 - margin / 100.0) * np.sqrt(row_or_col)
        tile_size = tile_size_rounding * int(tile_size / tile_size_rounding)

    if tile_size > max_tile_size:
        tile_size = max_tile_size
    elif tile_size < min_tile_size:
        tile_size = min_tile_size

    return tile_size
