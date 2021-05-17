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

# Standard imports
import logging
import os
from typing import Dict, List

# Third party imports
import numpy as np
import pandora
import pandora.marge
import xarray as xr
from pandora import constants as pcst
from pandora.constants import (
    PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING as P_MSK_PR_N_D,
)
from pandora.img_tools import check_dataset
from pandora.state_machine import PandoraMachine
from pkg_resources import iter_entry_points
from scipy import interpolate

# CARS imports
from cars.conf import input_parameters, mask_classes
from cars.core import constants as cst
from cars.core import datasets


def create_inside_sec_roi_mask(
    disp: np.ndarray, disp_msk: np.ndarray, sec_dataset: xr.Dataset
) -> np.ndarray:
    """
    Create mask of disp values which are in the secondary image roi
    (255 if in the roi, otherwise 0)

    :param disp: disparity map
    :param disp_msk: disparity map valid values mask
    :param sec_dataset: secondary image dataset
    :return: mask of valid pixels that are in the secondary image roi
    """
    # create mask of secondary image roi
    sec_up_margin = abs(sec_dataset.attrs[cst.EPI_MARGINS][1])
    sec_bottom_margin = abs(sec_dataset.attrs[cst.EPI_MARGINS][3])
    sec_right_margin = abs(sec_dataset.attrs[cst.EPI_MARGINS][2])
    sec_left_margin = abs(sec_dataset.attrs[cst.EPI_MARGINS][0])

    # valid pixels that are inside the secondary image roi
    in_sec_roi_msk = np.zeros(disp.shape, dtype=np.int16)
    for i in range(0, disp.shape[0]):
        for j in range(0, disp.shape[1]):

            # if the pixel is valid
            if disp_msk[i, j] == 255:
                idx = float(j) + disp[i, j]

                # if the pixel is in the roi in the secondary image
                if (
                    sec_left_margin <= idx < disp.shape[1] - sec_right_margin
                    and sec_up_margin <= i < disp.shape[0] - sec_bottom_margin
                ):
                    in_sec_roi_msk[i, j] = 255

    return in_sec_roi_msk


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
    disp: xr.Dataset, verbose: bool
) -> Dict[str, np.ndarray]:
    """
    Get masks dictionary from the disparity map in output of pandora.

    :param disp: disparity map (pandora output)
    :param verbose: verbose activation status
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
    validity_mask_cropped = disp["validity_mask"].values
    # Mask intialization to false (all is invalid)
    msk = np.full(validity_mask_cropped.shape, False)
    # Identify valid points
    msk[
        np.where((validity_mask_cropped & pcst.PANDORA_MSK_PIXEL_INVALID) == 0)
    ] = True

    masks["mask"] = msk

    # With verbose, produce one mask for each invalid flag in
    # TODO: refactor in function (many duplicated code)
    if verbose:
        # Bit 9: False match bit 9
        msk_false_match = np.full(validity_mask_cropped.shape, False)
        msk_false_match[
            np.where(
                (validity_mask_cropped & pcst.PANDORA_MSK_PIXEL_MISMATCH) == 0
            )
        ] = True
        # Bit 8: Occlusion
        msk_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_occlusion[
            np.where(
                (validity_mask_cropped & pcst.PANDORA_MSK_PIXEL_OCCLUSION) == 0
            )
        ] = True
        # Bit 7: Masked in secondary image
        msk_masked_sec = np.full(validity_mask_cropped.shape, False)
        msk_masked_sec[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                )
                == 0
            )
        ] = True
        # Bit 6: Masked in reference image
        msk_masked_ref = np.full(validity_mask_cropped.shape, False)
        msk_masked_ref[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                )
                == 0
            )
        ] = True
        # Bit 5: Filled false match
        msk_filled_false_match = np.full(validity_mask_cropped.shape, False)
        msk_filled_false_match[
            np.where(
                (validity_mask_cropped & pcst.PANDORA_MSK_PIXEL_FILLED_MISMATCH)
                == 0
            )
        ] = True
        # Bit 4: Filled occlusion
        msk_filled_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_filled_occlusion[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION
                )
                == 0
            )
        ] = True
        # Bit 3: Computation stopped during pixelic step, under pixelic
        # interpolation never ended
        msk_stopped_interp = np.full(validity_mask_cropped.shape, False)
        msk_stopped_interp[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION
                )
                == 0
            )
        ] = True
        # Bit 2: Disparity range to explore is incomplete (borders reached in
        # secondary image)
        msk_incomplete_disp = np.full(validity_mask_cropped.shape, False)
        msk_incomplete_disp[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                )
                == 0
            )
        ] = True
        # Bit 1: Invalid in secondary image
        # fmt: off
        msk_invalid_sec = np.full(validity_mask_cropped.shape, False)
        msk_invalid_sec[
            np.where(
                (
    validity_mask_cropped  # noqa: E122
    & P_MSK_PR_N_D  # noqa: E122
                )
                == 0
            )
        ] = True
        # fmt: on
        # Bit 0: Invalid in reference image
        msk_invalid_ref = np.full(validity_mask_cropped.shape, False)
        msk_invalid_ref[
            np.where(
                (
                    validity_mask_cropped
                    & pcst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                )
                == 0
            )
        ] = True

        masks["masked_ref"] = msk_masked_ref
        masks["masked_sec"] = msk_masked_sec
        masks["incomplete_disp"] = msk_incomplete_disp
        masks["stopped_interp"] = msk_stopped_interp
        masks["filled_occlusion"] = msk_filled_occlusion
        masks["filled_false_match"] = msk_filled_false_match
        masks["invalid_ref"] = msk_invalid_ref
        masks["invalid_sec"] = msk_invalid_sec
        masks["occlusion"] = msk_occlusion
        masks["false_match"] = msk_false_match

    # Build final mask with 255 for valid points and 0 for invalid points
    # The mask is used by rasterize method (non zero are valid points)
    for key in masks:
        final_msk = np.ndarray(masks[key].shape, dtype=np.int16)
        final_msk[masks[key]] = 255
        final_msk[np.equal(masks[key], False)] = 0
        masks[key] = final_msk

    return masks


def create_disp_dataset(
    disp: xr.Dataset,
    ref_dataset: xr.Dataset,
    sec_dataset: xr.Dataset = None,
    check_roi_in_sec: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param sec_dataset: secondary dataset for the considered disparity map
                        (needed only if the check_roi_in_sec is set to True)
    :param check_roi_in_sec: option to invalid the values of the disparity
                             which end up outside the secondary image roi
    :param verbose: verbose activation status
    :return: disparity dataset as used in cars
    """
    # Retrieve disparity values
    disp_map = disp["disparity_map"].values

    # retrieve masks
    masks = get_masks_from_pandora(disp, verbose)
    if check_roi_in_sec:
        masks["inside_sec_roi"] = create_inside_sec_roi_mask(
            disp_map, masks["mask"], sec_dataset
        )
        masks["mask"][masks["inside_sec_roi"] == 0] = 0

    # Crop disparity to ROI
    if not check_roi_in_sec:
        ref_roi = [
            int(-ref_dataset.attrs[cst.EPI_MARGINS][0]),
            int(-ref_dataset.attrs[cst.EPI_MARGINS][1]),
            int(
                ref_dataset.dims[cst.COL]
                - ref_dataset.attrs[cst.EPI_MARGINS][2]
            ),
            int(
                ref_dataset.dims[cst.ROW]
                - ref_dataset.attrs[cst.EPI_MARGINS][3]
            ),
        ]
        disp_map = disp_map[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]
        for key in masks:
            masks[key] = masks[key][
                ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
            ]

    # Fill disparity array with 0 value for invalid points
    disp_map[masks["mask"] == 0] = 0

    # Build output dataset
    if not check_roi_in_sec:
        row = np.array(
            range(ref_dataset.attrs[cst.ROI][1], ref_dataset.attrs[cst.ROI][3])
        )
        col = np.array(
            range(ref_dataset.attrs[cst.ROI][0], ref_dataset.attrs[cst.ROI][2])
        )
    else:
        row = np.array(
            range(
                ref_dataset.attrs[cst.ROI_WITH_MARGINS][1],
                ref_dataset.attrs[cst.ROI_WITH_MARGINS][3],
            )
        )
        col = np.array(
            range(
                ref_dataset.attrs[cst.ROI_WITH_MARGINS][0],
                ref_dataset.attrs[cst.ROI_WITH_MARGINS][2],
            )
        )

    disp_ds = xr.Dataset(
        {
            cst.DISP_MAP: ([cst.ROW, cst.COL], np.copy(disp_map)),
            cst.DISP_MSK: ([cst.ROW, cst.COL], np.copy(masks["mask"])),
        },
        coords={cst.ROW: row, cst.COL: col},
    )
    if verbose:
        disp_ds[cst.DISP_MSK_INVALID_REF] = xr.DataArray(
            np.copy(masks["invalid_ref"]), dims=[cst.ROW, cst.COL]
        )
        disp_ds[cst.DISP_MSK_INVALID_SEC] = xr.DataArray(
            np.copy(masks["invalid_sec"]), dims=[cst.ROW, cst.COL]
        )
        disp_ds[cst.DISP_MSK_MASKED_REF] = xr.DataArray(
            np.copy(masks["masked_ref"]), dims=[cst.ROW, cst.COL]
        )
        disp_ds[cst.DISP_MSK_MASKED_SEC] = xr.DataArray(
            np.copy(masks["masked_sec"]), dims=[cst.ROW, cst.COL]
        )
        disp_ds[cst.DISP_MSK_OCCLUSION] = xr.DataArray(
            np.copy(masks["occlusion"]), dims=[cst.ROW, cst.COL]
        )
        disp_ds[cst.DISP_MSK_FALSE_MATCH] = xr.DataArray(
            np.copy(masks["false_match"]), dims=[cst.ROW, cst.COL]
        )
        if check_roi_in_sec:
            disp_ds[cst.DISP_MSK_INSIDE_SEC_ROI] = xr.DataArray(
                np.copy(masks["inside_sec_roi"]), dims=[cst.ROW, cst.COL]
            )

    disp_ds.attrs = disp.attrs.copy()
    disp_ds.attrs[cst.ROI] = ref_dataset.attrs[cst.ROI]
    if check_roi_in_sec:
        disp_ds.attrs[cst.ROI_WITH_MARGINS] = ref_dataset.attrs[
            cst.ROI_WITH_MARGINS
        ]
    disp_ds.attrs[cst.EPI_FULL_SIZE] = ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


def compute_mask_to_use_in_pandora(
    dataset: xr.Dataset,
    msk_key: str,
    classes_to_ignore: List[int],
    out_msk_dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Compute the mask to use in Pandora.
    Valid pixels will be set to the value of the 'valid_pixels' field of the
    correlation configuration file. No data pixels will be set to the value of
    the 'no_data' field of the correlation configuration file. Nonvalid pixels
    will be set to a value automatically determined to be different from the
    'valid_pixels' and the 'no_data' fields of the correlation configuration
    file.

    :param dataset: dataset containing the multi-classes mask from which the
                    mask to used in Pandora will be computed
    :param msk_key: key to use to access the multi-classes mask in the dataset
    :param classes_to_ignore:
    :param out_msk_dtype: numpy dtype of the returned mask
    :return: the mask to use in Pandora
    """

    ds_values_list = [key for key, _ in dataset.items()]
    if msk_key not in ds_values_list:
        worker_logger = logging.getLogger("distributed.worker")
        worker_logger.fatal(
            "No value identified by {} is "
            "present in the dataset".format(msk_key)
        )
        raise Exception(
            "No value identified by {} is "
            "present in the dataset".format(msk_key)
        )

    # retrieve specific values from datasets
    # Valid values and nodata values
    valid_pixels = dataset.attrs[cst.EPI_VALID_PIXELS]
    nodata_pixels = dataset.attrs[cst.EPI_NO_DATA_MASK]

    info_dtype = np.iinfo(out_msk_dtype)

    # find a value to use for unvalid pixels
    unvalid_pixels = None
    for i in range(info_dtype.max):
        if i not in (valid_pixels, nodata_pixels):
            unvalid_pixels = i
            break

    # initialization of the mask to use in Pandora
    final_msk = np.full(
        dataset[msk_key].values.shape,
        dtype=out_msk_dtype,
        fill_value=valid_pixels,
    )

    # retrieve the unvalid and nodata pixels locations
    unvalid_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values, classes_to_ignore, out_msk_dtype=np.bool
    )
    nodata_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values, [nodata_pixels], out_msk_dtype=np.bool
    )

    # update the mask to use in pandora with the unvalid and
    # nodata pixels values
    final_msk = np.where(unvalid_pixels_mask, unvalid_pixels, final_msk)
    final_msk = np.where(nodata_pixels_mask, nodata_pixels, final_msk)

    return final_msk


def compute_disparity(
    left_dataset,
    right_dataset,
    input_stereo_cfg,
    corr_cfg,
    disp_min=None,
    disp_max=None,
    use_sec_disp=True,
    verbose=False,
) -> Dict[str, xr.Dataset]:
    """
    This function will compute disparity.

    :param left_dataset: Dataset containing left image and mask
    :type left_dataset: xarray.Dataset
    :param right_dataset: Dataset containing right image and mask
    :type right_dataset: xarray.Dataset
    :param input_stereo_cfg: input stereo configuration
    :type input_stereo_cfg: dict
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min: Minimum disparity
                     (if None, value is taken from left dataset)
    :type disp_min: int
    :param disp_max: Maximum disparity
                     (if None, value is taken from left dataset)
    :type disp_max: int
    :param use_sec_disp: Boolean activating the use of the secondary
                         disparity map
    :type use_sec_disp: bool
    :param verbose: Activation of verbose mode
    :type verbose: Boolean
    :returns: Dictionary of disparity dataset. Keys are:
        * 'ref' for the left to right disparity map
        * 'sec' for the right to left disparity map
        if it is computed by Pandora
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

    # Handle masks' classes if necessary
    # TODO : Refacto stereo to not change attributes here
    # but in stereo class or dedicated functional step
    mask1_classes = input_stereo_cfg[input_parameters.INPUT_SECTION_TAG].get(
        input_parameters.MASK1_CLASSES_TAG, None
    )
    mask2_classes = input_stereo_cfg[input_parameters.INPUT_SECTION_TAG].get(
        input_parameters.MASK2_CLASSES_TAG, None
    )
    mask1_use_classes = False
    mask2_use_classes = False

    if mask1_classes is not None:
        classes_dict = mask_classes.read_mask_classes(mask1_classes)
        if mask_classes.ignored_by_corr_tag in classes_dict.keys():
            left_msk = left_dataset[cst.EPI_MSK].values
            left_dataset[cst.EPI_MSK].values = compute_mask_to_use_in_pandora(
                left_dataset,
                cst.EPI_MSK,
                classes_dict[mask_classes.ignored_by_corr_tag],
            )
            mask1_use_classes = True

    if mask2_classes is not None:
        classes_dict = mask_classes.read_mask_classes(mask2_classes)
        if mask_classes.ignored_by_corr_tag in classes_dict.keys():
            right_msk = right_dataset[cst.EPI_MSK].values
            right_dataset[cst.EPI_MSK].values = compute_mask_to_use_in_pandora(
                right_dataset,
                cst.EPI_MSK,
                classes_dict[mask_classes.ignored_by_corr_tag],
            )
            mask2_use_classes = True

    # Update nodata values
    left_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_left"]
    right_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_right"]

    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()

    # check datasets
    checked_left_dataset = check_dataset(left_dataset)
    checked_right_dataset = check_dataset(right_dataset)

    # Run the Pandora pipeline
    ref, sec = pandora.run(
        pandora_machine,
        checked_left_dataset,
        checked_right_dataset,
        int(disp_min),
        int(disp_max),
        corr_cfg["pipeline"],
    )

    # Set the datasets' cst.EPI_MSK values back to the original
    # multi-classes masks
    if mask1_use_classes:
        left_dataset[cst.EPI_MSK].values = left_msk
    if mask2_use_classes:
        right_dataset[cst.EPI_MSK].values = right_msk

    disp = {}
    disp[cst.STEREO_REF] = create_disp_dataset(
        ref, left_dataset, verbose=verbose
    )

    if bool(sec.dims) and use_sec_disp:
        # for the secondary disparity map, the reference is the right dataset
        # and the secondary image is the left one
        logging.info(
            "Secondary disparity map will be used to densify "
            "the points cloud"
        )
        disp[cst.STEREO_SEC] = create_disp_dataset(
            sec,
            right_dataset,
            sec_dataset=left_dataset,
            check_roi_in_sec=True,
            verbose=verbose,
        )

    return disp


def optimal_tile_size_pandora_plugin_libsgm(
    disp_min: int,
    disp_max: int,
    min_tile_size: int,
    max_tile_size: int,
    otb_max_ram_hint: int = None,
    tile_size_rounding: int = 50,
    margin: int = 0,
) -> int:
    """
    Compute optimal tile size according to estimated memory usage
    (pandora_plugin_libsgm)
    Returned optimal tile size will be at least equal to tile_size_rounding.

    :param disp_min: Minimum disparity to explore
    :param disp_max: Maximum disparity to explore
    :param min_tile_size : Minimal tile size
    :param max_tile_size : Maximal tile size
    :param otb_max_ram_hint: amount of RAM allocated to OTB (if None, will try
                             to read it from environment variable)
    :param tile_size_rounding: Optimal tile size will be aligned to multiples
                               of tile_size_rounding
    :param margin: margin to remove to the computed tile size
                   (as a percent of the computed tile size)
    :returns: Optimal tile size according to benchmarked memory usage
    """

    if otb_max_ram_hint is None:
        if "OTB_MAX_RAM_HINT" in os.environ:
            otb_max_ram_hint = int(os.environ["OTB_MAX_RAM_HINT"])
        else:
            raise ValueError(
                "otb_max_ram_hint is None and OTB_MAX_RAM_HINT "
                "envvar is not set"
            )

    memory = otb_max_ram_hint
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

    row_or_col = float(((memory - import_) * 2 ** 23)) / tot

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


def estimate_color_from_disparity(
    disp_ref_to_sec: xr.Dataset, sec_ds: xr.Dataset, sec_color: xr.Dataset
) -> xr.Dataset:
    """
    Estimate color image of reference from the disparity map and the secondary
    color image.

    :param disp_ref_to_sec: disparity map
    :param sec_ds: secondary image dataset
    :param sec_color: secondary color image
    :return: interpolated reference color image dataset
    """
    # retrieve numpy arrays from input datasets
    disp_msk = disp_ref_to_sec[cst.DISP_MSK].values
    im_color = sec_color[cst.EPI_IMAGE].values
    if cst.EPI_MSK in sec_color.variables.keys():
        im_msk = sec_color[cst.EPI_MSK].values

    # retrieve image sizes
    nb_bands, nb_row, nb_col = im_color.shape
    nb_disp_row, nb_disp_col = disp_ref_to_sec[cst.DISP_MAP].values.shape

    sec_up_margin = abs(sec_ds.attrs[cst.EPI_MARGINS][1])
    sec_left_margin = abs(sec_ds.attrs[cst.EPI_MARGINS][0])

    # instantiate final image
    final_interp_color = np.zeros(
        (nb_disp_row, nb_disp_col, nb_bands), dtype=np.float
    )

    # construct secondary color image pixels positions
    clr_x_positions, clr_y_positions = np.meshgrid(
        np.linspace(0, nb_col - 1, nb_col), np.linspace(0, nb_row - 1, nb_row)
    )
    clr_xy_positions = np.concatenate(
        [
            clr_x_positions.reshape(1, nb_row * nb_col).transpose(),
            clr_y_positions.reshape(1, nb_row * nb_col).transpose(),
        ],
        axis=1,
    )

    # construct the positions for which the interpolation has to be done
    interpolated_points = np.zeros(
        (nb_disp_row * nb_disp_col, 2), dtype=np.float
    )
    for i in range(0, disp_ref_to_sec[cst.DISP_MAP].values.shape[0]):
        for j in range(0, disp_ref_to_sec[cst.DISP_MAP].values.shape[1]):

            # if the pixel is valid,
            # else the position is left to (0,0)
            # and the final image pixel value will be set to np.nan
            if disp_msk[i, j] == 255:
                idx = j + disp_ref_to_sec[cst.DISP_MAP].values[i, j]
                interpolated_points[i * nb_disp_col + j, 0] = (
                    idx - sec_left_margin
                )
                interpolated_points[i * nb_disp_col + j, 1] = i - sec_up_margin

    # construct final image mask
    final_msk = disp_msk
    if cst.EPI_MSK in sec_color.variables.keys():
        # interpolate the color image mask to the new image referential
        # (nearest neighbor interpolation)
        msk_values = im_msk.reshape(nb_row * nb_col, 1)
        interp_msk_value = interpolate.griddata(
            clr_xy_positions, msk_values, interpolated_points, method="nearest"
        )
        interp_msk = interp_msk_value.reshape(nb_disp_row, nb_disp_col)

        # remove from the final mask all values which are interpolated from non
        # valid values (strictly non equal to 255)
        final_msk[interp_msk == 0] = 0

    # interpolate each band of the color image
    for band in range(nb_bands):
        # get band values
        band_im = im_color[band, :, :]
        clr_values = band_im.reshape(nb_row * nb_col, 1)

        # interpolate values
        interp_values = interpolate.griddata(
            clr_xy_positions, clr_values, interpolated_points, method="nearest"
        )
        final_interp_color[:, :, band] = interp_values.reshape(
            nb_disp_row, nb_disp_col
        )

        # apply final mask
        final_interp_color[:, :, band][final_msk != 255] = np.nan

    # create interpolated color image dataset
    region = list(disp_ref_to_sec.attrs[cst.ROI_WITH_MARGINS])
    largest_size = disp_ref_to_sec.attrs[cst.EPI_FULL_SIZE]

    interp_clr_ds = datasets.create_im_dataset(
        final_interp_color, region, largest_size, band_coords=True, msk=None
    )
    interp_clr_ds.attrs[cst.ROI] = disp_ref_to_sec.attrs[cst.ROI]
    interp_clr_ds.attrs[cst.ROI_WITH_MARGINS] = disp_ref_to_sec.attrs[
        cst.ROI_WITH_MARGINS
    ]

    return interp_clr_ds
