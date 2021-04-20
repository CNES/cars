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
Stereo module:
contains stereo-rectification, disparity map estimation
"""

# Standard imports
from __future__ import absolute_import
from typing import Dict, List, Tuple
import warnings
import os
import math
import logging
from pkg_resources import iter_entry_points

# Third party imports
import numpy as np
from scipy import interpolate

from scipy.spatial import Delaunay #pylint: disable=no-name-in-module
from scipy.spatial import tsearch #pylint: disable=no-name-in-module
from scipy.spatial import cKDTree #pylint: disable=no-name-in-module

import rasterio as rio
import xarray as xr
from dask import sizeof
import pandora
import pandora.marge
from pandora import constants as pcst

# Cars imports
from cars.conf import input_parameters as in_params
from cars import projection
from cars import tiling
from cars.conf import mask_classes, output_prepare
from cars import constants as cst
from cars import matching_regularisation
from cars.lib.steps.epi_rectif.grids import compute_epipolar_grid_min_max
from cars.lib.steps.epi_rectif.resampling import epipolar_rectify_images
from cars.datasets import create_im_dataset
from cars.lib.steps import triangulation

# Register sizeof for xarray
@sizeof.sizeof.register_lazy("xarray")
def register_xarray():
    """
    Add hook to dask so it correctly estimates memory used by xarray
    """
    @sizeof.sizeof.register(xr.DataArray)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataarray(xarr):
        """
        Inner function for total size of xarray_dataarray
        """
        total_size = sizeof.sizeof(xarr.values)
        for __, carray in xarr.coords.items():
            total_size += sizeof.sizeof(carray.values)
        total_size += sizeof.sizeof(xarr.attrs)
        return total_size
    @sizeof.sizeof.register(xr.Dataset)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataset(xdat):
        """
        Inner function for total size of xarray_dataset
        """
        total_size = 0
        for __, varray in xdat.data_vars.items():
            total_size += sizeof.sizeof(varray.values)
        for __, carray in xdat.coords.items():
            total_size += sizeof.sizeof(carray)
        total_size += sizeof.sizeof(xdat.attrs)
        return total_size

# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def optimal_tile_size_pandora_plugin_libsgm(disp_min: int,
        disp_max: int, min_tile_size: int, max_tile_size: int,
        otb_max_ram_hint: int=None,
        tile_size_rounding: int=50,
        margin: int=0) -> int:
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
                'otb_max_ram_hint is None and OTB_MAX_RAM_HINT '
                'envvar is not set')

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
    tot += confidence + 2*cv_ + nan_ + cv_uint + penal + img_crop
    import_ = 200 #MiB

    row_or_col = float(((memory-import_)* 2**23)) / tot

    if row_or_col <= 0:
        logging.warning(
            "Optimal tile size is null, "
            "forcing it to {} pixels".format(tile_size_rounding))
        tile_size = tile_size_rounding
    else:
        tile_size = (1.-margin/100.)*np.sqrt(row_or_col)
        tile_size = tile_size_rounding * int(tile_size / tile_size_rounding)

    if tile_size > max_tile_size:
        tile_size = max_tile_size
    elif tile_size < min_tile_size:
        tile_size = min_tile_size

    return tile_size





def compute_disparity(left_dataset,
                      right_dataset,
                      input_stereo_cfg,
                      corr_cfg,
                      disp_min=None,
                      disp_max=None,
                      use_sec_disp=True,
                      verbose=False) -> Dict[str, xr.Dataset]:
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
                    disp_min, left_dataset['disp_min']))

    if disp_max is None:
        disp_max = left_dataset.attrs[cst.EPI_DISP_MAX]
    else:
        if disp_max > left_dataset.attrs[cst.EPI_DISP_MAX]:
            raise ValueError(
                "disp_max ({}) is greater than disp_max used to determine "
                "margin during rectification ({})".format(
                    disp_max, left_dataset['disp_max']))

    # Load pandora plugin
    for entry_point in iter_entry_points(group='pandora.plugin'):
        entry_point.load()

    if corr_cfg['image']\
               ['no_data'] != mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION:
        logging.warning('mask no data value defined in the correlation '
                        'configuration file does not match the internal no '
                        'data value used for epipolar rectification.')

    # Handle masks' classes if necessary
    # TODO ce serait le rêve un peu d'avoir tout ça déjà dans les attributs
    #  de la classe stereo pour pas le modifier dans chaque fonction ici ;-)
    mask1_classes = input_stereo_cfg[in_params.INPUT_SECTION_TAG]\
                                    .get(in_params.MASK1_CLASSES_TAG, None)
    mask2_classes = input_stereo_cfg[in_params.INPUT_SECTION_TAG]\
                                    .get(in_params.MASK2_CLASSES_TAG, None)
    mask1_use_classes = False
    mask2_use_classes = False

    if mask1_classes is not None:
        classes_dict = mask_classes.read_mask_classes(mask1_classes)
        if mask_classes.ignored_by_corr_tag in classes_dict.keys():
            left_msk = left_dataset[cst.EPI_MSK].values
            left_dataset[cst.EPI_MSK].values = \
                compute_mask_to_use_in_pandora(
                    corr_cfg,
                    left_dataset,
                    cst.EPI_MSK,
                    classes_dict[mask_classes.ignored_by_corr_tag])
            mask1_use_classes = True

    if mask2_classes is not None:
        classes_dict = mask_classes.read_mask_classes(mask2_classes)
        if mask_classes.ignored_by_corr_tag in classes_dict.keys():
            right_msk = right_dataset[cst.EPI_MSK].values
            right_dataset[cst.EPI_MSK].values = \
                compute_mask_to_use_in_pandora(
                    corr_cfg,
                    right_dataset,
                    cst.EPI_MSK,
                    classes_dict[mask_classes.ignored_by_corr_tag])
            mask2_use_classes = True

    # Run the Pandora pipeline
    ref, sec = pandora.run(left_dataset,
                           right_dataset,
                           int(disp_min),
                           int(disp_max),
                           corr_cfg)

    # Set the datasets' cst.EPI_MSK values back to the original
    # multi-classes masks
    if mask1_use_classes:
        left_dataset[cst.EPI_MSK].values = left_msk
    if mask2_use_classes:
        right_dataset[cst.EPI_MSK].values = right_msk

    disp = dict()
    disp[cst.STEREO_REF] = create_disp_dataset(ref,
                                               left_dataset,
                                               verbose=verbose)

    if bool(sec.dims) and use_sec_disp:
        # for the secondary disparity map, the reference is the right dataset
        # and the secondary image is the left one
        logging.info('Secondary disparity map will be used to densify '
                     'the points cloud')
        disp[cst.STEREO_SEC] = create_disp_dataset(sec,
                                                   right_dataset,
                                                   sec_dataset=left_dataset,
                                                   check_roi_in_sec=True,
                                                   verbose=verbose)

    return disp


def compute_mask_to_use_in_pandora(
    corr_cfg,
    dataset: xr.Dataset,
    msk_key: str,
    classes_to_ignore: List[int],
    out_msk_dtype: np.dtype=np.int16) -> np.ndarray:
    """
    Compute the mask to use in Pandora.
    Valid pixels will be set to the value of the 'valid_pixels' field of the
    correlation configuration file. No data pixels will be set to the value of
    the 'no_data' field of the correlation configuration file. Nonvalid pixels
    will be set to a value automatically determined to be different from the
    'valid_pixels' and the 'no_data' fields of the correlation configuration
    file.

    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param dataset: dataset containing the multi-classes mask from which the
                    mask to used in Pandora will be computed
    :param msk_key: key to use to access the multi-classes mask in the dataset
    :param classes_to_ignore:
    :param out_msk_dtype: numpy dtype of the returned mask
    :return: the mask to use in Pandora
    """

    ds_values_list = [key for key, _ in dataset.items()]
    if msk_key not in ds_values_list:
        worker_logger = logging.getLogger('distributed.worker')
        worker_logger.fatal('No value identified by {} is '
                            'present in the dataset'.format(msk_key))
        raise Exception('No value identified by {} is '
                        'present in the dataset'.format(msk_key))

    # retrieve specific values from the correlation configuration file
    valid_pixels = corr_cfg['image']['valid_pixels']
    nodata_pixels = corr_cfg['image']['no_data']

    info_dtype = np.iinfo(out_msk_dtype)

    # find a value to use for unvalid pixels
    unvalid_pixels = None
    for i in range(info_dtype.max):
        if i not in (valid_pixels, nodata_pixels):
            unvalid_pixels = i
            break

    # initialization of the mask to use in Pandora
    final_msk = np.full(dataset[msk_key].values.shape,
                        dtype=out_msk_dtype,
                        fill_value=valid_pixels)

    # retrieve the unvalid and nodata pixels locations
    unvalid_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values,
        classes_to_ignore,
        out_msk_dtype=np.bool)
    nodata_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values,
        [nodata_pixels],
        out_msk_dtype=np.bool)

    # update the mask to use in pandora with the unvalid and
    # nodata pixels values
    final_msk = np.where(unvalid_pixels_mask, unvalid_pixels, final_msk)
    final_msk = np.where(nodata_pixels_mask, nodata_pixels, final_msk)

    return final_msk


def create_disp_dataset(disp: xr.Dataset,
                        ref_dataset: xr.Dataset,
                        sec_dataset: xr.Dataset=None,
                        check_roi_in_sec:bool=False,
                        verbose:bool=False) -> xr.Dataset:
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
    disp_map = disp['disparity_map'].values

    # retrieve masks
    masks = get_masks_from_pandora(disp, verbose)
    if check_roi_in_sec:
        masks['inside_sec_roi'] = create_inside_sec_roi_mask(disp_map,
                                                             masks['mask'],
                                                             sec_dataset)
        masks['mask'][masks['inside_sec_roi'] == 0] = 0

    # Crop disparity to ROI
    if not check_roi_in_sec:
        ref_roi = [int(-ref_dataset.attrs[cst.EPI_MARGINS][0]),
                   int(-ref_dataset.attrs[cst.EPI_MARGINS][1]),
                   int(ref_dataset.dims[cst.COL] - \
                       ref_dataset.attrs[cst.EPI_MARGINS][2]),
                   int(ref_dataset.dims[cst.ROW] - \
                       ref_dataset.attrs[cst.EPI_MARGINS][3])]
        disp_map = disp_map[ref_roi[1]:ref_roi[3], ref_roi[0]:ref_roi[2]]
        for key in masks:
            masks[key] = masks[key][ref_roi[1]:ref_roi[3],
                                    ref_roi[0]:ref_roi[2]]

    # Fill disparity array with 0 value for invalid points
    disp_map[masks['mask'] == 0] = 0

    # Build output dataset
    if not check_roi_in_sec:
        row = np.array(range(ref_dataset.attrs[cst.ROI][1],
                             ref_dataset.attrs[cst.ROI][3]))
        col = np.array(range(ref_dataset.attrs[cst.ROI][0],
                             ref_dataset.attrs[cst.ROI][2]))
    else:
        row = np.array(range(ref_dataset.attrs[cst.ROI_WITH_MARGINS][1],
                             ref_dataset.attrs[cst.ROI_WITH_MARGINS][3]))
        col = np.array(range(ref_dataset.attrs[cst.ROI_WITH_MARGINS][0],
                             ref_dataset.attrs[cst.ROI_WITH_MARGINS][2]))

    disp_ds = xr.Dataset({cst.DISP_MAP: ([cst.ROW, cst.COL],
                                         np.copy(disp_map)),
                          cst.DISP_MSK: ([cst.ROW, cst.COL],
                                          np.copy(masks['mask']))},
                         coords={cst.ROW: row, cst.COL: col})
    if verbose:
        disp_ds[cst.DISP_MSK_INVALID_REF] = \
            xr.DataArray(np.copy(masks['invalid_ref']),
                         dims=[cst.ROW, cst.COL])
        disp_ds[cst.DISP_MSK_INVALID_SEC] = \
            xr.DataArray(np.copy(masks['invalid_sec']),
                         dims=[cst.ROW, cst.COL])
        disp_ds[cst.DISP_MSK_MASKED_REF] = \
            xr.DataArray(np.copy(masks['masked_ref']),
                         dims=[cst.ROW, cst.COL])
        disp_ds[cst.DISP_MSK_MASKED_SEC] = \
            xr.DataArray(np.copy(masks['masked_sec']),
                         dims=[cst.ROW, cst.COL])
        disp_ds[cst.DISP_MSK_OCCLUSION] = \
            xr.DataArray(np.copy(masks['occlusion']),
                         dims=[cst.ROW, cst.COL])
        disp_ds[cst.DISP_MSK_FALSE_MATCH] = \
            xr.DataArray(np.copy(masks['false_match']),
                         dims=[cst.ROW, cst.COL])
        if check_roi_in_sec:
            disp_ds[cst.DISP_MSK_INSIDE_SEC_ROI] = \
                xr.DataArray(np.copy(masks['inside_sec_roi']),
                             dims=[cst.ROW, cst.COL])

    disp_ds.attrs = disp.attrs.copy()
    disp_ds.attrs[cst.ROI] = ref_dataset.attrs[cst.ROI]
    if check_roi_in_sec:
        disp_ds.attrs[cst.ROI_WITH_MARGINS] = \
            ref_dataset.attrs[cst.ROI_WITH_MARGINS]
    disp_ds.attrs[cst.EPI_FULL_SIZE] = \
        ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


def create_inside_sec_roi_mask(disp: np.ndarray,
                               disp_msk: np.ndarray,
                               sec_dataset: xr.Dataset) -> np.ndarray:
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
            if disp_msk[i,j] == 255:
                idx = float(j) + disp[i,j]

                # if the pixel is in the roi in the secondary image
                if sec_left_margin <= idx < disp.shape[1] - sec_right_margin \
                    and sec_up_margin <= i < disp.shape[0] - sec_bottom_margin:
                    in_sec_roi_msk[i,j] = 255

    return in_sec_roi_msk


def get_masks_from_pandora(disp:xr.Dataset,
                           verbose: bool) -> Dict[str, np.ndarray]:
    """
    Get masks dictionary from the disparity map in output of pandora.

    :param disp: disparity map (pandora output)
    :param verbose: verbose activation status
    :return: masks dictionary
    """
    masks = dict()

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
    validity_mask_cropped = disp['validity_mask'].values
    # Mask intialization to false (all is invalid)
    msk = np.full(validity_mask_cropped.shape, False)
    # Identify valid points
    msk[np.where((validity_mask_cropped & \
                  pcst.PANDORA_MSK_PIXEL_INVALID) == 0)] = True

    masks['mask'] = msk

    # With verbose, produce one mask for each invalid flag in
    if verbose:
        # Bit 9: False match bit 9
        msk_false_match = np.full(validity_mask_cropped.shape, False)
        msk_false_match[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_MISMATCH) == 0)] = True
        # Bit 8: Occlusion
        msk_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_occlusion[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_OCCLUSION) == 0)] = True
        # Bit 7: Masked in secondary image
        msk_masked_sec = np.full(validity_mask_cropped.shape, False)
        msk_masked_sec[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC) == 0)] = True
        # Bit 6: Masked in reference image
        msk_masked_ref = np.full(validity_mask_cropped.shape, False)
        msk_masked_ref[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF) == 0)] = True
        # Bit 5: Filled false match
        msk_filled_false_match = np.full(validity_mask_cropped.shape, False)
        msk_filled_false_match[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_FILLED_MISMATCH) == 0)] = True
        # Bit 4: Filled occlusion
        msk_filled_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_filled_occlusion[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION) == 0)] = True
        # Bit 3: Computation stopped during pixelic step, under pixelic
        # interpolation never ended
        msk_stopped_interp = np.full(validity_mask_cropped.shape, False)
        msk_stopped_interp[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION) == 0)] = True
        # Bit 2: Disparity range to explore is incomplete (borders reached in
        # secondary image)
        msk_incomplete_disp = np.full(validity_mask_cropped.shape, False)
        msk_incomplete_disp[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE) == 0)] \
                 = True
        # Bit 1: Invalid in secondary image
        msk_invalid_sec = np.full(validity_mask_cropped.shape, False)
        msk_invalid_sec[np.where(
            (validity_mask_cropped & \
        pcst.PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING) == 0)] \
            = True
        # Bit 0: Invalid in reference image
        msk_invalid_ref = np.full(validity_mask_cropped.shape, False)
        msk_invalid_ref[np.where(
            (validity_mask_cropped & \
             pcst.PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER) == 0)] = True

        masks['masked_ref'] = msk_masked_ref
        masks['masked_sec'] = msk_masked_sec
        masks['incomplete_disp'] = msk_incomplete_disp
        masks['stopped_interp'] = msk_stopped_interp
        masks['filled_occlusion'] = msk_filled_occlusion
        masks['filled_false_match'] = msk_filled_false_match
        masks['invalid_ref'] = msk_invalid_ref
        masks['invalid_sec'] = msk_invalid_sec
        masks['occlusion'] = msk_occlusion
        masks['false_match'] = msk_false_match

    # Build final mask with 255 for valid points and 0 for invalid points
    # The mask is used by rasterize method (non zero are valid points)
    for key in masks:
        final_msk = np.ndarray(masks[key].shape, dtype=np.int16)
        final_msk[masks[key]] = 255
        final_msk[np.equal(masks[key], False)] = 0
        masks[key] = final_msk

    return masks


def estimate_color_from_disparity(disp_ref_to_sec: xr.Dataset,
                                  sec_ds: xr.Dataset,
                                  sec_color: xr.Dataset) -> xr.Dataset:
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
    final_interp_color = np.zeros((nb_disp_row, nb_disp_col, nb_bands),
                                  dtype=np.float)

    # construct secondary color image pixels positions
    clr_x_positions, clr_y_positions = np.meshgrid(np.linspace(0,
                                                               nb_col-1,
                                                               nb_col),
                                                   np.linspace(0,
                                                               nb_row-1,
                                                               nb_row))
    clr_xy_positions = np.concatenate([
        clr_x_positions.reshape(1,nb_row*nb_col).transpose(),
        clr_y_positions.reshape(1, nb_row*nb_col).transpose()],
        axis=1)

    # construct the positions for which the interpolation has to be done
    interpolated_points = np.zeros((nb_disp_row * nb_disp_col, 2),
                                   dtype=np.float)
    for i in range(0, disp_ref_to_sec[cst.DISP_MAP].values.shape[0]):
        for j in range(0, disp_ref_to_sec[cst.DISP_MAP].values.shape[1]):

            # if the pixel is valid,
            # else the position is left to (0,0)
            # and the final image pixel value will be set to np.nan
            if disp_msk[i, j] == 255:
                idx = j + disp_ref_to_sec[cst.DISP_MAP].values[i, j]
                interpolated_points[i * nb_disp_col + j,
                                    0] = idx - sec_left_margin
                interpolated_points[i * nb_disp_col + j,
                                    1] = i - sec_up_margin

    # construct final image mask
    final_msk = disp_msk
    if cst.EPI_MSK in sec_color.variables.keys():
        # interpolate the color image mask to the new image referential
        # (nearest neighbor interpolation)
        msk_values = im_msk.reshape(nb_row * nb_col, 1)
        interp_msk_value = interpolate.griddata(clr_xy_positions,
                                                msk_values,
                                                interpolated_points,
                                                method='nearest')
        interp_msk = interp_msk_value.reshape(nb_disp_row, nb_disp_col)

        # remove from the final mask all values which are interpolated from non
        # valid values (strictly non equal to 255)
        final_msk[interp_msk == 0] = 0

    # interpolate each band of the color image
    for band in range(nb_bands):
        # get band values
        band_im = im_color[band,:,:]
        clr_values = band_im.reshape(nb_row*nb_col, 1)

        # interpolate values
        interp_values = interpolate.griddata(clr_xy_positions,
                                             clr_values,
                                             interpolated_points,
                                             method='nearest')
        final_interp_color[:, :, band] = interp_values.reshape(nb_disp_row,
                                                               nb_disp_col)

        # apply final mask
        final_interp_color[:, :, band][final_msk != 255] = np.nan

    # create interpolated color image dataset
    region = list(disp_ref_to_sec.attrs[cst.ROI_WITH_MARGINS])
    largest_size = disp_ref_to_sec.attrs[cst.EPI_FULL_SIZE]

    interp_clr_ds = create_im_dataset(final_interp_color,
                                      region,
                                      largest_size,
                                      band_coords=True,
                                      msk=None)
    interp_clr_ds.attrs[cst.ROI] = disp_ref_to_sec.attrs[cst.ROI]
    interp_clr_ds.attrs[cst.ROI_WITH_MARGINS] = \
        disp_ref_to_sec.attrs[cst.ROI_WITH_MARGINS]

    return interp_clr_ds


def images_pair_to_3d_points(input_stereo_cfg,
                             region,
                             corr_cfg,
                             epsg=None,
                             disp_min=None,
                             disp_max=None,
                             out_epsg=None,
                             geoid_data=None,
                             use_sec_disp=False,
                             snap_to_img1=False,
                             align=False,
                             add_msk_info=False) -> Dict[str,
                                                         Tuple[xr.Dataset,
                                                               xr.Dataset]]:
    # Retrieve disp min and disp max if needed
    """
    This function will produce a 3D points cloud as an xarray.Dataset from the
    given stereo configuration (from both left to right disparity map and right
    to left disparity map if the latter is computed by Pandora).
    Clouds will be produced over the region with the specified EPSG, using
    disp_min and disp_max
    :param input_stereo_cfg: Configuration for stereo processing
    :type StereoConfiguration
    :param region: Array defining region.

    * For espg region as [lat_min, lon_min, lat_max, lon_max]
    * For epipolar region as [xmin, ymin, xmax, ymax]

    :type region: numpy.array
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param epsg: EPSG code for the region,
                 if None then epipolar geometry is considered
    :type epsg: int
    :param disp_min: Minimum disparity value
    :type disp_min: int
    :param disp_max: Maximum disparity value
    :type disp_max: int
    :param geoid_data: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_data: xarray.Dataset
    :param use_sec_disp: Boolean activating the use of the secondary
                         disparity map
    :type use_sec_disp: bool
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so
                         as to cross those of img1
    :param snap_to_img1: bool
    :param align: If True, apply correction to point after triangulation to
                  align with lowres DEM (if available. If not, no correction
                  is applied)
    :param align: bool
    :param add_msk_info: boolean enabling the addition of the masks'
                         information in the point clouds final dataset
    :returns: Dictionary of tuple. The tuple are constructed with the dataset
              containing the 3D points +
    A dataset containing color of left image, or None

    The dictionary keys are :
        * 'ref' to retrieve the dataset built from the left to right
          disparity map
        * 'sec' to retrieve the dataset built from the right to left
          disparity map (if computed in Pandora)
    """


    # Retrieve disp min and disp max if needed
    preprocessing_output_cfg = input_stereo_cfg\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_cfg[
        output_prepare.MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_cfg[
        output_prepare.MAXIMUM_DISPARITY_TAG]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Compute margins for the correlator
    margins = pandora.marge.get_margins(
        disp_min, disp_max, corr_cfg)

    # Reproject region to epipolar geometry if necessary
    if epsg is not None:
        region = transform_terrain_region_to_epipolar(
            region, input_stereo_cfg, epsg,  disp_min, disp_max)

    # Rectify images
    left, right, color = epipolar_rectify_images(input_stereo_cfg,
                                                 region,
                                                 margins)
    # Compute disparity
    disp = compute_disparity(
        left, right, input_stereo_cfg, corr_cfg, disp_min, disp_max,
        use_sec_disp=use_sec_disp)

    # If necessary, set disparity to 0 for classes to be set to input dem
    mask1_classes = input_stereo_cfg \
        [in_params.INPUT_SECTION_TAG].get(in_params.MASK1_CLASSES_TAG, None)
    mask2_classes = input_stereo_cfg \
        [in_params.INPUT_SECTION_TAG].get(in_params.MASK2_CLASSES_TAG, None)
    matching_regularisation.update_disp_to_0(
        disp, left, right, mask1_classes, mask2_classes)

    colors = dict()
    colors[cst.STEREO_REF] = color
    if cst.STEREO_SEC in disp:
        # compute right color image from right-left disparity map
        colors[cst.STEREO_SEC] = estimate_color_from_disparity(
            disp[cst.STEREO_SEC], left, color)

    im_ref_msk = None
    im_sec_msk = None
    if add_msk_info:
        ref_values_list = [key for key, _ in left.items()]
        if cst.EPI_MSK in ref_values_list:
            im_ref_msk = left
        else:
            worker_logger = logging.getLogger('distributed.worker')
            worker_logger.warning("Left image does not have a "
                                  "mask to rasterize")
        if cst.STEREO_SEC in disp:
            sec_values_list = [key for key, _ in right.items()]
            if cst.EPI_MSK in sec_values_list:
                im_sec_msk = right
            else:
                worker_logger = logging.getLogger('distributed.worker')
                worker_logger.warning("Right image does not have a "
                                      "mask to rasterize")

    # Triangulate
    if cst.STEREO_SEC in disp:
        points = triangulation.triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF], disp[cst.STEREO_SEC],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)
    else:
        points = triangulation.triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key in points:
            points[key] = triangulation.geoid_offset(points[key], geoid_data)

    if out_epsg is not None:
        for key in points:
            projection.points_cloud_conversion_dataset(points[key], out_epsg)

    return points, colors


def transform_terrain_region_to_epipolar(
        region, conf,
        epsg = 4326,
        disp_min = None,
        disp_max = None,
        step = 100):
    """
    Transform terrain region to epipolar region according to ground_positions

    :param region: The terrain region to transform to epipolar region
                   ([lat_min, lon_min, lat_max, lon_max])
    :type region: list of four float
    :param ground_positions: Grid of ground positions for epipolar geometry
    :type ground_positions: numpy array
    :param origin: origin of the grid
    :type origin: list of two float
    :param spacing: spacing of the grid
    :type spacing: list of two float
    :returns: The epipolar region as [xmin, ymin, xmax, ymax]
    :rtype: list of four float
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_conf = conf\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]
    minimum_disparity = preprocessing_output_conf[
        output_prepare.MINIMUM_DISPARITY_TAG]
    maximum_disparity = preprocessing_output_conf[
        output_prepare.MAXIMUM_DISPARITY_TAG]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    region_grid = np.array([[region[0],region[1]],
                            [region[2],region[1]],
                            [region[2],region[3]],
                            [region[0],region[3]]])

    epipolar_grid = tiling.grid(0, 0,
                         preprocessing_output_conf[
                             output_prepare.EPIPOLAR_SIZE_X_TAG],
                         preprocessing_output_conf[
                             output_prepare.EPIPOLAR_SIZE_Y_TAG],
                         step,
                         step)

    epi_grid_flat = epipolar_grid.reshape(-1, epipolar_grid.shape[-1])

    epipolar_grid_min, epipolar_grid_max = compute_epipolar_grid_min_max(
        epipolar_grid, epsg, conf,disp_min, disp_max)

    # Build Delaunay triangulations
    delaunay_min = Delaunay(epipolar_grid_min)
    delaunay_max = Delaunay(epipolar_grid_max)

    # Build kdtrees
    tree_min = cKDTree(epipolar_grid_min)
    tree_max = cKDTree(epipolar_grid_max)

    # Look-up terrain grid with Delaunay
    s_min = tsearch(delaunay_min, region_grid)
    s_max = tsearch(delaunay_max, region_grid)

    points_list = []
    # For each corner
    for i in range(0,4):
        # If we are inside triangulation of s_min
        if s_min[i] != -1:
            # Add points from surrounding triangle
            for point in epi_grid_flat[delaunay_min.simplices[s_min[i]]]:
                points_list.append(point)
        else:
            # else add nearest neighbor
            __, point_idx = tree_min.query(region_grid[i,:])
            points_list.append(epi_grid_flat[point_idx])
        # If we are inside triangulation of s_min
            if s_max[i] != -1:
                # Add points from surrounding triangle
                for point in epi_grid_flat[delaunay_max.simplices[s_max[i]]]:
                    points_list.append(point)
            else:
                # else add nearest neighbor
                __, point_nn_idx = tree_max.query(region_grid[i,:])
                points_list.append(epi_grid_flat[point_nn_idx])

    points_min = np.min(points_list, axis=0)
    points_max = np.max(points_list, axis=0)

    # Bouding region of corresponding cell
    epipolar_region_minx = points_min[0]
    epipolar_region_miny = points_min[1]
    epipolar_region_maxx = points_max[0]
    epipolar_region_maxy = points_max[1]

    # This mimics the previous code that was using
    # transform_terrain_region_to_epipolar
    epipolar_region = [
        epipolar_region_minx,
        epipolar_region_miny,
        epipolar_region_maxx,
        epipolar_region_maxy]
    return epipolar_region
