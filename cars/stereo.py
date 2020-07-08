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
This module contains functions related to stereo-rectification, disparity map estimation and triangulation
"""

# Standard imports
from typing import Dict, List, Tuple
import warnings
import os
import math
import logging
import pickle
from pkg_resources import iter_entry_points

# Third party imports
import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay, tsearch, cKDTree
import rasterio as rio
import xarray as xr
import otbApplication
from osgeo import osr
from dask import sizeof
import pandora
import pandora.marge
from pandora.constants import *

# Cars imports
from cars import pipelines
from cars import tiling
from cars import parameters as params
from cars import projection
from cars import utils
from cars.preprocessing import project_coordinates_on_line


# Register sizeof for xarray
@sizeof.sizeof.register_lazy("xarray")
def register_xarray():
    
    @sizeof.sizeof.register(xr.DataArray)
    def sizeof_xarray_dataarray(x):
        total_size = sizeof.sizeof(x.values)
        for cname, carray in x.coords.items():
            total_size += sizeof.sizeof(carray.values)
        total_size += sizeof.sizeof(x.attrs)
        return total_size
    
    @sizeof.sizeof.register(xr.Dataset)
    def sizeof_xarray_dataset(x):
        total_size = 0
        for vname, varray in x.data_vars.items():
            total_size += sizeof.sizeof(varray.values)
        for cname, carray in x.coords.items():
            total_size += sizeof.sizeof(carray)
        total_size += sizeof.sizeof(x.attrs)
        return total_size

# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def optimal_tile_size(
        disp_min,
        disp_max,
        otb_max_ram_hint=None,
        tile_size_rounding=50):
    """
    Compute optimal tile size according to estimated memory usage.
    Returned optimal tile size will be at least equal to tile_size_rounding.

    :param disp_min: Minimum disparity to explore
    :type disp_min: int
    :param disp_max: Maximum disparity to explore
    :type disp_max: int
    :param otb_max_ram_hint: amount of RAM allocated to OTB (if None, will try to read it from environment variable)
    :type otb_max_ram_hint: int
    :param tile_size_rounding: Optimal tile size will be aligned to multiples of tile_size_rounding
    :type tile_size_rounding: int
    :returns: Optimal tile size according to benchmarked memory usage
    :rtype: int
    """

    if otb_max_ram_hint is None:
        if "OTB_MAX_RAM_HINT" in os.environ:
            otb_max_ram_hint = os.environ["OTB_MAX_RAM_HINT"]
        else:
            raise ValueError(
                'otb_max_ram_hint is None and OTB_MAX_RAM_HINT envvar is not set')

    disp = disp_max - disp_min
    a = (0.02 * disp + 0.4)
    b = (0.008 * disp * disp + 2 * disp + 300)
    c = (0.9 * disp * disp - 250 * disp + 65000) - \
        1000 * (int(otb_max_ram_hint) / 4)
    delta = b * b - 4 * a * c

    tile_size = (-b + np.sqrt(delta)) / (2 * a)
    tile_size = tile_size_rounding * int(tile_size / tile_size_rounding)

    if tile_size <= 0:
        logging.warning(
            "Optimal tile size is null, forcing it to {} pixels".format(tile_size_rounding))
        tile_size = tile_size_rounding

    return tile_size


def resample_image(
        img,
        grid,
        largest_size,
        region=None,
        nodata=None,
        mask=None,
        band_coords=False,
        lowres_color=None):
    """
    Resample image according to grid and largest size.

    :param img: Path to the image to resample
    :type img: string
    :param grid: Path to the resampling grid
    :type grid: string
    :param largest_size: Size of full output image
    :type largest_size: list of two int
    :param region: A subset of the ouptut image to produce
    :type region: None (full output is produced) or array of four floats [xmin,ymin,xmax,ymax]
    :param nodata: Nodata value to use (both for input and output)
    :type nodata: None or float
    :param mask: Mask to resample as well
    :type mask: None or path to mask image
    :param band_coords: Force bands coordinate in output dataset
    :type band_coords: boolean
    :param lowres_color: Path to the multispectral image if p+xs fusion is needed
    :type lowres_color: string
    :rtype: xarray.Dataset with resampled image and mask
    """
    # Handle region is None
    if region is None:
        region = [0, 0, largest_size[0], largest_size[1]]
    else:
        region = [int(math.floor(region[0])),
                  int(math.floor(region[1])),
                  int(math.ceil(region[2])),
                  int(math.ceil(region[3]))]

    # Convert largest_size to int if needed
    largest_size = [int(x) for x in largest_size]

    # Build mask pipeline for img needed
    img_has_mask = nodata is not None or mask is not None
    mask_otb = None
    mask_pipeline = None
    if img_has_mask:
        mask_otb, mask_pipeline = pipelines.build_mask_pipeline(
            img, grid, nodata, mask, largest_size[0], largest_size[1], region)

    # Build bundletoperfectsensor (p+xs fusion) for images
    if lowres_color is not None:
        img, img_pxs_pipeline = pipelines.build_bundletoperfectsensor_pipeline(
            img, lowres_color)

    # Build resampling pipelines for images
    img_epi_otb, img_epi_pipeline = pipelines.build_image_resampling_pipeline(
        img, grid, largest_size[0], largest_size[1], region)

    # Retrieve data and build left dataset
    im = np.copy(
        img_epi_pipeline["extract_app"].GetVectorImageAsNumpyArray("out"))

    if img_has_mask:
        msk = np.copy(mask_pipeline["extract_app"].GetImageAsNumpyArray("out"))
    else:
        msk = None

    dataset = create_im_dataset(im, region, largest_size, band_coords, msk)

    return dataset


def create_im_dataset(im:np.ndarray, region:List[int], largest_size:List[int], band_coords:bool=False,
                      msk:np.ndarray=None) -> xr.Dataset:
    """
    Create image dataset as used in cars.

    :param im: image as a numpy array
    :param region: region as list [xmin ymin xmax ymax]
    :param largest_size: whole image size
    :param band_coords: set to true to add the coords 'band' to the dataset
    :param msk: image mask as a numpy array (default None)
    :return: The image dataset as used in cars
    """
    nb_bands = im.shape[-1]

    # Add band dimension if needed
    if band_coords or nb_bands > 1:
        bands = range(nb_bands)
        # Reorder dimensions in color dataset in order that the first dimension
        # is band.
        dataset = xr.Dataset({'im': (['band', 'row', 'col'],
                                     np.einsum('ijk->kij', im)
                                     )},
                             coords={'band': bands,
                                     'row': np.array(range(region[1], region[3])),
                                     'col': np.array(range(region[0], region[2]))
                                     })
    else:
        dataset = xr.Dataset({'im': (['row', 'col'], im[:, :, 0])},
                             coords={'row': np.array(range(region[1], region[3])),
                                     'col': np.array(range(region[0], region[2]))})

    if msk is not None:
        dataset['msk'] = xr.DataArray(
            msk.astype(np.int16), dims=['row', 'col'])

    dataset.attrs['full_epipolar_size'] = largest_size
    dataset.attrs['region'] = np.array(region)

    return dataset


def epipolar_rectify_images(
        configuration,
        region,
        margins):
    """
    This function will produce rectified images over a region.
    If espg is equal to None, geometry used is the epipolar geometry.
    This function returns a xarray Dataset containing images, masks and color.
    The parameters margin_x and margin_y are used to apply margins on the epipolar region.

    :param configuration: Configuration for stereo processing
    :type configuration: StereoConfiguration
    :param region: Array defining epipolar region as [xmin, ymin, xmax, ymax]
    :type region: list of four float
    :param margins: margins for the images
    :type: 2D (image, corner) DataArray, with the dimensions image = ['ref_margin', 'sec_margin'],
    corner = ['left','up', 'right', 'down']
    :return: Datasets containing:

    1. left image and mask,
    2. right image and mask,
    3. left color image or None

    :rtype: xarray.Dataset, xarray.Dataset, xarray.Dataset
    """
    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_configuration = configuration[
        params.preprocessing_section_tag][params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    nodata1 = input_configuration.get(params.nodata1_tag, None)
    nodata2 = input_configuration.get(params.nodata2_tag, None)
    mask1 = input_configuration.get(params.mask1_tag, None)
    mask2 = input_configuration.get(params.mask2_tag, None)
    color1 = input_configuration.get(params.color1_tag, None)

    grid1 = preprocessing_output_configuration[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_configuration[params.right_epipolar_grid_tag]

    epipolar_size_x = preprocessing_output_configuration[params.epipolar_size_x_tag]
    epipolar_size_y = preprocessing_output_configuration[params.epipolar_size_y_tag]

    epipolar_origin = [preprocessing_output_configuration[params.epipolar_origin_x_tag],
                       preprocessing_output_configuration[params.epipolar_origin_y_tag]]

    epipolar_spacing = [preprocessing_output_configuration[params.epipolar_spacing_x_tag],
                        preprocessing_output_configuration[params.epipolar_spacing_y_tag]]

    # Force region to be float
    region = [int(x) for x in region]
    # Apply margins to left image
    left_region = region.copy()
    left_margins = margins.loc[dict(image='ref_margin')].values
    left_roi = tiling.crop(
        left_region, [0, 0, epipolar_size_x, epipolar_size_y])
    left_region = tiling.crop(tiling.pad(left_region, left_margins), [
                              0, 0, epipolar_size_x, epipolar_size_y])

    left_margins = margins.loc[dict(image='ref_margin')].values
    # Get actual margin taking cropping into account
    left_margins[0] = left_region[0] - left_roi[0]
    left_margins[1] = left_region[1] - left_roi[1]
    left_margins[2] = left_region[2] - left_roi[2]
    left_margins[3] = left_region[3] - left_roi[3]

    # Apply margins to right image
    right_region = region.copy()
    right_margins = margins.loc[dict(image='sec_margin')].values
    right_roi = tiling.crop(
        right_region, [0, 0, epipolar_size_x, epipolar_size_y])
    right_region = tiling.crop(tiling.pad(right_region, right_margins), [
                               0, 0, epipolar_size_x, epipolar_size_y])

    # Get actual margin taking cropping into account
    right_margins[0] = right_region[0] - right_roi[0]
    right_margins[1] = right_region[1] - right_roi[1]
    right_margins[2] = right_region[2] - right_roi[2]
    right_margins[3] = right_region[3] - right_roi[3]

    # Resample left image
    left_dataset = resample_image(img1,
                                  grid1,
                                  [epipolar_size_x,
                                   epipolar_size_y],
                                  region=left_region,
                                  nodata=nodata1,
                                  mask=mask1)

    # Update attributes
    left_dataset.attrs['roi'] = np.array(left_roi)
    left_dataset.attrs['roi_with_margins'] = np.array(left_region)
    # Remove region key as it duplicates roi_with_margins key
    left_dataset.attrs.pop("region", None)
    left_dataset.attrs['margins'] = np.array(left_margins)
    left_dataset.attrs['disp_min'] = margins.attrs['disp_min']
    left_dataset.attrs['disp_max'] = margins.attrs['disp_max']

    # Resample right image
    right_dataset = resample_image(img2,
                                   grid2,
                                   [epipolar_size_x,
                                    epipolar_size_y],
                                   region=left_region,
                                   nodata=nodata2,
                                   mask=mask2)

    # Update attributes
    right_dataset.attrs['roi'] = np.array(right_roi)
    right_dataset.attrs['roi_with_margins'] = np.array(right_region)
    # Remove region key as it duplicates roi_with_margins key
    right_dataset.attrs.pop("region", None)
    right_dataset.attrs['margins'] = np.array(right_margins)
    right_dataset.attrs['disp_min'] = margins.attrs['disp_min']
    right_dataset.attrs['disp_max'] = margins.attrs['disp_max']

    # Build resampling pipeline for color image, and build datasets
    if color1 is None:
        color1 = img1

    # Ensure that region is cropped to largest
    left_roi = tiling.crop(left_roi, [0, 0, epipolar_size_x, epipolar_size_y])

    # Check if p+xs fusion is not needed (color1 and img1 have the same size)
    if utils.rasterio_get_size(color1) == utils.rasterio_get_size(img1):
        left_color_dataset = resample_image(
            color1, grid1, [
                epipolar_size_x, epipolar_size_y], region=left_roi, band_coords=True)
    else:
        left_color_dataset = resample_image(img1,
                                            grid1,
                                            [epipolar_size_x,
                                             epipolar_size_y],
                                            region=left_roi,
                                            band_coords=True,
                                            lowres_color=color1)

    # Remove region key as it duplicates coordinates span
    left_color_dataset.attrs.pop("region", None)

    return left_dataset, right_dataset, left_color_dataset


def compute_disparity(left_dataset, 
                      right_dataset, 
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
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min: Minimum disparity (if None, value is taken from left dataset)
    :type disp_min: int
    :param disp_max: Maximum disparity (if None, value is taken from left dataset)
    :type disp_max: int
    :param use_sec_disp: Boolean activating the use of the secondary disparity map
    :type use_sec_disp: bool
    :param verbose: Activation of verbose mode
    :type verbose: Boolean
    :returns: Dictionary of disparity dataset. Keys are:
        * 'ref' for the left to right disparity map
        * 'sec' for the right to left disparity map if it is computed by Pandora
    """

    # Check disp min and max bounds with respect to margin used for
    # rectification
    if disp_min is None:
        disp_min = left_dataset.attrs['disp_min']
    else:
        if disp_min < left_dataset.attrs['disp_min']:
            raise ValueError(
                "disp_min ({}) is lower than disp_min used to determine margin during rectification ({})".format(
                    disp_min, left_dataset['disp_min']))

    if disp_max is None:
        disp_max = left_dataset.attrs['disp_max']
    else:
        if disp_max > left_dataset.attrs['disp_max']:
            raise ValueError(
                "disp_max ({}) is greater than disp_max used to determine margin during rectification ({})".format(
                    disp_max, left_dataset['disp_max']))

    # Load pandora plugin
    for entry_point in iter_entry_points(group='pandora.plugin'):
        entry_point.load()

    # Run the Pandora pipeline
    ref, sec = pandora.run(left_dataset,
                           right_dataset,
                           int(disp_min),
                           int(disp_max),
                           corr_cfg)

    disp = dict()
    disp['ref'] = create_disp_dataset(ref, left_dataset, verbose=verbose)

    if bool(sec.dims) and use_sec_disp:
        # for the secondary disparity map, the reference is the right dataset and the secondary image is the left one
        logging.info('Secondary disparity map will be used to densify the points cloud')
        disp['sec'] = create_disp_dataset(sec, right_dataset,
                                          sec_dataset=left_dataset, check_roi_in_sec=True, verbose=verbose)

    return disp


def create_disp_dataset(disp: xr.Dataset, ref_dataset: xr.Dataset, sec_dataset:xr.Dataset=None,
                        check_roi_in_sec:bool=False, verbose:bool=False) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param sec_dataset: secondary dataset for the considered disparity map
    (needed only if the check_roi_in_sec is set to True)
    :param check_roi_in_sec: option to invalid the values of the disparity which end up outside the secondary image roi
    :param verbose: verbose activation status
    :return: disparity dataset as used in cars
    """
    # Retrieve disparity values
    disp_map = disp['disparity_map'].values

    # retrieve masks
    masks = get_masks_from_pandora(disp, verbose)
    if check_roi_in_sec:
        masks['inside_sec_roi'] = create_inside_sec_roi_mask(disp_map, masks['mask'], sec_dataset)
        masks['mask'][masks['inside_sec_roi'] == 0] = 0

    # Crop disparity to ROI
    if not check_roi_in_sec:
        ref_roi = [int(-ref_dataset.attrs['margins'][0]),
                   int(-ref_dataset.attrs['margins'][1]),
                   int(ref_dataset.dims['col'] - ref_dataset.attrs['margins'][2]),
                   int(ref_dataset.dims['row'] - ref_dataset.attrs['margins'][3])]
        disp_map = disp_map[ref_roi[1]:ref_roi[3], ref_roi[0]:ref_roi[2]]
        for key in masks:
            masks[key] = masks[key][ref_roi[1]:ref_roi[3], ref_roi[0]:ref_roi[2]]

    # Fill disparity array with 0 value for invalid points
    disp_map[masks['mask'] == 0] = 0

    # Build output dataset
    if not check_roi_in_sec:
        row = np.array(range(ref_dataset.attrs['roi'][1], ref_dataset.attrs['roi'][3]))
        col = np.array(range(ref_dataset.attrs['roi'][0], ref_dataset.attrs['roi'][2]))
    else:
        row = np.array(range(ref_dataset.attrs['roi_with_margins'][1], ref_dataset.attrs['roi_with_margins'][3]))
        col = np.array(range(ref_dataset.attrs['roi_with_margins'][0], ref_dataset.attrs['roi_with_margins'][2]))

    disp_ds = xr.Dataset({'disp': (['row', 'col'], np.copy(disp_map)),
                          'msk': (['row', 'col'], np.copy(masks['mask']))},
                         coords={'row': row, 'col': col})
    if verbose:
        disp_ds['msk_invalid_ref'] = xr.DataArray(np.copy(masks['invalid_ref']), dims=['row', 'col'])
        disp_ds['msk_invalid_sec'] = xr.DataArray(np.copy(masks['invalid_sec']), dims=['row', 'col'])
        disp_ds['msk_masked_ref'] = xr.DataArray(np.copy(masks['masked_ref']), dims=['row', 'col'])
        disp_ds['msk_masked_sec'] = xr.DataArray(np.copy(masks['masked_sec']), dims=['row', 'col'])
        disp_ds['msk_occlusion'] = xr.DataArray(np.copy(masks['occlusion']), dims=['row', 'col'])
        disp_ds['msk_false_match'] = xr.DataArray(np.copy(masks['false_match']), dims=['row', 'col'])
        if check_roi_in_sec:
            disp_ds['msk_inside_sec_roi'] = xr.DataArray(np.copy(masks['inside_sec_roi']), dims=['row', 'col'])

    disp_ds.attrs = disp.attrs.copy()
    disp_ds.attrs['roi'] = ref_dataset.attrs['roi']
    if check_roi_in_sec:
        disp_ds.attrs['roi_with_margins'] = ref_dataset.attrs['roi_with_margins']
    disp_ds.attrs['full_epipolar_size'] = ref_dataset.attrs['full_epipolar_size']

    return disp_ds


def create_inside_sec_roi_mask(disp: np.ndarray, disp_msk: np.ndarray, sec_dataset: xr.Dataset) -> np.ndarray:
    """
    Create mask of disp values which are in the secondary image roi (255 if in the roi, otherwise 0)

    :param disp: disparity map
    :param disp_msk: disparity map valid values mask
    :param sec_dataset: secondary image dataset
    :return: mask of valid pixels that are in the secondary image roi
    """
    # create mask of secondary image roi
    sec_up_margin = abs(sec_dataset.attrs['margins'][1])
    sec_bottom_margin = abs(sec_dataset.attrs['margins'][3])
    sec_right_margin = abs(sec_dataset.attrs['margins'][2])
    sec_left_margin = abs(sec_dataset.attrs['margins'][0])

    # valid pixels that are inside the secondary image roi
    in_sec_roi_msk = np.zeros(disp.shape, dtype=np.int16)
    for i in range(0, disp.shape[0]):
        for j in range(0, disp.shape[1]):

            # if the pixel is valid
            if disp_msk[i,j] == 255:
                idx = float(j) + disp[i,j]

                # if the pixel is in the roi in the secondary image
                if sec_left_margin <= idx < disp.shape[1] - sec_right_margin\
                        and sec_up_margin <= i < disp.shape[0] - sec_bottom_margin:
                    in_sec_roi_msk[i,j] = 255

    return in_sec_roi_msk


def get_masks_from_pandora(disp:xr.Dataset, verbose: bool) -> Dict[str, np.ndarray]:
    """
    Get masks dictionary from the disparity map in output of pandora.

    :param disp: disparity map (pandora output)
    :param verbose: verbose activation status
    :return: masks dictionary
    """
    masks = dict()

    # Retrieve validity mask from pandora
    # Invalid pixels in validity mask are:
    #  * Edge of the reference image or nodata in reference image (bit 0)
    #  * Disparity interval to explore is missing or nodata in the secondary image (bit 1)
    #  * Pixel is masked on the mask of the reference image (bit 6)
    #  * Disparity to explore is masked on the mask of the secondary image (bit 7)
    #  * Pixel located in an occlusion region (bit 8)
    #  * Fake match (bit 9)
    validity_mask_cropped = disp['validity_mask'].values
    # Mask intialization to false (all is invalid)
    msk = np.full(validity_mask_cropped.shape, False)
    # Identify valid points
    msk[np.where((validity_mask_cropped & PANDORA_MSK_PIXEL_INVALID) == 0)] = True

    masks['mask'] = msk

    # With verbose, produce one mask for each invalid flag in
    if verbose:
        # Bit 9: False match bit 9
        msk_false_match = np.full(validity_mask_cropped.shape, False)
        msk_false_match[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_MISMATCH) == 0)] = True
        # Bit 8: Occlusion
        msk_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_occlusion[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_OCCLUSION) == 0)] = True
        # Bit 7: Masked in secondary image
        msk_masked_sec = np.full(validity_mask_cropped.shape, False)
        msk_masked_sec[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC) == 0)] = True
        # Bit 6: Masked in reference image
        msk_masked_ref = np.full(validity_mask_cropped.shape, False)
        msk_masked_ref[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF) == 0)] = True
        # Bit 5: Filled false match
        msk_filled_false_match = np.full(validity_mask_cropped.shape, False)
        msk_filled_false_match[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_FILLED_MISMATCH) == 0)] = True
        # Bit 4: Filled occlusion
        msk_filled_occlusion = np.full(validity_mask_cropped.shape, False)
        msk_filled_occlusion[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_FILLED_OCCLUSION) == 0)] = True
        # Bit 3: Computation stopped during pixelic step, under pixelic interpolation never ended
        msk_stopped_interp = np.full(validity_mask_cropped.shape, False)
        msk_stopped_interp[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION) == 0)] = True
        # Bit 2: Disparity range to explore is incomplete (borders reached in secondary image)
        msk_incomplete_disp = np.full(validity_mask_cropped.shape, False)
        msk_incomplete_disp[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE) == 0)] = True
        # Bit 1: Invalid in secondary image
        msk_invalid_sec = np.full(validity_mask_cropped.shape, False)
        msk_invalid_sec[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING) == 0)] = True
        # Bit 0: Invalid in reference image 
        msk_invalid_ref = np.full(validity_mask_cropped.shape, False)
        msk_invalid_ref[np.where(
            (validity_mask_cropped & PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER) == 0)] = True

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
        final_msk[masks[key] == False] = 0
        masks[key] = final_msk

    return masks

def estimate_color_from_disparity(disp_ref_to_sec: xr.Dataset, sec_ds: xr.Dataset,
                                  sec_color: xr.Dataset) -> xr.Dataset:
    """
    Estimate color image of reference from the disparity map and the secondary color image.

    :param disp_ref_to_sec: disparity map
    :param sec_ds: secondary image dataset
    :param sec_color: secondary color image
    :return: interpolated reference color image dataset
    """
    # retrieve numpy arrays from input datasets
    disp_msk = disp_ref_to_sec['msk'].values
    im_color = sec_color['im'].values
    if 'msk' in sec_color.variables.keys():
        im_msk = sec_color['msk'].values

    # retrieve image sizes
    nb_bands, nb_row, nb_col = im_color.shape
    nb_disp_row, nb_disp_col = disp_ref_to_sec['disp'].values.shape

    sec_up_margin = abs(sec_ds.attrs['margins'][1])
    sec_left_margin = abs(sec_ds.attrs['margins'][0])

    # instantiate final image
    final_interp_color = np.zeros((nb_disp_row, nb_disp_col, nb_bands), dtype=np.float)

    # construct secondary color image pixels positions
    clr_x_positions, clr_y_positions = np.meshgrid(np.linspace(0, nb_col-1, nb_col), np.linspace(0, nb_row-1, nb_row))
    clr_xy_positions = np.concatenate([clr_x_positions.reshape(1, nb_row*nb_col).transpose(),
                                       clr_y_positions.reshape(1, nb_row*nb_col).transpose()], axis=1)

    # construct the positions for which the interpolation has to be done
    interpolated_points = np.zeros((nb_disp_row * nb_disp_col, 2), dtype=np.float)
    for i in range(0, disp_ref_to_sec['disp'].values.shape[0]):
        for j in range(0, disp_ref_to_sec['disp'].values.shape[1]):

            # if the pixel is valid,
            # else the position is left to (0,0) and the final image pixel value will be set to np.nan
            if disp_msk[i, j] == 255:
                idx = j + disp_ref_to_sec['disp'].values[i, j]
                interpolated_points[i * nb_disp_col + j, 0] = idx - sec_left_margin
                interpolated_points[i * nb_disp_col + j, 1] = i - sec_up_margin

    # construct final image mask
    final_msk = disp_msk
    if 'msk' in sec_color.variables.keys():
        # interpolate the color image mask to the new image referential (nearest neighbor interpolation)
        msk_values = im_msk.reshape(nb_row * nb_col, 1)
        interp_msk_value = interpolate.griddata(clr_xy_positions, msk_values, interpolated_points, method='nearest')
        interp_msk = interp_msk_value.reshape(nb_disp_row, nb_disp_col)

        # remove from the final mask all values which are interpolated from non valid values (strictly non equal to 255)
        final_msk[interp_msk == 0] = 0

    # interpolate each band of the color image
    for band in range(nb_bands):
        # get band values
        band_im = im_color[band,:,:]
        clr_values = band_im.reshape(nb_row*nb_col, 1)

        # interpolate values
        interp_values = interpolate.griddata(clr_xy_positions, clr_values, interpolated_points, method='nearest')
        final_interp_color[:, :, band] = interp_values.reshape(nb_disp_row, nb_disp_col)

        # apply final mask
        final_interp_color[:, :, band][final_msk != 255] = np.nan

    # create interpolated color image dataset
    region = list(disp_ref_to_sec.attrs['roi_with_margins'])
    largest_size = disp_ref_to_sec.attrs['full_epipolar_size']

    interp_clr_ds = create_im_dataset(final_interp_color, region, largest_size, band_coords=True, msk=None)
    interp_clr_ds.attrs['roi'] = disp_ref_to_sec.attrs['roi']
    interp_clr_ds.attrs['roi_with_margins'] = disp_ref_to_sec.attrs['roi_with_margins']

    return interp_clr_ds

def get_elevation_range_from_metadata(img:str, default_min:float=0, default_max:float=300) -> (float, float):
    """
    This function will try to derive a valid RPC altitude range from img metadata.
    It will first try to read metadata with gdal.
    If it fails, it will look for values in the geom file if it exists
    If it fails, it will return the default range

    :param img: Path to the img for which the elevation range is required
    :param default_min: Default minimum value to return if everything else fails
    :param default_max: Default minimum value to return if everything else fails
    :returns: (elev_min, elev_max) float tuple
    """
    # First, try to get range from gdal metadata
    with rio.open(img) as ds:
        gdal_height_offset = ds.get_tag_item('HEIGHT_OFF','RPC')
        gdal_height_scale  = ds.get_tag_item('HEIGHT_SCALE','RPC')
                
        if gdal_height_scale is not None and gdal_height_offset is not None:
            if isinstance(gdal_height_offset, str):
                gdal_height_offset = float(gdal_height_offset)
            if isinstance(gdal_height_scale, str):
                gdal_height_scale = float(gdal_height_scale)
            return (float(gdal_height_offset-gdal_height_scale/2.),
                    float(gdal_height_offset+gdal_height_scale/2.))

    # If we are still here, try to get range from OTB/OSSIM geom file if exists
    geom_file, _ = os.path.splitext(img)
    geom_file = geom_file+".geom"
    
    # If geom file exists
    if os.path.isfile(geom_file):
        with open(geom_file,'r') as f:
            geom_height_offset = None
            geom_height_scale = None

            for line in f:
                if line.startswith("height_off:"):
                    geom_height_offset = float(line.split(sep=':')[1])

                if line.startswith("height_scale:"):
                    geom_height_scale = float(line.split(sep=':')[1])
            if geom_height_offset is not None and geom_height_scale is not None:
                return (float(geom_height_offset-geom_height_scale/2),
                        float(geom_height_offset+geom_height_scale/2))
    
    # If we are still here, return a default range:
    return (default_min, default_max)

def triangulate(configuration, disp_ref: xr.Dataset, disp_sec:xr.Dataset=None, left_dataset: xr.Dataset=None,
                snap_to_img1:bool = False, align:bool = False) -> Dict[str, xr.Dataset]:
    """
    This function will perform triangulation from a disparity map

    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param disp_ref: left to right disparity map dataset
    :param disp_sec: if available, the right to left disparity map dataset
    :param left_dataset:
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so as to cross those of img1
    :param snap_to_img1: bool
    :param align: If True, apply correction to point after triangulation to align with lowres DEM (if available. If not, no correction is applied)
    :param align: bool
    :returns: point_cloud as a dictionary of dataset containing:

        * Array with shape (roi_size_x,roi_size_y,3),
          with last dimension corresponding to longitude, lattitude and elevation
        * Array with shape (roi_size_x,roi_size_y) with output mask
        * Array for color (optional): only if color1 is not None

    The dictionary keys are :
        * 'ref' to retrieve the dataset built from the left to right disparity map
        * 'sec' to retrieve the dataset built from the right to left disparity map (if provided in input)
    """

    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_configuration = configuration[
        params.preprocessing_section_tag][params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    grid1 = preprocessing_output_configuration[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_configuration[params.right_epipolar_grid_tag]
    if snap_to_img1:
        grid2 = preprocessing_output_configuration[params.right_epipolar_uncorrected_grid_tag]

    point_clouds = dict()
    point_clouds['ref'] = compute_points_cloud(disp_ref, img1, img2, grid1, grid2, roi_key='roi',
                                               left_dataset=left_dataset)
    if disp_sec is not None:
        point_clouds['sec'] = compute_points_cloud(disp_sec, img2, img1, grid2, grid1, roi_key='roi_with_margins')
        
    # Handle alignment with lowres DEM
    if align and params.lowres_dem_splines_fit_tag in preprocessing_output_configuration:        

        # Read splines file
        splines_file = preprocessing_output_configuration[params.lowres_dem_splines_fit_tag]
        splines_coefs = None
        with open(splines_file,'rb') as f:
            splines_coefs = pickle.load(f)

        # Read time direction line parameters
        time_direction_origin = [preprocessing_output_configuration[params.time_direction_line_origin_x_tag],
                                 preprocessing_output_configuration[params.time_direction_line_origin_y_tag]]
        time_direction_vector = [preprocessing_output_configuration[params.time_direction_line_vector_x_tag],
                                 preprocessing_output_configuration[params.time_direction_line_vector_y_tag]]

        disp_to_alt_ratio = preprocessing_output_configuration[params.disp_to_alt_ratio_tag]

        # Interpolate correction
        point_cloud_z_correction = splines_coefs(project_coordinates_on_line(point_clouds['ref'].x.values.ravel(), 
                                                                                 point_clouds['ref'].y.values.ravel(), 
                                                                                 time_direction_origin, 
                                                                                 time_direction_vector))
        point_cloud_z_correction = np.reshape(point_cloud_z_correction, point_clouds['ref'].x.shape)

        # Convert to disparity correction
        point_cloud_disp_correction = point_cloud_z_correction/disp_to_alt_ratio

        # Correct disparity
        disp_ref['disp'] = disp_ref['disp'] - point_cloud_disp_correction

        # Triangulate again
        point_clouds['ref'] = compute_points_cloud(disp_ref, img1, img2, grid1, grid2, roi_key='roi')
        values_list = [key for key, _ in point_clouds['ref'].items()]
        print('in triangulate', values_list)
        # TODO handle sec case
        if disp_sec is not None:
            # Interpolate correction
            point_cloud_z_correction = splines_coefs(project_coordinates_on_line(point_clouds['sec'].x.values.ravel(), 
                                                                                 point_clouds['sec'].y.values.ravel(), 
                                                                                 time_direction_origin, 
                                                                                 time_direction_vector))
            point_cloud_z_correction = np.reshape(point_cloud_z_correction, point_clouds['sec'].x.shape)

            # Convert to disparity correction
            point_cloud_disp_correction = point_cloud_z_correction/disp_to_alt_ratio

            # Correct disparity
            disp_sec['disp'] = disp_sec['disp'] + point_cloud_disp_correction

            # Triangulate again
            point_clouds['sec'] = compute_points_cloud(disp_sec, img2, img1, grid2, grid1, roi_key='roi_with_margins')

    return point_clouds


def compute_points_cloud(data: xr.Dataset, img1:xr.Dataset, img2: xr.Dataset,
                         grid1:str, grid2:str, roi_key:str, left_dataset: xr.Dataset=None) -> xr.Dataset:
    """
    Compute points cloud

    :param data: The reference to secondary disparity map dataset
    :param img1: reference image dataset
    :param img2: secondary image dataset
    :param grid1: path to the reference image grid file
    :param grid2: path to the secondary image grid file
    :param roi_key: roi of the disparity map key
    ('roi' if cropped while calling create_disp_dataset, otherwise 'roi_with_margins')
    :param left_dataset:
    :return: the points cloud dataset
    """
    disp = pipelines.encode_to_otb(
        data['disp'].values,
        data.attrs['full_epipolar_size'],
        data.attrs[roi_key])
    msk = pipelines.encode_to_otb(
        data['msk'].values,
        data.attrs['full_epipolar_size'],
        data.attrs[roi_key])

    # Retrieve elevation range from imgs
    (min_elev1, max_elev1) = get_elevation_range_from_metadata(img1)
    (min_elev2, max_elev2) = get_elevation_range_from_metadata(img2)

    # Build triangulation app
    triangulation_app = otbApplication.Registry.CreateApplication(
        "EpipolarTriangulation")

    triangulation_app.SetParameterString("mode","disp");
    triangulation_app.ImportImage(
        "mode.disp.indisp", disp)

    triangulation_app.SetParameterString("leftgrid", grid1)
    triangulation_app.SetParameterString("rightgrid", grid2)
    triangulation_app.SetParameterString("leftimage", img1)
    triangulation_app.SetParameterString("rightimage", img2)
    triangulation_app.SetParameterFloat("leftminelev",min_elev1)
    triangulation_app.SetParameterFloat("leftmaxelev",max_elev1)
    triangulation_app.SetParameterFloat("rightminelev",min_elev2)
    triangulation_app.SetParameterFloat("rightmaxelev",max_elev2)

    triangulation_app.Execute()

    llh = np.copy(triangulation_app.GetVectorImageAsNumpyArray("out"))

    row = np.array(range(data.attrs[roi_key][1], data.attrs[roi_key][3]))
    col = np.array(range(data.attrs[roi_key][0], data.attrs[roi_key][2]))

    values = {
        'x': (['row', 'col'], llh[:, :, 0]),  # longitudes
        'y': (['row', 'col'], llh[:, :, 1]),  # latitudes
        'z': (['row', 'col'], llh[:, :, 2]),
        'msk': (['row', 'col'], data['msk'].values)
    }

    if left_dataset is not None:
        values_list = [key for key, _ in left_dataset.items()]
        print(values_list)
        if 'msk' in values_list:
            print('laaaaaaaaaaaaaaaaaaaa')
            ref_roi = [int(-left_dataset.attrs['margins'][0]),
                       int(-left_dataset.attrs['margins'][1]),
                       int(left_dataset.dims['col'] - left_dataset.attrs['margins'][2]),
                       int(left_dataset.dims['row'] - left_dataset.attrs['margins'][3])]
            left_msk = left_dataset.msk.values[ref_roi[1]:ref_roi[3], ref_roi[0]:ref_roi[2]]
            values['left_msk'] = (['row', 'col'], left_msk)

    point_cloud = xr.Dataset(values,
                             coords={'row': row, 'col': col})

    point_cloud.attrs['roi'] = data.attrs['roi']
    if roi_key == 'roi_with_margins':
        point_cloud.attrs['roi_with_margins'] = data.attrs['roi_with_margins']
    point_cloud.attrs['full_epipolar_size'] = data.attrs['full_epipolar_size']
    point_cloud.attrs['epsg'] = int(4326)

    return point_cloud


def triangulate_matches(configuration, matches, snap_to_img1 = False):
    """
    This function will perform triangulation from sift matches

    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param matches: numpy.array of matches of shape (nb_matches, 4)
    :type data: numpy.ndarray
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so as to cross those of img1
    :param snap_to_img1: bool
    :returns: point_cloud as a dataset containing:

        * Array with shape (nb_matches,1,3),
          with last dimension corresponding to longitude, lattitude and elevation
        * Array with shape (nb_matches,1) with output mask

    :rtype: xarray.Dataset
    """

    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_configuration = configuration[params.preprocessing_section_tag][params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    grid1 = preprocessing_output_configuration[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_configuration[params.right_epipolar_grid_tag]
    if snap_to_img1:
        grid2 = preprocessing_output_configuration[params.right_epipolar_uncorrected_grid_tag]

    # Retrieve elevation range from imgs
    (min_elev1, max_elev1) = get_elevation_range_from_metadata(img1)
    (min_elev2, max_elev2) = get_elevation_range_from_metadata(img2)

    # Build triangulation app
    triangulation_app = otbApplication.Registry.CreateApplication(
        "EpipolarTriangulation")

    triangulation_app.SetParameterString("mode","sift")
    triangulation_app.SetImageFromNumpyArray("mode.sift.inmatches",matches)

    triangulation_app.SetParameterString("leftgrid", grid1)
    triangulation_app.SetParameterString("rightgrid", grid2)
    triangulation_app.SetParameterString("leftimage", img1)
    triangulation_app.SetParameterString("rightimage", img2)
    triangulation_app.SetParameterFloat("leftminelev",min_elev1)
    triangulation_app.SetParameterFloat("leftmaxelev",max_elev1)
    triangulation_app.SetParameterFloat("rightminelev",min_elev2)
    triangulation_app.SetParameterFloat("rightmaxelev",max_elev2)

    triangulation_app.Execute()

    llh = np.copy(triangulation_app.GetVectorImageAsNumpyArray("out"))

    row = np.array(range(llh.shape[0]))
    col = np.array([0])

    msk = np.full(llh.shape[0:2],255, dtype=np.uint8)

    point_cloud = xr.Dataset({'x': (['row', 'col'], llh[:,:,0]),
                              'y': (['row', 'col'], llh[:,:,1]),
                              'z': (['row', 'col'], llh[:,:,2]),
                              'msk' : (['row', 'col'], msk)},
                             coords={'row':row,'col':col})
    point_cloud.attrs['epsg'] = int(4326)

    return point_cloud


def images_pair_to_3d_points(configuration,
                             region,
                             corr_cfg,
                             epsg=None,
                             disp_min=None,
                             disp_max=None,
                             out_epsg=None,
                             geoid_data=None,
                             use_sec_disp=False,
                             snap_to_img1=False,
                             align=False) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    # Retrieve disp min and disp max if needed
    """
    This function will produce a 3D points cloud as an xarray.Dataset from the given stereo configuration (from both
    left to right disparity map and right to left disparity map if the latter is computed by Pandora).
    Clouds will be produced over the region with the specified EPSG, using disp_min and disp_max
    :param configuration: Configuration for stereo processing
    :type StereoConfiguration
    :param region: Array defining region.

    * For espg region as [lat_min, lon_min, lat_max, lon_max]
    * For epipolar region as [xmin, ymin, xmax, ymax]

    :type region: numpy.array
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param epsg: EPSG code for the region, if None then epipolar geometry is considered
    :type epsg: int
    :param disp_min: Minimum disparity value
    :type disp_min: int
    :param disp_max: Maximum disparity value
    :type disp_max: int
    :param geoid_data: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_data: xarray.Dataset
    :param use_sec_disp: Boolean activating the use of the secondary disparity map
    :type use_sec_disp: bool
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so as to cross those of img1
    :param snap_to_img1: bool
    :param align: If True, apply correction to point after triangulation to align with lowres DEM (if available. If not, no correction is applied)
    :param align: bool
    :returns: Dictionary of tuple. The tuple are constructed with the dataset containing the 3D points +
    A dataset containing color of left image, or None

    The dictionary keys are :
        * 'ref' to retrieve the dataset built from the left to right disparity map
        * 'sec' to retrieve the dataset built from the right to left disparity map (if computed in Pandora)
    """


    # Retrieve disp min and disp max if needed
    preprocessing_output_configuration = configuration[
        params.preprocessing_section_tag][params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_configuration[params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_configuration[params.maximum_disparity_tag]

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
        region = transform_terrain_region_to_epipolar(region, conf, epsg,  disp_min, disp_max)

    # Rectify images
    left, right, color = epipolar_rectify_images(configuration,
                                                 region,
                                                 margins)
    # Compute disparity
    disp = compute_disparity(left, right, corr_cfg, disp_min, disp_max, use_sec_disp=use_sec_disp)

    colors = dict()
    colors['ref'] = color
    if 'sec' in disp:
        # compute right color image from right-left disparity map
        colors['sec'] = estimate_color_from_disparity(disp['sec'], left, color)

    # Triangulate
    if 'sec' in disp:
        points = triangulate(configuration, disp['ref'], disp['sec'], snap_to_img1 = snap_to_img1, align=align,
                             left_dataset=left)
    else:
        points = triangulate(configuration, disp['ref'], snap_to_img1=snap_to_img1, align=align)

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key in points:
            points[key] = geoid_offset(points[key], geoid_data)

    if out_epsg is not None:
        for key in points:
            points[key] = projection.points_cloud_conversion_dataset(points[key], out_epsg)

    return points, colors


def geoid_offset(points, geoid):
    """
    Compute the point cloud height offset from geoid.

    :param points: point cloud data in lat/lon/alt WGS84 (EPSG 4326)
        coordinates.
    :type points: xarray.Dataset
    :param geoid: geoid elevation data.
    :type geoid: xarray.Dataset
    :return: the same point cloud but using geoid as altimetric reference.
    :rtype: xarray.Dataset
    """

    # deep copy the given point cloud that will be used as output
    out_pc = points.copy(deep=True)

    # currently assumes that the OTB EGM96 geoid will be used with longitude
    # ranging from 0 to 360, so we must unwrap longitudes to this range.
    longitudes = np.copy(out_pc.x.values)
    longitudes[longitudes < 0] += 360

    # perform interpolation using point cloud coordinates.
    if not geoid.lat_min <= out_pc.y.min() <= out_pc.y.max() <= geoid.lat_max \
            and geoid.lon_min <= np.min(longitudes) <= np.max(longitudes) <= \
            geoid.lat_max:
        raise RuntimeError('Geoid does not fully cover the area spanned by '
                           'the point cloud.')

    # interpolate data
    ref_interp = geoid.interp({'lat': out_pc.y,
                               'lon':xr.DataArray(longitudes,
                                                  dims=('row', 'col'))})
    # offset using geoid height
    out_pc['z'] = points.z - ref_interp.hgt

    # remove coordinates lat & lon added by the interpolation
    out_pc = out_pc.reset_coords(['lat', 'lon'], drop=True)

    return out_pc


def compute_epipolar_grid_min_max(grid, epsg, conf, disp_min = None, disp_max = None):
    """
    Compute ground terrain location of epipolar grids at disp_min and disp_max

    :param grid: The epipolar grid to project
    :type grid: np.ndarray of shape (N,M,2)
    :param epsg: EPSG code of the terrain projection
    :type epsg: Int
    :param conf: Configuration dictionnary from prepare step
    :type conf: Dict
    :param disp_min: Minimum disparity (if None, read from configuration dictionnary)
    :type disp_min: Float or None
    :param disp_max: Maximum disparity (if None, read from configuration dictionnary)
    :type disp_max: Float or None
    :returns: a tuple of location grid at disp_min and disp_max
    :rtype: Tuple(np.ndarray, np.ndarray) same shape as grid param
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_configuration = conf[
        params.preprocessing_section_tag][params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_configuration[params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_configuration[params.maximum_disparity_tag]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Generate disp_min and disp_max matches
    matches_min = np.stack((grid[:,:,0].flatten(), grid[:,:,1].flatten(), grid[:,:,0].flatten()+disp_min, grid[:,:,1].flatten()), axis=1)
    matches_max = np.stack((grid[:,:,0].flatten(), grid[:,:,1].flatten(), grid[:,:,0].flatten()+disp_max, grid[:,:,1].flatten()), axis=1)

    # Generate corresponding points clouds
    pc_min = triangulate_matches(conf, matches_min)
    pc_max = triangulate_matches(conf, matches_max)

    # Convert to correct EPSG
    pc_min_epsg = projection.points_cloud_conversion_dataset(pc_min, epsg)
    pc_max_epsg = projection.points_cloud_conversion_dataset(pc_max, epsg)

    # Form grid_min and grid_max
    grid_min = np.concatenate((pc_min_epsg.x.values,pc_min_epsg.y.values), axis=1)
    grid_max = np.concatenate((pc_max_epsg.x.values,pc_max_epsg.y.values), axis=1)

    return grid_min, grid_max


def transform_terrain_region_to_epipolar(
        region, conf, epsg = 4326, disp_min = None, disp_max = None, step = 100):
    """
    Transform terrain region to epipolar region according to ground_positions

    :param region: The terrain region to transform to epipolar region ([lat_min, lon_min, lat_max, lon_max])
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
    preprocessing_output_config = conf[
        params.preprocessing_section_tag][params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_config[params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_config[params.maximum_disparity_tag]
    
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
                         preprocessing_output_config[params.epipolar_size_x_tag],
                         preprocessing_output_config[params.epipolar_size_y_tag],
                         step,
                         step)

    epipolar_grid_flat = epipolar_grid.reshape(-1, epipolar_grid.shape[-1])

    epipolar_grid_min, epipolar_grid_max = compute_epipolar_grid_min_max(epipolar_grid, epsg, conf,disp_min, disp_max)

    # Build Delaunay triangulations
    delaunay_min = Delaunay(epipolar_grid_min)
    delaunay_max = Delaunay(epipolar_grid_max)

    # Build kdtrees
    tree_min = cKDTree(epipolar_grid_min)
    tree_max = cKDTree(epipolar_grid_max)

        
    # Look-up terrain grid with Delaunay
    s_min = tsearch(delaunay_min, region_grid)
    s_max = tsearch(delaunay_max, region_grid)
    
    points = []
    # For ecah corner
    for i in range(0,4):
        # If we are inside triangulation of s_min
        if s_min[i] != -1:
            # Add points from surrounding triangle
            for p in  epipolar_grid_flat[delaunay_min.simplices[s_min[i]]]:
                points.append(p)
        else:
            # else add nearest neighbor
            di,pi = tree_min.query(region_grid[i,:])
            points.append(epipolar_grid_flat[pi])
        # If we are inside triangulation of s_min
            if s_max[i] != -1:
                # Add points from surrounding triangle
                for p in  epipolar_grid_flat[delaunay_max.simplices[s_max[i]]]:
                    points.append(p)
            else:
                # else add nearest neighbor
                di,pi = tree_max.query(region_grid[i,:])
                points.append(epipolar_grid_flat[pi])

    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)

    # Bouding region of corresponding cell
    epipolar_region_minx = points_min[0]
    epipolar_region_miny = points_min[1]
    epipolar_region_maxx = points_max[0]
    epipolar_region_maxy = points_max[1]

    # This mimics the previous code that was using
    # transform_terrain_region_to_epipolar
    epipolar_region = [epipolar_region_minx, epipolar_region_miny,
                       epipolar_region_maxx, epipolar_region_maxy]
    
    return epipolar_region
