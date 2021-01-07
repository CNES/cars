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
contains stereo-rectification, disparity map estimation, triangulation functions
"""

# Standard imports
from __future__ import absolute_import
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

from scipy.spatial import Delaunay #pylint: disable=no-name-in-module
from scipy.spatial import tsearch #pylint: disable=no-name-in-module
from scipy.spatial import cKDTree #pylint: disable=no-name-in-module

import rasterio as rio
import xarray as xr
import otbApplication
from dask import sizeof
import pandora
import pandora.marge
from pandora import constants as pcst

# Cars imports
from cars import pipelines
from cars import tiling
from cars import parameters as params
from cars import projection
from cars import utils
from cars import mask_classes
from cars.preprocessing import project_coordinates_on_line
from cars import constants as cst
from cars import matching_regularisation


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
        disp_max: int,
        otb_max_ram_hint: int=None,
        tile_size_rounding: int=50,
        margin: int=0) -> int:
    """
    Compute optimal tile size according to estimated memory usage
    (pandora_plugin_libsgm)
    Returned optimal tile size will be at least equal to tile_size_rounding.

    :param disp_min: Minimum disparity to explore
    :param disp_max: Maximum disparity to explore
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
    :type region: None (full output is produced) or array of four floats
                  [xmin,ymin,xmax,ymax]
    :param nodata: Nodata value to use (both for input and output)
    :type nodata: None or float
    :param mask: Mask to resample as well
    :type mask: None or path to mask image
    :param band_coords: Force bands coordinate in output dataset
    :type band_coords: boolean
    :param lowres_color: Path to the multispectral image
                         if p+xs fusion is needed
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
    msk = None
    if img_has_mask:
        msk = pipelines.build_mask_pipeline(
            img, grid, nodata, mask, largest_size[0], largest_size[1], region)

    # Build resampling pipelines for images
    resamp = pipelines.build_image_resampling_pipeline(
        img, grid, largest_size[0], largest_size[1], region, lowres_color)

    dataset = create_im_dataset(resamp, region, largest_size, band_coords, msk)

    return dataset


def create_im_dataset(img: np.ndarray,
                      region: List[int],
                      largest_size: List[int],
                      band_coords: bool=False,
                      msk: np.ndarray=None) -> xr.Dataset:
    """
    Create image dataset as used in cars.

    :param img: image as a numpy array
    :param region: region as list [xmin ymin xmax ymax]
    :param largest_size: whole image size
    :param band_coords: set to true to add the coords 'band' to the dataset
    :param msk: image mask as a numpy array (default None)
    :return: The image dataset as used in cars
    """
    nb_bands = img.shape[-1]

    # Add band dimension if needed
    if band_coords or nb_bands > 1:
        bands = range(nb_bands)
        # Reorder dimensions in color dataset in order that the first dimension
        # is band.
        dataset = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL],
                                     np.einsum('ijk->kij', img)
                                     )},
                             coords={cst.BAND: bands,
                                     cst.ROW: np.array(range(region[1],
                                                             region[3])),
                                     cst.COL: np.array(range(region[0],
                                                             region[2]))
                                     })
    else:
        dataset = xr.Dataset({cst.EPI_IMAGE: ([cst.ROW, cst.COL],
                             img[:, :, 0])},
                             coords={cst.ROW: np.array(range(region[1],
                                                             region[3])),
                                     cst.COL: np.array(range(region[0],
                                                             region[2]))})

    if msk is not None:
        dataset[cst.EPI_MSK] = xr.DataArray(msk.astype(np.int16),
                                            dims=[cst.ROW,
                                            cst.COL])

    dataset.attrs[cst.EPI_FULL_SIZE] = largest_size
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
    The parameters margin_x and margin_y are used to apply margins on the
    epipolar region.

    :param configuration: Configuration for stereo processing
    :type configuration: StereoConfiguration
    :param region: Array defining epipolar region as [xmin, ymin, xmax, ymax]
    :type region: list of four float
    :param margins: margins for the images
    :type: 2D (image, corner) DataArray, with the
           dimensions image = ['ref_margin', 'sec_margin'],
           corner = ['left','up', 'right', 'down']
    :return: Datasets containing:

    1. left image and mask,
    2. right image and mask,
    3. left color image or None

    :rtype: xarray.Dataset, xarray.Dataset, xarray.Dataset
    """
    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_conf = configuration\
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    nodata1 = input_configuration.get(params.nodata1_tag, None)
    nodata2 = input_configuration.get(params.nodata2_tag, None)
    mask1 = input_configuration.get(params.mask1_tag, None)
    mask1_classes = input_configuration.get(params.mask1_classes_tag, None)
    mask2 = input_configuration.get(params.mask2_tag, None)
    mask2_classes = input_configuration.get(params.mask2_classes_tag, None)
    color1 = input_configuration.get(params.color1_tag, None)

    grid1 = preprocessing_output_conf[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_conf[params.right_epipolar_grid_tag]

    epipolar_size_x = preprocessing_output_conf[params.epipolar_size_x_tag]
    epipolar_size_y = preprocessing_output_conf[params.epipolar_size_y_tag]

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

    # Check masks' classes consistency
    if mask1_classes is None and mask1 is not None:
        if mask_classes.is_multiclasses_mask(left_dataset[cst.EPI_MSK].values):
            logging.warning('Left mask seems to have several classes but no '
                            'classes usage json file has been indicated in the '
                            'configuration file. All classes will be '
                            'considered as unvalid data.')

    # Update attributes
    left_dataset.attrs[cst.ROI] = np.array(left_roi)
    left_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(left_region)
    # Remove region key as it duplicates roi_with_margins key
    left_dataset.attrs.pop('region', None)
    left_dataset.attrs[cst.EPI_MARGINS] = np.array(left_margins)
    left_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs['disp_min']
    left_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs['disp_max']

    # Resample right image
    right_dataset = resample_image(img2,
                                   grid2,
                                   [epipolar_size_x,
                                    epipolar_size_y],
                                   region=left_region,
                                   nodata=nodata2,
                                   mask=mask2)

    # Check masks' classes consistency
    if mask2_classes is None and mask2 is not None:
        if mask_classes.is_multiclasses_mask(right_dataset[cst.EPI_MSK].values):
            logging.warning('Right mask seems to have several classes but no '
                            'classes usage json file has been indicated in the '
                            'configuration file. All classes will be '
                            'considered as unvalid data.')

    # Update attributes
    right_dataset.attrs[cst.ROI] = np.array(right_roi)
    right_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(right_region)
    # Remove region key as it duplicates roi_with_margins key
    right_dataset.attrs.pop('region', None)
    right_dataset.attrs[cst.EPI_MARGINS] = np.array(right_margins)
    right_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs['disp_min']
    right_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs['disp_max']

    # Build resampling pipeline for color image, and build datasets
    if color1 is None:
        color1 = img1

    # Ensure that region is cropped to largest
    left_roi = tiling.crop(left_roi, [0, 0, epipolar_size_x, epipolar_size_y])

    # Check if p+xs fusion is not needed (color1 and img1 have the same size)
    if utils.rasterio_get_size(color1) == utils.rasterio_get_size(img1):
        left_color_dataset = resample_image(
            color1, grid1,
            [epipolar_size_x, epipolar_size_y],
            region=left_roi, band_coords=True)
    else:
        left_color_dataset = resample_image(img1,
                                            grid1,
                                            [epipolar_size_x,
                                             epipolar_size_y],
                                            region=left_roi,
                                            band_coords=True,
                                            lowres_color=color1)

    # Remove region key as it duplicates coordinates span
    left_color_dataset.attrs.pop('region', None)

    return left_dataset, right_dataset, left_color_dataset


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
    mask1_classes = input_stereo_cfg[params.input_section_tag]\
                                    .get(params.mask1_classes_tag, None)
    mask2_classes = input_stereo_cfg[params.input_section_tag]\
                                    .get(params.mask2_classes_tag, None)
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

def triangulate(configuration,
                disp_ref: xr.Dataset,
                disp_sec:xr.Dataset=None,
                im_ref_msk_ds: xr.Dataset=None,
                im_sec_msk_ds: xr.Dataset=None,
                snap_to_img1:bool = False,
                align:bool = False) -> Dict[str, xr.Dataset]:
    """
    This function will perform triangulation from a disparity map

    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param disp_ref: left to right disparity map dataset
    :param disp_sec: if available, the right to left disparity map dataset
    :param im_ref_msk_ds: reference image dataset (image and
                          mask (if indicated by the user) in epipolar geometry)
    :param im_sec_msk_ds: secondary image dataset (image and
                          mask (if indicated by the user) in epipolar geometry)
    :param snap_to_img1: If True, Lines of Sight of img2 are moved so as to
                         cross those of img1
    :param snap_to_img1: bool
    :param align: If True, apply correction to point after triangulation to
                  align with lowres DEM (if available. If not, no correction
                  is applied)
    :param align: bool
    :returns: point_cloud as a dictionary of dataset containing:

        * Array with shape (roi_size_x,roi_size_y,3), with last dimension
          corresponding to longitude, lattitude and elevation
        * Array with shape (roi_size_x,roi_size_y) with output mask
        * Array for color (optional): only if color1 is not None

    The dictionary keys are :
        * 'ref' to retrieve the dataset built from the left to
           right disparity map
        * 'sec' to retrieve the dataset built from the right to
           left disparity map
        (if provided in input)
    """

    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_conf = configuration\
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    grid1 = preprocessing_output_conf[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_conf[params.right_epipolar_grid_tag]
    if snap_to_img1:
        grid2 = preprocessing_output_conf[
            params.right_epipolar_uncorrected_grid_tag]

    point_clouds = dict()
    point_clouds[cst.STEREO_REF] = compute_points_cloud(
        disp_ref,
        img1,
        img2,
        grid1,
        grid2,
        roi_key=cst.ROI,
        dataset_msk=im_ref_msk_ds)

    if disp_sec is not None:
        point_clouds[cst.STEREO_SEC] = compute_points_cloud(
            disp_sec,
            img2,
            img1,
            grid2,
            grid1,
            roi_key=cst.ROI_WITH_MARGINS,
            dataset_msk=im_sec_msk_ds)

    # Handle alignment with lowres DEM
    if align and params.lowres_dem_splines_fit_tag in preprocessing_output_conf:
        # Read splines file
        splines_file = preprocessing_output_conf[
            params.lowres_dem_splines_fit_tag]
        splines_coefs = None
        with open(splines_file,'rb') as splines_file_reader:
            splines_coefs = pickle.load(splines_file_reader)

        # Read time direction line parameters
        time_direction_origin = [preprocessing_output_conf[
                                 params.time_direction_line_origin_x_tag],
                                 preprocessing_output_conf[
                                 params.time_direction_line_origin_y_tag]]
        time_direction_vector = [preprocessing_output_conf[
                                 params.time_direction_line_vector_x_tag],
                                 preprocessing_output_conf[
                                 params.time_direction_line_vector_y_tag]]

        disp_to_alt_ratio = preprocessing_output_conf[
                            params.disp_to_alt_ratio_tag]

        # Interpolate correction
        point_cloud_z_correction = \
            splines_coefs(project_coordinates_on_line(
                point_clouds[cst.STEREO_REF][cst.X].values.ravel(),
                point_clouds[cst.STEREO_REF][cst.Y].values.ravel(),
                time_direction_origin,
                time_direction_vector))
        point_cloud_z_correction = np.reshape(
            point_cloud_z_correction,
            point_clouds[cst.STEREO_REF][cst.X].shape)

        # Convert to disparity correction
        point_cloud_disp_correction = point_cloud_z_correction/disp_to_alt_ratio

        # Correct disparity
        disp_ref[cst.DISP_MAP] = disp_ref[cst.DISP_MAP] - \
                                 point_cloud_disp_correction

        # Triangulate again
        point_clouds[cst.STEREO_REF] = compute_points_cloud(
            disp_ref, img1, img2, grid1, grid2, roi_key=cst.ROI)

        # TODO handle sec case
        if disp_sec is not None:
            # Interpolate correction
            point_cloud_z_correction = \
                splines_coefs(
                    project_coordinates_on_line(
                        point_clouds[cst.STEREO_SEC][cst.X].values.ravel(),
                        point_clouds[cst.STEREO_SEC][cst.Y].values.ravel(),
                        time_direction_origin,
                        time_direction_vector))
            point_cloud_z_correction = np.reshape(
                point_cloud_z_correction,
                point_clouds[cst.STEREO_SEC][cst.X].shape)

            # Convert to disparity correction
            point_cloud_disp_correction = \
                point_cloud_z_correction/disp_to_alt_ratio

            # Correct disparity
            disp_sec[cst.DISP_MAP] = \
                disp_sec[cst.DISP_MAP] + point_cloud_disp_correction

            # Triangulate again
            point_clouds[cst.STEREO_SEC] = compute_points_cloud(
                disp_sec, img2, img1, grid2, grid1,
                roi_key=cst.ROI_WITH_MARGINS)

    return point_clouds


def compute_points_cloud(data: xr.Dataset,
                         img1: xr.Dataset,
                         img2: xr.Dataset,
                         grid1: str,
                         grid2: str,
                         roi_key: str,
                         dataset_msk: xr.Dataset=None) -> xr.Dataset:
    """
    Compute points cloud

    :param data: The reference to disparity map dataset
    :param img1: reference image dataset
    :param img2: secondary image dataset
    :param grid1: path to the reference image grid file
    :param grid2: path to the secondary image grid file
    :param roi_key: roi of the disparity map key
    ('roi' if cropped while calling create_disp_dataset,
    otherwise 'roi_with_margins')
    :param dataset_msk: dataset with mask information to use
    :return: the points cloud dataset
    """
    disp = pipelines.encode_to_otb(
        data[cst.DISP_MAP].values,
        data.attrs[cst.EPI_FULL_SIZE],
        data.attrs[roi_key])

    # Retrieve elevation range from imgs
    (min_elev1, max_elev1) = utils.get_elevation_range_from_metadata(img1)
    (min_elev2, max_elev2) = utils.get_elevation_range_from_metadata(img2)

    # Build triangulation app
    triangulation_app = otbApplication.Registry.CreateApplication(
        "EpipolarTriangulation")

    triangulation_app.SetParameterString("mode","disp")
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
        cst.X: ([cst.ROW, cst.COL], llh[:, :, 0]),  # longitudes
        cst.Y: ([cst.ROW, cst.COL], llh[:, :, 1]),  # latitudes
        cst.Z: ([cst.ROW, cst.COL], llh[:, :, 2]),
        cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL],
                                    data[cst.DISP_MSK].values)
    }

    if dataset_msk is not None:
        ds_values_list = [key for key, _ in dataset_msk.items()]

        if cst.EPI_MSK in ds_values_list:
            if roi_key == cst.ROI_WITH_MARGINS:
                ref_roi = [0,
                           0,
                           int(dataset_msk.dims[cst.COL]),
                           int(dataset_msk.dims[cst.ROW])]
            else:
                ref_roi = [int(-dataset_msk.attrs[cst.EPI_MARGINS][0]),
                           int(-dataset_msk.attrs[cst.EPI_MARGINS][1]),
                           int(dataset_msk.dims[cst.COL] \
                               - dataset_msk.attrs[cst.EPI_MARGINS][2]),
                           int(dataset_msk.dims[cst.ROW] \
                               - dataset_msk.attrs[cst.EPI_MARGINS][3])]
            im_msk = dataset_msk[cst.EPI_MSK].values[ref_roi[1]:ref_roi[3],
                                                     ref_roi[0]:ref_roi[2]]
            values[cst.POINTS_CLOUD_MSK] = ([cst.ROW, cst.COL], im_msk)
        else:
            worker_logger = logging.getLogger('distributed.worker')
            worker_logger.warning("No mask is present in the image dataset")

    point_cloud = xr.Dataset(values,
                             coords={cst.ROW: row, cst.COL: col})

    point_cloud.attrs[cst.ROI] = data.attrs[cst.ROI]
    if roi_key == cst.ROI_WITH_MARGINS:
        point_cloud.attrs[cst.ROI_WITH_MARGINS] = \
            data.attrs[cst.ROI_WITH_MARGINS]
    point_cloud.attrs[cst.EPI_FULL_SIZE] = \
        data.attrs[cst.EPI_FULL_SIZE]
    point_cloud.attrs[cst.EPSG] = int(4326)

    return point_cloud


def triangulate_matches(configuration, matches, snap_to_img1=False):
    """
    This function will perform triangulation from sift matches

    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param matches: numpy.array of matches of shape (nb_matches, 4)
    :type data: numpy.ndarray
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so
                         as to cross those of img1
    :param snap_to_img1: bool
    :returns: point_cloud as a dataset containing:

        * Array with shape (nb_matches,1,3), with last dimension
        corresponding to longitude, lattitude and elevation
        * Array with shape (nb_matches,1) with output mask

    :rtype: xarray.Dataset
    """

    # Retrieve information from configuration
    input_configuration = configuration[params.input_section_tag]
    preprocessing_output_configuration = configuration\
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]

    img1 = input_configuration[params.img1_tag]
    img2 = input_configuration[params.img2_tag]

    grid1 = preprocessing_output_configuration[params.left_epipolar_grid_tag]
    grid2 = preprocessing_output_configuration[params.right_epipolar_grid_tag]
    if snap_to_img1:
        grid2 = preprocessing_output_configuration\
            [params.right_epipolar_uncorrected_grid_tag]

    # Retrieve elevation range from imgs
    (min_elev1, max_elev1) = utils.get_elevation_range_from_metadata(img1)
    (min_elev2, max_elev2) = utils.get_elevation_range_from_metadata(img2)

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

    point_cloud = xr.Dataset({cst.X: ([cst.ROW, cst.COL], llh[:, :, 0]),
                              cst.Y: ([cst.ROW, cst.COL], llh[:, :, 1]),
                              cst.Z: ([cst.ROW, cst.COL], llh[:, :, 2]),
                              cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL],
                                                          msk)},
                             coords={cst.ROW: row,cst.COL: col})
    point_cloud.attrs[cst.EPSG] = int(4326)

    return point_cloud


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
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_cfg[params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_cfg[params.maximum_disparity_tag]

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
        [params.input_section_tag].get(params.mask1_classes_tag, None)
    mask2_classes = input_stereo_cfg \
        [params.input_section_tag].get(params.mask2_classes_tag, None)
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
        points = triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF], disp[cst.STEREO_SEC],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)
    else:
        points = triangulate(
            input_stereo_cfg, disp[cst.STEREO_REF],
            snap_to_img1=snap_to_img1, align=align,
            im_ref_msk_ds=im_ref_msk, im_sec_msk_ds=im_sec_msk)

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key in points:
            points[key] = geoid_offset(points[key], geoid_data)

    if out_epsg is not None:
        for key in points:
            projection.points_cloud_conversion_dataset(points[key], out_epsg)

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
    longitudes = np.copy(out_pc[cst.X].values)
    longitudes[longitudes < 0] += 360

    # perform interpolation using point cloud coordinates.
    if not geoid.lat_min <= out_pc[cst.Y].min() \
        <= out_pc[cst.Y].max() <= geoid.lat_max \
       and geoid.lon_min <= np.min(longitudes) \
        <= np.max(longitudes) <= geoid.lat_max:
        raise RuntimeError('Geoid does not fully cover the area spanned by '
                           'the point cloud.')

    # interpolate data
    ref_interp = geoid.interp({'lat': out_pc[cst.Y],
                               'lon':xr.DataArray(longitudes,
                                                  dims=(cst.ROW, cst.COL))})
    # offset using geoid height
    out_pc[cst.Z] = points[cst.Z] - ref_interp.hgt

    # remove coordinates lat & lon added by the interpolation
    out_pc = out_pc.reset_coords(['lat', 'lon'], drop=True)

    return out_pc


def compute_epipolar_grid_min_max(grid,
                                  epsg,
                                  conf,
                                  disp_min = None,
                                  disp_max = None):
    """
    Compute ground terrain location of epipolar grids at disp_min and disp_max

    :param grid: The epipolar grid to project
    :type grid: np.ndarray of shape (N,M,2)
    :param epsg: EPSG code of the terrain projection
    :type epsg: Int
    :param conf: Configuration dictionnary from prepare step
    :type conf: Dict
    :param disp_min: Minimum disparity
                     (if None, read from configuration dictionnary)
    :type disp_min: Float or None
    :param disp_max: Maximum disparity
                     (if None, read from configuration dictionnary)
    :type disp_max: Float or None
    :returns: a tuple of location grid at disp_min and disp_max
    :rtype: Tuple(np.ndarray, np.ndarray) same shape as grid param
    """
    # Retrieve disp min and disp max if needed
    preprocessing_output_configuration = conf\
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_configuration\
                        [params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_configuration\
                        [params.maximum_disparity_tag]

    if disp_min is None:
        disp_min = int(math.floor(minimum_disparity))
    else:
        disp_min = int(math.floor(disp_min))

    if disp_max is None:
        disp_max = int(math.ceil(maximum_disparity))
    else:
        disp_max = int(math.ceil(disp_max))

    # Generate disp_min and disp_max matches
    matches_min = np.stack((grid[:,:,0].flatten(),
                            grid[:,:,1].flatten(),
                            grid[:,:,0].flatten()+disp_min,
                            grid[:,:,1].flatten()), axis=1)
    matches_max = np.stack((grid[:,:,0].flatten(),
                            grid[:,:,1].flatten(),
                            grid[:,:,0].flatten()+disp_max,
                            grid[:,:,1].flatten()), axis=1)

    # Generate corresponding points clouds
    pc_min = triangulate_matches(conf, matches_min)
    pc_max = triangulate_matches(conf, matches_max)

    # Convert to correct EPSG
    projection.points_cloud_conversion_dataset(pc_min, epsg)
    projection.points_cloud_conversion_dataset(pc_max, epsg)

    # Form grid_min and grid_max
    grid_min = np.concatenate((pc_min[cst.X].values,
                               pc_min[cst.Y].values),
                              axis=1)
    grid_max = np.concatenate((pc_max[cst.X].values,
                               pc_max[cst.Y].values),
                              axis=1)

    return grid_min, grid_max


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
        [params.preprocessing_section_tag]\
        [params.preprocessing_output_section_tag]
    minimum_disparity = preprocessing_output_conf[params.minimum_disparity_tag]
    maximum_disparity = preprocessing_output_conf[params.maximum_disparity_tag]

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
                         preprocessing_output_conf[params.epipolar_size_x_tag],
                         preprocessing_output_conf[params.epipolar_size_y_tag],
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
