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
Resampling module:
contains functions used for epipolar resampling
"""

# Standard imports
import logging
import math

# Third party imports
import numpy as np

# CARS imports
from cars.conf import input_parameters, mask_classes, output_prepare
from cars.core import constants as cst
from cars.core import datasets, inputs, tiling
from cars.externals import otb_pipelines


def epipolar_rectify_images(configuration, region, margins):
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
           dimensions image = ['left_margin', 'right_margin'],
           corner = ['left','up', 'right', 'down']
    :return: Datasets containing:

    1. left image and mask,
    2. right image and mask,
    3. left color image or None

    :rtype: xarray.Dataset, xarray.Dataset, xarray.Dataset
    """
    # Retrieve information from configuration
    input_configuration = configuration[input_parameters.INPUT_SECTION_TAG]
    preprocessing_output_conf = configuration[
        output_prepare.PREPROCESSING_SECTION_TAG
    ][output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]

    img1 = input_configuration[input_parameters.IMG1_TAG]
    img2 = input_configuration[input_parameters.IMG2_TAG]

    nodata1 = input_configuration.get(input_parameters.NODATA1_TAG, None)
    nodata2 = input_configuration.get(input_parameters.NODATA2_TAG, None)
    mask1 = input_configuration.get(input_parameters.MASK1_TAG, None)
    mask1_classes = input_configuration.get(
        input_parameters.MASK1_CLASSES_TAG, None
    )
    mask2 = input_configuration.get(input_parameters.MASK2_TAG, None)
    mask2_classes = input_configuration.get(
        input_parameters.MASK2_CLASSES_TAG, None
    )
    color1 = input_configuration.get(input_parameters.COLOR1_TAG, None)

    grid1 = preprocessing_output_conf[output_prepare.LEFT_EPIPOLAR_GRID_TAG]
    grid2 = preprocessing_output_conf[output_prepare.RIGHT_EPIPOLAR_GRID_TAG]

    epipolar_size_x = preprocessing_output_conf[
        output_prepare.EPIPOLAR_SIZE_X_TAG
    ]
    epipolar_size_y = preprocessing_output_conf[
        output_prepare.EPIPOLAR_SIZE_Y_TAG
    ]

    # Force region to be float
    region = [int(x) for x in region]

    # Apply margins to left image
    # TODO: tiled region should be given in parameter
    # TODO: keep only resampling here (keep functional unitary approach)
    left_region = region.copy()
    left_margins = margins["left_margin"].data
    left_roi = tiling.crop(
        left_region, [0, 0, epipolar_size_x, epipolar_size_y]
    )
    left_region = tiling.crop(
        tiling.pad(left_region, left_margins),
        [0, 0, epipolar_size_x, epipolar_size_y],
    )

    left_margins = margins["left_margin"].data
    # Get actual margin taking cropping into account
    left_margins[0] = left_region[0] - left_roi[0]
    left_margins[1] = left_region[1] - left_roi[1]
    left_margins[2] = left_region[2] - left_roi[2]
    left_margins[3] = left_region[3] - left_roi[3]

    # Apply margins to right image
    right_region = region.copy()
    right_margins = margins["right_margin"].data
    right_roi = tiling.crop(
        right_region, [0, 0, epipolar_size_x, epipolar_size_y]
    )
    right_region = tiling.crop(
        tiling.pad(right_region, right_margins),
        [0, 0, epipolar_size_x, epipolar_size_y],
    )

    # Get actual margin taking cropping into account
    right_margins[0] = right_region[0] - right_roi[0]
    right_margins[1] = right_region[1] - right_roi[1]
    right_margins[2] = right_region[2] - right_roi[2]
    right_margins[3] = right_region[3] - right_roi[3]

    # Resample left image
    left_dataset = resample_image(
        img1,
        grid1,
        [epipolar_size_x, epipolar_size_y],
        region=left_region,
        nodata=nodata1,
        mask=mask1,
    )

    # Check masks' classes consistency
    if mask1_classes is None and mask1 is not None:
        if mask_classes.is_multiclasses_mask(left_dataset[cst.EPI_MSK].values):
            logging.warning(
                "Left mask seems to have several classes but no "
                "classes usage json file has been indicated in the "
                "configuration file. All classes will be "
                "considered as invalid data."
            )

    # Update attributes
    left_dataset.attrs[cst.ROI] = np.array(left_roi)
    left_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(left_region)
    # Remove region key as it duplicates roi_with_margins key
    left_dataset.attrs.pop("region", None)
    left_dataset.attrs[cst.EPI_MARGINS] = np.array(left_margins)
    left_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs["disp_min"]
    left_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs["disp_max"]

    # Resample right image
    right_dataset = resample_image(
        img2,
        grid2,
        [epipolar_size_x, epipolar_size_y],
        region=left_region,
        nodata=nodata2,
        mask=mask2,
    )

    # Check masks' classes consistency
    if mask2_classes is None and mask2 is not None:
        if mask_classes.is_multiclasses_mask(right_dataset[cst.EPI_MSK].values):
            logging.warning(
                "Right mask seems to have several classes but no "
                "classes usage json file has been indicated in the "
                "configuration file. All classes will be "
                "considered as invalid data."
            )

    # Update attributes
    right_dataset.attrs[cst.ROI] = np.array(right_roi)
    right_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(right_region)
    # Remove region key as it duplicates roi_with_margins key
    right_dataset.attrs.pop("region", None)
    right_dataset.attrs[cst.EPI_MARGINS] = np.array(right_margins)
    right_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs["disp_min"]
    right_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs["disp_max"]

    # Build resampling pipeline for color image, and build datasets
    if color1 is None:
        color1 = img1

    # Ensure that region is cropped to largest
    left_roi = tiling.crop(left_roi, [0, 0, epipolar_size_x, epipolar_size_y])

    # Check if p+xs fusion is not needed (color1 and img1 have the same size)
    # TODO : Refactor inputs dependency as only here ?
    if inputs.rasterio_get_size(color1) == inputs.rasterio_get_size(img1):
        left_color_dataset = resample_image(
            color1,
            grid1,
            [epipolar_size_x, epipolar_size_y],
            region=left_roi,
            band_coords=True,
        )
    else:
        left_color_dataset = resample_image(
            img1,
            grid1,
            [epipolar_size_x, epipolar_size_y],
            region=left_roi,
            band_coords=True,
            lowres_color=color1,
        )

    # Remove region key as it duplicates coordinates span
    left_color_dataset.attrs.pop("region", None)

    return left_dataset, right_dataset, left_color_dataset


def resample_image(
    img,
    grid,
    largest_size,
    region=None,
    nodata=None,
    mask=None,
    band_coords=False,
    lowres_color=None,
):
    """
    Resample image according to grid and largest size.

    :param img: Path to the image to resample
    :type img: string
    :param grid: Path to the resampling grid
    :type grid: string
    :param largest_size: Size of full output image
    :type largest_size: list of two int
    :param region: A subset of the output image to produce
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
        region = [
            int(math.floor(region[0])),
            int(math.floor(region[1])),
            int(math.ceil(region[2])),
            int(math.ceil(region[3])),
        ]

    # Convert largest_size to int if needed
    largest_size = [int(x) for x in largest_size]

    # Build mask pipeline for img needed
    img_has_mask = nodata is not None or mask is not None
    msk = None
    if img_has_mask:
        msk = otb_pipelines.build_mask_pipeline(
            img,
            mask,
            nodata,
            mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION,
            mask_classes.VALID_VALUE,
            grid,
            largest_size[0],
            largest_size[1],
            region,
        )

    # Build resampling pipelines for images
    resamp = otb_pipelines.build_image_resampling_pipeline(
        img, grid, largest_size[0], largest_size[1], region, lowres_color
    )

    dataset = datasets.create_im_dataset(
        resamp, region, largest_size, img, band_coords, msk
    )

    return dataset
