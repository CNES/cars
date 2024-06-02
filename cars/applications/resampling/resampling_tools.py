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

import logging

# Standard imports
import math

# Third party imports
import numpy as np
import rasterio as rio
import resample as cresample
from rasterio.windows import Window, bounds, from_bounds

from cars.conf import mask_cst as msk_cst

# CARS imports
from cars.core import constants as cst
from cars.core import datasets, inputs, tiling
from cars.data_structures import cars_dataset


def epipolar_rectify_images(
    img1,
    img2,
    grid1,
    grid2,
    region,
    margins,
    epipolar_size_x,
    epipolar_size_y,
    step=None,
    color1=None,
    mask1=None,
    mask2=None,
    classif1=None,
    classif2=None,
    nodata1=0,
    nodata2=0,
    add_color=True,
):
    """
    Resample left and right images, with color on left
    """

    # Force region to be float
    region = [int(x) for x in region]

    # Apply margins to left image
    # TODO: tiled region should be given in parameter
    # TODO: keep only rectification here (keep functional unitary approach)
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
        step=step,
        region=left_region,
        nodata=nodata1,
        mask=mask1,
    )

    # Update attributes
    left_dataset.attrs[cst.ROI] = np.array(left_roi)
    left_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(left_region)
    # Remove region key as it duplicates roi_with_margins key
    # left_dataset.attrs.pop("region", None)
    left_dataset.attrs[cst.EPI_MARGINS] = np.array(left_margins)
    if "disp_min" in margins.attrs:
        left_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs["disp_min"]
    if "disp_max" in margins.attrs:
        left_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs["disp_max"]

    # Resample right image
    right_dataset = resample_image(
        img2,
        grid2,
        [epipolar_size_x, epipolar_size_y],
        step=step,
        region=right_region,
        nodata=nodata2,
        mask=mask2,
    )

    # Update attributes
    right_dataset.attrs[cst.ROI] = np.array(right_roi)
    right_dataset.attrs[cst.ROI_WITH_MARGINS] = np.array(right_region)
    # Remove region key as it duplicates roi_with_margins key
    # right_dataset.attrs.pop("region", None)
    right_dataset.attrs[cst.EPI_MARGINS] = np.array(right_margins)
    if "disp_min" in margins.attrs:
        right_dataset.attrs[cst.EPI_DISP_MIN] = margins.attrs["disp_min"]
    if "disp_max" in margins.attrs:
        right_dataset.attrs[cst.EPI_DISP_MAX] = margins.attrs["disp_max"]

    left_color_dataset = None
    if add_color:
        # Build rectification pipeline for color image, and build datasets
        if color1 is None:
            color1 = img1

        if inputs.rasterio_get_size(color1) == inputs.rasterio_get_size(img1):
            left_color_dataset = resample_image(
                color1,
                grid1,
                [epipolar_size_x, epipolar_size_y],
                region=left_region,
                band_coords=cst.BAND_IM,
            )
        else:
            raise RuntimeError(
                "The image and the color "
                "haven't the same sizes "
                "{} != {}".format(
                    inputs.rasterio_get_size(color1),
                    inputs.rasterio_get_size(img1),
                )
            )

    # resample the mask images
    left_classif_dataset = None
    if classif1:
        left_classif_dataset = resample_image(
            classif1,
            grid1,
            [epipolar_size_x, epipolar_size_y],
            region=left_region,
            band_coords=cst.BAND_CLASSIF,
            interpolator="nearest",
        )

    right_classif_dataset = None
    if classif2:
        right_classif_dataset = resample_image(
            classif2,
            grid2,
            [epipolar_size_x, epipolar_size_y],
            region=right_region,
            band_coords=cst.BAND_CLASSIF,
            interpolator="nearest",
        )

    return (
        left_dataset,
        right_dataset,
        left_color_dataset,
        left_classif_dataset,
        right_classif_dataset,
    )


def resample_image(
    img,
    grid,
    largest_size,
    step=None,
    region=None,
    nodata=None,
    mask=None,
    band_coords=False,
    interpolator="bicubic",
):
    """
    Resample image according to grid and largest size.

    :param img: Path to the image to resample
    :type img: string
    :param grid: Path to the rectification grid
    :type grid: string
    :param largest_size: Size of full output image
    :type largest_size: list of two int
    :param step: horizontal step of resampling (useful for strip resampling)
    :type step: int
    :param region: A subset of the output image to produce
    :type region: None (full output is produced) or array of four floats
                  [xmin,ymin,xmax,ymax]
    :param nodata: Nodata value to use (both for input and output)
    :type nodata: None or float
    :param mask: Mask to resample as well
    :type mask: None or path to mask image
    :param band_coords: Force bands coordinate in output dataset
    :type band_coords: boolean
    :param interpolator: interpolator type (bicubic (default) or nearest)
    :type interpolator: str ("nearest" "linear" "bco")
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

    # Get path if grid is of type CarsDataset TODO remove
    if isinstance(grid, cars_dataset.CarsDataset):
        grid = grid.attributes["path"]

    # Localize blocks of the tile to resample
    if step is None:
        step = region[2] - region[0]
    xmin_of_blocks = np.arange(region[0], region[2], step)
    xmax_of_blocks = np.append(
        np.arange(region[0] + step, region[2], step), region[2]
    )
    ymin = region[1]
    ymax = region[3]
    # Initialize outputs of the entire tile
    nb_bands = inputs.rasterio_get_nb_bands(img)
    resamp = np.empty((nb_bands, region[3] - region[1], 0), dtype=np.float32)
    if nodata is not None or mask is not None:
        msk = np.empty((1, region[3] - region[1], 0), dtype=np.float32)
    else:
        msk = None

    with rio.open(grid) as grid_reader, rio.open(img) as img_reader:
        for xmin, xmax in zip(xmin_of_blocks, xmax_of_blocks):  # noqa: B905
            block_region = [xmin, ymin, xmax, ymax]
            # Build rectification pipelines for images
            res_x, res_y = grid_reader.res
            assert res_x == res_y
            oversampling = int(res_x)
            assert res_x == oversampling

            # Convert resampled region to grid region with oversampling
            grid_region = [
                math.floor(xmin / oversampling),
                math.floor(ymin / oversampling),
                math.ceil(xmax / oversampling),
                math.ceil(ymax / oversampling),
            ]

            grid_window = Window.from_slices(
                (grid_region[1], grid_region[3] + 1),
                (grid_region[0], grid_region[2] + 1),
            )
            grid_as_array = grid_reader.read(window=grid_window)
            grid_as_array = grid_as_array.astype(np.float32)
            grid_as_array = grid_as_array.astype(np.float64)

            # deformation to localization
            grid_as_array[0, ...] += np.arange(
                oversampling * grid_region[0],
                oversampling * (grid_region[2] + 1),
                step=oversampling,
            )
            grid_as_array[1, ...] += np.arange(
                oversampling * grid_region[1],
                oversampling * (grid_region[3] + 1),
                step=oversampling,
            ).T[..., np.newaxis]

            # get needed source bounding box
            left = math.floor(np.amin(grid_as_array[0, ...]))
            right = math.ceil(np.amax(grid_as_array[0, ...]))
            top = math.floor(np.amin(grid_as_array[1, ...]))
            bottom = math.ceil(np.amax(grid_as_array[1, ...]))

            # filter margin for bicubic = 2
            filter_margin = 2
            top -= filter_margin
            bottom += filter_margin
            left -= filter_margin
            right += filter_margin

            # extract src according to grid values
            transform = img_reader.transform
            res_x = int(transform[0] / abs(transform[0]))
            res_y = int(transform[4] / abs(transform[4]))

            (full_left, full_bottom, full_right, full_top) = img_reader.bounds

            left, right, top, bottom = (
                res_x * left,
                res_x * right,
                res_y * top,
                res_y * bottom,
            )

            full_bounds_window = from_bounds(
                full_left, full_bottom, full_right, full_top, transform
            )
            img_window = from_bounds(left, bottom, right, top, transform)

            # Crop window to be in image
            in_sensor = True
            try:
                img_window = img_window.intersection(full_bounds_window)
            except rio.errors.WindowError:
                # Window not in sensor image
                logging.debug("Window not in sensor image")
                in_sensor = False

            # round window
            img_window = img_window.round_offsets()
            img_window = img_window.round_lengths()

            # Compute offset
            tile_bounds = bounds(img_window, transform)
            tile_bounds_with_res = [
                res_x * tile_bounds[0],
                res_y * tile_bounds[1],
                res_x * tile_bounds[2],
                res_y * tile_bounds[3],
            ]

            x_offset = min(tile_bounds_with_res[0], tile_bounds_with_res[2])
            y_offset = min(tile_bounds_with_res[1], tile_bounds_with_res[3])

            if in_sensor:
                # Get sensor data
                img_as_array = img_reader.read(window=img_window)

                # shift grid regarding the img extraction
                grid_as_array[0, ...] -= x_offset
                grid_as_array[1, ...] -= y_offset

                block_resamp = cresample.grid(
                    img_as_array,
                    grid_as_array,
                    oversampling,
                    interpolator=interpolator,
                    nodata=0,
                ).astype(np.float32)

                # extract exact region
                out_region = oversampling * np.array(grid_region)
                ext_region = block_region - out_region
                block_resamp = block_resamp[
                    ...,
                    ext_region[1] : ext_region[3] - 1,
                    ext_region[0] : ext_region[2] - 1,
                ]
            else:
                block_resamp = np.zeros(
                    (
                        img_reader.count,
                        block_region[3] - block_region[1],
                        block_region[2] - block_region[0],
                    )
                )

            resamp = np.concatenate((resamp, block_resamp), axis=2)

            # create msk
            if nodata is not None or mask is not None:
                if in_sensor:
                    # get mask in source geometry
                    nodata_index = img_as_array == nodata

                    if mask is not None:
                        with rio.open(mask) as msk_reader:
                            msk_as_array = msk_reader.read(window=img_window)
                    else:
                        msk_as_array = np.zeros(img_as_array.shape)

                    nodata_msk = msk_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
                    msk_as_array[nodata_index] = nodata_msk

                    # resample mask
                    block_msk = cresample.grid(
                        msk_as_array,
                        grid_as_array,
                        oversampling,
                        interpolator="nearest",
                        nodata=nodata_msk,
                    )

                    block_msk = block_msk[
                        ...,
                        ext_region[1] : ext_region[3] - 1,
                        ext_region[0] : ext_region[2] - 1,
                    ]
                else:
                    nodata_msk = msk_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
                    block_msk = np.full(
                        (
                            1,
                            block_region[3] - block_region[1],
                            block_region[2] - block_region[0],
                        ),
                        fill_value=nodata_msk,
                    )

                msk = np.concatenate((msk, block_msk), axis=2)
    dataset = datasets.create_im_dataset(
        resamp, region, largest_size, img, band_coords, msk
    )

    return dataset
