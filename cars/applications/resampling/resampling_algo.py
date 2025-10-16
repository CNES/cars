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
import math

# Third party imports
import numpy as np
import rasterio as rio
import resample as cresample
from rasterio.windows import Window, bounds
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from cars.conf import mask_cst as msk_cst

# CARS imports
from cars.core import constants as cst
from cars.core import datasets, inputs, tiling
from cars.core.geometry import abstract_geometry


def epipolar_rectify_images(  # pylint: disable=too-many-positional-arguments
    left_imgs,
    right_imgs,
    grid1,
    grid2,
    region,
    margins,
    epipolar_size_x,
    epipolar_size_y,
    interpolator_image="bicubic",
    interpolator_classif="nearest",
    interpolator_mask="nearest",
    step=None,
    resolution=1,
    mask1=None,
    mask2=None,
    left_classifs=None,
    right_classifs=None,
    nodata1=0,
    nodata2=0,
    add_classif=True,
):
    """
    Resample left and right images
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

    # Resample left images
    left_img_transform = inputs.rasterio_get_transform(
        next(iter(left_imgs)), convention="north"
    )

    left_dataset = resample_image(
        left_imgs,
        grid1,
        [epipolar_size_x, epipolar_size_y],
        step=step,
        resolution=resolution,
        region=left_region,
        nodata=nodata1,
        mask=mask1,
        band_coords=cst.BAND_IM,
        interpolator_img=interpolator_image,
        interpolator_mask=interpolator_mask,
        img_transform=left_img_transform,
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
    right_img_transform = inputs.rasterio_get_transform(next(iter(right_imgs)))
    right_dataset = resample_image(
        right_imgs,
        grid2,
        [epipolar_size_x, epipolar_size_y],
        step=step,
        resolution=resolution,
        region=right_region,
        nodata=nodata2,
        mask=mask2,
        band_coords=cst.BAND_IM,
        interpolator_img=interpolator_image,
        interpolator_mask=interpolator_mask,
        img_transform=right_img_transform,
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

    # Resample classifications
    left_classif_dataset = None
    right_classif_dataset = None
    if add_classif:
        if left_classifs:
            left_classif_dataset = resample_image(
                left_classifs,
                grid1,
                [epipolar_size_x, epipolar_size_y],
                resolution=resolution,
                region=left_region,
                band_coords=cst.BAND_CLASSIF,
                interpolator_img=interpolator_classif,
                interpolator_mask=interpolator_mask,
                img_transform=left_img_transform,
            )
        if right_classifs:
            right_classif_dataset = resample_image(
                right_classifs,
                grid2,
                [epipolar_size_x, epipolar_size_y],
                resolution=resolution,
                region=right_region,
                band_coords=cst.BAND_CLASSIF,
                interpolator_img=interpolator_classif,
                interpolator_mask=interpolator_mask,
                img_transform=right_img_transform,
            )

    return (
        left_dataset,
        right_dataset,
        left_classif_dataset,
        right_classif_dataset,
    )


# pylint: disable=too-many-positional-arguments
def resample_image(  # noqa: C901
    imgs,
    grid,
    largest_size,
    step=None,
    resolution=1,
    region=None,
    nodata=None,
    mask=None,
    band_coords=False,
    interpolator_img="bicubic",
    interpolator_mask="nearest",
    img_transform=None,
):
    """
    Resample image according to grid and largest size.

    :param img: Path to the image to resample
    :type img: string
    :param grid: rectification grid dict
    :type grid: dict
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
    img_sample = next(iter(imgs))
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

    if img_transform is None:
        img_transform = inputs.rasterio_get_transform(
            img_sample, convention="north"
        )

    # Convert largest_size to int if needed
    largest_size = [int(x) for x in largest_size]

    # Localize blocks of the tile to resample
    if step is None:
        step = region[2] - region[0]

    xmin_of_blocks = np.arange(region[0], region[2], step)
    xmax_of_blocks = np.append(
        np.arange(region[0] + step, region[2], step), region[2]
    )

    ymin_of_blocks = np.arange(region[1], region[3], step)
    ymax_of_blocks = np.append(
        np.arange(region[1] + step, region[3], step), region[3]
    )

    resampled_images_list = []
    resampled_masks_list = []
    band_names = []
    data_types = []
    nodata_msk = msk_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
    for img in imgs:
        bands = imgs[img]
        nb_bands = len(bands["band_id"])
        # Initialize outputs of the entire tile
        resamp = np.empty(
            (nb_bands, region[3] - region[1], region[2] - region[0]),
            dtype=np.float32,
        )
        msk = np.empty(
            (nb_bands, region[3] - region[1], region[2] - region[0]),
            dtype=np.float32,
        )

        ystart = 0
        with rio.open(grid["path"]) as grid_reader, rio.open(img) as img_reader:
            for ymin, ymax in zip(ymin_of_blocks, ymax_of_blocks):  # noqa: B905
                ysize = ymax - ymin
                xstart = 0
                for xmin, xmax in zip(  # noqa: B905
                    xmin_of_blocks, xmax_of_blocks
                ):
                    block_region = [xmin, ymin, xmax, ymax]
                    xsize = xmax - xmin
                    resamp, msk = oversampling_func(
                        grid_reader,
                        img_reader,
                        img_transform,
                        block_region,
                        resolution,
                        interpolator_img,
                        band_coords,
                        nb_bands,
                        bands,
                        resamp,
                        nodata,
                        msk,
                        mask,
                        nodata_msk,
                        interpolator_mask,
                        ysize,
                        xsize,
                        ystart,
                        xstart,
                    )
                    xstart += xsize

                ystart += ysize
        band_names += bands["band_name"]
        data_types += [inputs.rasterio_get_image_type(img)] * nb_bands
        resampled_images_list.append(resamp)
        resampled_masks_list.append(msk)

    resamp_final = np.concatenate(resampled_images_list, axis=0)
    msk_final = np.concatenate(resampled_masks_list, axis=0)
    dataset = datasets.create_im_dataset(
        resamp_final,
        region,
        largest_size,
        img_sample,
        band_coords,
        band_names,
        data_types,
        msk_final,
    )

    return dataset


def oversampling_func(  # pylint: disable=too-many-positional-arguments
    grid_reader,
    img_reader,
    img_transform,
    block_region,
    resolution,
    interpolator_img,
    band_coords,
    nb_bands,
    bands,
    resamp,
    nodata,
    msk,
    mask,
    nodata_msk,
    interpolator_mask,
    ysize,
    xsize,
    ystart,
    xstart,
):
    """
    Do the resampling calculus
    """

    xmin = block_region[0]
    ymin = block_region[1]
    xmax = block_region[2]
    ymax = block_region[3]

    # Build rectification pipelines for images
    res_x, res_y = grid_reader.res
    assert res_x == res_y
    oversampling = int(res_x)
    assert res_x == oversampling

    grid_origin_x = grid_reader.transform[2]
    grid_origin_y = grid_reader.transform[5]
    assert grid_origin_x == grid_origin_y
    grid_margin = int(-grid_origin_x / oversampling - 0.5)

    grid_margin = int(grid_margin)

    # Convert resampled region to grid region with oversampling
    grid_region = np.array(
        [
            math.floor(xmin / oversampling),
            math.floor(ymin / oversampling),
            math.ceil(xmax / oversampling),
            math.ceil(ymax / oversampling),
        ]
    )

    # Out region of epipolar image
    out_region = oversampling * grid_region
    # Grid region
    grid_region += grid_margin

    grid_window = Window.from_slices(
        (grid_region[1], grid_region[3] + 1),
        (grid_region[0], grid_region[2] + 1),
    )
    grid_as_array = grid_reader.read(window=grid_window)
    grid_as_array = grid_as_array.astype(np.float32)
    grid_as_array = grid_as_array.astype(np.float64)

    # get needed source bounding box
    left = math.floor(np.amin(grid_as_array[0, ...]))
    right = math.ceil(np.amax(grid_as_array[0, ...]))
    top = math.floor(np.amin(grid_as_array[1, ...]))
    bottom = math.ceil(np.amax(grid_as_array[1, ...]))

    transform = rio.Affine(*np.abs(img_transform))
    # transform xmin and xmax positions to index
    (top, bottom, left, right) = abstract_geometry.min_max_to_index_min_max(
        left, right, top, bottom, transform
    )

    # filter margin for bicubic = 2
    filter_margin = 2
    top -= filter_margin
    bottom += filter_margin
    left -= filter_margin
    right += filter_margin

    left, right = list(np.clip([left, right], 0, img_reader.shape[0]))
    top, bottom = list(np.clip([top, bottom], 0, img_reader.shape[1]))

    img_window = Window.from_slices([left, right], [top, bottom])

    in_sensor = True
    if right - left == 0 or bottom - top == 0:
        in_sensor = False

    # round window
    img_window = img_window.round_offsets()
    img_window = img_window.round_lengths()

    # Compute offset
    res_x = float(abs(transform[0]))
    res_y = float(abs(transform[4]))
    tile_bounds = list(bounds(img_window, transform))

    x_offset = min(tile_bounds[0], tile_bounds[2])
    y_offset = min(tile_bounds[1], tile_bounds[3])

    if in_sensor:
        # Get sensor data
        img_as_array = img_reader.read(bands["band_id"], window=img_window)
        # get the nodata mask before blurring
        img_nan_mask = img_as_array == nodata

        # blur the image to avoid moiré artefacts if downsampling
        if resolution != 1:
            fourier = fftshift(fft2(img_as_array))

            _, rows, cols = img_as_array.shape
            crow, ccol = rows // 2, cols // 2
            radius = min(rows, cols) / (2 * resolution)

            row_mesh, col_mesh = np.ogrid[:rows, :cols]
            dist = (col_mesh - ccol) ** 2 + (row_mesh - crow) ** 2
            gaussian_mask = np.exp(-dist / (2 * radius**2))
            f_filtered = fourier * gaussian_mask

            img_as_array = np.real(ifft2(ifftshift(f_filtered)))

        # set the nodata values back
        if nodata is not None:
            img_as_array[img_nan_mask] = nodata

        # shift grid regarding the img extraction
        grid_as_array[0, ...] -= x_offset
        grid_as_array[1, ...] -= y_offset

        # apply input resolution
        grid_as_array[0, ...] /= res_x
        grid_as_array[1, ...] /= res_y

        block_resamp = cresample.grid(
            img_as_array,
            grid_as_array,
            oversampling,
            interpolator=interpolator_img,
            nodata=0,
        ).astype(np.float32)
        if interpolator_img == "bicubic" and band_coords == cst.BAND_CLASSIF:
            block_resamp = np.where(
                block_resamp >= 0.5,
                1,
                np.where(block_resamp < 0.5, 0, block_resamp),
            ).astype(int)

        # extract exact region
        ext_region = block_region - out_region
        block_resamp = block_resamp[
            ...,
            ext_region[1] : ext_region[3] - 1,
            ext_region[0] : ext_region[2] - 1,
        ]
    else:
        block_resamp = np.zeros(
            (
                nb_bands,
                block_region[3] - block_region[1],
                block_region[2] - block_region[0],
            )
        )

    resamp[:, ystart : ystart + ysize, xstart : xstart + xsize] = block_resamp

    # create msk
    if in_sensor:
        # get mask in source geometry
        if mask is not None:
            with rio.open(mask) as msk_reader:
                msk_as_array = msk_reader.read(1, window=img_window)
            msk_as_array = np.array([msk_as_array] * img_as_array.shape[0])
        else:
            msk_as_array = np.zeros(img_as_array.shape)

        if nodata is not None:
            nodata_index = img_as_array == nodata
            msk_as_array[nodata_index] = nodata_msk

        # resample mask
        block_msk = cresample.grid(
            msk_as_array,
            grid_as_array,
            oversampling,
            interpolator=interpolator_mask,
            nodata=nodata_msk,
        )

        if interpolator_mask == "bicubic":
            block_msk = np.where(
                block_msk >= 0.5,
                1,
                np.where(block_msk < 0.5, 0, block_msk),
            ).astype(int)

        block_msk = block_msk[
            ...,
            ext_region[1] : ext_region[3] - 1,
            ext_region[0] : ext_region[2] - 1,
        ]
    else:
        block_msk = np.full(
            (
                nb_bands,
                block_region[3] - block_region[1],
                block_region[2] - block_region[0],
            ),
            fill_value=nodata_msk,
        )

    msk[:, ystart : ystart + ysize, xstart : xstart + xsize] = block_msk

    return resamp, msk
