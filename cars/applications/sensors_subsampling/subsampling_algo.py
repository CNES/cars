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
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from cars.core import datasets


def resample_image(  # pylint: disable=too-many-positional-arguments
    image,
    tile_window,
    tile_size,
    key,
    scale_factor=1,
    interpolator="bilinear",
):
    """
    Resample the image

    :param image: the image to resample
    :type image: str
    :param tile_window: the tile window
    :type tile_window: Window
    :param tile_size: the tile size
    :type tile_size: int
    :param key: the key in the path dictionary
    :type key: str
    :param scale_factor: the scaling factor
    :type scale_factor: float
    :param interpolator: the interpolator
    :type interpolator: str
    """

    x = tile_window["col_min"]
    y = tile_window["row_min"]

    x_read = max(x, 0)
    y_read = max(y, 0)

    width = tile_window["col_max"]
    height = tile_window["row_max"]
    w = min(tile_size, width - x_read)
    h = min(tile_size, height - y_read)

    window_in = Window(x_read, y_read, w, h)

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    resampling_image = getattr(Resampling, interpolator)

    band_coords = None
    with rasterio.open(image) as src:
        data_resampled = src.read(
            out_shape=(src.count, new_h, new_w),
            window=window_in,
            resampling=resampling_image,
        )

        description = list(src.descriptions)

        if len(description) > 1:
            band_coords = "band" + key

    out_x = int(x * scale_factor)
    out_y = int(y * scale_factor)

    region = [
        out_x,
        out_y,
        out_x + data_resampled.shape[2],
        out_y + data_resampled.shape[1],
    ]

    dataset = datasets.create_im_dataset(
        data_resampled,
        region,
        [data_resampled.shape[1], data_resampled.shape[2]],
        band_coords=band_coords,
        descriptions=description,
    )

    return dataset
