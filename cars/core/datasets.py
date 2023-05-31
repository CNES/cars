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
CARS Datasets module
"""

# Standard imports
from typing import List

# Third party imports
import numpy as np
import rasterio as rio
import xarray as xr

# CARS imports
from cars.core import constants as cst

# TODO: refacto constants: define constants here as only concerning datasets


def create_im_dataset(
    img: np.ndarray,
    region: List[int],
    largest_size: List[int],
    img_path: str = None,
    band_coords: str = None,
    msk: np.ndarray = None,
) -> xr.Dataset:
    """
    Create image dataset as used in cars.

    :param img: image as a numpy array
    :param region: region as list [xmin ymin xmax ymax]
    :param largest_size: whole image size
    :param img_path: path to image
    :param band_type: set to band coord names (cst.BAND_IM or BAND_CLASSIF)
     to add band description in the dataset
    :param msk: image mask as a numpy array (default None)
    :return: The image dataset as used in cars
    """
    nb_bands = img.shape[0]

    # Get georef and transform
    img_crs = None
    img_transform = None
    descriptions = None
    if img_path is not None:
        with rio.open(img_path) as img_srs:
            img_crs = img_srs.profile["crs"]
            img_transform = img_srs.profile["transform"]
            descriptions = list(img_srs.descriptions)

    if img_crs is None:
        img_crs = "None"
    if img_transform is None:
        img_transform = "None"

    # Add band dimension if needed
    if band_coords or nb_bands > 1:
        # Reorder dimensions in color dataset in order that the first dimension
        # is band.
        if band_coords == cst.BAND_IM:
            if np.any(descriptions) is None:
                default_band = ["R", "G", "B", "N"]
                descriptions = default_band[:nb_bands]

        dataset = xr.Dataset(
            {
                cst.EPI_IMAGE: (
                    [band_coords, cst.ROW, cst.COL],
                    img,
                )
            },
            coords={
                band_coords: descriptions,
                cst.ROW: np.array(range(region[1], region[3])),
                cst.COL: np.array(range(region[0], region[2])),
            },
        )
    else:
        if np.any(descriptions) is None:
            descriptions = None
        dataset = xr.Dataset(
            {cst.EPI_IMAGE: ([cst.ROW, cst.COL], img[0, ...])},
            coords={
                cst.ROW: np.array(range(region[1], region[3])),
                cst.COL: np.array(range(region[0], region[2])),
            },
        )

    if msk is not None:
        dataset[cst.EPI_MSK] = xr.DataArray(
            msk[0, ...].astype(np.int16), dims=[cst.ROW, cst.COL]
        )

    dataset.attrs[cst.EPI_VALID_PIXELS] = 0
    dataset.attrs[cst.EPI_NO_DATA_MASK] = 255
    dataset.attrs[cst.EPI_FULL_SIZE] = largest_size
    dataset.attrs[cst.EPI_CRS] = img_crs
    dataset.attrs[cst.EPI_TRANSFORM] = img_transform
    dataset.attrs["region"] = np.array(region)
    if descriptions is not None:
        dataset.attrs[cst.BAND_NAMES] = descriptions
    return dataset


def get_color_bands(dataset, key):
    """
    Get band names list from the cardataset color

    :param dataset: carsdataset with the color data
    :type dataset: CarsDataset
    :param key: dataset color data key
    :param key: string

    :return: list of color band names
    """
    nb_bands = dataset[key].values.shape[0]
    band_im = None
    if cst.BAND_NAMES in dataset.attrs.keys():
        band_im = dataset.attrs[cst.BAND_NAMES]
    # if description is None, fill with default value
    # according with the number of channels

    if band_im is None:  # and np.any(band_im) is None
        default_band = ["Red", "Green", "Blue", "NIR"]
        band_im = default_band[:nb_bands]

    return band_im
