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
This module contains function for holes detection.
"""
# pylint: disable=too-many-lines

import logging

# Standard imports
from typing import List

# Third party imports
import numpy as np
import rasterio
import rasterio.features
import xarray as xr
from affine import Affine
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from shapely.geometry import Polygon

from cars.core import constants as cst


def get_roi_coverage_as_poly_with_margins(
    msk_values: np.ndarray, row_offset=0, col_offset=0, margin=0
) -> List[Polygon]:
    """
    Finds all roi existing in binary msk and stores their coverage as
    list of Polygon

    :param msk_values: msk layer of left/right epipolar image dataset
    :type msk_values: np.ndarray
    :param row_offset: offset on row to apply
    :type row_offset: int
    :param col_offset: offset on col to apply
    :type col_offset: int
    :param margin: margin added to bbox in case masked region is
        localized at tile border (to ensure later disparity values
        at mask border extraction)
    :type margin: int

    :return: list of polygon

    """

    bbox = []
    coord_shapes = []
    # Check if at least one masked area in roi
    if np.sum(msk_values) != 0:
        msk_values_dil = msk_values
        if margin != 0:
            # Dilates areas in mask according to parameter 'margin' in order
            # to get enough disparity values if region is near a tile border
            # 1. Generates a structuring element that will consider features
            # connected even if they touch diagonally
            struct = generate_binary_structure(2, 2)
            # 2. Dilation operation
            msk_values_dil = binary_dilation(
                msk_values, structure=struct, iterations=margin
            )
        labeled_array, __ = label(np.array(msk_values_dil).astype("int"))
        shapes = rasterio.features.shapes(
            labeled_array,
            transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        for geometry, value in shapes:
            if value != 0:
                # Get polygon coordinates of labelled region
                coords = geometry["coordinates"][0]
                coords = [
                    (c[1] + row_offset, c[0] + col_offset) for c in coords
                ]
                coord_shapes.append(coords)
        bbox.extend([Polygon(c) for c in coord_shapes])
    return bbox


def localize_masked_areas(
    dataset: xr.Dataset,
    classification: List[str],
    row_offset: int = 0,
    col_offset: int = 0,
    margin: int = 0,
) -> np.ndarray:
    """
    Calculates bbox of masked region(s) if mask exists for
    input image file (see configuration "mask" and "mask_classes"
    in input .json configuration file)

    :param dataset: epipolar image dataset
    :type dataset: CarsDataset
    :param classification: label of masked region to use
    :type classification: List of str
    :param row_offset: offset on row to apply
    :type row_offset: int
    :param col_offset: offset on col to apply
    :type col_offset: int
    :param margin: margin added to bbox in case masked region is
        localized at tile border (to ensure later disparity values
        at mask border extraction)
    :type margin: int

    :return: bounding box of masked area(s)

    """
    # binarize msk layer of epipolar image dataset
    # 0: 'valid' data, 1: masked data according to key_id
    if cst.EPI_CLASSIFICATION not in dataset:
        logging.debug("No classif provided")
        bbox = []
    else:
        if not isinstance(classification, list):
            logging.error("no mask classes provided for DisparityFilling")
            raise RuntimeError("no mask classes provided for DisparityFilling")
        msk_values = classif_to_stacked_array(dataset, classification)
        # Finds roi in msk and stores its localization as polygon list
        bbox = get_roi_coverage_as_poly_with_margins(
            msk_values,
            row_offset=row_offset,
            col_offset=col_offset,
            margin=margin,
        )
    return bbox


def classif_to_stacked_array(disp_map, class_index):
    """
    Convert disparity dataset to mask correspoding to all classes

    :param disp_map: disparity dataset
    :type disp_map: xarray Dataset
    :param class_index: classification tags
    :type class_index: list of str

    """

    index_class = np.where(
        np.isin(
            np.array(disp_map.coords[cst.BAND_CLASSIF].values),
            np.array(class_index),
        )
    )[0].tolist()
    # get index for each band classification of the non zero values
    stack_index = np.any(
        disp_map[cst.EPI_CLASSIFICATION].values[index_class, :, :] > 0,
        axis=0,
    )

    return stack_index
