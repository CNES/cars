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
# pylint: disable=too-many-lines

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


def get_msk_roi_to_fill(
    msk_values: np.ndarray,
    key_id: List[int],
) -> np.ndarray:
    """
    Calculates binary mask. All regions to fill specified in
    multi_mask configuration will be set to 1.

    :param msk_values: msk layer of left or right epipolar image dataset
    :type msk_values: np.ndarray
    :param key_id: label of masked region specified in
     "mask_classes" of input .json file
    :type key_id: List of int

    :return: binary mask

    """
    if len(key_id) == 1:
        msk = msk_values == key_id[0]
    else:
        msk = msk_values == key_id[0]
        for k_val in range(1, len(key_id)):
            tmp_mask = msk_values == key_id[k_val]
            msk = np.logical_or(msk, tmp_mask)
    return msk


def get_roi_coverage_as_poly_with_margins(
    msk_values: np.ndarray, row_offset=0, col_offset=0, margin=0
) -> List[Polygon]:
    """
    Finds all roi existing in binary msk and stores their coverage as
    list of Polygon

    :param roi: ['roi'] attribute of epipolar image dataset
    :type roi: List[int]
    :param msk_values: msk layer of left/right epipolar image dataset
    :type msk_values: np.ndarray
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
        # TODO: create function to avoid code duplication
        labeled_array, __ = label(np.array(msk_values_dil).astype("int"))
        shapes = rasterio.features.shapes(
            labeled_array,
            transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        for (geometry, value) in shapes:
            if value != 0:
                # Get polygon coordinates of labelled region
                coords = geometry["coordinates"][0]
                coords = [
                    (c[0] + row_offset, c[1] + col_offset) for c in coords
                ]
                coord_shapes.append(coords)
        bbox.extend([Polygon(c) for c in coord_shapes])
    return bbox


def localize_masked_areas(
    dataset: xr.Dataset,
    key_id: List[int],
    row_offset: int = 0,
    col_offset: int = 0,
    margin: int = 0,
) -> np.ndarray:
    """
    Calculates bbox of masked region(s) if mask exists for
    input image file (see configuration "mask" and "mask_classes"
    in input .json configuration file)

    :param dataset_left: left epipolar image dataset
    :type dataset_left: CarsDataset
    :param dataset_right: right epipolar image dataset
    :type dataset_right: CarsDataset
    :param key_id: label used in mask and specified in
     "mask_classes" of input .json configuration file
    :type key_id: List of int
    :param margin: margin added to bbox in case masked region is
        localized at tile border (to ensure later disparity values
        at mask border extraction)
    :type margin: int

    :return: bounding box of masked area(s)

    """

    # binarize msk layer of epipolar image dataset
    # 0: 'valid' data, 1: masked data according to key_id
    msk_values = get_msk_roi_to_fill(dataset["msk"].values, key_id)
    # Finds roi in msk and stores its localization as polygon list
    bbox = get_roi_coverage_as_poly_with_margins(
        msk_values, row_offset=row_offset, col_offset=col_offset, margin=margin
    )
    return bbox


def find_tiles_including_mask_roi(
    dataset_left: xr.Dataset, dataset_right: xr.Dataset
) -> dict:
    """
    Retrieves all previously calculated bbox in left and right tiles
    and calculates their intersection to identify each masked region
    and its tile coverage.

    :param dataset_left: left epipolar image dataset
    :type dataset_left: CarsDataset
    :param dataset_right: right epipolar image dataset
    :type dataset_right: CarsDataset

    :return: dict containing masked area id and its tile coverage

    """

    # Storage of tile definition and their Polygon shape for later
    # intersection test.
    # Retrieve all bbox of masked areas previously calculated in
    # fill_disp_tools.find_bbox_for_masked_areas()
    tiles = []
    poly_tiles = []
    msk_areas = []
    shape = dataset_left.tiling_grid.shape
    if len(shape) == 3:
        tiles = dataset_left.tiling_grid.reshape(shape[0] * shape[1], shape[2])
    else:
        tiles = dataset_left.tiling_grid
    poly_tiles = []
    for tile in tiles:
        poly_tiles.append(
            Polygon(
                [
                    [tile[2], tile[0]],
                    [tile[3], tile[0]],
                    [tile[3], tile[1]],
                    [tile[2], tile[1]],
                    [tile[2], tile[0]],
                ]
            )
        )
    # retrieve msk_bbox
    for x_val in range(dataset_left.shape[0]):
        for y_val in range(dataset_left.shape[1]):
            poly_right = dataset_right[x_val, y_val].attrs["msk_bbox"]
            if poly_right is not None:
                msk_areas.extend(poly_right)
            poly_left = dataset_left[x_val, y_val].attrs["msk_bbox"]
            if poly_left is not None:
                msk_areas.extend(poly_left)

    # Calculate intersection of left_msk_areas VS right_msk_areas and
    # msk_areas VS tile definitions
    mapping_poly = {}
    ind = 0
    if len(msk_areas) == 0:
        mapping_poly = None
    else:
        while len(msk_areas) != 0:
            find_intersections = [
                msk_areas[0].intersects(m) for m in msk_areas[1:]
            ]
            intersect_id = np.where(find_intersections)[0] + 1
            msk_of_interest = msk_areas[0]
            # Create union between areas that intersect
            # Remove area in list of masked areas and intersection id
            # in intersect_id
            while len(intersect_id) != 0:
                msk_of_interest = msk_of_interest.union(
                    msk_areas[intersect_id[-1]]
                )
                msk_areas.pop(intersect_id[-1])
                intersect_id = intersect_id[:-1]
            # Finds tiles that intersects with this region and stores
            # their region definition
            find_tiles_intersect = [
                msk_of_interest.intersects(t) for t in poly_tiles
            ]
            # print(f'find_tiles_intersect : {find_tiles_intersect}')
            linked_tiles = np.array(
                [tiles[el] for el in np.where(find_tiles_intersect)[0]]
            )
            # print(f'linked_tiles : {linked_tiles}')
            # Store linked masked area and associated tiles in dict
            mapping_poly[str(ind)] = {
                "msk_roi": [msk_of_interest],
                "associated_tiles": linked_tiles,
            }
            msk_areas.pop(0)
            ind = ind + 1

    return mapping_poly
