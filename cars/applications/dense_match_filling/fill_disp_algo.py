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
# pylint: disable=too-many-lines
"""
This module is responsible for the filling disparity algorithms:
thus it fills the disparity map with values estimated according to
their neighbourhood.
"""

# Standard imports

import logging

import numpy as np

# Third party imports
import xarray as xr
from scipy.linalg import lstsq
from scipy.ndimage import binary_dilation, binary_erosion, label

from cars.applications.dense_match_filling import (
    fill_disp_wrappers as fill_wrap,
)
from cars.applications.hole_detection import (
    hole_detection_algo,
    hole_detection_wrappers,
)
from cars.conf import mask_cst

# Cars import
from cars.core import constants as cst


def fill_central_area_using_plane(  # noqa: C901
    disp_map: xr.Dataset,
    corresponding_poly,
    row_min,
    col_min,
    ignore_nodata: bool,
    ignore_zero_fill: bool,
    ignore_extrema: bool,
    nb_pix: int,
    percent_to_erode: float,
    class_index: list,
    fill_valid_pixels: bool,
):
    """
    Finds central area of invalid region and estimates disparity values
    in this area according to a plan model estimation. The estimation
    of this model is done using disparity values at invalid region
    borders.

    :param disp_map: disparity map with several layers ('disp',
        'disp_msk', 'msk_invalid_sec')
    :type disp_map: 2D np.array (row, col)
    :param corresponding_poly: list of holes polygons
    :type corresponding_poly: list(Polygon)
    :param row_min: row offset of combined tile
    :type row_min: int
    :param col_min: col offset of combined tile
    :type col_min: int
    :param ignore_nodata: option to activate to
        ignore nodata values at disp mask borders
    :type ignore_nodata: bool
    :param ignore_zero_fill: option to activate to
        ignore zero values at disp mask borders
    :type ignore_zero_fill: bool
    :param ignore_extrema: option to activate to ignore
        extrema values at disp mask borders
    :type ignore_extrema: bool
    :param nb_pix: pixel number used to define disparity values band
        at invalid region borders that will be considered for disp estimation
    :type nb_pix: int
    :param percent_to_erode: percentage to define size of central area
    :type percent_to_erode: float
    :param class_index: list of tag to use
    :type class_index: list(str)
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool

    :return: mask of invalid region that hasn't been filled yet
        (original invalid region - central area)
    :rtype: 2D np.array (row, col)
    """

    # Generate a structuring element that will consider features

    # connected even if they touch diagonally
    struct = fill_wrap.generate_binary_structure(2, 2)

    disp_mask = np.copy(disp_map["disp_msk"].values)
    disp_values = np.copy(disp_map["disp"].values)

    # Find invalid region of interest in disp data from polygon info
    classif_mask = hole_detection_wrappers.classif_to_stacked_array(
        disp_map, class_index
    )

    classif_mask_arrays, num_features = label(
        (classif_mask > 0).astype(int), structure=struct
    )

    list_roi_msk = []

    for segm in range(1, num_features + 1):
        roi_msk = classif_mask_arrays == segm

        # Create Polygon of current mask
        mask_polys = hole_detection_algo.get_roi_coverage_as_poly_with_margins(
            roi_msk, row_offset=row_min, col_offset=col_min, margin=0
        )
        # Clean mask polygons, remove artefacts
        cleaned_mask_poly = []
        for msk_pol in mask_polys:
            # if area > 20  : small groups to remove
            if msk_pol.area > 20:
                cleaned_mask_poly.append(msk_pol)

        if len(cleaned_mask_poly) > 1:
            # polygons due to surrounding no data
            # use biggest poly
            main_poly = None
            biggest_area = 0
            for curent_poly in cleaned_mask_poly:
                current_area = curent_poly.area
                if current_area > biggest_area:
                    main_poly = curent_poly
            cleaned_mask_poly = [main_poly]
            logging.debug("Not single polygon for current mask")

        intersect_holes = False
        # Check if main poly intersect found classif polygons
        if len(cleaned_mask_poly) > 0:
            for hole_poly in corresponding_poly:
                if hole_poly.intersects(cleaned_mask_poly[0]):
                    intersect_holes = True

        if intersect_holes:
            # is a hole to fill, not nodata in the border

            # Option 'ignore_nodata' adds invalid values of disp mask at roi_msk
            # invalid region borders
            if ignore_nodata:
                fill_wrap.add_surrounding_nodata_to_roi(
                    roi_msk, disp_values, disp_mask
                )

            # Selected invalid region dilation
            dilatation = binary_dilation(
                roi_msk, structure=struct, iterations=nb_pix
            )
            # dilated mask - initial mask = band of 'nb_pix' pix around roi
            roi_msk_tmp = np.logical_xor(dilatation, roi_msk)

            # do not use nan
            roi_msk_tmp = np.logical_and(
                roi_msk_tmp,
                ~np.isnan(disp_values),
            )
            roi_msk_tmp = np.logical_and(
                roi_msk_tmp,
                ~np.isnan(disp_mask),
            )

            # Band disp values retrieval
            # Optional filter processing n°1 : ignore invalid values in band
            if ignore_zero_fill:
                initial_len = np.sum(roi_msk_tmp)
                roi_msk_tmp = np.logical_and(
                    roi_msk_tmp,
                    disp_map["disp"].values.astype(bool),
                )
                logging.info(
                    "Zero_fill_disp_mask - Filtering {} \
                    disparity values, equivalent to {}% of data".format(
                        initial_len - np.sum(roi_msk_tmp),
                        100 - (100 * np.sum(roi_msk_tmp)) / initial_len,
                    )
                )
            band_disp_values = disp_values[roi_msk_tmp]

            # Optional filter processing n°2 : remove extreme values (10%)
            if ignore_extrema and len(band_disp_values) != 0:
                initial_len = len(band_disp_values)
                msk_extrema = np.copy(roi_msk_tmp)
                msk_extrema[:] = 0
                msk_extrema[
                    np.where(
                        abs(disp_values - np.mean(band_disp_values))
                        < 1.65 * np.std(band_disp_values)
                    )
                ] = 1
                roi_msk_tmp = np.logical_and(
                    roi_msk_tmp,
                    msk_extrema,
                )

                band_disp_values = disp_values[roi_msk_tmp]
                logging.info(
                    "Extrema values - Filtering {} disparity values,\
                    equivalent to {}% of data".format(
                        initial_len - len(band_disp_values),
                        100 - (100 * len(band_disp_values)) / initial_len,
                    )
                )

            if len(band_disp_values) != 0:
                disp_moy = np.mean(band_disp_values)
                logging.info("Disparity mean comptuted : {}".format(disp_moy))

            # roi_msk can be filled with 0 if neighbours have filled mask
            if np.sum(~roi_msk) > 0:
                # Definition of central area to fill using plane model
                erosion_value = fill_wrap.define_interpolation_band_width(
                    roi_msk, percent_to_erode
                )
                central_area = binary_erosion(
                    roi_msk, structure=struct, iterations=erosion_value
                )

                # Exclude pixels outside of epipolar footprint
                mask = (
                    disp_map[cst.EPI_MSK].values
                    != mask_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
                )
                if not fill_valid_pixels:
                    # Exclude valid pixels
                    mask = np.logical_and(
                        mask, disp_map["disp_msk"].values == 0
                    )
                central_area = np.logical_and(central_area, mask)

                variable_disp = calculate_disp_plane(
                    band_disp_values,
                    roi_msk_tmp,
                    central_area,
                )

                disp_map["disp"].values[central_area] = variable_disp
                disp_map["disp_msk"].values[central_area] = 255
                disp_map[cst.EPI_MSK].values[central_area] = 0
                fill_wrap.update_filling(
                    disp_map, central_area, "plane.hole_center"
                )

                # Retrieve borders that weren't filled yet
                roi_msk[central_area] = 0

                list_roi_msk.append(roi_msk)
    return list_roi_msk


def calculate_disp_plane(
    values,
    mask,
    central_area,
    display=False,
):
    """
    Estimates disparity values in disparity map which contains invalid
    area using valid data and a plane model.

    :param values: valid disparity values
    :type values: 3D np.array (row, col)
    :param mask: validity mask
    :type mask: 3D np.array (row, col)
    :param central_area: mask of disparity values to fill
    :type central_area: 3D np.array (row, col)
    :param display: plot interpolation fct in region to fill
    :type display: boolean

    :return: central interpolated disparity values
    :rtype: list
    """
    data_to_fill = np.where(central_area)
    data = np.vstack([np.where(mask), values]).T
    b_mat = data[:, 2]
    # Calcul coefficient fonction optimale plan/quadratique
    # ORDRE 1
    a_mat = np.vstack((data[:, :2].T, np.ones_like(b_mat))).T
    fit, __, __, __ = lstsq(a_mat, b_mat)

    # Détermination des valeurs optimales pour les coords centrales
    x_to_fill = data_to_fill[0]
    y_to_fill = data_to_fill[1]
    val = list(
        map(lambda x, y: fit[0] * x + fit[1] * y + fit[2], x_to_fill, y_to_fill)
    )
    # Option d'affichage de la fonction plan
    if display:
        fill_wrap.plot_function(data, fit)

    return val


def fill_area_borders_using_interpolation(
    disp_map,
    masks_to_fill,
    options,
    fill_valid_pixels,
):
    """
    Raster interpolation command
    :param disp_map: disparity values
    :type disp_map: 2D np.array (row, col)
    :param masks_to_fill: masks to locate disp values to fill
    :type masks_to_fill: list(2D np.array (row, col))
    :param options: parameters for interpolation methods
    :type options: dict
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool
    """
    # Copy input data - disparity values + mask with values to fill
    raster = np.copy(disp_map["disp"].values)

    # Interpolation step
    for mask_to_fill in masks_to_fill:
        # Exclude pixels outside of epipolar footprint
        mask = (
            disp_map[cst.EPI_MSK].values
            != mask_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
        )
        if not fill_valid_pixels:
            # Exclude valid pixels
            mask = np.logical_and(mask, disp_map["disp_msk"].values == 0)
        mask_to_fill = np.logical_and(mask_to_fill, mask)

        interpol_raster = fill_wrap.make_raster_interpolation(
            raster, mask_to_fill, options
        )
        # Insertion of interpolated data into disparity map
        disp_map["disp"].values[mask_to_fill] = interpol_raster[mask_to_fill]
        disp_map["disp_msk"].values[mask_to_fill] = 255
        disp_map[cst.EPI_MSK].values[mask_to_fill] = 0
        fill_wrap.update_filling(disp_map, mask_to_fill, "plane.hole_border")


def fill_disp_using_plane(
    disp_map: xr.Dataset,
    corresponding_poly,
    row_min,
    col_min,
    ignore_nodata: bool,
    ignore_zero_fill: bool,
    ignore_extrema: bool,
    nb_pix: int,
    percent_to_erode: float,
    interp_options: dict,
    classification,
    fill_valid_pixels,
) -> xr.Dataset:
    """
    Fill disparity map holes

    :param disp_map: disparity map
    :type disp_map: xr.Dataset
    :param corresponding_poly: list of holes polygons
    :type corresponding_poly: list(Polygon)
    :param row_min: row offset of combined tile
    :type row_min: int
    :param col_min: col offset of combined tile
    :type col_min: int
    :param ignore_nodata: ingore nodata
    :type ignore_nodata: bool
    :param ignore_zero_fill: ingnore zero fill
    :type ignore_zero_fill: bool
    :param ignore_extrema: ignore extrema
    :type ignore_extrema: bool
    :param nb_pix: margin to use
    :type nb_pix: int
    :param percent_to_erode: percent to erode
    :type percent_to_erode: float
    :param interp_options: interp_options
    :type interp_options: dict
    :param classification: list of tag to use
    :type classification: list(str)
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool
    """

    border_region = fill_central_area_using_plane(
        disp_map,
        corresponding_poly,
        row_min,
        col_min,
        ignore_nodata,
        ignore_zero_fill,
        ignore_extrema,
        nb_pix,
        percent_to_erode,
        classification,
        fill_valid_pixels,
    )

    fill_area_borders_using_interpolation(
        disp_map,
        border_region,
        interp_options,
        fill_valid_pixels,
    )


def fill_disp_using_zero_padding(
    disp_map: xr.Dataset,
    class_index,
    fill_valid_pixels,
) -> xr.Dataset:
    """
    Fill disparity map holes

    :param disp_map: disparity map
    :type disp_map: xr.Dataset
    :param class_index: class index according to the classification tag
    :type class_index: int
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool
    """
    # get index of the application class config
    # according the coords classif band
    if cst.BAND_CLASSIF in disp_map.coords:
        # get index for each band classification
        stack_index = hole_detection_wrappers.classif_to_stacked_array(
            disp_map, class_index
        )
        # Exclude pixels outside of epipolar footprint
        mask = (
            disp_map[cst.EPI_MSK].values
            != mask_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION
        )
        if not fill_valid_pixels:
            # Exclude valid pixels
            mask = np.logical_and(mask, disp_map["disp_msk"].values == 0)
        stack_index = np.logical_and(stack_index, mask)
        # set disparity value to zero where the class is
        # non zero value and masked region
        disp_map["disp"].values[stack_index] = 0
        disp_map["disp_msk"].values[stack_index] = 255
        disp_map[cst.EPI_MSK].values[stack_index] = 0
        # Add a band to disparity dataset to memorize which pixels are filled
        fill_wrap.update_filling(disp_map, stack_index, "zeros_padding")
