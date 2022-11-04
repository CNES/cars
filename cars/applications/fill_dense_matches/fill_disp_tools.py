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
This module is responsible for the filling disparity algorithms:
- thus it fills the disparity map with values estimated according to
  their neighbourhood.
"""
# pylint: disable=too-many-lines

# Third party imports
# Standard imports
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numba import njit
from rasterio.fill import fillnodata
from scipy.linalg import lstsq
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    label,
    measurements,
    median_filter,
)
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries


def fill_central_area_using_plane(
    disp_map: xr.Dataset,
    ignore_nodata: bool,
    ignore_zero_fill: bool,
    ignore_extrema: bool,
    nb_pix: int,
    percent_to_erode: float,
):
    """
    Finds central area of invalid region and estimates disparity values
    in this area according to a plan model estimation. The estimation
    of this model is done using disparity values at invalid region
    borders.
    :param disp_map: disparity map with several layers ('disp',
    'disp_msk', 'msk_invalid_sec')
    :type disp_map: 2D np.array (row, col)
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
    :return: mask of invalid region that hasn't be filled yet (original
    invalid region - central area)
    :rtype: 2D np.array (row, col)
    """

    disp_mask = np.copy(disp_map["disp_msk"].values)
    disp_values = np.copy(disp_map["disp"].values)

    # Find invalid region of interest in disp data from polygon info
    disp_inv_mask = np.copy(disp_map["msk_invalid_sec"].values)
    roi_msk = np.zeros(disp_mask.shape)
    roi_msk = np.logical_and(disp_inv_mask == 0, disp_mask == 0)

    # Generate a structuring element that will consider features
    # connected even if they touch diagonally
    struct = generate_binary_structure(2, 2)

    # Option 'ignore_nodata' adds invalid values of disp mask at roi_msk
    # invalid region borders
    if ignore_nodata:
        add_surrounding_nodata_to_roi(roi_msk, disp_values, disp_mask)

    # Selected invalid region dilation
    dilatation = binary_dilation(roi_msk, structure=struct, iterations=nb_pix)
    # dilated mask - initial mask = band of 'nb_pix' pix around roi
    roi_msk_tmp = np.logical_xor(dilatation, roi_msk)

    # Band disp values retrieval
    # Optional filter processing n°1 : ignore invalid values in band
    if ignore_zero_fill:
        initial_len = np.sum(roi_msk_tmp)
        roi_msk_tmp = np.logical_and(
            roi_msk_tmp,
            disp_map["disp_msk"].values.astype(bool),
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
        logging.info("Valeur disparité moyenne calculée : {}".format(disp_moy))
    # Definition of central area to fill using plane model
    erosion_value = define_interpolation_band_width(roi_msk, percent_to_erode)
    central_area = binary_erosion(
        roi_msk, structure=struct, iterations=erosion_value
    )

    variable_disp = calculate_disp_plane(
        band_disp_values,
        roi_msk_tmp,
        central_area,
    )

    disp_map["disp"].values[central_area] = variable_disp
    disp_map["disp_msk"].values[central_area] = 255

    # Retrieve borders that weren't filled yet
    roi_msk[central_area] = 0
    return roi_msk


def add_surrounding_nodata_to_roi(
    roi_mask: xr.Dataset,
    disp: xr.Dataset,
    disp_mask: xr.Dataset,
):
    """
    Add surounding nodata to invalidity region
    :param roi_mask: invalidity mask (values to fill)
    :type roi_mask: 2D np.array (row, col)
    :param disp: disparity values
    :type disp: 2D np.array (row, col)
    :param disp_mask: disparity values mask
    :type disp_mask: 2D np.array (row, col)
    """

    struct = generate_binary_structure(2, 2)
    all_mask = np.logical_or(roi_mask.astype(bool), ~disp_mask.astype(bool))

    # Added because zero values not included in disp_mask are present
    all_mask = np.logical_or(all_mask, disp == 0)
    labeled_msk_array, __ = label(all_mask.astype(int), structure=struct)
    label_of_interest = np.unique(labeled_msk_array[np.where(roi_mask == 1)])

    if len(label_of_interest) != 1:
        raise Exception(
            "More than one label found for ROI :\
            {}".format(
                label_of_interest
            )
        )
    roi_mask[labeled_msk_array == label_of_interest] = 1


def define_interpolation_band_width(binary_image, percentage):
    """
    Define number of pixel for later erosion operation
    :param binary_image: invalidity mask (values to fill)
    :type binary_image: 2D np.array (row, col)
    :param percentage: percentage of border compared to center region
    :type percentage: dict
    :return: pixel number to erode
    :rtype: int
    """

    # Recherche des pixels de contour de la région masquée
    contours = find_boundaries(binary_image, mode="inner")
    # Localisation du centre de la zone masquée
    # TODO: cas d'une zone bien concave --> résultat ok?
    centroid = measurements.center_of_mass(binary_image)
    # Recherche de la dist moy entre contours et centre de la zone
    coords_contours = list(zip(*np.where(contours)))
    centroid_list = (centroid,) * len(coords_contours)
    all_dist = cdist(centroid_list, coords_contours, "euclidean")
    mean_dist = np.mean(all_dist)
    erosion_value = np.round((percentage * mean_dist))
    return int(erosion_value)


def plot_function(
    data,
    fit,
):
    """
    Displays shape of plane/quadratic function used in region to fill.
    :param data: coords and value of valid disparity values
    :type data: 3D np.array (row, col)
    :param fit: Least-squares solution.
    :type fit: ndarray
    :return: plot of function used in region to fill
    :rtype: matplotlib plot
    """

    plt.figure()
    plt_ax = plt.subplot(111, projection="3d")
    plt_ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="b", alpha=0.5)
    plt_ax.set_xlabel("x")
    plt_ax.set_ylabel("y")
    plt_ax.set_zlabel("z")
    xlim = plt_ax.get_xlim()
    ylim = plt_ax.get_ylim()
    v_x, v_y = np.meshgrid(
        np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1])
    )
    val_ref = fit[0] * v_x + fit[1] * v_y + fit[2]
    plt_ax.plot_wireframe(v_x, v_y, val_ref, color="r", alpha=0.5)
    plt.show()


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
    :return: val
    :rtype: central interpolated disparity values
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
    val = np.zeros(x_to_fill.shape)
    val = list(
        map(lambda x, y: fit[0] * x + fit[1] * y + fit[2], x_to_fill, y_to_fill)
    )
    # Option d'affichage de la fonction plan
    if display:
        plot_function(data, fit)
    return val


def fill_area_borders_using_interpolation(disp_map, mask_to_fill, options):
    """
    Raster interpolation command
    :param disp_map: disparity values
    :type disp_map: 2D np.array (row, col)
    :param mask_to_fill: mask to locate disp values to fill
    :type mask_to_fill: 2D np.array (row, col)
    :param options: parameters for interpolation methods
    :type options: dict
    """
    # Copy input data - disparity values + mask with values to fill
    raster = np.copy(disp_map["disp"].values)
    # Interpolation step
    interpol_raster = make_raster_interpolation(raster, mask_to_fill, options)
    # Insertion of interpolated data into disparity map
    disp_map["disp"].values[mask_to_fill] = interpol_raster[mask_to_fill]
    disp_map["disp_msk"].values[mask_to_fill] = 255


# --------------------------------------------------------------------
#     Global functions for interpolation process
# --------------------------------------------------------------------
def make_raster_interpolation(
    raster: np.ndarray, mask: np.ndarray, options: dict
):
    """
    Raster interpolation (scipy, rasterio or pandora)
    :param raster: disparity values
    :type raster: 2D np.array (row, col)
    :param mask: invalidity mask (values to fill)
    :type mask: 2D np.array (row, col)
    :param options: parameters for interpolation methods
    :type options: dict
    :return: interpolated raster
    :rtype: 2D np.array
    """

    if options["type"] == "fillnodata":
        interpol_raster_tmp = np.copy(raster)
        interpol_raster_tmp = fillnodata(
            interpol_raster_tmp,
            mask=~mask,
            max_search_distance=options["max_search_distance"],
            smoothing_iterations=options["smoothing_iterations"],
        )
        interpol_raster = median_filter(interpol_raster_tmp, size=(3, 3))
    elif options["type"] == "pandora" and options["method"] == "mc_cnn":
        interpol_raster, __ = fill_disp_pandora(raster, mask, 16)
    elif options["type"] == "pandora" and options["method"] == "sgm":
        interpol_raster, __ = fill_disp_pandora(raster, mask, 8)
    else:
        raise Exception("Invalid interpolation type.")
    return interpol_raster


# Copied/adapted fct from pandora/validation/interpolated_disparity.py @njit()
def fill_disp_pandora(
    disp: np.ndarray, msk_fill_disp: np.ndarray, nb_directions: int
):
    """
    Interpolation of the left disparity map to fill holes.
    Interpolate invalid pixel by finding the nearest correct pixels in
    8/16 different directions and use the median of their disparities.
    ?bontar, J., & LeCun, Y. (2016). Stereo matching by training
    a convolutional neural network to compare image
    patches. The journal of machine learning research, 17(1), 2287-2318.
    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching
    and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence,
    2007, vol. 30, no 2, p. 328-341.
    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param msk_fill_disp: validity mask
    :type msk_fill_disp: 2D np.array (row, col)
    :param nb_directions: nb directions to explore
    :type nb_directions: integer
    :return: the interpolate left disparity map,
    with the validity mask update :
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    # Output disparity map and validity mask
    out_disp = np.copy(disp)
    out_msk = np.copy(msk_fill_disp)
    ncol, nrow = disp.shape
    if nb_directions == 8:
        # 8 directions : [row, col]
        dirs = np.array(
            [
                [0.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.0],
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
    elif nb_directions == 16:
        # 16 directions : [row, col]
        dirs = np.array(
            [
                [0.0, 1.0],
                [-0.5, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.5],
                [-1.0, 0.0],
                [-1.0, -0.5],
                [-1.0, -1.0],
                [-0.5, -1.0],
                [0.0, -1.0],
                [0.5, -1.0],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [0.5, 1.0],
            ]
        )
    for col in range(ncol):
        for row in range(nrow):
            if msk_fill_disp[col, row]:
                valid_neighbors = find_valid_neighbors(
                    dirs, disp, msk_fill_disp, row, col, nb_directions
                )
                # Median of the 8/16 pixels
                out_disp[col, row] = np.nanmedian(valid_neighbors)
                # Update the validity mask : Information : filled disp
                out_msk[col, row] = False
    return out_disp, out_msk


@njit()
# Copied/adapted fct from pandora/validation/interpolated_disparity.py
def find_valid_neighbors(
    dirs: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    row: int,
    col: int,
    nb_directions: int,
):
    """
    Find valid neighbors along directions
    :param dirs: directions
    :type dirs: 2D np.array (row, col)
    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param row: row current value
    :type row: int
    :param col: col current value
    :type col: int
    :param nb_directions: nb directions to explore
    :type nb_directions: int
    :return: valid neighbors
    :rtype: 2D np.array
    """
    ncol, nrow = disp.shape
    # Maximum path length
    max_path_length = max(nrow, ncol)
    # For each directions
    valid_neighbors = np.zeros(nb_directions, dtype=np.float32)
    for direction in range(nb_directions):
        # Find the first valid pixel in the current path
        for i in range(1, max_path_length):
            tmp_row = row + int(dirs[direction][0] * i)
            tmp_col = col + int(dirs[direction][1] * i)
            tmp_row = math.floor(tmp_row)
            tmp_col = math.floor(tmp_col)
            # Edge of the image reached:
            # there is no valid pixel in the current path
            if (
                (tmp_col < 0)
                | (tmp_col >= ncol)
                | (tmp_row < 0)
                | (tmp_row >= nrow)
            ):
                valid_neighbors[direction] = np.nan
                break
                # First valid pixel
            if not valid[tmp_col, tmp_row]:
                valid_neighbors[direction] = disp[tmp_col, tmp_row]
                break
    return valid_neighbors
