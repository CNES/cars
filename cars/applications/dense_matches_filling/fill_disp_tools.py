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
thus it fills the disparity map with values estimated according to
their neighbourhood.
"""
# pylint: disable=too-many-lines


import copy

# Standard imports
import logging
import math
from typing import Dict, Tuple

# Third party imports
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
from shapely import affinity
from skimage.segmentation import find_boundaries

# Cars import
from cars.applications.holes_detection import holes_detection_tools
from cars.core import constants as cst


def fill_central_area_using_plane(
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

    :return: mask of invalid region that hasn't been filled yet (original
        invalid region - central area)
    :rtype: 2D np.array (row, col)
    """

    # Generate a structuring element that will consider features
    # connected even if they touch diagonally
    struct = generate_binary_structure(2, 2)

    disp_mask = np.copy(disp_map["disp_msk"].values)
    disp_values = np.copy(disp_map["disp"].values)

    # Find invalid region of interest in disp data from polygon info
    classif_mask = holes_detection_tools.classif_to_stacked_array(
        disp_map, class_index
    )

    classif_mask_arrays, num_features = label(
        (classif_mask > 0).astype(int), structure=struct
    )

    list_roi_msk = []

    for segm in range(1, num_features + 1):
        roi_msk = classif_mask_arrays == segm

        # Create Polygon of current mask
        mask_polys = (
            holes_detection_tools.get_roi_coverage_as_poly_with_margins(
                roi_msk, row_offset=row_min, col_offset=col_min, margin=0
            )
        )
        # Clean mask polygons, remove artefacts
        cleaned_mask_poly = []
        for msk_pol in mask_polys:
            # if area > 20  : small groups to remove
            if msk_pol.area > 20:
                cleaned_mask_poly.append(msk_pol)

        if len(cleaned_mask_poly) > 1:
            raise RuntimeError("Not single polygon for current mask")

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
                add_surrounding_nodata_to_roi(roi_msk, disp_values, disp_mask)

            # Selected invalid region dilation
            dilatation = binary_dilation(
                roi_msk, structure=struct, iterations=nb_pix
            )
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
                logging.info(
                    "Valeur disparité moyenne calculée : {}".format(disp_moy)
                )

            # roi_msk can be filled with 0 if neighbours have filled mask
            if np.sum(~roi_msk) > 0:
                # Definition of central area to fill using plane model
                erosion_value = define_interpolation_band_width(
                    roi_msk, percent_to_erode
                )
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

                list_roi_msk.append(roi_msk)
    return list_roi_msk


def add_surrounding_nodata_to_roi(
    roi_mask,
    disp,
    disp_mask,
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
    modified_nan_disp_mask = np.nan_to_num(disp_mask, nan=0, posinf=0)

    all_mask = np.logical_or(
        roi_mask.astype(bool), ~modified_nan_disp_mask.astype(bool)
    )

    # Added because zero values not included in disp_mask are present
    all_mask = np.logical_or(all_mask, disp == 0)
    labeled_msk_array, __ = label(all_mask.astype(int), structure=struct)
    label_of_interest = np.unique(labeled_msk_array[np.where(roi_mask == 1)])
    if len(label_of_interest) != 1:
        logging.error(
            "More than one label found for ROI :\
            {}".format(
                label_of_interest
            )
        )

        for label_o_i in label_of_interest:
            roi_mask[labeled_msk_array == label_o_i] = 1

    else:
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
    coords_contours = list(zip(*np.where(contours)))  # noqa: B905
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
    val = np.zeros(x_to_fill.shape)
    val = list(
        map(lambda x, y: fit[0] * x + fit[1] * y + fit[2], x_to_fill, y_to_fill)
    )
    # Option d'affichage de la fonction plan
    if display:
        plot_function(data, fit)
    return val


def fill_area_borders_using_interpolation(disp_map, masks_to_fill, options):
    """
    Raster interpolation command
    :param disp_map: disparity values
    :type disp_map: 2D np.array (row, col)
    :param masks_to_fill: masks to locate disp values to fill
    :type masks_to_fill: list(2D np.array (row, col))
    :param options: parameters for interpolation methods
    :type options: dict
    """
    # Copy input data - disparity values + mask with values to fill
    raster = np.copy(disp_map["disp"].values)
    # Interpolation step
    for mask_to_fill in masks_to_fill:
        interpol_raster = make_raster_interpolation(
            raster, mask_to_fill, options
        )
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
        raise RuntimeError("Invalid interpolation type.")
    return interpol_raster


@njit()
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

    Copied/adapted fct from pandora/validation/interpolated_disparity.py

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

    Copied/adapted fct from pandora/validation/interpolated_disparity.py

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


def estimate_poly_with_disp(poly, dmin=0, dmax=0):
    """
    Estimate new polygone using disparity range

    :param poly: polygone to estimate
    :type poly: Polygon
    :param dmin: minimum disparity
    :type dmin: int
    :param dmax: maximum disparity
    :type dmax: int

    :return: polygon in disparity range
    :rtype: Polygon

    """

    new_poly = copy.copy(poly)
    for disp in range(dmin, dmax + 1):
        translated_poly = affinity.translate(poly, xoff=0.0, yoff=disp)
        new_poly = new_poly.union(translated_poly)

    return poly


def get_corresponding_holes(tile_poly, holes_poly_list):
    """
    Get list of holes situated in tile

    :param tile_poly: envelop of tile
    :type tile_poly: Polygon
    :param holes_poly_list: envelop of holes
    :type holes_poly_list: list(Polygon)


    :return: list of holes envelops
    :rtype: list(Polygon)

    """

    corresponding_holes = []
    for poly in holes_poly_list:
        if tile_poly.intersects(poly):
            corresponding_holes.append(poly)

    return corresponding_holes


def get_corresponding_tiles(tiles_polygones, corresponding_holes, epi_disp_map):
    """
    Get list of tiles intersecting with holes

    :param tiles_polygones: envelop of tiles
    :type tiles_polygones: list(Polygon)
    :param corresponding_holes: envelop of holes
    :type corresponding_holes: list(Polygon)
    :param epi_disp_map: disparity map cars dataset
    :type epi_disp_map: CarsDataset


    :return: list of tiles to use (window, overlap, xr.Dataset)
    :rtype: list(tuple)

    """
    corresponding_tiles_row_col = []
    corresponding_tiles = []

    for key_tile, poly_tile in tiles_polygones.items():
        for poly_hole in corresponding_holes:
            if poly_tile.intersects(poly_hole):
                if key_tile not in corresponding_tiles_row_col:
                    corresponding_tiles_row_col.append(key_tile)

    for row, col in corresponding_tiles_row_col:
        corresponding_tiles.append(
            (
                epi_disp_map.tiling_grid[row, col],
                epi_disp_map.overlaps[row, col],
                epi_disp_map[row, col],
            )
        )

    return corresponding_tiles


def get_polygons_from_cars_ds(cars_ds):
    """
    Get the holes envelops computed in holes detection application
    cars_ds must contain dicts, and not delayed.
    This function must be called after an orchestrator.breakpoint()

    :param cars_ds: holes cars dataset
    :type cars_ds: CarsDataset


    :return: list of holes
    :rtype: list(Polygon)
    """

    list_poly = []

    if cars_ds is not None:
        for row in range(cars_ds.shape[0]):
            for col in range(cars_ds.shape[1]):
                if cars_ds[row, col] is not None:
                    list_poly += cars_ds[row, col].data["list_bbox"]

    return list_poly


def merge_intersecting_polygones(list_poly):
    """
    Merge polygons that intersects each other

    :param list_poly: list of holes
    :type list_poly: list(Polygon)


    :return: list of holes
    :rtype: list(Polygon)
    """

    new_list_poly = list_poly

    merged_list = []

    while len(new_list_poly) > 0:
        current_poly = new_list_poly[0]

        new_poly = current_poly
        to_delete = [0]

        for element_id in range(1, len(new_list_poly)):
            if new_poly.intersects(new_list_poly[element_id]):
                # Delete from list
                to_delete.append(element_id)
                # Merge with current
                new_poly = new_poly.union(new_list_poly[element_id])

        # Add new poly to merged list
        merged_list.append(new_poly)

        # Delete element
        for _ in range(len(to_delete)):
            # Start with last ones
            pos = to_delete.pop()
            new_list_poly.pop(pos)

    return merged_list


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
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
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

    :return: overloaded configuration
    :rtype: dict

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
    )

    fill_area_borders_using_interpolation(
        disp_map,
        border_region,
        interp_options,
    )


def fill_disp_using_zero_padding(
    disp_map: xr.Dataset,
    class_index,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Fill disparity map holes

    :param disp_map: disparity map
    :type disp_map: xr.Dataset
    :param class_index: class index according to the classification tag
    :type class_index: int

    :return: overloaded configuration
    :rtype: dict

    """
    # Generate a structuring element that will consider features
    # connected even if they touch diagonally

    # get index of the application class config
    # according the coords classif band
    if cst.BAND_CLASSIF in disp_map.coords:
        # get index for each band classification
        stack_index = holes_detection_tools.classif_to_stacked_array(
            disp_map, class_index
        )
        # set disparity value to zero where the class is
        # non zero value and masked region
        disp_map["disp"].values[stack_index] = 0
        disp_map["disp_msk"].values[stack_index] = 255
