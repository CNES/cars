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

import copy

# Standard imports
import logging
import math
from typing import Tuple

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rasterio.fill import fillnodata
from scipy.ndimage import (
    generate_binary_structure,
    label,
    measurements,
    median_filter,
)
from scipy.spatial.distance import cdist
from shapely import affinity
from skimage.segmentation import find_boundaries

# Cars import
from cars.core import constants as cst

from .cpp import dense_match_filling_cpp


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
    all_mask = np.logical_or(all_mask, disp == np.nan)
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


def add_empty_filling_band(
    output_dataset: xr.Dataset,
    filling_types: list,
):
    """
    Add filling attribute to dataset or band to filling attribute
    if it already exists

    :param output_dataset: output dataset
    :param filling: input mask of filled pixels
    :param band_filling: type of filling (zero padding or plane)

    """
    nb_band = len(filling_types)
    nb_row = len(output_dataset.coords[cst.ROW])
    nb_col = len(output_dataset.coords[cst.COL])
    filling = np.zeros((nb_band, nb_row, nb_col), dtype=bool)
    filling = xr.Dataset(
        data_vars={
            cst.EPI_FILLING: ([cst.BAND_FILLING, cst.ROW, cst.COL], filling)
        },
        coords={
            cst.BAND_FILLING: filling_types,
            cst.ROW: output_dataset.coords[cst.ROW],
            cst.COL: output_dataset.coords[cst.COL],
        },
    )
    # Add band to EPI_FILLING attribute or create the attribute
    return xr.merge([output_dataset, filling])


def update_filling(
    output_dataset: xr.Dataset,
    filling: np.ndarray = None,
    filling_type: str = None,
):
    """
    Update filling attribute of dataset with an additional mask

    :param output_dataset: output dataset
    :param filling: input mask of filled pixels
    :param band_filling: type of filling (zero padding or plane)

    """
    # Select accurate band of output according to the type of filling
    filling_type = {cst.BAND_FILLING: filling_type}
    # Add True values from inputmask to output accurate band
    filling = filling.astype(bool)
    output_dataset[cst.EPI_FILLING].sel(**filling_type).values[filling] = True


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


def fill_disp_pandora(
    disp: np.ndarray, msk_fill_disp: np.ndarray, nb_directions: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    return dense_match_filling_cpp.fill_disp_pandora(
        disp, msk_fill_disp, nb_directions
    )


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

    dmin = int(math.floor(dmin))
    dmax = int(math.ceil(dmax))

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
