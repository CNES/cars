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
epipolar outlier filtering module:
contains functions to filter point cloud in epipolar geometry
"""

import time

import numpy as np
import pyproj

from cars.core import constants as cst


def epipolar_knn(x_coords, y_coords, z_coords, k, idx, window_half_size):
    """
    find K neighbors of input point, limiting the search to a window in the
    epipolar image
    """
    start_row = max(0, idx[0] - window_half_size)
    end_row = min(x_coords.shape[0], idx[0] + window_half_size)
    start_col = max(0, idx[1] - window_half_size)
    end_col = min(x_coords.shape[1], idx[1] + window_half_size)

    x_extract = x_coords[start_row:end_row, start_col:end_col]
    y_extract = y_coords[start_row:end_row, start_col:end_col]
    z_extract = z_coords[start_row:end_row, start_col:end_col]

    x_ref = x_coords[idx[0], idx[1]]
    y_ref = y_coords[idx[0], idx[1]]
    z_ref = z_coords[idx[0], idx[1]]

    squared_euclidian_distances = (
        (x_extract - x_ref) ** 2
        + (y_extract - y_ref) ** 2
        + (z_extract - z_ref) ** 2
    )

    ksmallest = np.argpartition(squared_euclidian_distances, k, axis=None)[:k]

    k_distances = np.sqrt(squared_euclidian_distances.flatten()[ksmallest])

    # Convert 1D indices to 2D
    k_row = ksmallest // (end_row - start_row) + start_row
    k_col = ksmallest % (end_col - start_col) + start_col

    return k_row, k_col, k_distances


def epipolar_neighbors_in_ball(
    x_coords, y_coords, z_coords, radius, idx, window_half_size
):
    """
    find all point in a radius around the input point, limiting the search to a
    window in the epipolar image
    """

    sq_radius = radius * radius

    start_row = max(0, idx[0] - window_half_size)
    end_row = min(x_coords.shape[0], idx[0] + window_half_size)
    start_col = max(0, idx[1] - window_half_size)
    end_col = min(x_coords.shape[1], idx[1] + window_half_size)

    x_extract = x_coords[start_row:end_row, start_col:end_col]
    y_extract = y_coords[start_row:end_row, start_col:end_col]
    z_extract = z_coords[start_row:end_row, start_col:end_col]

    x_ref = x_coords[idx[0], idx[1]]
    y_ref = y_coords[idx[0], idx[1]]
    z_ref = z_coords[idx[0], idx[1]]

    squared_euclidian_distances = (
        (x_extract - x_ref) ** 2
        + (y_extract - y_ref) ** 2
        + (z_extract - z_ref) ** 2
    )

    neighbors = np.where(squared_euclidian_distances < sq_radius)

    k_row = neighbors[0] + start_row
    k_col = neighbors[1] + start_col

    return (k_row, k_col)


def statistical_filtering(x_coords, y_coords, z_coords, k, dev_factor):
    """
    Statistical outlier filtering
    """
    start = time.time()

    mean_neighbors_distances = np.zeros(x_coords.shape)

    for idx, _ in np.ndenumerate(x_coords):
        _, _, distances = epipolar_knn(x_coords, y_coords, z_coords, k, idx, 10)
        mean_neighbors_distances[idx] = np.mean(distances)

    mean_distances = np.nanmean(mean_neighbors_distances)
    std_distances = np.nanstd(mean_neighbors_distances)
    # compute distance threshold and
    # apply it to determine which points will be removed
    dist_thresh = mean_distances + dev_factor * std_distances

    points_to_remove = np.where(mean_neighbors_distances > dist_thresh)

    end = time.time()
    print(f"statistical_filtering duration: {end - start}")
    return points_to_remove, mean_neighbors_distances


def small_component_filtering(
    x_coords, y_coords, z_coords, radius, num_elem=15
):
    """
    Small component filtering
    """
    start = time.time()
    visited_pixels = np.zeros(x_coords.shape, dtype=bool)

    points_to_remove = []

    for idx, value in np.ndenumerate(x_coords):
        if np.isnan(value):
            continue
        cluster = [idx]
        neighbors_row, neighbors_col = epipolar_neighbors_in_ball(
            x_coords, y_coords, z_coords, radius, idx, 5
        )
        neighbors = set(zip(neighbors_row, neighbors_col, strict=True))

        if visited_pixels[idx]:
            continue
        visited_pixels[idx] = True
        while neighbors:
            current_idx = neighbors.pop()
            if visited_pixels[current_idx]:
                continue
            cluster += [current_idx]
            neighbors_row, neighbors_col = epipolar_neighbors_in_ball(
                x_coords, y_coords, z_coords, 2, current_idx, 5
            )
            visited_pixels[current_idx] = True
            neighbors.update(
                [
                    (row, col)
                    for row, col in zip(
                        neighbors_row, neighbors_col, strict=True
                    )
                    if not visited_pixels[row, col]
                ]
            )
        if len(cluster) < num_elem:
            points_to_remove += cluster

    points_to_remove_row = [elem[0] for elem in points_to_remove]
    points_to_remove_col = [elem[1] for elem in points_to_remove]
    end = time.time()
    print(f"small_component_filtering duration: {end - start}")
    return (points_to_remove_row, points_to_remove_col)


def filter_pc(points, method=None):
    """
    Outlier filtering in epipolar geometry
    """

    start = time.time()

    # hard code UTM 36N for Gizeh for now
    transformer = pyproj.Transformer.from_crs(4326, 32636)
    # X-Y inversion required because WGS84 is lat first ?
    # pylint: disable=unpacking-non-sequence
    x_utm, y_utm = transformer.transform(
        points[cst.Y].values, points[cst.X].values
    )

    end = time.time()
    print(f"projection from geo to UTM duration: {end - start}")

    # Debug: replace lon/lat by UTM in depth map (but will probably crash later)
    # points[cst.X] = ([cst.ROW, cst.COL], x_utm)
    # points[cst.Y] =  ([cst.ROW, cst.COL], y_utm)

    if method == "statistical":
        points_to_remove, distances = statistical_filtering(
            x_utm, y_utm, points[cst.Z].values, 15, 1
        )
    elif method == "small_components":
        points_to_remove = small_component_filtering(
            x_utm, y_utm, points[cst.Z].values, 2
        )
        distances = np.zeros(x_utm.shape)
    else:
        print("warning: invalid method selected")
        points_to_remove = []
        distances = np.zeros(x_utm.shape)

    # print(f"points_to_remove {points_to_remove}")

    outliers = np.zeros(x_utm.shape)
    outliers[points_to_remove] = 1

    points[cst.X].values[points_to_remove] = np.nan
    points[cst.Y].values[points_to_remove] = np.nan
    points[cst.Z].values[points_to_remove] = np.nan

    # offset using geoid height
    return outliers, distances
