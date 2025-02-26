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
This module contains functions used in outlier removal
"""

# Standard imports
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import outlier_filter  # pylint:disable=E0401
import pandas
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from cars.applications.point_cloud_fusion.point_cloud_tools import filter_cloud

# CARS imports
from cars.core import constants as cst
from cars.core import projection

# ##### Small component filtering ######


def small_component_filtering(
    cloud: pandas.DataFrame,
    connection_val: float,
    nb_pts_threshold: int,
    clusters_distance_threshold: float = None,
    filtered_elt_pos: bool = False,
) -> Tuple[pandas.DataFrame, Union[None, pandas.DataFrame]]:
    """
    Filter point cloud to remove small clusters of points
    (see the detect_small_components function).

    :param cloud: combined cloud
        as returned by the create_combined_cloud function
    :param connection_val: distance to use
        to consider that two points are connected
    :param nb_pts_threshold: number of points to use
        to identify small clusters to filter
    :param clusters_distance_threshold: distance to use
        to consider if two points clusters are far from each other or not
        (set to None to deactivate this level of filtering)
    :param filtered_elt_pos: if filtered_elt_pos is set to True,
        the removed points positions in their original
        epipolar images are returned, otherwise it is set to None
    :return: Tuple made of the filtered cloud and
        the removed elements positions in their epipolar images
    """

    clusters_distance_threshold_float = (
        np.nan
        if clusters_distance_threshold is None
        else clusters_distance_threshold
    )

    index_elt_to_remove = outlier_filter.pc_small_component_outlier_filtering(
        cloud.loc[:, cst.X].values,
        cloud.loc[:, cst.Y].values,
        cloud.loc[:, cst.Z].values,
        radius=connection_val,
        min_cluster_size=nb_pts_threshold,
        clusters_distance_threshold=clusters_distance_threshold_float,
    )

    return filter_cloud(cloud, index_elt_to_remove, filtered_elt_pos)


def detect_small_components(
    cloud_xyz: np.ndarray,
    connection_val: float,
    nb_pts_threshold: int,
    clusters_distance_threshold: float = None,
) -> List[int]:
    """
    Determine the indexes of the points of cloud_xyz to filter.
    The clusters are made of 'connected' points
    (2 connected points have a distance smaller than connection_val)

    The removed clusters are composed of less than nb_pts_threshold points and
    are also far from other clusters
    (points are further than clusters_distance_threshold).

    If clusters_distance_threshold is set to None, all the clusters that are
    composed of less than nb_pts_threshold points are filtered.

    :param cloud_xyz: points kdTree
    :param connection_val: distance to use
        to consider that two points are connected
    :param nb_pts_threshold: number of points to use
        to identify small clusters to filter
    :param clusters_distance_threshold: distance to use
        to consider if two points clusters are far from each other or not
        (set to None to deactivate this level of filtering)
    :return: list of the points to filter indexes
    """
    cloud_tree = cKDTree(cloud_xyz)

    # extract connected components
    processed = [False] * len(cloud_xyz)
    connected_components = []
    for idx, xyz_point in enumerate(cloud_xyz):
        # if point has already been added to a cluster
        if processed[idx]:
            continue

        # get point neighbors
        neighbors_list = cloud_tree.query_ball_point(xyz_point, connection_val)

        # add them to the current cluster
        seed = []
        seed.extend(neighbors_list)
        for neigh_idx in neighbors_list:
            processed[neigh_idx] = True

        # iteratively add all the neighbors of the points
        # which were added to the current cluster (if there are some)
        while len(neighbors_list) != 0:
            all_neighbors = cloud_tree.query_ball_point(
                cloud_xyz[neighbors_list], connection_val
            )

            # flatten neighbors
            new_neighbors = []
            for neighbor_item in all_neighbors:
                new_neighbors.extend(neighbor_item)

            # retrieve only new neighbors
            neighbors_list = list(set(new_neighbors) - set(seed))

            # add them to the current cluster
            seed.extend(neighbors_list)
            for neigh_idx in neighbors_list:
                processed[neigh_idx] = True

        connected_components.append(seed)

    # determine clusters to remove
    cluster_to_remove = []
    for _, connected_components_item in enumerate(connected_components):
        if len(connected_components_item) < nb_pts_threshold:
            if clusters_distance_threshold is not None:
                # search if the current cluster has any neighbors
                # in the clusters_distance_threshold radius
                all_neighbors = cloud_tree.query_ball_point(
                    cloud_xyz[connected_components_item],
                    clusters_distance_threshold,
                )

                # flatten neighbors
                new_neighbors = []
                for neighbor_item in all_neighbors:
                    new_neighbors.extend(neighbor_item)

                # retrieve only new neighbors
                neighbors_list = list(
                    set(new_neighbors) - set(connected_components_item)
                )

                # if there are no new neighbors, the cluster will be removed
                if len(neighbors_list) == 0:
                    cluster_to_remove.extend(connected_components_item)
            else:
                cluster_to_remove.extend(connected_components_item)

    return cluster_to_remove


# ##### statistical filtering ######


def statistical_outlier_filtering(
    cloud: pandas.DataFrame,
    k: int,
    dev_factor: float,
    use_median: bool = False,
    filtered_elt_pos: bool = False,
) -> Tuple[pandas.DataFrame, Union[None, pandas.DataFrame]]:
    """
    Filter point cloud to remove statistical outliers
    (see the detect_statistical_outliers function).

    :param cloud: combined cloud
        as returned by the create_combined_cloud function
    :param k: number of neighbors
    :param dev_factor: multiplication factor of deviation used
        to compute the distance threshold
    :param use_median: choice of statistical measure used to filter
    :param filtered_elt_pos: if filtered_elt_pos is set to True,
        the removed points positions in their original
        epipolar images are returned, otherwise it is set to None
    :return: Tuple made of the filtered cloud and
        the removed elements positions in their epipolar images
    """

    index_elt_to_remove = outlier_filter.pc_statistical_outlier_filtering(
        cloud.loc[:, cst.X].values,
        cloud.loc[:, cst.Y].values,
        cloud.loc[:, cst.Z].values,
        dev_factor=dev_factor,
        k=k,
        use_median=use_median,
    )

    return filter_cloud(cloud, index_elt_to_remove, filtered_elt_pos)


def detect_statistical_outliers(
    cloud_xyz: np.ndarray, k: int, dev_factor: float, use_median: bool
) -> List[int]:
    """
    Determine the indexes of the points of cloud_xyz to filter.
    The removed points have mean distances with their k nearest neighbors
    that are greater than a distance threshold (dist_thresh).

    This threshold is computed from the mean (or median) and
    standard deviation (or interquartile range) of all the points mean
    distances with their k nearest neighbors:

        (1) dist_thresh = mean_distances + dev_factor * std_distances
        or
        (2) dist_thresh = median_distances + dev_factor * iqr_distances

    :param cloud_xyz: points kdTree
    :param k: number of neighbors
    :param dev_factor: multiplication factor of deviation used
        to compute the distance threshold
    :param use_median: if True formula (2) is used for threshold, else
        formula (1)
    :return: list of the points to filter indexes
    """
    # compute for each points, all the distances to their k neighbors
    cloud_tree = cKDTree(cloud_xyz)
    neighbors_distances, _ = cloud_tree.query(cloud_xyz, k + 1)

    # Compute the mean of those distances for each point
    # Mean is not used directly as each line
    #           contained the distance value to the point itself
    mean_neighbors_distances = np.sum(neighbors_distances, axis=1)
    mean_neighbors_distances /= k

    if use_median:
        # compute median and interquartile range of those mean distances
        # for the whole point cloud
        median_distances = np.median(mean_neighbors_distances)
        iqr_distances = np.percentile(
            mean_neighbors_distances, 75
        ) - np.percentile(mean_neighbors_distances, 25)

        # compute distance threshold and
        # apply it to determine which points will be removed
        dist_thresh = median_distances + dev_factor * iqr_distances
    else:
        # compute median and interquartile range of those mean distances
        # for the whole point cloud
        mean_distances = np.mean(mean_neighbors_distances)
        std_distances = np.std(mean_neighbors_distances)
        # compute distance threshold and
        # apply it to determine which points will be removed
        dist_thresh = mean_distances + dev_factor * std_distances

    points_to_remove = np.argwhere(mean_neighbors_distances > dist_thresh)

    # flatten points_to_remove
    detected_points = []
    for removed_point in points_to_remove:
        detected_points.extend(removed_point)

    return detected_points


def epipolar_small_components(
    cloud,
    epsg,
    min_cluster_size=15,
    radius=1.0,
    half_window_size=5,
    clusters_distance_threshold=np.nan,
):
    """
    Filter outliers using the small components method in epipolar geometry

    :param epipolar_ds: epipolar dataset to filter
    :type epipolar_ds: xr.Dataset
    :param epsg: epsg code of the CRS used to compute distances
    :type epsg: int
    :param statistical_k: k
    :type statistical_k: int
    :param std_dev_factor: std factor
    :type std_dev_factor: float
    :param half_window_size: use median and quartile instead of mean and std
    :type half_window_size: int
    :param use_median: use median and quartile instead of mean and std
    :type use_median: bool

    :return: filtered dataset
    :rtype:  xr.Dataset

    """

    projection.point_cloud_conversion_dataset(cloud, epsg)

    if clusters_distance_threshold is None:
        clusters_distance_threshold = np.nan

    outlier_filter.epipolar_small_component_outlier_filtering(
        cloud[cst.X],
        cloud[cst.Y],
        cloud[cst.Z],
        min_cluster_size,
        radius,
        half_window_size,
        clusters_distance_threshold,
    )

    projection.point_cloud_conversion_dataset(cloud, 4326)

    return cloud


def epipolar_statistical_filtering(
    epipolar_ds,
    epsg,
    k=15,
    dev_factor=1.0,
    half_window_size=5,
    use_median=False,
):
    """
    Filter outliers using the statistical method in epipolar geometry

    :param epipolar_ds: epipolar dataset to filter
    :type epipolar_ds: xr.Dataset
    :param epsg: epsg code of the CRS used to compute distances
    :type epsg: int
    :param statistical_k: k
    :type statistical_k: int
    :param std_dev_factor: std factor
    :type std_dev_factor: float
    :param half_window_size: use median and quartile instead of mean and std
    :type half_window_size: int
    :param use_median: use median and quartile instead of mean and std
    :type use_median: bool

    :return: filtered dataset
    :rtype:  xr.Dataset

    """

    projection.point_cloud_conversion_dataset(epipolar_ds, epsg)

    if not np.all(np.isnan(epipolar_ds[cst.Z])):

        outlier_filter.epipolar_statistical_outlier_filtering(
            epipolar_ds[cst.X],
            epipolar_ds[cst.Y],
            epipolar_ds[cst.Z],
            k,
            half_window_size,
            dev_factor,
            use_median,
        )

    projection.point_cloud_conversion_dataset(epipolar_ds, 4326)

    return epipolar_ds
