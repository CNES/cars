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
This module contains functions used in outlier removing
"""

# Standard imports
import logging
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas
import xarray as xr
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

# CARS imports
from cars.core import constants as cst

# ##### Small components filtering ######


def small_components_filtering(
    cloud: pandas.DataFrame,
    connection_val: float,
    nb_pts_threshold: int,
    clusters_distance_threshold: float = None,
    filtered_elt_pos: bool = False,
) -> Tuple[pandas.DataFrame, Union[None, pandas.DataFrame]]:
    """
    Filter points cloud to remove small clusters of points
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
    cloud_xyz = cloud.loc[:, [cst.X, cst.Y, cst.Z]].values
    index_elt_to_remove = detect_small_components(
        cloud_xyz, connection_val, nb_pts_threshold, clusters_distance_threshold
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


def statistical_outliers_filtering(
    cloud: pandas.DataFrame,
    k: int,
    std_factor: float,
    filtered_elt_pos: bool = False,
) -> Tuple[pandas.DataFrame, Union[None, pandas.DataFrame]]:
    """
    Filter points cloud to remove statistical outliers
    (see the detect_statistical_outliers function).

    :param cloud: combined cloud
        as returned by the create_combined_cloud function
    :param k: number of neighbors
    :param std_factor: multiplication factor to use
        to compute the distance threshold
    :param filtered_elt_pos: if filtered_elt_pos is set to True,
        the removed points positions in their original
        epipolar images are returned, otherwise it is set to None
    :return: Tuple made of the filtered cloud and
        the removed elements positions in their epipolar images
    """
    cloud_xyz = cloud.loc[:, [cst.X, cst.Y, cst.Z]].values
    index_elt_to_remove = detect_statistical_outliers(cloud_xyz, k, std_factor)

    return filter_cloud(cloud, index_elt_to_remove, filtered_elt_pos)


def detect_statistical_outliers(
    cloud_xyz: np.ndarray, k: int, std_factor: float = 3.0
) -> List[int]:
    """
    Determine the indexes of the points of cloud_xyz to filter.
    The removed points have mean distances with their k nearest neighbors
    that are greater than a distance threshold (dist_thresh).

    This threshold is computed from the mean (mean_distances) and
    standard deviation (stddev_distances) of all the points mean distances
    with their k nearest neighbors:

        dist_thresh = mean_distances + std_factor * stddev_distances

    :param cloud_xyz: points kdTree
    :param k: number of neighbors
    :param std_factor: multiplication factor to use
        to compute the distance threshold
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

    # compute mean and standard deviation of those mean distances
    #           for the whole point cloud
    mean_distances = np.mean(mean_neighbors_distances)
    stddev_distances = np.std(mean_neighbors_distances)

    # compute distance threshold and
    # apply it to determine which points will be removed
    dist_thresh = mean_distances + std_factor * stddev_distances
    points_to_remove = np.argwhere(mean_neighbors_distances > dist_thresh)

    # flatten points_to_remove
    detected_points = []
    for removed_point in points_to_remove:
        detected_points.extend(removed_point)

    return detected_points


# ##### common filtering tools ######


def filter_cloud(
    cloud: pandas.DataFrame,
    index_elt_to_remove: List[int],
    filtered_elt_pos: bool = False,
) -> Tuple[pandas.DataFrame, Union[None, pandas.DataFrame]]:
    """
    Filter all points of the cloud DataFrame
    which index is in the index_elt_to_remove list.

    If filtered_elt_pos is set to True, the information of the removed elements
    positions in their original epipolar images are returned.

    To do so the cloud DataFrame has to be build
    with the 'with_coords' option activated.

    :param cloud: combined cloud
        as returned by the create_combined_cloud function
    :param index_elt_to_remove: indexes of lines
        to filter in the cloud DataFrame
    :param filtered_elt_pos: if filtered_elt_pos is set to True,
        the removed points positions in their original epipolar images are
        returned, otherwise it is set to None
    :return: Tuple composed of the filtered cloud DataFrame and
        the filtered elements epipolar position information
        (or None for the latter if filtered_elt_pos is set to False
        or if the cloud Dataframe has not been build with with_coords option)
    """
    if filtered_elt_pos and not (
        cst.POINTS_CLOUD_COORD_EPI_GEOM_I in cloud.columns
        and cst.POINTS_CLOUD_COORD_EPI_GEOM_J in cloud.columns
        and cst.POINTS_CLOUD_ID_IM_EPI in cloud.columns
    ):
        logging.warning(
            "In filter_cloud: the filtered_elt_pos has been activated but "
            "the cloud Datafram has not been build with option with_coords. "
            "The positions cannot be retrieved."
        )
        filtered_elt_pos = False

    # retrieve removed points position in their original epipolar images
    if filtered_elt_pos:
        labels = [
            cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
            cst.POINTS_CLOUD_COORD_EPI_GEOM_J,
            cst.POINTS_CLOUD_ID_IM_EPI,
        ]

        removed_elt_pos_infos = cloud.loc[
            cloud.index.values[index_elt_to_remove], labels
        ].values

        removed_elt_pos_infos = pandas.DataFrame(
            removed_elt_pos_infos, columns=labels
        )
    else:
        removed_elt_pos_infos = None

    # remove points from the cloud
    cloud = cloud.drop(index=cloud.index.values[index_elt_to_remove])

    return cloud, removed_elt_pos_infos


def add_cloud_filtering_msk(
    clouds_list: List[xr.Dataset],
    elt_pos_infos: pandas.DataFrame,
    mask_label: str,
    mask_value: int = 255,
):
    """
    Add a uint16 mask labeled 'mask_label' to the clouds in clouds_list.
    (in-line function)

    :param clouds_list: Input list of clouds
    :param elt_pos_infos: pandas dataframe
        composed of cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
        cst.POINTS_CLOUD_COORD_EPI_GEOM_J, cst.POINTS_CLOUD_ID_IM_EPI columns
        as computed in the create_combined_cloud function.
        Those information are used to retrieve the point position
        in its original epipolar image.
    :param mask_label: label to give to the mask in the datasets
    :param mask_value: filtered elements value in the mask
    """

    # Verify that the elt_pos_infos is consistent
    if (
        elt_pos_infos is None
        or cst.POINTS_CLOUD_COORD_EPI_GEOM_I not in elt_pos_infos.columns
        or cst.POINTS_CLOUD_COORD_EPI_GEOM_J not in elt_pos_infos.columns
        or cst.POINTS_CLOUD_ID_IM_EPI not in elt_pos_infos.columns
    ):
        logging.warning(
            "Cannot generate filtered elements mask, "
            "no information about the point's"
            " original position in the epipolar image is given"
        )

    else:
        elt_index = elt_pos_infos.loc[:, cst.POINTS_CLOUD_ID_IM_EPI].to_numpy()

        min_elt_index = np.min(elt_index)
        max_elt_index = np.max(elt_index)

        if min_elt_index < 0 or max_elt_index > len(clouds_list) - 1:
            raise RuntimeError(
                "Index indicated in the elt_pos_infos pandas. "
                "DataFrame is not coherent with the clouds list given in input"
            )

        # create and add mask to each element of clouds_list
        for cloud_idx, cloud_item in enumerate(clouds_list):
            if mask_label not in cloud_item:
                nb_row = cloud_item.coords[cst.ROW].data.shape[0]
                nb_col = cloud_item.coords[cst.COL].data.shape[0]
                msk = np.zeros((nb_row, nb_col), dtype=np.uint16)
            else:
                msk = cloud_item[mask_label].values

            cur_elt_index = np.argwhere(elt_index == cloud_idx)

            for elt_pos in range(cur_elt_index.shape[0]):
                i = int(
                    elt_pos_infos.loc[
                        cur_elt_index[elt_pos],
                        cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
                    ].iat[0]
                )
                j = int(
                    elt_pos_infos.loc[
                        cur_elt_index[elt_pos],
                        cst.POINTS_CLOUD_COORD_EPI_GEOM_J,
                    ].iat[0]
                )

                try:
                    msk[i, j] = mask_value
                except Exception as index_error:
                    raise RuntimeError(
                        "Point at location ({},{}) is not accessible "
                        "in an image of size ({},{})".format(
                            i, j, msk.shape[0], msk.shape[1]
                        )
                    ) from index_error

            cloud_item[mask_label] = ([cst.ROW, cst.COL], msk)
