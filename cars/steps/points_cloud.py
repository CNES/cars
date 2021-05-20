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
This module is responsible for the transition between triangulation and
rasterization steps
"""

# Standard imports
import logging
from collections import namedtuple
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas
import xarray as xr
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

# CARS imports
from cars.core import constants as cst
from cars.core import projection


def create_combined_cloud(  # noqa: C901
    cloud_list: List[xr.Dataset],
    dsm_epsg: int,
    color_list: List[xr.Dataset] = None,
    resolution: float = None,
    xstart: float = None,
    ystart: float = None,
    xsize: int = None,
    ysize: int = None,
    on_ground_margin: int = 0,
    epipolar_border_margin: int = 0,
    radius: float = 1,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        * if no colors in input and no mask data present in cloud_list datasets:
            >>> labels=[cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z]

            The combined cloud has x, y, z columns along with 'valid data' one.
            The valid data is a mask set to True if the data
            are not on the epipolar image margin (epipolar_border_margin),
             otherwise it is set to False.

        * if no colors in input and mask data present in cloud_list datasets:
            >>> labels=[cst.POINTS_CLOUD_VALID_DATA,
            >>>         cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK]

            The mask values are added to the dataframe.

        * if colors are set in input and mask data are present
            in the cloud_list datasets:
            >>> labels=[cst.POINTS_CLOUD_VALID_DATA,
            >>>         cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"0",
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"1",
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"2"]

            Color channels information are added to the dataframe.

        * if colors in input, mask data present in the cloud_list datasets and
            the with_coords option is activated:
            >>> labels=[cst.POINTS_CLOUD_VALID_DATA,
            >>>         cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"0",
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"1",
            >>>         cst.POINTS_CLOUD_CLR_KEY_ROOT+"2"
            >>>         cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
            >>>         cst.POINTS_CLOUD_COORD_EPI_GEOM_J,
            >>>         cst.POINTS_CLOUD_IDX_IM_EPI]

            The pixel position of the xyz point in the original epipolar
            image (coord_epi_geom_i, coord_epi_geom_j) are added
            to the dataframe along with the index of its original cloud
            in the cloud_list input.

    :raise Exception: if a color_list is set
        but does not have the same length as the cloud list

    :param cloud_list: list of cloud points to rasterize
    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param color_list: Additional list of images
        with bands to rasterize (same size as cloud_list), or None
    :param resolution: Resolution of rasterized cells, in cloud CRS units
        (if None, the whole clouds are combined)
    :param xstart: xstart of the rasterization grid
        (if None, the whole clouds are combined)
    :param ystart: ystart of the rasterization grid
        (if None, the whole clouds are combined)
    :param xsize: xsize of the rasterization grid
        (if None, the whole clouds are combined)
    :param ysize: ysize of the rasterization grid
        (if None, the whole clouds are combined)
    :param on_ground_margin: Margin added to the rasterization grid
        (default value: 0)
    :param epipolar_border_margin: Margin used
        to invalidate cells too close to epipolar border. (default value: 0)
    :param radius: Radius for hole filling
        (if None, the whole clouds are combined).
    :param with_coords: Option enabling the adding to the combined cloud
        of information of each point to retrieve their positions
        in the original epipolar images
    :return: Tuple formed with the combined clouds and color
        in a single pandas dataframe and the epsg code
    """
    worker_logger = logging.getLogger("distributed.worker")

    # check input data consistency
    if color_list is not None and len(cloud_list) != len(color_list):
        raise Exception("There shall be as many cloud elements as color ones")

    epsg = None
    for cloud_list_item in cloud_list:
        if epsg is None:
            epsg = int(cloud_list_item.attrs[cst.EPSG])
        elif int(cloud_list_item.attrs[cst.EPSG]) != epsg:
            worker_logger.error(
                "All points clouds do not have the same epsg code"
            )

    # compute margin/roi and final number of data to add to the combined cloud
    roi = (
        resolution is not None
        and xstart is not None
        and ystart is not None
        and xsize is not None
        and ysize is not None
    )
    if roi:
        total_margin = (on_ground_margin + radius + 1) * resolution
        xend = xstart + (xsize + 1) * resolution
        yend = ystart - (ysize + 1) * resolution

    nb_data = [cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z]

    # check if the input mask values are present in the dataset
    nb_data_msk = 0
    for cloud_list_item in cloud_list:
        ds_values_list = [key for key, _ in cloud_list_item.items()]
        if cst.POINTS_CLOUD_MSK in ds_values_list:
            nb_data.append(cst.POINTS_CLOUD_MSK)
            nb_data_msk = 1
            break

    if color_list is not None:
        clr_im = color_list[0].im.values
        nb_band_clr = clr_im.shape[0]
        list_clr = [
            "{}{}".format(cst.POINTS_CLOUD_CLR_KEY_ROOT, i)
            for i in range(nb_band_clr)
        ]
        nb_data.extend(list_clr)
    else:
        nb_band_clr = 0

    if with_coords:
        nb_data.extend(
            [
                cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
                cst.POINTS_CLOUD_COORD_EPI_GEOM_J,
                cst.POINTS_CLOUD_IDX_IM_EPI,
            ]
        )

    # iterate trough input clouds
    cloud = np.zeros((0, len(nb_data)), dtype=np.float64)
    nb_points = 0
    for cloud_list_idx, cloud_list_item in enumerate(cloud_list):
        full_x = cloud_list_item[cst.X].values
        full_y = cloud_list_item[cst.Y].values
        full_z = cloud_list_item[cst.Z].values

        # get mask of points inside the roi (plus margins)
        if roi:

            # if the points clouds are not in the same referential as the roi,
            # it is converted using the dsm_epsg
            if epsg != dsm_epsg:
                (
                    full_x,
                    full_y,
                ) = projection.get_converted_xy_np_arrays_from_dataset(
                    cloud_list_item, dsm_epsg
                )

            msk_xstart = np.where(full_x > xstart - total_margin, True, False)
            msk_xend = np.where(full_x < xend + total_margin, True, False)
            msk_yend = np.where(full_y > yend - total_margin, True, False)
            msk_ystart = np.where(full_y < ystart + total_margin, True, False)
            terrain_tile_data_msk = np.logical_and(
                msk_xstart,
                np.logical_and(msk_xend, np.logical_and(msk_ystart, msk_yend)),
            )
            terrain_tile_data_msk_pos = terrain_tile_data_msk.astype(
                np.int8
            ).nonzero()

            # if the points clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = cloud_list_item[cst.X].values
                full_y = cloud_list_item[cst.Y].values

            # if no point is found, continue
            if terrain_tile_data_msk_pos[0].shape[0] == 0:
                continue

            # get useful data bounding box
            bbox = [
                np.min(terrain_tile_data_msk_pos[0]),
                np.min(terrain_tile_data_msk_pos[1]),
                np.max(terrain_tile_data_msk_pos[0]),
                np.max(terrain_tile_data_msk_pos[1]),
            ]
        else:
            bbox = [0, 0, full_y.shape[0] - 1, full_y.shape[1] - 1]

        # add (x, y, z) information to the current cloud
        c_x = full_x[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        c_y = full_y[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        c_z = full_z[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]

        c_cloud = np.zeros(
            (len(nb_data), (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
        )
        c_cloud[1, :] = np.ravel(c_x)
        c_cloud[2, :] = np.ravel(c_y)
        c_cloud[3, :] = np.ravel(c_z)

        ds_values_list = [key for key, _ in cloud_list_item.items()]

        if cst.POINTS_CLOUD_MSK in ds_values_list:
            c_msk = cloud_list_item[cst.POINTS_CLOUD_MSK].values[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            c_cloud[4, :] = np.ravel(c_msk)

        # add data valid mask
        # (points that are not in the border of the epipolar image)
        if epipolar_border_margin == 0:
            epipolar_margin_mask = np.full(
                (
                    cloud_list_item[cst.X].values.shape[0],
                    cloud_list_item[cst.X].values.shape[1],
                ),
                True,
            )
        else:
            epipolar_margin_mask = np.full(
                (
                    cloud_list_item[cst.X].values.shape[0],
                    cloud_list_item[cst.X].values.shape[1],
                ),
                False,
            )
            epipolar_margin_mask[
                epipolar_border_margin:-epipolar_border_margin,
                epipolar_border_margin:-epipolar_border_margin,
            ] = True

        c_epipolar_margin_mask = epipolar_margin_mask[
            bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
        ]
        c_cloud[0, :] = np.ravel(c_epipolar_margin_mask)

        # add the color information to the current cloud
        if color_list is not None:
            c_color = color_list[cloud_list_idx].im.values[
                :, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]

            for band in range(nb_band_clr):
                c_cloud[4 + nb_data_msk + band, :] = np.ravel(
                    c_color[band, :, :]
                )

        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(bbox[0], bbox[2], bbox[2] - bbox[0] + 1)
            coords_col = np.linspace(bbox[1], bbox[3], bbox[3] - bbox[1] + 1)
            coords_col, coords_line = np.meshgrid(coords_col, coords_line)

            c_cloud[4 + nb_data_msk + nb_band_clr, :] = np.ravel(coords_line)
            c_cloud[4 + nb_data_msk + nb_band_clr + 1, :] = np.ravel(coords_col)
            c_cloud[4 + nb_data_msk + nb_band_clr + 2, :] = cloud_list_idx

        # remove masked data (pandora + out of the terrain tile points)
        c_terrain_tile_data_msk = (
            cloud_list_item[cst.POINTS_CLOUD_CORR_MSK].values[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            == 255
        )

        if roi:
            c_terrain_tile_data_msk = np.logical_and(
                c_terrain_tile_data_msk,
                terrain_tile_data_msk[
                    bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                ],
            )

        c_terrain_tile_data_msk = np.ravel(c_terrain_tile_data_msk)

        c_terrain_tile_data_msk_pos = np.nonzero(~c_terrain_tile_data_msk)

        nb_points += c_cloud.shape[1]

        c_cloud = np.delete(
            c_cloud.transpose(), c_terrain_tile_data_msk_pos[0], 0
        )

        # add current cloud to the combined one
        cloud = np.concatenate([cloud, c_cloud], axis=0)

    worker_logger.debug("Received {} points to rasterize".format(nb_points))
    worker_logger.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(cloud, columns=nb_data)

    return pd_cloud, epsg


# ##### Parameters structures ######

# Cloud small components filtering parameters :
# ---------------------------------------------
#   * on_ground_margin:
#           margin added to the on ground region to not filter points clusters
#           that were incomplete because they were on the edges
#
#   * pts_connection_dist:
#           distance to use to consider that two points are connected
#
#   * nb_pts_threshold:
#           points clusters that have less than this number of points
#           will be filtered
#
#   * dist_between_clusters:
#           distance to use to consider that two points clusters
#           are far from each other or not.
#       If a small points cluster is near to another one, it won't be filtered.
#          (None = deactivated)
#
#   * construct_removed_elt_msk:
#           if True, the removed points mask will be added to the cloud datasets
#           in input of the simple_rasterization_dataset
#
#   * mask_value:
#           value to use to identify the removed points in the mask
#
SmallComponentsFilterParams = namedtuple(
    "SmallComponentsFilterParams",
    [
        "on_ground_margin",
        "connection_val",
        "nb_pts_threshold",
        "clusters_distance_threshold",
        "filtered_elt_msk",
        "msk_value",
    ],
)

# Cloud statistical filtering :
# ------------------------------
#   * k:
#       number of neighbors
#
#   * stdev_factor:
#       factor to apply in the distance threshold computation
#
#   * construct_removed_elt_msk:
#       if True, the removed points mask will be added to the cloud datasets
#       in input of the simple_rasterization_dataset
#
#   * mask_value:
#       value to use to identify the removed points in the mask
#
StatisticalFilterParams = namedtuple(
    "StatisticalFilterParams",
    ["k", "std_dev_factor", "filtered_elt_msk", "msk_value"],
)


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
    for idx, _ in enumerate(cloud_xyz):

        # if point has already been added to a cluster
        if processed[idx]:
            continue

        # get point neighbors
        neighbors_list = cloud_tree.query_ball_point(
            cloud_xyz[idx], connection_val
        )

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
    for connected_components_idx, connected_components_item in enumerate(
        connected_components
    ):
        if len(connected_components_item) < nb_pts_threshold:
            if clusters_distance_threshold is not None:
                # search if the current cluster has any neighbors
                # in the clusters_distance_threshold radius
                all_neighbors = cloud_tree.query_ball_point(
                    cloud_xyz[connected_components[connected_components_idx]],
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
        and cst.POINTS_CLOUD_IDX_IM_EPI in cloud.columns
    ):
        worker_logger = logging.getLogger("distributed.worker")
        worker_logger.warning(
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
            cst.POINTS_CLOUD_IDX_IM_EPI,
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
        cst.POINTS_CLOUD_COORD_EPI_GEOM_J, cst.POINTS_CLOUD_IDX_IM_EPI columns
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
        or cst.POINTS_CLOUD_IDX_IM_EPI not in elt_pos_infos.columns
    ):
        worker_logger = logging.getLogger("distributed.worker")
        worker_logger.warning(
            "Cannot generate filtered elements mask, "
            "no information about the point's"
            " original position in the epipolar image is given"
        )

    else:
        elt_index = elt_pos_infos.loc[:, cst.POINTS_CLOUD_IDX_IM_EPI].to_numpy()

        min_elt_index = np.min(elt_index)
        max_elt_index = np.max(elt_index)

        if min_elt_index < 0 or max_elt_index > len(clouds_list) - 1:
            raise Exception(
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
                    raise Exception(
                        "Point at location ({},{}) is not accessible "
                        "in an image of size ({},{})".format(
                            i, j, msk.shape[0], msk.shape[1]
                        )
                    ) from index_error

            cloud_item[mask_label] = ([cst.ROW, cst.COL], msk)
