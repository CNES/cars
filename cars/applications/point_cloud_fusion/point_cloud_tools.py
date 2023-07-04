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
# pylint: disable=too-many-lines

# Standard imports
import logging
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.core import projection


def create_combined_cloud(  # noqa: C901
    cloud_list: List[xr.Dataset] or List[pandas.DataFrame],
    cloud_ids: List[int],
    dsm_epsg: int,
    xmin: float = None,
    xmax: float = None,
    ymin: int = None,
    ymax: int = None,
    epipolar_border_margin: int = 0,
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds from sparse or dense matching
    into a pandas dataframe.
    The detailed cases for each cloud type are in the derived function
    create_combined_sparse_cloud and create_combined_dense_cloud.

    :param cloud_list: list of every point cloud to merge
    :param cloud_ids: list of global identificators of clouds in cloud_list
    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param xmin: xmin of the rasterization grid
        (if None, the whole clouds are combined)
    :param xmax: xmax of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymin: ymin of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymax: ymax of the rasterization grid
        (if None, the whole clouds are combined)
    :param margin: Margin added for each tile, in meter or degree.
        (default value: 0)
    :param epipolar_border_margin: Margin used
        to invalidate cells too close to epipolar border. (default value: 0)
    :param with_coords: Option enabling the adding to the combined cloud
        of information of each point to retrieve their positions
        in the original epipolar images
    :return: Tuple formed with the combined clouds and color
        in a single pandas dataframe and the epsg code
    """
    if isinstance(cloud_list[0], xr.Dataset):
        return create_combined_dense_cloud(
            cloud_list,
            cloud_ids,
            dsm_epsg,
            xmin,
            xmax,
            ymin,
            ymax,
            epipolar_border_margin,
            margin,
            with_coords,
        )
    # case of pandas.DataFrame cloud
    return create_combined_sparse_cloud(
        cloud_list,
        cloud_ids,
        dsm_epsg,
        xmin,
        xmax,
        ymin,
        ymax,
        epipolar_border_margin,
        margin,
        with_coords,
    )


def create_combined_sparse_cloud(  # noqa: C901
    cloud_list: List[pandas.DataFrame],
    cloud_ids: List[int],
    dsm_epsg: int,
    xmin: float = None,
    xmax: float = None,
    ymin: int = None,
    ymax: int = None,
    epipolar_border_margin: int = 0,
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        - if no mask data present in cloud_list datasets:
            labels=[cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z] \
            The combined cloud has x, y, z columns along with 'valid data' one.\
            The valid data is a mask set to True if the data \
            are not on the epipolar image margin (epipolar_border_margin), \
             otherwise it is set to False.

        - if mask data present in cloud_list datasets:
            labels=[cst.POINTS_CLOUD_VALID_DATA,\
                    cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK]\
            The mask values are added to the dataframe.

    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param xmin: xmin of the rasterization grid
        (if None, the whole clouds are combined)
    :param xmax: xmax of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymin: ymin of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymax: ymax of the rasterization grid
        (if None, the whole clouds are combined)
    :param margin: Margin added for each tile, in meter or degree.
        (default value: 0)
    :param epipolar_border_margin: Margin used
        to invalidate cells too close to epipolar border. (default value: 0)
    :param with_coords: Option enabling the adding to the combined cloud
        of information of each point to retrieve their positions
        in the original epipolar images
    :return: Tuple formed with the combined clouds and color
        in a single pandas dataframe and the epsg code
    """

    epsg = get_epsg(cloud_list)

    # compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    nb_data = create_point_cloud_index(cloud_list)

    if with_coords:
        nb_data.extend([cst.POINTS_CLOUD_COORD_EPI_GEOM_I])

    # iterate through input clouds
    cloud = np.zeros((0, len(nb_data)), dtype=np.float64)
    nb_points = 0
    for cloud_global_id, cloud_list_item in zip(  # noqa: B905
        cloud_ids, cloud_list
    ):
        full_x = cloud_list_item[cst.X]
        full_y = cloud_list_item[cst.Y]
        full_z = cloud_list_item[cst.Z]

        # get mask of points inside the roi (plus margins)
        if roi:
            # Compute terrain tile bounds
            # if the points clouds are not in the same referential as the roi,
            # it is converted using the dsm_epsg
            (
                terrain_tile_data_msk,
                terrain_tile_data_msk_pos,
            ) = compute_terrain_msk(
                dsm_epsg,
                xmin,
                xmax,
                ymin,
                ymax,
                margin,
                epsg,
                cloud_list_item,
                full_x,
                full_y,
            )

            # if the points clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = cloud_list_item[cst.X]
                full_y = cloud_list_item[cst.Y]

            # if no point is found, continue
            if terrain_tile_data_msk_pos[0].shape[0] == 0:
                continue
            # get useful data bounding box
            bbox = [
                np.min(terrain_tile_data_msk_pos),
                np.max(terrain_tile_data_msk_pos),
            ]

        else:
            bbox = [0, full_y.shape[0] - 1]

        # add (x, y, z) information to the current cloud
        c_x = full_x[bbox[0] : bbox[1] + 1]
        c_y = full_y[bbox[0] : bbox[1] + 1]
        c_z = full_z[bbox[0] : bbox[1] + 1]

        c_cloud = np.zeros((len(nb_data), (bbox[1] - bbox[0] + 1)))
        c_cloud[nb_data.index(cst.X), :] = c_x
        c_cloud[nb_data.index(cst.Y), :] = c_y
        c_cloud[nb_data.index(cst.Z), :] = c_z

        # add data valid mask
        # (points that are not in the border of the epipolar image)
        if epipolar_border_margin == 0:
            epipolar_margin_mask = np.full(
                cloud_list_item[cst.X].size,
                True,
            )
        else:
            epipolar_margin_mask = np.full(
                cloud_list_item[cst.X].size,
                False,
            )
            epipolar_margin_mask[
                epipolar_border_margin:-epipolar_border_margin,
            ] = True

        c_epipolar_margin_mask = epipolar_margin_mask[bbox[0] : bbox[1] + 1]
        c_cloud[nb_data.index(cst.POINTS_CLOUD_VALID_DATA), :] = np.ravel(
            c_epipolar_margin_mask
        )

        # add index of original point cloud
        c_cloud[nb_data.index(cst.POINTS_CLOUD_GLOBAL_ID), :] = cloud_global_id

        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(
                bbox[0], bbox[1], num=bbox[1] - bbox[0] + 1
            )
            c_cloud[
                nb_data.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_I), :
            ] = coords_line

        # remove masked data (pandora + out of the terrain tile points)
        c_terrain_tile_data_msk = (
            cloud_list_item[cst.POINTS_CLOUD_CORR_MSK][bbox[0] : bbox[1]] == 255
        )
        if roi:
            c_terrain_tile_data_msk = np.logical_and(
                c_terrain_tile_data_msk,
                terrain_tile_data_msk[bbox[0] : bbox[1]],
            )

        c_cloud = filter_cloud_with_mask(
            nb_points, c_cloud, c_terrain_tile_data_msk
        )

        # add current cloud to the combined one
        cloud = np.concatenate([cloud, c_cloud], axis=0)

    logging.debug("Received {} points to rasterize".format(nb_points))
    logging.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(cloud, columns=nb_data)

    return pd_cloud, epsg


def get_epsg(cloud_list):
    """
    Extract epsg from cloud list and check if all the same

    :param cloud_list: list of the point clouds
    """
    epsg = None
    for cloud_list_item in cloud_list:
        if epsg is None:
            epsg = int(cloud_list_item.attrs[cst.EPSG])
        elif int(cloud_list_item.attrs[cst.EPSG]) != epsg:
            logging.error("All points clouds do not have the same epsg code")

    return epsg


def filter_cloud_with_mask(nb_points, c_cloud, c_terrain_tile_data_msk):
    """
    Delete masked points with terrain tile mask

    :param nb_points: total number of point cloud
        (increase at each point cloud)
    :param c_cloud: the point cloud
    :param c_terrain_tile_data_msk: terrain tile mask
    """
    c_terrain_tile_data_msk = np.ravel(c_terrain_tile_data_msk)

    c_terrain_tile_data_msk_pos = np.nonzero(~c_terrain_tile_data_msk)

    # compute nb points before apply the mask
    nb_points += c_cloud.shape[1]

    c_cloud = np.delete(c_cloud.transpose(), c_terrain_tile_data_msk_pos[0], 0)

    return c_cloud


def compute_terrain_msk(
    dsm_epsg,
    xmin,
    xmax,
    ymin,
    ymax,
    margin,
    epsg,
    cloud_list_item,
    full_x,
    full_y,
):
    """
    Compute terrain tile msk bounds

    If the points clouds are not in the same referential as the roi,
    it is converted using the dsm_epsg

    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param xmin: xmin of the rasterization grid
        (if None, the whole clouds are combined)
    :param xmax: xmax of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymin: ymin of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymax: ymax of the rasterization grid
        (if None, the whole clouds are combined)
    :param margin: Margin added for each tile, in meter or degree.
        (default value: 0)
    :param epsg: epsg code of the input cloud
    :param cloud_list_item: the point cloud
    :param full_x: cloud_list_item[X]
    :param full_y: cloud_list_item[Y]
    """
    if epsg != dsm_epsg:
        (
            full_x,
            full_y,
        ) = projection.get_converted_xy_np_arrays_from_dataset(
            cloud_list_item, dsm_epsg
        )
    msk_xmin = np.where(full_x > xmin - margin, True, False)
    msk_xmax = np.where(full_x < xmax + margin, True, False)
    msk_ymin = np.where(full_y > ymin - margin, True, False)
    msk_ymax = np.where(full_y < ymax + margin, True, False)
    terrain_tile_data_msk = np.logical_and(
        msk_xmin,
        np.logical_and(msk_xmax, np.logical_and(msk_ymin, msk_ymax)),
    )
    terrain_tile_data_msk_pos = terrain_tile_data_msk.astype(np.int8).nonzero()

    return terrain_tile_data_msk, terrain_tile_data_msk_pos


def create_combined_dense_cloud(  # noqa: C901
    cloud_list: List[xr.Dataset],
    cloud_id: List[int],
    dsm_epsg: int,
    xmin: float = None,
    xmax: float = None,
    ymin: int = None,
    ymax: int = None,
    epipolar_border_margin: int = 0,
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        - if no colors in input and no mask data present in cloud_list datasets:
            labels=[cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z] \
            The combined cloud has x, y, z columns along with 'valid data' one.\
            The valid data is a mask set to True if the data \
            are not on the epipolar image margin (epipolar_border_margin), \
             otherwise it is set to False.

        - if no colors in input and mask data present in cloud_list datasets:
            labels=[cst.POINTS_CLOUD_VALID_DATA,\
                    cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK]\
            The mask values are added to the dataframe.

        - if colors are set in input and mask data are present \
            in the cloud_list datasets:
           labels=[cst.POINTS_CLOUD_VALID_DATA,\
                     cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"0",\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"1",\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"2"]\
            Color channels information are added to the dataframe.

        - if colors in input, mask data present in the cloud_list datasets and\
            the with_coords option is activated:
             labels=[cst.POINTS_CLOUD_VALID_DATA,\
                     cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"0",\
                    cst.POINTS_CLOUD_CLR_KEY_ROOT+"1",\
                    cst.POINTS_CLOUD_CLR_KEY_ROOT+"2"\
                     cst.POINTS_CLOUD_COORD_EPI_GEOM_I,\
                     cst.POINTS_CLOUD_COORD_EPI_GEOM_J,\
                     cst.POINTS_CLOUD_ID_IM_EPI]\
            The pixel position of the xyz point in the original epipolar\
            image (coord_epi_geom_i, coord_epi_geom_j) are added\
            to the dataframe along with the index of its original cloud\
            in the cloud_list input.


    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param xmin: xmin of the rasterization grid
        (if None, the whole clouds are combined)
    :param xmax: xmax of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymin: ymin of the rasterization grid
        (if None, the whole clouds are combined)
    :param ymax: ymax of the rasterization grid
        (if None, the whole clouds are combined)
    :param margin: Margin added for each tile, in meter or degree.
        (default value: 0)
    :param epipolar_border_margin: Margin used
        to invalidate cells too close to epipolar border. (default value: 0)
    :param with_coords: Option enabling the adding to the combined cloud
        of information of each point to retrieve their positions
        in the original epipolar images
    :return: Tuple formed with the combined clouds and color
        in a single pandas dataframe and the epsg code
    """
    epsg = get_epsg(cloud_list)

    # compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    # create point cloud index
    nb_data = create_point_cloud_index(cloud_list)

    # Find number of bands
    # get max of bands
    nb_band_clr = get_number_bands(cloud_list)

    # Extend list of data
    list_clr = [
        "{}{}".format(cst.POINTS_CLOUD_CLR_KEY_ROOT, i)
        for i in range(nb_band_clr)
    ]
    nb_data.extend(list_clr)

    if with_coords:
        nb_data.extend(
            [
                cst.POINTS_CLOUD_COORD_EPI_GEOM_I,
                cst.POINTS_CLOUD_COORD_EPI_GEOM_J,
                cst.POINTS_CLOUD_ID_IM_EPI,
            ]
        )

    # add classif indexes
    band_classif = None
    if cst.EPI_CLASSIFICATION in cloud_list[0]:
        band_classif = list(cloud_list[0].coords[cst.BAND_CLASSIF].to_numpy())
        for band in band_classif:
            band_index = "{}_{}".format(cst.POINTS_CLOUD_CLASSIF_KEY_ROOT, band)
            nb_data.extend(
                [
                    band_index,
                ]
            )

    confidence_list = []
    for key in cloud_list[0].keys():
        if cst.POINTS_CLOUD_CONFIDENCE in key:
            nb_data.append(key)
            confidence_list.append(key)

    # iterate through input clouds
    cloud = np.zeros((0, len(nb_data)), dtype=np.float64)
    nb_points = 0
    for cloud_global_id, (cloud_list_id, cloud_list_item) in zip(  # noqa: B905
        cloud_id, enumerate(cloud_list)
    ):
        full_x = cloud_list_item[cst.X].values
        full_y = cloud_list_item[cst.Y].values
        full_z = cloud_list_item[cst.Z].values

        # get mask of points inside the roi (plus margins)
        if roi:
            # Compute terrain tile bounds
            # if the points clouds are not in the same referential as the roi,
            # it is converted using the dsm_epsg
            (
                terrain_tile_data_msk,
                terrain_tile_data_msk_pos,
            ) = compute_terrain_msk(
                dsm_epsg,
                xmin,
                xmax,
                ymin,
                ymax,
                margin,
                epsg,
                cloud_list_item,
                full_x,
                full_y,
            )

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
        c_cloud[nb_data.index(cst.X), :] = np.ravel(c_x)
        c_cloud[nb_data.index(cst.Y), :] = np.ravel(c_y)
        c_cloud[nb_data.index(cst.Z), :] = np.ravel(c_z)
        ds_values_list = [key for key, _ in cloud_list_item.items()]
        for confidence_name in confidence_list:
            if cst.POINTS_CLOUD_CONFIDENCE in " ".join(ds_values_list):
                c_confidence = cloud_list_item[confidence_name].values[
                    bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                ]
                c_cloud[nb_data.index(confidence_name), :] = np.ravel(
                    c_confidence
                )

        if cst.POINTS_CLOUD_MSK in ds_values_list:
            c_msk = cloud_list_item[cst.POINTS_CLOUD_MSK].values[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            c_cloud[nb_data.index(cst.POINTS_CLOUD_MSK), :] = np.ravel(c_msk)

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
        c_cloud[nb_data.index(cst.POINTS_CLOUD_VALID_DATA), :] = np.ravel(
            c_epipolar_margin_mask
        )

        # add index of original point cloud
        c_cloud[nb_data.index(cst.POINTS_CLOUD_GLOBAL_ID), :] = cloud_global_id

        # add the color information to the current cloud
        if cst.EPI_COLOR in cloud_list[cloud_list_id]:
            add_color_information(
                cloud_list,
                nb_data,
                nb_band_clr,
                cloud_list_id,
                bbox,
                c_cloud,
            )

        # add classification to the current cloud
        if cst.EPI_CLASSIFICATION in cloud_list[cloud_list_id]:
            add_classification_information(
                cloud_list,
                nb_data,
                band_classif,
                cloud_list_id,
                bbox,
                c_cloud,
            )
        # add mask to the current cloud
        if cst.EPI_COLOR_MSK in cloud_list[cloud_list_id]:
            add_msk_information(
                cloud_list,
                nb_data,
                cloud_list_id,
                bbox,
                c_cloud,
            )
        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(bbox[0], bbox[2], bbox[2] - bbox[0] + 1)
            coords_col = np.linspace(bbox[1], bbox[3], bbox[3] - bbox[1] + 1)
            coords_col, coords_line = np.meshgrid(coords_col, coords_line)

            c_cloud[
                nb_data.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_I), :
            ] = np.ravel(coords_line)
            c_cloud[
                nb_data.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_J), :
            ] = np.ravel(coords_col)
            c_cloud[
                nb_data.index(cst.POINTS_CLOUD_ID_IM_EPI), :
            ] = cloud_list_id

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

        c_cloud = filter_cloud_with_mask(
            nb_points, c_cloud, c_terrain_tile_data_msk
        )

        # add current cloud to the combined one
        cloud = np.concatenate([cloud, c_cloud], axis=0)

    logging.debug("Received {} points to rasterize".format(nb_points))
    logging.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(cloud, columns=nb_data)

    return pd_cloud, epsg


def create_point_cloud_index(cloud_list):
    """
    Create point cloud index from cloud list keys and color inputs
    """
    nb_data = [
        cst.POINTS_CLOUD_GLOBAL_ID,
        cst.POINTS_CLOUD_VALID_DATA,
        cst.X,
        cst.Y,
        cst.Z,
    ]

    # check if the input mask values are present in the dataset
    for cloud_list_item in cloud_list:
        ds_values_list = [key for key, _ in cloud_list_item.items()]
        if cst.POINTS_CLOUD_MSK in ds_values_list:
            nb_data.append(cst.POINTS_CLOUD_MSK)
            break

    return nb_data


def add_color_information(
    cloud_list,
    nb_data,
    nb_band_clr,
    cloud_list_id,
    bbox,
    c_cloud,
):
    """
    Add color information for a current cloud_list item

    :param cloud_list: point cloud dataset
    :type cloud_list: List(Dataset)
    :param nb_data: list of band data
    :type nb_data: list[str]
    :param nb_band_clr: number of color band
    :type nb_band_clr: int
    :param cloud_list_id: index of the current point cloud
    :type cloud_list_id: int
    :param bbox: bbox of interest
    :type bbox: list[int]
    :param c_cloud: arranged point cloud
    :type c_cloud: NDArray[float64]
    """
    if nb_band_clr == 1:
        if len(cloud_list[cloud_list_id][cst.EPI_COLOR].values.shape) == 3:
            c_color = np.squeeze(
                cloud_list[cloud_list_id][cst.EPI_COLOR].values
            )[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        else:
            c_color = cloud_list[cloud_list_id][cst.EPI_COLOR].values[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]

        c_cloud[
            nb_data.index("{}{}".format(cst.POINTS_CLOUD_CLR_KEY_ROOT, 0)),
            :,
        ] = np.ravel(c_color[:, :])
    else:
        color_array = cloud_list[cloud_list_id][cst.EPI_COLOR].values
        if len(color_array.shape) == 2:
            # point cloud created with pancro, needs to duplicate
            logging.debug(
                "Not the same number of color bands for all point clouds"
            )
            color_array = np.stack(
                [color_array for _ in range(nb_band_clr)], axis=0
            )

        c_color = color_array[:, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        for band in range(nb_band_clr):
            c_cloud[
                nb_data.index(
                    "{}{}".format(cst.POINTS_CLOUD_CLR_KEY_ROOT, band)
                ),
                :,
            ] = np.ravel(c_color[band, :, :])


def add_classification_information(
    cloud_list,
    nb_data,
    band_classif,
    cloud_list_id,
    bbox,
    c_cloud,
):
    """
    Add color information for a current cloud_list item

    :param cloud_list: point cloud dataset
    :type cloud_list: List(Dataset)
    :param band_classif: list of band classif
    :type band_classif: list[str]
    :param nb_data: list of band data
    :type nb_data: list[str]
    :param cloud_list_id: index of the current point cloud
    :type cloud_list_id: int
    :param bbox: bbox of interest
    :type bbox: list[int]
    :param c_cloud: arranged point cloud
    :type c_cloud: NDArray[float64]
    """
    classif_array = cloud_list[cloud_list_id][cst.EPI_CLASSIFICATION].values
    c_classif = classif_array[:, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    for idx, band in enumerate(band_classif):
        band_index = "{}_{}".format(cst.POINTS_CLOUD_CLASSIF_KEY_ROOT, band)
        c_cloud[
            nb_data.index(band_index),
            :,
        ] = np.ravel(c_classif[idx, :, :])


def add_msk_information(
    cloud_list,
    nb_data,
    cloud_list_id,
    bbox,
    c_cloud,
):
    """
    Add mask information for a current cloud_list item

    :param cloud_list: point cloud dataset
    :type cloud_list: List(Dataset)
    :param nb_data: list of band data
    :type nb_data: list[str]
    :param cloud_list_id: index of the current point cloud
    :type cloud_list_id: int
    :param bbox: bbox of interest
    :type bbox: list[int]
    :param c_cloud: arranged point cloud
    :type c_cloud: NDArray[float64]
    """
    msk_array = cloud_list[cloud_list_id][cst.EPI_COLOR_MSK].values
    c_msk = msk_array[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    c_cloud[
        nb_data.index(cst.POINTS_CLOUD_MSK),
        :,
    ] = np.ravel(c_msk[:, :])


def get_color_type(clouds):
    """
    Get color type of the tiles and if the same type.

    :param cloud_list: list of clouds
    :type cloud_list: xarray Dataset

    :return: color type of the tiles list
    :rtype: str

    """
    color_types = []
    for cloud_id, cloud_item in enumerate(clouds):
        if cst.EPI_COLOR in clouds[cloud_id]:
            if "color_type" in cloud_item.attrs:
                color_types.append(cloud_item.attrs["color_type"])
    if color_types:
        color_type_set = set(color_types)
        if len(color_type_set) > 1:
            logging.warning("The tiles colors don't have the same type.")
        return color_types[0]

    return None


def get_number_bands(cloud_list):
    """
    Get max number of bands of clouds

    :param cloud_list: list of clouds
    :type cloud_list: xarray Dataset

    :return: max number of band
    :rtype: int

    """

    nb_band_clr = 0
    for current_cloud in cloud_list:
        current_cloud_nb_bands = 0
        if cst.EPI_COLOR in current_cloud:
            clr_im = current_cloud[cst.EPI_COLOR].values
            if len(clr_im.shape) == 2:
                current_cloud_nb_bands = 1
            else:
                current_cloud_nb_bands = clr_im.shape[0]

        nb_band_clr = max(nb_band_clr, current_cloud_nb_bands)

    return nb_band_clr


# ONLY USED IN TEST
# TODO remove


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

    TODO only used in tests

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
        for cloud_id, cloud_item in enumerate(clouds_list):
            if mask_label not in cloud_item:
                nb_row = cloud_item.coords[cst.ROW].data.shape[0]
                nb_col = cloud_item.coords[cst.COL].data.shape[0]
                msk = np.zeros((nb_row, nb_col), dtype=np.uint16)
            else:
                msk = cloud_item[mask_label].values

            cur_elt_index = np.argwhere(elt_index == cloud_id)

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
