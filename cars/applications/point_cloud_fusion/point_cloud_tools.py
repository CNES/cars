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
# pylint: disable=too-many-lines
"""
This module is responsible for the transition between triangulation and
rasterization steps
"""


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
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        - if no mask data present in cloud_list datasets:
            labels=[ cst.X, cst.Y, cst.Z] \
            The combined cloud has x, y, z columns

        - if mask data present in cloud_list datasets:
            labels=[cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK]\
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

    cloud_indexes_with_types = create_points_cloud_index(cloud_list[0])

    if with_coords:
        cloud_indexes_with_types.update(
            {cst.POINTS_CLOUD_COORD_EPI_GEOM_I: "uint16"}
        )

    cloud_indexes = list(cloud_indexes_with_types.keys())

    # iterate through input clouds
    combined_cloud = np.zeros((0, len(cloud_indexes)))
    nb_points = 0
    for cloud_global_id, points_cloud in zip(  # noqa: B905
        cloud_ids, cloud_list
    ):
        full_x = points_cloud[cst.X]
        full_y = points_cloud[cst.Y]
        full_z = points_cloud[cst.Z]

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
                points_cloud,
                full_x,
                full_y,
            )

            # if the points clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = points_cloud[cst.X]
                full_y = points_cloud[cst.Y]

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
        crop_x = full_x[bbox[0] : bbox[1] + 1]
        crop_y = full_y[bbox[0] : bbox[1] + 1]
        crop_z = full_z[bbox[0] : bbox[1] + 1]

        crop_cloud = np.zeros((len(cloud_indexes), (bbox[1] - bbox[0] + 1)))
        crop_cloud[cloud_indexes.index(cst.X), :] = crop_x
        crop_cloud[cloud_indexes.index(cst.Y), :] = crop_y
        crop_cloud[cloud_indexes.index(cst.Z), :] = crop_z

        # add index of original point cloud
        crop_cloud[cloud_indexes.index(cst.POINTS_CLOUD_GLOBAL_ID), :] = (
            cloud_global_id
        )

        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(
                bbox[0], bbox[1], num=bbox[1] - bbox[0] + 1
            )
            crop_cloud[
                cloud_indexes.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_I), :
            ] = coords_line

        # remove masked data (pandora + out of the terrain tile points)
        crop_terrain_tile_data_msk = (
            points_cloud[cst.POINTS_CLOUD_CORR_MSK][bbox[0] : bbox[1]] == 255
        )
        if roi:
            crop_terrain_tile_data_msk = np.logical_and(
                crop_terrain_tile_data_msk,
                terrain_tile_data_msk[bbox[0] : bbox[1]],
            )

        crop_cloud = filter_cloud_with_mask(
            nb_points, crop_cloud, crop_terrain_tile_data_msk
        )

        # add current cloud to the combined one
        combined_cloud = np.concatenate([combined_cloud, crop_cloud], axis=0)

    logging.debug("Received {} points to rasterize".format(nb_points))
    logging.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(combined_cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(combined_cloud, columns=cloud_indexes)

    return pd_cloud, epsg


def get_epsg(cloud_list):
    """
    Extract epsg from cloud list and check if all the same

    :param cloud_list: list of the point clouds
    """
    epsg = None
    for points_cloud in cloud_list:
        if epsg is None:
            epsg = int(points_cloud.attrs[cst.EPSG])
        elif int(points_cloud.attrs[cst.EPSG]) != epsg:
            logging.error("All points clouds do not have the same epsg code")

    return epsg


def filter_cloud_with_mask(nb_points, crop_cloud, crop_terrain_tile_data_msk):
    """
    Delete masked points with terrain tile mask

    :param nb_points: total number of point cloud
        (increase at each point cloud)
    :param crop_cloud: the point cloud
    :param crop_terrain_tile_data_msk: terrain tile mask
    """
    crop_terrain_tile_data_msk = np.ravel(crop_terrain_tile_data_msk)

    crop_terrain_tile_data_msk_pos = np.nonzero(~crop_terrain_tile_data_msk)

    # compute nb points before apply the mask
    nb_points += crop_cloud.shape[1]

    crop_cloud = np.delete(
        crop_cloud.transpose(), crop_terrain_tile_data_msk_pos[0], 0
    )

    return crop_cloud


def compute_terrain_msk(
    dsm_epsg,
    xmin,
    xmax,
    ymin,
    ymax,
    margin,
    epsg,
    points_cloud,
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
    :param points_cloud: the point cloud
    :param full_x: points_cloud[X]
    :param full_y: points_cloud[Y]
    """
    if epsg != dsm_epsg:
        (
            full_x,
            full_y,
        ) = projection.get_converted_xy_np_arrays_from_dataset(
            points_cloud, dsm_epsg
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
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        - if no colors in input and no mask data present in cloud_list datasets:
            labels=[cst.X, cst.Y, cst.Z] \
            The combined cloud has x, y, z columns

        - if no colors in input and mask data present in cloud_list datasets:
            labels=[cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK]\
            The mask values are added to the dataframe.

        - if colors are set in input and mask data are present \
            in the cloud_list datasets:
           labels=[cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"0",\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"1",\
                     cst.POINTS_CLOUD_CLR_KEY_ROOT+"2"]\
            Color channels information are added to the dataframe.

        - if colors in input, mask data present in the cloud_list datasets and\
            the with_coords option is activated:
             labels=[cst.X, cst.Y, cst.Z, cst.POINTS_CLOUD_MSK,\
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
        - if confidence intervals on Z in input, then\
            [cst.Z_INF, cst.Z_SUP] are also added to the labels


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
    :param with_coords: Option enabling the adding to the combined cloud
        of information of each point to retrieve their positions
        in the original epipolar images
    :return: Tuple formed with the combined clouds and color
        in a single pandas dataframe and the epsg code
    """
    epsg = get_epsg(cloud_list)

    # Compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    # Create point cloud index
    cloud_indexes_with_types = create_points_cloud_index(cloud_list[0])

    # Add coords
    if with_coords:
        cloud_indexes_with_types.update(
            {
                cst.POINTS_CLOUD_COORD_EPI_GEOM_I: "uint16",
                cst.POINTS_CLOUD_COORD_EPI_GEOM_J: "uint16",
                cst.POINTS_CLOUD_ID_IM_EPI: "uint16",
            }
        )

    cloud_indexes = list(cloud_indexes_with_types.keys())

    # Iterate through input clouds
    combined_cloud = np.zeros((0, len(cloud_indexes)))
    nb_points = 0
    for cloud_global_id, (cloud_list_id, points_cloud) in zip(  # noqa: B905
        cloud_id, enumerate(cloud_list)
    ):
        full_x = points_cloud[cst.X].values
        full_y = points_cloud[cst.Y].values
        full_z = points_cloud[cst.Z].values

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
                points_cloud,
                full_x,
                full_y,
            )

            # if the points clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = points_cloud[cst.X].values
                full_y = points_cloud[cst.Y].values

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
        crop_x = full_x[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        crop_y = full_y[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        crop_z = full_z[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]

        flatten_cloud = np.zeros(
            (
                len(cloud_indexes),
                (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1),
            )
        )
        flatten_cloud[cloud_indexes.index(cst.X), :] = np.ravel(crop_x)
        flatten_cloud[cloud_indexes.index(cst.Y), :] = np.ravel(crop_y)
        flatten_cloud[cloud_indexes.index(cst.Z), :] = np.ravel(crop_z)

        if (cst.Z_INF in cloud_indexes) and (cst.Z_SUP in cloud_indexes):
            full_z_inf = points_cloud[cst.Z_INF].values
            full_z_sup = points_cloud[cst.Z_SUP].values
            crop_z_inf = full_z_inf[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            crop_z_sup = full_z_sup[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            flatten_cloud[cloud_indexes.index(cst.Z_INF), :] = np.ravel(
                crop_z_inf
            )
            flatten_cloud[cloud_indexes.index(cst.Z_SUP), :] = np.ravel(
                crop_z_sup
            )

        # add index of original point cloud
        flatten_cloud[cloud_indexes.index(cst.POINTS_CLOUD_GLOBAL_ID), :] = (
            cloud_global_id
        )

        # add additional information to point cloud
        arrays_to_add_to_points_cloud = [
            (cst.EPI_COLOR, cst.POINTS_CLOUD_CLR_KEY_ROOT),
            (cst.EPI_MSK, cst.POINTS_CLOUD_MSK),
            (cst.EPI_CLASSIFICATION, cst.POINTS_CLOUD_CLASSIF_KEY_ROOT),
            (cst.EPI_FILLING, cst.POINTS_CLOUD_FILLING_KEY_ROOT),
        ]

        # add confidence layers
        for array_name in points_cloud:
            if cst.EPI_CONFIDENCE_KEY_ROOT in array_name:
                arrays_to_add_to_points_cloud.append((array_name, array_name))

        # add denoising info layers
        for array_name in points_cloud:
            if cst.EPI_DENOISING_INFO_KEY_ROOT in array_name:
                arrays_to_add_to_points_cloud.append((array_name, array_name))

        for input_band, output_column in arrays_to_add_to_points_cloud:
            add_information_to_cloud(
                points_cloud,
                cloud_indexes,
                bbox,
                flatten_cloud,
                input_band,
                output_column,
            )

        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(bbox[0], bbox[2], bbox[2] - bbox[0] + 1)
            coords_col = np.linspace(bbox[1], bbox[3], bbox[3] - bbox[1] + 1)
            coords_col, coords_line = np.meshgrid(coords_col, coords_line)

            flatten_cloud[
                cloud_indexes.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_I), :
            ] = np.ravel(coords_line)
            flatten_cloud[
                cloud_indexes.index(cst.POINTS_CLOUD_COORD_EPI_GEOM_J), :
            ] = np.ravel(coords_col)
            flatten_cloud[
                cloud_indexes.index(cst.POINTS_CLOUD_ID_IM_EPI), :
            ] = cloud_list_id

        # remove masked data (pandora + out of the terrain tile points)
        crop_terrain_tile_data_msk = (
            points_cloud[cst.POINTS_CLOUD_CORR_MSK].values[
                bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            == 255
        )

        if roi:
            crop_terrain_tile_data_msk = np.logical_and(
                crop_terrain_tile_data_msk,
                terrain_tile_data_msk[
                    bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                ],
            )

        flatten_cloud = filter_cloud_with_mask(
            nb_points, flatten_cloud, crop_terrain_tile_data_msk
        )

        # add current cloud to the combined one
        combined_cloud = np.concatenate([combined_cloud, flatten_cloud], axis=0)

    logging.debug("Received {} points to rasterize".format(nb_points))
    logging.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(combined_cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(combined_cloud, columns=cloud_indexes)
    pd_cloud = pd_cloud.astype(cloud_indexes_with_types)

    return pd_cloud, epsg


def create_points_cloud_index(cloud_sample):
    """
    Create point cloud index from cloud list keys and color inputs
    """
    cloud_indexes_with_types = {
        cst.POINTS_CLOUD_GLOBAL_ID: "uint16",
        cst.X: "float64",
        cst.Y: "float64",
        cst.Z: "float64",
    }

    # Add Z_inf and Z_sup if intervals have been computed
    if (cst.Z_INF in cloud_sample) and (cst.Z_SUP in cloud_sample):
        cloud_indexes_with_types[cst.Z_INF] = "float64"
        cloud_indexes_with_types[cst.Z_SUP] = "float64"

    # Add mask index
    if cst.EPI_MSK in cloud_sample:
        cloud_indexes_with_types[cst.POINTS_CLOUD_MSK] = "uint8"

    # Add color indexes
    if cst.EPI_COLOR in cloud_sample:
        band_color = list(cloud_sample.coords[cst.BAND_IM].to_numpy())
        color_type = "float32"
        if "color_type" in cloud_sample.attrs:
            color_type = cloud_sample.attrs["color_type"]
        for band in band_color:
            band_index = "{}_{}".format(cst.POINTS_CLOUD_CLR_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = color_type

    # Add classif indexes
    if cst.EPI_CLASSIFICATION in cloud_sample:
        band_classif = list(cloud_sample.coords[cst.BAND_CLASSIF].to_numpy())
        for band in band_classif:
            band_index = "{}_{}".format(cst.POINTS_CLOUD_CLASSIF_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = "boolean"

    # Add filling information indexes
    if cst.EPI_FILLING in cloud_sample:
        band_filling = list(cloud_sample.coords[cst.BAND_FILLING].to_numpy())
        for band in band_filling:
            band_index = "{}_{}".format(cst.POINTS_CLOUD_FILLING_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = "boolean"

    # Add confidence indexes
    for key in cloud_sample:
        if cst.EPI_CONFIDENCE_KEY_ROOT in key:
            cloud_indexes_with_types[key] = "float32"

    return cloud_indexes_with_types


def add_information_to_cloud(
    input_cloud, cloud_indexes, bbox, target_cloud, input_array, output_column
):
    """
    Add color information for a current cloud_list item

    :param cloud: source point cloud dataset
    :type cloud: xr.Dataset
    :param cloud_indexes: list of band data to extract
    :type cloud_indexes: list[str]
    :param bbox: bbox of interest
    :type bbox: list[int]
    :param crop_cloud: target flatten point cloud
    :type crop_cloud: np.array[columns, points]
    :param input_array: index of input to extract from cloud
    :type input_array: str
    :param output_column: index of crop_cloud to fill
    :type input_array: str
    """
    if input_array in input_cloud:
        full_array = input_cloud[input_array]
        if len(full_array.shape) == 3:
            # Array with multiple bands
            array = full_array[:, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
            for column_name in cloud_indexes:
                if output_column in column_name:
                    band_name = column_name.replace(output_column + "_", "")
                    band = array.loc[band_name]
                    index = cloud_indexes.index(column_name)
                    target_cloud[index, :] = np.ravel(band.values)
        elif len(full_array.shape) == 2:
            # Array with single band
            array = full_array[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
            index = cloud_indexes.index(output_column)
            target_cloud[index, :] = np.ravel(array.values)


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
