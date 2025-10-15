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
# pylint: disable=C0302

# Standard imports
import logging
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import xarray as xr

from cars.applications.dense_matching import dense_matching_wrappers

# CARS imports
from cars.core import constants as cst
from cars.core import projection


def filter_cloud_with_mask(crop_cloud, crop_terrain_tile_data_msk):
    """
    Delete masked points with terrain tile mask

    :param crop_cloud: the point cloud
    :param crop_terrain_tile_data_msk: terrain tile mask
    """
    crop_terrain_tile_data_msk = np.ravel(crop_terrain_tile_data_msk)

    crop_terrain_tile_data_msk_pos = np.nonzero(~crop_terrain_tile_data_msk)

    crop_cloud = np.delete(crop_cloud, crop_terrain_tile_data_msk_pos[0], 0)

    return crop_cloud


# pylint: disable=too-many-positional-arguments
def compute_terrain_msk(
    dsm_epsg,
    xmin,
    xmax,
    ymin,
    ymax,
    margin,
    epsg,
    point_cloud,
    full_x,
    full_y,
):
    """
    Compute terrain tile msk bounds

    If the point clouds are not in the same referential as the roi,
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
    :param point_cloud: the point cloud
    :param full_x: point_cloud[X]
    :param full_y: point_cloud[Y]
    """
    if epsg != dsm_epsg:
        (
            full_x,
            full_y,
        ) = projection.get_converted_xy_np_arrays_from_dataset(
            point_cloud, dsm_epsg
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


def create_point_cloud_index(cloud_sample):
    """
    Create point cloud index from cloud list keys and color inputs
    """
    cloud_indexes_with_types = {
        cst.X: "float64",
        cst.Y: "float64",
        cst.Z: "float64",
    }

    # Add Z_inf Z_sup, and performance maps if computed
    for key in list(cloud_sample.keys()):
        if (
            cst.POINT_CLOUD_LAYER_INF in key
            or cst.POINT_CLOUD_LAYER_SUP in key
            or cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT in key
        ):
            cloud_indexes_with_types[key] = "float32"

    # Add mask index
    if cst.EPI_MSK in cloud_sample:
        cloud_indexes_with_types[cst.POINT_CLOUD_MSK] = "uint8"

    # Add color indexes
    if cst.EPI_TEXTURE in cloud_sample:
        band_color = list(cloud_sample.coords[cst.BAND_IM].to_numpy())
        color_type = "float32"
        if "color_type" in cloud_sample.attrs:
            color_type = cloud_sample.attrs["color_type"]
        for band in band_color:
            band_index = "{}_{}".format(cst.POINT_CLOUD_CLR_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = color_type

    # Add classif indexes
    if cst.EPI_CLASSIFICATION in cloud_sample:
        band_classif = list(cloud_sample.coords[cst.BAND_CLASSIF].to_numpy())
        for band in band_classif:
            band_index = "{}_{}".format(cst.POINT_CLOUD_CLASSIF_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = "boolean"

    # Add filling information indexes
    if cst.EPI_FILLING in cloud_sample:
        band_filling = list(cloud_sample.coords[cst.BAND_FILLING].to_numpy())
        for band in band_filling:
            band_index = "{}_{}".format(cst.POINT_CLOUD_FILLING_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = "uint8"

    # Add ambiguity information index
    if cst.EPI_AMBIGUITY in cloud_sample:
        cloud_indexes_with_types[cst.EPI_AMBIGUITY] = "float32"

    return cloud_indexes_with_types


# pylint: disable=too-many-positional-arguments
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
        if cst.EPI_TEXTURE in clouds[cloud_id]:
            if "color_type" in cloud_item.attrs:
                color_types.append(cloud_item.attrs["color_type"])
    if color_types:
        color_type_set = set(color_types)
        if len(color_type_set) > 1:
            logging.warning("The tiles colors don't have the same type.")
        return color_types[0]

    return None


def filter_cloud(
    cloud: pd.DataFrame,
    index_elt_to_remove: List[int],
    filtered_elt_pos: bool = False,
) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame]]:
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
        cst.POINT_CLOUD_COORD_EPI_GEOM_I in cloud.columns
        and cst.POINT_CLOUD_COORD_EPI_GEOM_J in cloud.columns
        and cst.POINT_CLOUD_ID_IM_EPI in cloud.columns
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
            cst.POINT_CLOUD_COORD_EPI_GEOM_I,
            cst.POINT_CLOUD_COORD_EPI_GEOM_J,
            cst.POINT_CLOUD_ID_IM_EPI,
        ]

        removed_elt_pos_infos = cloud.loc[
            cloud.index.values[index_elt_to_remove], labels
        ].values

        removed_elt_pos_infos = pd.DataFrame(
            removed_elt_pos_infos, columns=labels
        )
    else:
        removed_elt_pos_infos = None

    # remove points from the cloud
    cloud = cloud.drop(index=cloud.index.values[index_elt_to_remove])

    return cloud, removed_elt_pos_infos


# pylint: disable=too-many-positional-arguments
def depth_map_dataset_to_dataframe(  # noqa: C901
    cloud_dataset: xr.Dataset,
    dsm_epsg: int,
    xmin: float = None,
    xmax: float = None,
    ymin: int = None,
    ymax: int = None,
    margin: float = 0,
    with_coords: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe
    structured with the following labels:

        - if no colors in input and no mask data present in cloud_list datasets:
            labels=[cst.X, cst.Y, cst.Z] \
            The combined cloud has x, y, z columns

        - if no colors in input and mask data present in cloud_list datasets:
            labels=[cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_MSK]\
            The mask values are added to the dataframe.

        - if colors are set in input and mask data are present \
            in the cloud_list datasets:
           labels=[cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_MSK,\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"0",\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"1",\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"2"]\
            Color channels information are added to the dataframe.

        - if colors in input, mask data present in the cloud_list datasets and\
            the with_coords option is activated:
             labels=[cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_MSK,\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"0",\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"1",\
                     cst.POINT_CLOUD_CLR_KEY_ROOT+"2"\
                     cst.POINT_CLOUD_COORD_EPI_GEOM_I,\
                     cst.POINT_CLOUD_COORD_EPI_GEOM_J,\
                     cst.POINT_CLOUD_ID_IM_EPI]\
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
    epsg = int(cloud_dataset.attrs[cst.EPSG])

    # Compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    # Create point cloud index
    cloud_indexes_with_types = create_point_cloud_index(cloud_dataset)

    # Add coords
    if with_coords:
        cloud_indexes_with_types.update(
            {
                cst.POINT_CLOUD_COORD_EPI_GEOM_I: "uint16",
                cst.POINT_CLOUD_COORD_EPI_GEOM_J: "uint16",
            }
        )

    cloud_indexes = list(cloud_indexes_with_types.keys())

    # crop point cloud if is not created from tif depth maps
    if (
        cst.EPI_MARGINS in cloud_dataset.attrs
        and cst.ROI in cloud_dataset.attrs
    ):
        ref_roi, _, _ = dense_matching_wrappers.compute_cropped_roi(
            cloud_dataset.attrs[cst.EPI_MARGINS],
            0,
            cloud_dataset.attrs[cst.ROI],
            cloud_dataset.sizes[cst.ROW],
            cloud_dataset.sizes[cst.COL],
        )
        cloud_dataset = cloud_dataset.isel(
            row=slice(ref_roi[1], ref_roi[3]),
            col=slice(ref_roi[0], ref_roi[2]),
        )

    full_x = cloud_dataset[cst.X].values
    full_y = cloud_dataset[cst.Y].values
    full_z = cloud_dataset[cst.Z].values

    # get mask of points inside the roi (plus margins)
    if roi:
        # Compute terrain tile bounds
        # if the point clouds are not in the same referential as the roi,
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
            cloud_dataset,
            full_x,
            full_y,
        )

        # if the point clouds are not in the same referential as the roi,
        # retrieve the initial values
        if epsg != dsm_epsg:
            full_x = cloud_dataset[cst.X].values
            full_y = cloud_dataset[cst.Y].values

        # if no point is found, return Empty cloud
        if terrain_tile_data_msk_pos[0].shape[0] == 0:
            logging.error("No points found to transform")
            return pd.DataFrame(columns=cloud_indexes), epsg

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

    # add additional information to point cloud
    arrays_to_add_to_point_cloud = [
        (cst.EPI_TEXTURE, cst.POINT_CLOUD_CLR_KEY_ROOT),
        (cst.EPI_MSK, cst.POINT_CLOUD_MSK),
        (cst.EPI_CLASSIFICATION, cst.POINT_CLOUD_CLASSIF_KEY_ROOT),
        (cst.EPI_FILLING, cst.POINT_CLOUD_FILLING_KEY_ROOT),
    ]

    # Add layer inf and sup
    for array_name in cloud_dataset:
        if cst.POINT_CLOUD_LAYER_SUP_OR_INF_ROOT in array_name:
            arrays_to_add_to_point_cloud.append((array_name, array_name))

    # add performance map
    for array_name in cloud_dataset:
        if cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT in array_name:
            arrays_to_add_to_point_cloud.append((array_name, array_name))

    # add ambiguity layer, drop confidence_* layers
    for array_name in cloud_dataset:
        if (
            cst.EPI_AMBIGUITY in array_name
            and cst.EPI_CONFIDENCE_KEY_ROOT not in array_name
        ):
            arrays_to_add_to_point_cloud.append((array_name, array_name))

    # add denoising info layers
    for array_name in cloud_dataset:
        if cst.EPI_DENOISING_INFO_KEY_ROOT in array_name:
            arrays_to_add_to_point_cloud.append((array_name, array_name))

    for input_band, output_column in arrays_to_add_to_point_cloud:
        add_information_to_cloud(
            cloud_dataset,
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
            cloud_indexes.index(cst.POINT_CLOUD_COORD_EPI_GEOM_I), :
        ] = np.ravel(coords_line)
        flatten_cloud[
            cloud_indexes.index(cst.POINT_CLOUD_COORD_EPI_GEOM_J), :
        ] = np.ravel(coords_col)

    # Transpose point cloud
    flatten_cloud = flatten_cloud.transpose()

    # remove masked data (pandora + out of the terrain tile points)
    crop_terrain_tile_data_msk = (
        cloud_dataset[cst.POINT_CLOUD_CORR_MSK].values[
            bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
        ]
        == 255
    )

    if roi:
        crop_terrain_tile_data_msk = np.logical_and(
            crop_terrain_tile_data_msk,
            terrain_tile_data_msk[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1],
        )

    flatten_cloud = filter_cloud_with_mask(
        flatten_cloud, crop_terrain_tile_data_msk
    )

    # Remove points with nan values on X, Y or Z
    xyz_indexes = np.array(
        [
            cloud_indexes.index(cst.X),
            cloud_indexes.index(cst.Y),
            cloud_indexes.index(cst.Z),
        ]
    )
    flatten_cloud = flatten_cloud[
        ~np.any(np.isnan(flatten_cloud[:, xyz_indexes]), axis=1)
    ]

    pd_cloud = pd.DataFrame(flatten_cloud, columns=cloud_indexes)
    pd_cloud = pd_cloud.astype(cloud_indexes_with_types)

    return pd_cloud, epsg
