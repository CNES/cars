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
from typing import List, Tuple

# Third party imports
import numpy as np
import pandas
import rasterio as rio
import xarray as xr

import cars.orchestrator.orchestrator as ocht
from cars.applications.dense_matching import dense_matching_wrappers
from cars.applications.point_cloud_fusion import pc_fusion_wrappers as pc_wrap

# CARS imports
from cars.core import constants as cst
from cars.core import inputs, preprocessing, projection, tiling
from cars.data_structures import cars_dataset, cars_dict


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
            labels=[cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_MSK]\
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

    epsg = pc_wrap.get_epsg(cloud_list)

    # compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    cloud_indexes_with_types = pc_wrap.create_point_cloud_index(cloud_list[0])

    if with_coords:
        cloud_indexes_with_types.update(
            {cst.POINT_CLOUD_COORD_EPI_GEOM_I: "uint16"}
        )

    cloud_indexes = list(cloud_indexes_with_types.keys())

    # iterate through input clouds
    combined_cloud = np.zeros((0, len(cloud_indexes)))
    nb_points = 0
    for cloud_global_id, point_cloud in zip(  # noqa: B905
        cloud_ids, cloud_list
    ):
        full_x = point_cloud[cst.X]
        full_y = point_cloud[cst.Y]
        full_z = point_cloud[cst.Z]

        # get mask of points inside the roi (plus margins)
        if roi:
            # Compute terrain tile bounds
            # if the point clouds are not in the same referential as the roi,
            # it is converted using the dsm_epsg
            (
                terrain_tile_data_msk,
                terrain_tile_data_msk_pos,
            ) = pc_wrap.compute_terrain_msk(
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
            )

            # if the point clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = point_cloud[cst.X]
                full_y = point_cloud[cst.Y]

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
        crop_cloud[cloud_indexes.index(cst.POINT_CLOUD_GLOBAL_ID), :] = (
            cloud_global_id
        )

        # add the original image coordinates information to the current cloud
        if with_coords:
            coords_line = np.linspace(
                bbox[0], bbox[1], num=bbox[1] - bbox[0] + 1
            )
            crop_cloud[
                cloud_indexes.index(cst.POINT_CLOUD_COORD_EPI_GEOM_I), :
            ] = coords_line

        # Transpose point cloud
        crop_cloud = crop_cloud.transpose()

        # remove masked data (pandora + out of the terrain tile points)
        crop_terrain_tile_data_msk = (
            point_cloud[cst.POINT_CLOUD_CORR_MSK][bbox[0] : bbox[1]] == 255
        )
        if roi:
            crop_terrain_tile_data_msk = np.logical_and(
                crop_terrain_tile_data_msk,
                terrain_tile_data_msk[bbox[0] : bbox[1]],
            )

        crop_cloud = pc_wrap.filter_cloud_with_mask(
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
    epsg = pc_wrap.get_epsg(cloud_list)

    # Compute margin/roi and final number of data to add to the combined cloud
    roi = (
        xmin is not None
        and xmax is not None
        and ymin is not None
        and ymax is not None
    )

    # Create point cloud index
    cloud_indexes_with_types = pc_wrap.create_point_cloud_index(cloud_list[0])

    # Add coords
    if with_coords:
        cloud_indexes_with_types.update(
            {
                cst.POINT_CLOUD_COORD_EPI_GEOM_I: "uint16",
                cst.POINT_CLOUD_COORD_EPI_GEOM_J: "uint16",
                cst.POINT_CLOUD_ID_IM_EPI: "uint16",
            }
        )

    cloud_indexes = list(cloud_indexes_with_types.keys())

    # Iterate through input clouds
    combined_cloud = np.zeros((0, len(cloud_indexes)))
    nb_points = 0
    for cloud_global_id, (cloud_list_id, point_cloud) in zip(  # noqa: B905
        cloud_id, enumerate(cloud_list)
    ):
        # crop point cloud if is not created from tif depth maps
        if (
            cst.EPI_MARGINS in point_cloud.attrs
            and cst.ROI in point_cloud.attrs
        ):
            ref_roi, _, _ = dense_matching_wrappers.compute_cropped_roi(
                point_cloud.attrs[cst.EPI_MARGINS],
                0,
                point_cloud.attrs[cst.ROI],
                point_cloud.sizes[cst.ROW],
                point_cloud.sizes[cst.COL],
            )
            point_cloud = point_cloud.isel(
                row=slice(ref_roi[1], ref_roi[3]),
                col=slice(ref_roi[0], ref_roi[2]),
            )

        full_x = point_cloud[cst.X].values
        full_y = point_cloud[cst.Y].values
        full_z = point_cloud[cst.Z].values

        # get mask of points inside the roi (plus margins)
        if roi:
            # Compute terrain tile bounds
            # if the point clouds are not in the same referential as the roi,
            # it is converted using the dsm_epsg
            (
                terrain_tile_data_msk,
                terrain_tile_data_msk_pos,
            ) = pc_wrap.compute_terrain_msk(
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
            )

            # if the point clouds are not in the same referential as the roi,
            # retrieve the initial values
            if epsg != dsm_epsg:
                full_x = point_cloud[cst.X].values
                full_y = point_cloud[cst.Y].values

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

        # add index of original point cloud
        flatten_cloud[cloud_indexes.index(cst.POINT_CLOUD_GLOBAL_ID), :] = (
            cloud_global_id
        )

        # add additional information to point cloud
        arrays_to_add_to_point_cloud = [
            (cst.EPI_TEXTURE, cst.POINT_CLOUD_CLR_KEY_ROOT),
            (cst.EPI_MSK, cst.POINT_CLOUD_MSK),
            (cst.EPI_CLASSIFICATION, cst.POINT_CLOUD_CLASSIF_KEY_ROOT),
            (cst.EPI_FILLING, cst.POINT_CLOUD_FILLING_KEY_ROOT),
        ]

        # Add layer inf and sup
        for array_name in point_cloud:
            if cst.POINT_CLOUD_LAYER_SUP_OR_INF_ROOT in array_name:
                arrays_to_add_to_point_cloud.append((array_name, array_name))

        # add performance map
        for array_name in point_cloud:
            if cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT in array_name:
                arrays_to_add_to_point_cloud.append((array_name, array_name))

        # add confidence layers
        for array_name in point_cloud:
            if cst.EPI_CONFIDENCE_KEY_ROOT in array_name:
                arrays_to_add_to_point_cloud.append((array_name, array_name))

        # add denoising info layers
        for array_name in point_cloud:
            if cst.EPI_DENOISING_INFO_KEY_ROOT in array_name:
                arrays_to_add_to_point_cloud.append((array_name, array_name))

        for input_band, output_column in arrays_to_add_to_point_cloud:
            pc_wrap.add_information_to_cloud(
                point_cloud,
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
            flatten_cloud[cloud_indexes.index(cst.POINT_CLOUD_ID_IM_EPI), :] = (
                cloud_list_id
            )

        # Transpose point cloud
        flatten_cloud = flatten_cloud.transpose()

        # remove masked data (pandora + out of the terrain tile points)
        crop_terrain_tile_data_msk = (
            point_cloud[cst.POINT_CLOUD_CORR_MSK].values[
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

        flatten_cloud = pc_wrap.filter_cloud_with_mask(
            nb_points, flatten_cloud, crop_terrain_tile_data_msk
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

        # Add current cloud to the combined one
        combined_cloud = np.concatenate([combined_cloud, flatten_cloud], axis=0)

    logging.debug("Received {} points to rasterize".format(nb_points))
    logging.debug(
        "Keeping {}/{} points "
        "inside rasterization grid".format(combined_cloud.shape[0], nb_points)
    )

    pd_cloud = pandas.DataFrame(combined_cloud, columns=cloud_indexes)
    pd_cloud = pd_cloud.astype(cloud_indexes_with_types)

    return pd_cloud, epsg


def create_combined_cloud_from_tif(
    clouds,
    clouds_id,
    epsg,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    margin=0,
):
    """
    Create combined cloud from tif point clouds

    :param clouds: list of clouds
    :type clouds: list(dict)
    :param clouds_id: list of global identificators associated to clouds
    :type clouds_id: list(str)
    :param epsg: epsg to convert point clouds to
    :type epsg: int or str
    :param xmin: min x coordinate
    :type xmin: float
    :param xmax: max x coordinate
    :type xmax: float
    :param ymin: min y coordinate
    :type ymin: float
    :param ymax: max y coordinate
    :type ymax: float

    :return: combined cloud, point cloud epsg
    :rtype: pandas Dataframe, int
    """
    clouds_pd_list = []
    color_types = []
    for cloud in clouds:
        for band_name in cloud["data"].keys():
            band_path = cloud["data"][band_name]
    # Create multiple pc pandas dataframes
    for cloud_file_id, cloud in zip(clouds_id, clouds):  # noqa: B905
        window = cloud["window"]
        cloud_epsg = cloud["cloud_epsg"]
        cloud_data_bands = []
        cloud_data_types = []
        cloud_data = {}
        for band_name in cloud["data"].keys():
            # open file and get data
            band_path = cloud["data"][band_name]

            if band_path is not None:
                if cst.POINT_CLOUD_CLR_KEY_ROOT in band_name:
                    # Get color type
                    color_types.append(
                        inputs.rasterio_get_image_type(band_path)
                    )

                if isinstance(band_path, dict):
                    for key in band_path:
                        sub_band_path = band_path[key]
                        sub_band_name = key
                        pc_wrap.read_band(
                            sub_band_name,
                            sub_band_path,
                            window,
                            cloud_data_bands,
                            cloud_data_types,
                            cloud_data,
                        )
                else:
                    pc_wrap.read_band(
                        band_name,
                        band_path,
                        window,
                        cloud_data_bands,
                        cloud_data_types,
                        cloud_data,
                    )

        # add source file id
        cloud_data[cst.POINT_CLOUD_GLOBAL_ID] = (
            np.ones(cloud_data[cst.X].shape) * cloud_file_id
        )
        cloud_data_bands.append(cst.POINT_CLOUD_GLOBAL_ID)
        cloud_data_types.append("uint16")

        # Create cloud pandas
        cloud_pd = pandas.DataFrame(cloud_data, columns=cloud_data_bands)

        # Post processing if 0 in data
        cloud_pd = cloud_pd.drop(
            cloud_pd.index[
                (cloud_pd[cst.X] == 0.0)  # pylint: disable=E1136
                | (cloud_pd[cst.Y] == 0.0)  # pylint: disable=E1136
            ]
        )

        cloud_pd = cloud_pd.drop(
            cloud_pd.index[
                (np.isnan(cloud_pd[cst.X]))  # pylint: disable=E1136
                | (np.isnan(cloud_pd[cst.Y]))  # pylint: disable=E1136
            ]
        )

        # Cast types according to band
        cloud_data_types = dict(
            zip(cloud_data_bands, cloud_data_types)  # noqa: B905
        )
        cloud_pd = cloud_pd.astype(cloud_data_types)

        # Convert pc if necessary
        if cloud_epsg != epsg:
            projection.point_cloud_conversion_dataframe(
                cloud_pd, cloud_epsg, epsg
            )

        # filter outside points considering mmargins
        pc_wrap.filter_cloud_tif(
            cloud_pd,
            list(
                np.array([xmin, xmax, ymin, ymax])
                + np.array([-margin, margin, -margin, margin])
            ),
        )

        # add to list of pandas pc
        clouds_pd_list.append(cloud_pd)

    # Merge pandas point clouds
    combined_pd_cloud = pandas.concat(
        clouds_pd_list,
        axis=0,
        join="outer",
    )

    # Get color type
    color_type_set = set(color_types)
    if len(color_type_set) > 1:
        logging.warning("The tiles colors don't have the same type.")
    color_type = None
    if len(color_types) > 0:
        color_type = color_types[0]

    return combined_pd_cloud, epsg, color_type


def generate_point_clouds(list_clouds, orchestrator, tile_size=1000):
    """
    Generate point cloud  cars Datasets from list

    :param list_clouds: list of clouds
    :type list_clouds: dict
    :param orchestrator: orchestrator
    :type orchestrator: Orchestrator
    :param tile_size: tile size
    :type tile_size: int

    :return list of point clouds
    :rtype: list(CarsDataset)
    """
    list_epipolar_point_clouds = []

    # Create cars datasets

    list_names = list(list_clouds.keys())

    for cloud_id, key in enumerate(list_clouds):
        cloud = list_clouds[key]
        cars_ds = cars_dataset.CarsDataset(dataset_type="arrays")

        epipolar_size_x, epipolar_size_y = inputs.rasterio_get_size(cloud["x"])

        # Generate tiling grid
        cars_ds.tiling_grid = tiling.generate_tiling_grid(
            0,
            0,
            epipolar_size_y,
            epipolar_size_x,
            tile_size,
            tile_size,
        )

        color_type = None
        if cst.POINT_CLOUD_CLR_KEY_ROOT in cloud:
            # Get color type
            color_type = inputs.rasterio_get_image_type(
                cloud[cst.POINT_CLOUD_CLR_KEY_ROOT]
            )
        cars_ds.attributes = {
            "color_type": color_type,
            "source_pc_names": list_names,
        }

        for col in range(cars_ds.shape[1]):
            for row in range(cars_ds.shape[0]):
                # get window
                window = cars_ds.get_window_as_dict(row, col)
                rio_window = cars_dataset.generate_rasterio_window(window)

                # Generate tile
                cars_ds[row, col] = orchestrator.cluster.create_task(
                    generate_pc_wrapper, nout=1
                )(
                    cloud,
                    rio_window,
                    color_type=color_type,
                    cloud_id=cloud_id,
                    list_cloud_ids=list_names,
                )

        list_epipolar_point_clouds.append(cars_ds)

    return list_epipolar_point_clouds


def generate_pc_wrapper(  # noqa: C901
    cloud, window, color_type=None, cloud_id=None, list_cloud_ids=None
):
    """
    Generate point cloud  dataset

    :param cloud: cloud dict
    :param window: window
    :param color_type: color type
    :param cloud_id: cloud id
    :param list_cloud_ids: list of global cloud ids

    :return cloud
    :rtype: xr.Dataset
    """

    list_keys = cloud.keys()
    # x y z
    data_x = pc_wrap.read_image_full(cloud["x"], window=window, squeeze=True)
    data_y = pc_wrap.read_image_full(cloud["y"], window=window, squeeze=True)
    data_z = pc_wrap.read_image_full(cloud["z"], window=window, squeeze=True)

    shape = data_x.shape

    row = np.arange(0, shape[0])
    col = np.arange(0, shape[1])

    values = {
        cst.X: ([cst.ROW, cst.COL], data_x),  # longitudes
        cst.Y: ([cst.ROW, cst.COL], data_y),  # latitudes
        cst.Z: ([cst.ROW, cst.COL], data_z),
    }

    coords = {cst.ROW: row, cst.COL: col}

    attributes = {"cloud_id": cloud_id, "number_of_pc": len(list_cloud_ids)}

    for key in list_keys:
        if cloud[key] is None and key != "mask":
            pass
        elif key in ["x", "y", "z"]:
            pass
        elif key == cst.POINT_CLOUD_LAYER_INF:
            data_z_inf = pc_wrap.read_image_full(
                cloud[cst.POINT_CLOUD_LAYER_INF], window=window, squeeze=True
            )
            values[cst.POINT_CLOUD_LAYER_INF] = ([cst.ROW, cst.COL], data_z_inf)
        elif key == cst.POINT_CLOUD_LAYER_SUP:
            data_z_sup = pc_wrap.read_image_full(
                cloud[cst.POINT_CLOUD_LAYER_SUP], window=window, squeeze=True
            )
            values[cst.POINT_CLOUD_LAYER_SUP] = ([cst.ROW, cst.COL], data_z_sup)
        elif key == "point_cloud_epsg":
            attributes["epsg"] = cloud[key]
        elif key == "mask":
            if cloud[key] is None:
                data = ~np.isnan(data_x) * 255
            else:
                data = pc_wrap.read_image_full(
                    cloud[key], window=window, squeeze=True
                )
            values[cst.POINT_CLOUD_CORR_MSK] = ([cst.ROW, cst.COL], data)

        elif key == cst.EPI_CLASSIFICATION:
            data = pc_wrap.read_image_full(
                cloud[key], window=window, squeeze=False
            )
            descriptions = list(inputs.get_descriptions_bands(cloud[key]))
            values[cst.EPI_CLASSIFICATION] = (
                [cst.BAND_CLASSIF, cst.ROW, cst.COL],
                data,
            )
            if cst.BAND_CLASSIF not in coords:
                coords[cst.BAND_CLASSIF] = descriptions

        elif key == cst.EPI_TEXTURE:
            data = pc_wrap.read_image_full(
                cloud[key], window=window, squeeze=False
            )
            descriptions = list(inputs.get_descriptions_bands(cloud[key]))
            attributes["color_type"] = color_type
            values[cst.EPI_TEXTURE] = ([cst.BAND_IM, cst.ROW, cst.COL], data)

            if cst.EPI_TEXTURE not in coords:
                coords[cst.BAND_IM] = descriptions

        elif key == cst.EPI_CONFIDENCE_KEY_ROOT:
            for sub_key in cloud[key].keys():
                data = pc_wrap.read_image_full(
                    cloud[key][sub_key], window=window, squeeze=True
                )
                values[sub_key] = ([cst.ROW, cst.COL], data)

        elif key == cst.EPI_FILLING:
            data = pc_wrap.read_image_full(
                cloud[key], window=window, squeeze=False
            )
            descriptions = list(inputs.get_descriptions_bands(cloud[key]))
            values[cst.EPI_FILLING] = (
                [cst.BAND_FILLING, cst.ROW, cst.COL],
                data,
            )
            if cst.BAND_FILLING not in coords:
                coords[cst.BAND_FILLING] = descriptions

        elif key == cst.EPI_PERFORMANCE_MAP:
            data = pc_wrap.read_image_full(
                cloud[key], window=window, squeeze=True
            )
            descriptions = list(inputs.get_descriptions_bands(cloud[key]))
            values[cst.EPI_PERFORMANCE_MAP] = (
                [cst.ROW, cst.COL],
                data,
            )
            if cst.BAND_PERFORMANCE_MAP not in coords:
                coords[cst.BAND_PERFORMANCE_MAP] = descriptions

        else:
            data = pc_wrap.read_image_full(
                cloud[key], window=window, squeeze=True
            )
            if data.shape == 2:
                values[key] = ([cst.ROW, cst.COL], data)
            else:
                logging.error(" {} data not managed".format(key))

    xr_cloud = xr.Dataset(values, coords=coords)
    xr_cloud.attrs = attributes

    return xr_cloud


def transform_input_pc(
    list_epipolar_point_clouds,
    epsg,
    roi_poly=None,
    epipolar_tile_size=600,
    orchestrator=None,
):
    """
    Transform point clouds from inputs into point cloud fusion application
    format.
    Create tiles, with x y min max informations.

    :param list_epipolar_point_clouds: list of epipolar point clouds
    :type list_epipolar_point_clouds: dict
    :param epsg: epsg
    :type epsg: int, str
    :param roi_poly: roi polygon
    :type roi_poly: Polygon
    :param epipolar_tile_size: size of tile used for tiling the tif files
    :type epipolar_tile_size: int

    :return list of point clouds
    :rtype: list(CarsDataset type dict)

    """

    if orchestrator is None:
        # Create default sequential orchestrator for current application
        # be awere, no out_json will be shared between orchestrators
        # No files saved
        cars_orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )
    else:
        cars_orchestrator = orchestrator

    list_epipolar_point_clouds_by_tiles = []

    # For each stereo pair
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []
    for pair_key, items in list_epipolar_point_clouds.items():
        # Generate CarsDataset
        epi_pc = cars_dataset.CarsDataset("dict")
        tif_size = inputs.rasterio_get_size(items[cst.X])
        epi_pc.tiling_grid = tiling.generate_tiling_grid(
            0,
            0,
            tif_size[0],
            tif_size[1],
            epipolar_tile_size,
            epipolar_tile_size,
        )

        # Add to replace list so tiles will be readable at the same time
        [saving_info_pc] = cars_orchestrator.get_saving_infos([epi_pc])
        cars_orchestrator.add_to_replace_lists(
            epi_pc, cars_ds_name="epi_pc_min_max"
        )

        # Open the TIFF and get bounds from lon/lat min and max values
        for row in range(epi_pc.shape[0]):
            for col in range(epi_pc.shape[1]):
                window = rio.windows.Window.from_slices(
                    (
                        epi_pc.tiling_grid[row, col, 0],
                        epi_pc.tiling_grid[row, col, 1],
                    ),
                    (
                        epi_pc.tiling_grid[row, col, 2],
                        epi_pc.tiling_grid[row, col, 3],
                    ),
                )

                # Update saving info for row and col
                # /!\ BE AWARE : this is not the conventionnal way
                # to parallelise tasks in CARS
                full_saving_info_pc = ocht.update_saving_infos(
                    saving_info_pc, row=row, col=col
                )

                epi_pc[row, col] = cars_orchestrator.cluster.create_task(
                    compute_x_y_min_max_wrapper, nout=1
                )(
                    items,
                    epsg,
                    window,
                    saving_info=full_saving_info_pc,
                )
        epi_pc.attributes["source_pc_name"] = pair_key
        list_epipolar_point_clouds_by_tiles.append(epi_pc)

    # Breakpoint : compute
    # /!\ BE AWARE : this is not the conventionnal way
    # to parallelise tasks in CARS
    cars_orchestrator.breakpoint()

    # Get all local min and max
    for computed_epi_pc in list_epipolar_point_clouds_by_tiles:
        pc_xmin_list, pc_ymin_list, pc_xmax_list, pc_ymax_list = [], [], [], []
        for row in range(computed_epi_pc.shape[0]):
            for col in range(computed_epi_pc.shape[1]):
                local_x_y_min_max = computed_epi_pc[row, col].data[
                    "x_y_min_max"
                ]

                if np.all(np.isfinite(local_x_y_min_max)):
                    # Add for global
                    xmin_list.append(local_x_y_min_max[0])
                    xmax_list.append(local_x_y_min_max[1])
                    ymin_list.append(local_x_y_min_max[2])
                    ymax_list.append(local_x_y_min_max[3])
                    # Add for current CarsDS
                    pc_xmin_list.append(local_x_y_min_max[0])
                    pc_xmax_list.append(local_x_y_min_max[1])
                    pc_ymin_list.append(local_x_y_min_max[2])
                    pc_ymax_list.append(local_x_y_min_max[3])

                # Simplify data
                computed_epi_pc[row, col] = computed_epi_pc[row, col].data

        # Add min max for current point cloud CarsDataset
        if len(pc_xmin_list) > 0:
            computed_epi_pc.attributes["xmin"] = min(pc_xmin_list)
            computed_epi_pc.attributes["ymin"] = min(pc_ymin_list)
            computed_epi_pc.attributes["xmax"] = max(pc_xmax_list)
            computed_epi_pc.attributes["ymax"] = max(pc_ymax_list)
            computed_epi_pc.attributes["epsg"] = epsg

    # Define a terrain tiling from the terrain bounds (in terrain epsg)
    if len(xmin_list) > 0:
        global_xmin = min(xmin_list)
        global_xmax = max(xmax_list)
        global_ymin = min(ymin_list)
        global_ymax = max(ymax_list)
    else:
        raise RuntimeError("All the depth maps are full of nan")

    if roi_poly is not None:
        (
            global_xmin,
            global_ymin,
            global_xmax,
            global_ymax,
        ) = preprocessing.crop_terrain_bounds_with_roi(
            roi_poly, global_xmin, global_ymin, global_xmax, global_ymax
        )

    terrain_bbox = [global_xmin, global_ymin, global_xmax, global_ymax]

    logging.info("terrain bbox in epsg {}: {}".format(str(epsg), terrain_bbox))

    return (terrain_bbox, list_epipolar_point_clouds_by_tiles)


def compute_x_y_min_max_wrapper(items, epsg, window, saving_info=None):
    """
    Compute bounds from item and create CarsDict filled with point cloud
    information: file paths, bounds, epsg, window

    :param items: point cloud
    :type items: dict
    :param epsg: epsg
    :type epsg: int
    :param window: window to use
    :type window: dict
    :param saving_info: saving infos
    :type saving_info: dict

    :return: Tile ready to use
    :rtype: CarsDict

    """
    x_y_min_max = pc_wrap.get_min_max_band(
        items[cst.X],
        items[cst.Y],
        items[cst.Z],
        items[cst.PC_EPSG],
        epsg,
        window=window,
    )

    data_dict = {
        cst.X: items[cst.X],
        cst.Y: items[cst.Y],
        cst.Z: items[cst.Z],
        cst.POINT_CLOUD_CLR_KEY_ROOT: items[cst.POINT_CLOUD_CLR_KEY_ROOT],
    }
    if cst.POINT_CLOUD_MSK in items:
        data_dict[cst.POINT_CLOUD_MSK] = items[cst.POINT_CLOUD_MSK]
    if cst.POINT_CLOUD_CLASSIF_KEY_ROOT in items:
        data_dict[cst.POINT_CLOUD_CLASSIF_KEY_ROOT] = items[
            cst.POINT_CLOUD_CLASSIF_KEY_ROOT
        ]
    if cst.POINT_CLOUD_FILLING_KEY_ROOT in items:
        data_dict[cst.POINT_CLOUD_FILLING_KEY_ROOT] = items[
            cst.POINT_CLOUD_FILLING_KEY_ROOT
        ]
    if cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT in items:
        data_dict[cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT] = items[
            cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT
        ]
    if cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT in items:
        data_dict[cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT] = items[
            cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT
        ]
    if cst.EPI_Z_INF in items:
        data_dict[cst.POINT_CLOUD_LAYER_SUP] = items[cst.EPI_Z_INF]
    if cst.EPI_Z_SUP in items:
        data_dict[cst.POINT_CLOUD_LAYER_INF] = items[cst.EPI_Z_SUP]

    # create dict
    tile = {
        "data": data_dict,
        "x_y_min_max": x_y_min_max,
        "window": window,
        "cloud_epsg": items[cst.PC_EPSG],
    }

    # add saving infos
    res = cars_dict.CarsDict(tile)
    cars_dataset.fill_dict(res, saving_info=saving_info)

    return res


def get_corresponding_tiles_tif(
    terrain_tiling_grid,
    list_epipolar_point_clouds_with_loc,
    margins=0,
    orchestrator=None,
):
    """
    Get point cloud tiles to use for terrain region

    :param terrain_tiling_grid: tiling grid
    :type terrain_tiling_grid: np.ndarray
    :param row: tiling row
    :type row: int
    :param col: col
    :type col: int
    :param list_epipolar_point_clouds_with_loc: list of left point clouds
    :type list_epipolar_point_clouds_with_loc: list(CarsDataset)
    :param margins: margin to use in point clouds
    :type margins: float

    :return: CarsDataset containing list of point cloud tiles to use
        to terrain tile
    :rtype: CarsDataset

    """

    if orchestrator is None:
        # Create default sequential orchestrator for current
        # application
        # be aware, no out_json will be shared between orchestrators
        # No files saved
        cars_orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )
    else:
        cars_orchestrator = orchestrator

    # Compute correspondances for every tile
    # Create Carsdataset containing a tile for every point cloud
    list_corresp_cars_ds = cars_dataset.CarsDataset("dict")
    # Create fake tiling grid , not used later
    list_corresp_cars_ds.tiling_grid = tiling.generate_tiling_grid(
        0,
        0,
        len(list_epipolar_point_clouds_with_loc),
        len(list_epipolar_point_clouds_with_loc),
        1,
        len(list_epipolar_point_clouds_with_loc),
    )
    # Add to replace list so tiles will be readable at the same time
    [saving_info_pc] = cars_orchestrator.get_saving_infos(
        [list_corresp_cars_ds]
    )
    cars_orchestrator.add_to_replace_lists(
        list_corresp_cars_ds, cars_ds_name="epi_pc_corresp"
    )

    for row_fake_cars_ds in range(list_corresp_cars_ds.shape[0]):
        # Update saving info for row and col
        full_saving_info_pc = ocht.update_saving_infos(
            saving_info_pc, row=row_fake_cars_ds, col=0
        )

        # /!\ BE AWARE : this is not the conventionnal way
        # to parallelise tasks in CARS

        list_corresp_cars_ds[
            row_fake_cars_ds, 0
        ] = cars_orchestrator.cluster.create_task(
            compute_correspondance_single_pc_terrain, nout=1
        )(
            list_epipolar_point_clouds_with_loc[row_fake_cars_ds],
            row_fake_cars_ds,
            terrain_tiling_grid,
            margins=margins,
            saving_info=full_saving_info_pc,
        )

    # Breakpoint : compute
    # /!\ BE AWARE : this is not the conventionnal way
    # to parallelise tasks in CARS
    cars_orchestrator.breakpoint()

    # Create res
    terrain_correspondances = cars_dataset.CarsDataset("dict")
    terrain_correspondances.tiling_grid = terrain_tiling_grid

    for row in range(terrain_correspondances.shape[0]):
        for col in range(terrain_correspondances.shape[1]):
            # get terrain region

            # Terrain grid [row, j, :] = [xmin, xmax, ymin, ymax]
            # terrain region = [xmin, ymin, xmax, ymax]
            terrain_region = [
                terrain_tiling_grid[row, col, 0],
                terrain_tiling_grid[row, col, 2],
                terrain_tiling_grid[row, col, 1],
                terrain_tiling_grid[row, col, 3],
            ]

            # Get required_point_clouds_left
            required_point_clouds = []
            for corresp_row in range(list_corresp_cars_ds.shape[0]):
                # each tile in list_corresp contains a CarsDict,
                # containing a CarsDataset filled with list
                corresp = list_corresp_cars_ds[corresp_row, 0].data[
                    "corresp_cars_ds"
                ][row, col]
                required_point_clouds += corresp

            terrain_correspondances[row, col] = {
                "terrain_region": terrain_region,
                "required_point_clouds": required_point_clouds,
            }

    return terrain_correspondances


def compute_correspondance_single_pc_terrain(
    epi_pc,
    epi_pc_id,
    terrain_tiling_grid,
    margins=0,
    saving_info=None,
):
    """
    Compute correspondances for each terrain tile, with current point cloud

    :param epi_pc: point cloud
    :type epi_pc: dict
    :param epi_pc_id: identificator of the file of the point cloud
    :type epi_pc_id: int
    :param terrain_tiling_grid: tiling grid
    :type terrain_tiling_grid: np.ndarray
    :param margins: margin to use in point clouds
    :type margins: float

    :return: CarsDict containing list of point cloud tiles to use for each
         terrain tile:

    :rtype: CarsDict

    """

    # Create fake CarsDataset only for 2d structure
    terrain_corresp = cars_dataset.CarsDataset("dict")
    terrain_corresp.tiling_grid = terrain_tiling_grid

    for terrain_row in range(terrain_corresp.shape[0]):
        for terrain_col in range(terrain_corresp.shape[1]):
            # Initialisae to empty list
            terrain_corresp[terrain_row, terrain_col] = []

            # Terrain grid [row, j, :] = [xmin, xmax, ymin, ymax]
            # terrain region = [xmin, ymin, xmax, ymax]
            terrain_region = [
                terrain_tiling_grid[terrain_row, terrain_col, 0],
                terrain_tiling_grid[terrain_row, terrain_col, 2],
                terrain_tiling_grid[terrain_row, terrain_col, 1],
                terrain_tiling_grid[terrain_row, terrain_col, 3],
            ]
            region_with_margin = list(
                np.array(terrain_region)
                + np.array([-margins, margins, -margins, margins])
            )

            # Convert the bounds of the terrain tile into shapely polygon
            # region: [xmin, ymin, xmax, ymax],
            # convert_to_polygon needs : [xmin, xmax, ymin, ymax]

            terrain_tile_polygon = pc_wrap.convert_to_polygon(
                [
                    region_with_margin[0],
                    region_with_margin[2],
                    region_with_margin[1],
                    region_with_margin[3],
                ]
            )

            for tile_row in range(epi_pc.shape[0]):
                for tile_col in range(epi_pc.shape[1]):
                    x_y_min_max = epi_pc[tile_row, tile_col]["x_y_min_max"]

                    # Convert the bounds of the point cloud tile into
                    #  shapely point
                    if np.all(np.isfinite(x_y_min_max)):
                        point_cloud_tile_polygon = pc_wrap.convert_to_polygon(
                            x_y_min_max
                        )

                        if pc_wrap.intersect_polygons(
                            terrain_tile_polygon, point_cloud_tile_polygon
                        ):
                            # add to required
                            terrain_corresp[terrain_row, terrain_col].append(
                                (epi_pc[tile_row, tile_col], epi_pc_id)
                            )

    # add saving infos
    dict_with_corresp_cars_ds = cars_dict.CarsDict(
        {"corresp_cars_ds": terrain_corresp}
    )
    cars_dataset.fill_dict(dict_with_corresp_cars_ds, saving_info=saving_info)

    return dict_with_corresp_cars_ds
