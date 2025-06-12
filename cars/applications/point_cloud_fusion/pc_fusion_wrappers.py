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
import rasterio as rio
import xarray as xr
from shapely import geometry, length

# CARS imports
from cars.core import constants as cst
from cars.core import inputs, preprocessing, projection, tiling


def get_epsg(cloud_list):
    """
    Extract epsg from cloud list and check if all the same

    :param cloud_list: list of the point clouds
    """
    epsg = None
    for point_cloud in cloud_list:
        if epsg is None:
            epsg = int(point_cloud.attrs[cst.EPSG])
        elif int(point_cloud.attrs[cst.EPSG]) != epsg:
            logging.error("All point clouds do not have the same epsg code")

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

    crop_cloud = np.delete(crop_cloud, crop_terrain_tile_data_msk_pos[0], 0)

    return crop_cloud


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
        cst.POINT_CLOUD_GLOBAL_ID: "uint16",
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
    if cst.EPI_IMAGE in cloud_sample:
        band_color = list(cloud_sample.coords[cst.BAND_IM].to_numpy())
        image_type = "float32"
        if "image_type" in cloud_sample.attrs:
            image_type = cloud_sample.attrs["image_type"]
        for band in band_color:
            band_index = "{}_{}".format(cst.POINT_CLOUD_CLR_KEY_ROOT, band)
            cloud_indexes_with_types[band_index] = image_type

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


def add_cloud_filtering_msk(
    clouds_list: List[xr.Dataset],
    elt_pos_infos: pd.DataFrame,
    mask_label: str,
    mask_value: int = 255,
):
    """
    Add a uint16 mask labeled 'mask_label' to the clouds in clouds_list.
    (in-line function)

    TODO only used in tests

    :param clouds_list: Input list of clouds
    :param elt_pos_infos: pandas dataframe
        composed of cst.POINT_CLOUD_COORD_EPI_GEOM_I,
        cst.POINT_CLOUD_COORD_EPI_GEOM_J, cst.POINT_CLOUD_ID_IM_EPI columns
        as computed in the create_combined_cloud function.
        Those information are used to retrieve the point position
        in its original epipolar image.
    :param mask_label: label to give to the mask in the datasets
    :param mask_value: filtered elements value in the mask
    """

    # Verify that the elt_pos_infos is consistent
    if (
        elt_pos_infos is None
        or cst.POINT_CLOUD_COORD_EPI_GEOM_I not in elt_pos_infos.columns
        or cst.POINT_CLOUD_COORD_EPI_GEOM_J not in elt_pos_infos.columns
        or cst.POINT_CLOUD_ID_IM_EPI not in elt_pos_infos.columns
    ):
        logging.warning(
            "Cannot generate filtered elements mask, "
            "no information about the point's"
            " original position in the epipolar image is given"
        )

    else:
        elt_index = elt_pos_infos.loc[:, cst.POINT_CLOUD_ID_IM_EPI].to_numpy()

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
                        cst.POINT_CLOUD_COORD_EPI_GEOM_I,
                    ].iat[0]
                )
                j = int(
                    elt_pos_infos.loc[
                        cur_elt_index[elt_pos],
                        cst.POINT_CLOUD_COORD_EPI_GEOM_J,
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


def create_polygon_from_list_points(list_points):
    """
    Create a Shapely polygon from list of points

    :param list_points: list of (x, y) coordinates
    :type list_points: list

    :return: Polygon
    :rtype: shapely Polygon

    """

    list_shapely_points = []
    for point in list_points:
        list_shapely_points.append((point[0], point[1]))

    # Add first
    first_point = list_points[0]
    list_shapely_points.append((first_point[0], first_point[1]))

    poly = geometry.Polygon(list_shapely_points)

    return poly


def compute_epsg_from_point_cloud(list_epipolar_point_clouds):
    """
    Compute epsg to use from list of tif point clouds

    :param list_epipolar_point_clouds: list of epipolar point clouds
    :type list_epipolar_point_clouds: list(dict)

    :return: epsg
    :rtype: int
    """

    # Get epsg from first point cloud
    pc_keys = list(list_epipolar_point_clouds.keys())
    point_cloud = list_epipolar_point_clouds[pc_keys[0]]
    tif_size = inputs.rasterio_get_size(point_cloud[cst.X])

    tile_size = 100
    grid = tiling.generate_tiling_grid(
        0,
        0,
        tif_size[0],
        tif_size[1],
        tile_size,
        tile_size,
    )

    can_compute_epsg = False
    x_y_min_max = None
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if not can_compute_epsg:
                # define window
                window = rio.windows.Window.from_slices(
                    (
                        grid[row, col, 0],
                        grid[row, col, 1],
                    ),
                    (
                        grid[row, col, 2],
                        grid[row, col, 3],
                    ),
                )
                # compute min max
                x_y_min_max = get_min_max_band(
                    point_cloud[cst.X],
                    point_cloud[cst.Y],
                    point_cloud[cst.Z],
                    point_cloud[cst.PC_EPSG],
                    4326,
                    window=window,
                )

                if not any(np.isnan(x_y_min_max)):
                    can_compute_epsg = True

    x_mean = (x_y_min_max[0] + x_y_min_max[1]) / 2
    y_mean = (x_y_min_max[2] + x_y_min_max[3]) / 2

    epsg = preprocessing.get_utm_zone_as_epsg_code(x_mean, y_mean)

    logging.info("EPSG code: {}".format(epsg))

    return epsg


def intersect_polygons(poly1, poly2):
    """
    Check if two polygons intersect each other

    :param poly1: polygon
    :type poly1: shapely Polygon
    :param poly2: polygon
    :type poly2: shapely Polygon

    :return: True ff intersects
    :rtype: bool

    """

    return poly1.intersects(poly2)


def get_min_max_band(
    image_path_x, image_path_y, image_path_z, epsg_in, epsg_utm, window=None
):
    """
    The purpose of this function is only to get the min and max values in
    row and col.
    of these input images, to do so:
    - Convert input images into a point cloud
    - Project the point cloud using the global EPSG code
    - Get the min and max values in row and col

    :param image_path_x: path to the X image to read
    :type image_path_x: str
    :param image_path_y: path to the Y image to read
    :type image_path_y: str
    :param image_path_z: path to the Z image to read
    :type image_path_z: str
    :param epsg_in: the EPSG in what the input images coordinates are
    :type epsg_in: integer
    :param epsg_utm: the EPSG code of the UTM referencial in common
        for all the input clouds
    :type epsg_utm: integer
    :param window: specify a region to open inside the image
    :type window: rasterio window
    :return: an array that contains [xmin, xmax, ymin, ymax] and the code epsg
        in which the point cloud is projected
    """
    cloud_data = {}
    with rio.open(image_path_x) as image_x:
        with rio.open(image_path_y) as image_y:
            with rio.open(image_path_z) as image_z:
                if window is None:
                    band_x = image_x.read(1)
                    band_y = image_y.read(1)
                    band_z = image_z.read(1)
                else:
                    band_x = image_x.read(1, window=window)
                    band_y = image_y.read(1, window=window)
                    band_z = image_z.read(1, window=window)

                cloud_data[cst.X] = np.ravel(band_x)
                cloud_data[cst.Y] = np.ravel(band_y)
                cloud_data[cst.Z] = np.ravel(band_z)

    pd_cloud = pd.DataFrame(cloud_data, columns=[cst.X, cst.Y, cst.Z])

    pd_cloud = pd_cloud.drop(
        pd_cloud.index[
            (pd_cloud[cst.X] == 0.0)  # pylint: disable=E1136
            | (pd_cloud[cst.Y] == 0.0)  # pylint: disable=E1136
        ]
    )
    pd_cloud = pd_cloud.drop(
        pd_cloud.index[
            (np.isnan(pd_cloud[cst.X]))  # pylint: disable=E1136
            | (np.isnan(pd_cloud[cst.Y]))  # pylint: disable=E1136
        ]
    )

    lon_med, lat_med, _ = pd_cloud.median()
    xmin = np.nan
    xmax = np.nan
    ymin = np.nan
    ymax = np.nan
    if not np.isnan(lon_med) and not np.isnan(lat_med):
        projection.point_cloud_conversion_dataframe(pd_cloud, epsg_in, epsg_utm)

        xmin = pd_cloud[cst.X].min()
        xmax = pd_cloud[cst.X].max()
        ymin = pd_cloud[cst.Y].min()
        ymax = pd_cloud[cst.Y].max()

    return [xmin, xmax, ymin, ymax]


def convert_to_polygon(x_y_min_max):
    """
    Resample a bounding box and convert it into an shapely polygon

    :param x_y_min_max: the x/y coordinates of the upper left and lower
     right points
    :type x_y_min_max: an array of float [x_min, x_max, y_min, y_max]

    :return: an shapely polygon
    """

    x_min, x_max, y_min, y_max = x_y_min_max
    points = []
    for x_coord in np.linspace(x_min, x_max, 5):
        points.append((x_coord, y_min, 0))
    for y_cord in np.linspace(y_min, y_max, 5):
        points.append((x_max, y_cord, 0))
    for x_coord in np.linspace(x_min, x_max, 5)[::-1]:
        points.append((x_coord, y_max, 0))
    for y_cord in np.linspace(y_min, y_max, 5)[::-1]:
        points.append((x_min, y_cord, 0))

    return create_polygon_from_list_points(points)


def filter_cloud_tif(pd_cloud, bounds):
    """
    Remove from the merged cloud all points that are out of the
    terrain tile boundaries.

    :param pd_cloud: point cloud
    :type pd_cloud: pandas dataframe
    :param bounds: terrain tile bounds
    :type bounds: array of float [xmin, ymin, xmax, ymax]

    :return: the epsg out
    :rtype: int
    """
    cond_x_min = pd_cloud[cst.X] < bounds[0]
    cond_x_max = pd_cloud[cst.X] > bounds[1]
    cond_y_min = pd_cloud[cst.Y] < bounds[2]
    cond_y_max = pd_cloud[cst.Y] > bounds[3]
    pd_cloud = pd_cloud.drop(
        pd_cloud.index[cond_x_min | cond_x_max | cond_y_min | cond_y_max]
    )


def read_band(
    band_name, band_path, window, cloud_data_bands, cloud_data_types, cloud_data
):
    """
    Extract from tif point cloud and put in carsdataset point cloud

    :param band_name: type of point cloud data
    :type band_name: str
    :param band_path: path of the tif point cloud file
    :type band_path: str
    :param window: window to use
    :type window: dict
    :param cloud_data_bands: list of point cloud
    :type cloud_data_bands: list
    :param cloud_data: point cloud numpy dict
    :type cloud_data: dict
    """
    # Determine type
    band_type = inputs.rasterio_get_image_type(band_path)
    if cst.POINT_CLOUD_MSK in band_name:
        band_type = "uint8"
    if (
        cst.POINT_CLOUD_CLASSIF_KEY_ROOT in band_name
        or cst.POINT_CLOUD_FILLING_KEY_ROOT in band_name
    ):
        band_type = "boolean"
    with rio.open(band_path) as band_file:
        if band_file.count == 1:
            cloud_data_bands.append(band_name)
            cloud_data_types.append(band_type)
            cloud_data[band_name] = np.ravel(band_file.read(1, window=window))
        else:
            descriptions = inputs.get_descriptions_bands(band_path)
            for id_band, band_desc in enumerate(descriptions):
                band_full_name = "{}_{}".format(band_name, band_desc)
                cloud_data_bands.append(band_full_name)
                cloud_data_types.append(band_type)
                cloud_data[band_full_name] = np.ravel(
                    band_file.read(1 + id_band, window=window)
                )


def read_image_full(band_path, window=None, squeeze=False):
    """
    Read image with window

    :param band_path: path to image
    :param window: window
    :param squeeze: squeeze data if true

    :return array
    """

    with rio.open(band_path) as desc_band:
        data = desc_band.read(window=window)
    if squeeze:
        data = np.squeeze(data)

    return data


def get_bounds(
    list_epipolar_point_clouds,
    epsg,
    roi_poly=None,
):
    """
    Get bounds of clouds

    :param list_epipolar_point_clouds: list of clouds
    :type list_epipolar_point_clouds: dict
    :param epsg: epsg of wanted roi
    :param roi_poly: crop with given roi

    :return bounds
    """
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []

    for _, point_cloud in list_epipolar_point_clouds.items():

        local_x_y_min_max = get_min_max_band(
            point_cloud[cst.X],
            point_cloud[cst.Y],
            point_cloud[cst.Z],
            point_cloud[cst.PC_EPSG],
            epsg,
        )

        xmin_list.append(local_x_y_min_max[0])
        xmax_list.append(local_x_y_min_max[1])
        ymin_list.append(local_x_y_min_max[2])
        ymax_list.append(local_x_y_min_max[3])

    # Define a terrain tiling from the terrain bounds (in terrain epsg)
    global_xmin = min(xmin_list)
    global_xmax = max(xmax_list)
    global_ymin = min(ymin_list)
    global_ymax = max(ymax_list)

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

    return terrain_bbox


def compute_max_nb_point_clouds(list_epipolar_point_clouds_by_tiles):
    """
    Compute the maximum number of point clouds superposing.

    :param list_epipolar_point_clouds_by_tiles: list of tiled point clouds
    :type list_epipolar_point_clouds_by_tiles: list(CarsDataset)

    :return: max number of point clouds
    :rtype: int

    """

    # Create polygon for each CarsDataset

    list_pc_polygon = []
    for epi_pc_cars_ds in list_epipolar_point_clouds_by_tiles:
        if "xmin" in epi_pc_cars_ds.attributes:
            xmin = epi_pc_cars_ds.attributes["xmin"]
            xmax = epi_pc_cars_ds.attributes["xmax"]
            ymin = epi_pc_cars_ds.attributes["ymin"]
            ymax = epi_pc_cars_ds.attributes["ymax"]

            x_y_min_max = [xmin, xmax, ymin, ymax]
            list_pc_polygon.append((convert_to_polygon(x_y_min_max), 1))

    # Compute polygon intersection. A polygon is reprensented with a tuple:
    # (shapely_polygon, nb_polygon intersection)

    list_intersected_polygons = []

    for poly in list_pc_polygon:
        if len(list_intersected_polygons) == 0:
            list_intersected_polygons.append(poly)
        else:
            new_poly_list = []
            for seen_poly in list_intersected_polygons:
                if poly[0].intersects(seen_poly[0]):
                    # Compute intersection
                    intersect_poly = poly[0].intersection(seen_poly[0])
                    new_poly_list.append(
                        (intersect_poly, poly[1] + seen_poly[1])
                    )
            list_intersected_polygons += new_poly_list

    # Get max of intersection
    nb_pc = 0
    for poly in list_intersected_polygons:
        nb_pc = max(nb_pc, poly[1])

    return nb_pc


def compute_average_distance(list_epipolar_point_clouds_by_tiles):
    """
    Compute average distance between points


    :param list_epipolar_point_clouds_by_tiles: list of tiled point clouds
    :type list_epipolar_point_clouds_by_tiles: list(CarsDataset)

    :return: average distance between points
    :rtype: float

    """

    # Get average for each point
    list_average_dist = []
    for epi_pc_cars_ds in list_epipolar_point_clouds_by_tiles:
        if "xmin" in epi_pc_cars_ds.attributes:
            xmin = epi_pc_cars_ds.attributes["xmin"]
            xmax = epi_pc_cars_ds.attributes["xmax"]
            ymin = epi_pc_cars_ds.attributes["ymin"]
            ymax = epi_pc_cars_ds.attributes["ymax"]
            data_epsg = epi_pc_cars_ds.attributes["epsg"]

            x_y_min_max = [xmin, xmax, ymin, ymax]
            # Create polygon
            poly = convert_to_polygon(x_y_min_max)
            # Transform polygon to epsg meter
            epsg_meter = 4978
            meter_poly = projection.polygon_projection(
                poly, data_epsg, epsg_meter
            )

            # Compute perimeter in meter
            perimeter_meters = length(meter_poly)
            # Compute perimeter in pixel
            nb_row = np.max(epi_pc_cars_ds.tiling_grid[:, :, 1])
            nb_col = np.max(epi_pc_cars_ds.tiling_grid[:, :, 3])
            perimeter_pixels = 2 * nb_row + 2 * nb_col
            # Compute average distance
            list_average_dist.append(perimeter_meters / perimeter_pixels)

    return max(list_average_dist)
