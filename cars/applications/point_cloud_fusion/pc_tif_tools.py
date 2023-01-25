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
This module contains functions used during the fusion of
point clouds from .tif files.
"""

# Standard imports
import logging

# Third party imports
import numpy as np
import pandas as pd
import rasterio as rio
from shapely import geometry

# CARS imports
from cars.core import constants as cst
from cars.core import inputs, preprocessing, projection, tiling
from cars.data_structures import cars_dataset


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

    poly = geometry.Polygon(list_shapely_points)

    return poly


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
    - Project the points cloud using the global EPSG code
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
    :type window: an array of integer (ex: [col_off, row_off, width, height])
    :return: an array that contains [xmin, xmax, ymin, ymax] and the code epsg
        in which
    the points cloud is projected
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
        projection.points_cloud_conversion_dataframe(
            pd_cloud, epsg_in, epsg_utm
        )

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


def filter_cloud(pd_cloud, bounds):
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


def create_combined_cloud_from_tif(
    clouds, epsg, xmin=None, xmax=None, ymin=None, ymax=None, margin=0
):
    """
    Create combined cloud from tif point clouds

    :param clouds: list of clouds
    :type clouds: list(dict)
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

    :return: combined cloud
    :rtype: pandas Dataframe

    """

    clouds_pd_list = []

    # Create multiple pc pandas dataframes
    for cloud in clouds:
        window = cloud["window"]
        cloud_epsg = cloud["cloud_epsg"]
        cloud_data_bands = []
        cloud_data = {}
        for type_band in cloud["data"].keys():
            # open file and get data
            band_path = cloud["data"][type_band]
            if band_path is not None:
                with rio.open(band_path) as desc_band:
                    if desc_band.count == 1:
                        cloud_data_bands.append(type_band)
                        cloud_data[type_band] = np.ravel(
                            desc_band.read(1, window=window)
                        )
                    else:
                        for index_band in range(desc_band.count):
                            band_name = "{}{}".format(type_band, index_band + 1)
                            cloud_data_bands.append(band_name)
                            cloud_data[band_name] = np.ravel(
                                desc_band.read(1 + index_band, window=window)
                            )

        # add mask if not given
        if cst.POINTS_CLOUD_VALID_DATA not in cloud_data_bands:
            cloud_data[cst.POINTS_CLOUD_VALID_DATA] = np.ones(
                cloud_data[cst.X].shape
            )
            cloud_data_bands.append(cst.POINTS_CLOUD_VALID_DATA)

        # Create cloud pandas
        cloud_pd = pd.DataFrame(cloud_data, columns=cloud_data_bands)

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

        # Convert pc if necessary
        if cloud_epsg != epsg:
            projection.points_cloud_conversion_dataframe(
                cloud_pd, cloud_epsg, epsg
            )

        # add to list of pandas pc
        clouds_pd_list.append(cloud_pd)

    # Merge pandas point clouds
    combined_pd_cloud = pd.concat(
        clouds_pd_list,
        axis=0,
        join="outer",
    )

    # filter outside points considering mmargins
    filter_cloud(
        combined_pd_cloud,
        list(
            np.array([xmin, xmax, ymin, ymax])
            + np.array([-margin, margin, -margin, margin])
        ),
    )

    return combined_pd_cloud, epsg


def transform_input_pc(
    list_epipolar_points_cloud, epsg, roi_poly=None, epipolar_tile_size=600
):
    """
    Transform point clouds from inputs into point cloud fusion application
    format.
    Create tiles, with x y min max informations.

    :param list_epipolar_points_cloud: list of epipolar point clouds
    :type list_epipolar_points_cloud: list(dict)
    :param epsg: epsg
    :type epsg: int, str
    :param roi_poly: roi polygon
    :type roi_poly: Polygon
    :param epipolar_tile_size: size of tile used for tiling the tif files
    :type epipolar_tile_size: int

    :return list of point clouds
    :rtype: list(CarsDataset type dict)

    """

    list_epipolar_points_cloud_left_by_tiles = []

    # For each stereo pair
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []
    for _, items in list_epipolar_points_cloud.items():
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

                x_y_min_max = get_min_max_band(
                    items[cst.X],
                    items[cst.Y],
                    items[cst.Z],
                    items[cst.PC_EPSG],
                    epsg,
                    window=window,
                )

                # fill CarsDataset
                epi_pc[row, col] = {
                    "data": {
                        cst.X: items[cst.X],
                        cst.Y: items[cst.Y],
                        cst.Z: items[cst.Z],
                        cst.POINTS_CLOUD_CLR_KEY_ROOT: items[
                            cst.POINTS_CLOUD_CLR_KEY_ROOT
                        ],
                        cst.POINTS_CLOUD_VALID_DATA: items[
                            cst.POINTS_CLOUD_VALID_DATA
                        ],
                    },
                    "x_y_min_max": x_y_min_max,
                    "window": window,
                    "cloud_epsg": items[cst.PC_EPSG],
                }

                if not any(np.isnan(x_y_min_max)):
                    xmin_list.append(x_y_min_max[0])
                    xmax_list.append(x_y_min_max[1])
                    ymin_list.append(x_y_min_max[2])
                    ymax_list.append(x_y_min_max[3])

        list_epipolar_points_cloud_left_by_tiles.append(epi_pc)

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

    return (terrain_bbox, list_epipolar_points_cloud_left_by_tiles)


def get_tiles_row_col(
    terrain_tiling_grid,
    row,
    col,
    list_epipolar_points_cloud_left_with_loc,
    list_epipolar_points_cloud_right_with_loc,
    margins=0,
):
    """
    Get point cloud tiles to use for terrain region

    :param terrain_tiling_grid: tiling grid
    :type terrain_tiling_grid: np.ndarray
    :param row: tiling row
    :type row: int
    :param col: col
    :type col: int
    :param list_epipolar_points_cloud_left_with_loc: list of left point clouds
    :type list_epipolar_points_cloud_left_with_loc: list(CarsDataset)
    :param list_epipolar_points_cloud_right_with_loc: list of right point clouds
    :type list_epipolar_points_cloud_right_with_loc: list(CarsDataset)
    :param margins: margin to use in point clouds
    :type margins: float

    :return: list of point cloud tiles to use to terrain tile
    :rtype: list(dict)

    """

    if list_epipolar_points_cloud_right_with_loc is not None:
        if len(list_epipolar_points_cloud_right_with_loc) > 0:
            logging.warning("Right point clouds given as input, not considered")

    # Terrain grid [row, j, :] = [xmin, xmax, ymin, ymax]
    # terrain region = [xmin, ymin, xmax, ymax]
    terrain_region = [
        terrain_tiling_grid[row, col, 0],
        terrain_tiling_grid[row, col, 2],
        terrain_tiling_grid[row, col, 1],
        terrain_tiling_grid[row, col, 3],
    ]

    region_with_margin = list(
        np.array(terrain_region)
        + np.array([-margins, margins, -margins, margins])
    )

    # Convert the bounds of the terrain tile into shapely polygon
    terrain_tile_polygon = convert_to_polygon(region_with_margin)

    required_point_clouds_left = []
    required_point_clouds_right = []

    for epi_pc in list_epipolar_points_cloud_left_with_loc:
        for tile_row in range(epi_pc.shape[0]):
            for tile_col in range(epi_pc.shape[1]):
                x_y_min_max = epi_pc[tile_row, tile_col]["x_y_min_max"]

                # Convert the bounds of the point cloud tile into shapely point
                if any(np.isnan(x_y_min_max)):
                    continue

                point_cloud_tile_polygon = convert_to_polygon(x_y_min_max)

                if intersect_polygons(
                    terrain_tile_polygon, point_cloud_tile_polygon
                ):
                    # add to required
                    required_point_clouds_left.append(
                        epi_pc[tile_row, tile_col]
                    )

    return (
        terrain_region,
        required_point_clouds_left,
        required_point_clouds_right,
    )
