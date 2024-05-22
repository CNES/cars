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

# pylint: disable=C0302

# Standard imports
import logging

# Third party imports
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from shapely import geometry, length

import cars.orchestrator.orchestrator as ocht
from cars.core import constants as cst
from cars.core import inputs, preprocessing, projection, tiling

# CARS imports
from cars.data_structures import cars_dataset, cars_dict


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


def compute_epsg_from_point_cloud(list_epipolar_points_cloud):
    """
    Compute epsg to use from list of tif point clouds

    :param list_epipolar_points_cloud: list of epipolar point clouds
    :type list_epipolar_points_cloud: list(dict)

    :return: epsg
    :rtype: int
    """

    # Get epsg from first point cloud
    pc_keys = list(list_epipolar_points_cloud.keys())
    point_cloud = list_epipolar_points_cloud[pc_keys[0]]
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
    :type window: rasterio window
    :return: an array that contains [xmin, xmax, ymin, ymax] and the code epsg
        in which the points cloud is projected
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
                if cst.POINTS_CLOUD_CLR_KEY_ROOT in band_name:
                    # Get color type
                    color_types.append(
                        inputs.rasterio_get_image_type(band_path)
                    )

                if isinstance(band_path, dict):
                    for key in band_path:
                        sub_band_path = band_path[key]
                        sub_band_name = key
                        read_band(
                            sub_band_name,
                            sub_band_path,
                            window,
                            cloud_data_bands,
                            cloud_data_types,
                            cloud_data,
                        )
                else:
                    read_band(
                        band_name,
                        band_path,
                        window,
                        cloud_data_bands,
                        cloud_data_types,
                        cloud_data,
                    )

        # add source file id
        cloud_data[cst.POINTS_CLOUD_GLOBAL_ID] = (
            np.ones(cloud_data[cst.X].shape) * cloud_file_id
        )
        cloud_data_bands.append(cst.POINTS_CLOUD_GLOBAL_ID)
        cloud_data_types.append("uint16")

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

        # Cast types according to band
        cloud_data_types = dict(
            zip(cloud_data_bands, cloud_data_types)  # noqa: B905
        )
        cloud_pd = cloud_pd.astype(cloud_data_types)

        # Convert pc if necessary
        if cloud_epsg != epsg:
            projection.points_cloud_conversion_dataframe(
                cloud_pd, cloud_epsg, epsg
            )

        # filter outside points considering mmargins
        filter_cloud(
            cloud_pd,
            list(
                np.array([xmin, xmax, ymin, ymax])
                + np.array([-margin, margin, -margin, margin])
            ),
        )

        # add to list of pandas pc
        clouds_pd_list.append(cloud_pd)

    # Merge pandas point clouds
    combined_pd_cloud = pd.concat(
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
    if cst.POINTS_CLOUD_MSK in band_name:
        band_type = "uint8"
    if (
        cst.POINTS_CLOUD_CLASSIF_KEY_ROOT in band_name
        or cst.POINTS_CLOUD_FILLING_KEY_ROOT in band_name
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
    list_epipolar_points_cloud = []

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
        if cst.POINTS_CLOUD_CLR_KEY_ROOT in cloud:
            # Get color type
            color_type = inputs.rasterio_get_image_type(
                cloud[cst.POINTS_CLOUD_CLR_KEY_ROOT]
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

        list_epipolar_points_cloud.append(cars_ds)

    return list_epipolar_points_cloud


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


def generate_pc_wrapper(
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
    data_x = read_image_full(cloud["x"], window=window, squeeze=True)
    data_y = read_image_full(cloud["y"], window=window, squeeze=True)
    data_z = read_image_full(cloud["z"], window=window, squeeze=True)

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
        elif key == "point_cloud_epsg":
            attributes["epsg"] = cloud[key]
        elif key == "mask":
            if cloud[key] is None:
                data = ~np.isnan(data_x) * 255
            else:
                data = read_image_full(cloud[key], window=window, squeeze=True)
            values[cst.POINTS_CLOUD_CORR_MSK] = ([cst.ROW, cst.COL], data)

        elif key == cst.EPI_CLASSIFICATION:
            data = read_image_full(cloud[key], window=window, squeeze=False)
            descriptions = inputs.get_descriptions_bands(cloud[key])
            values[cst.EPI_CLASSIFICATION] = (
                [cst.BAND_CLASSIF, cst.ROW, cst.COL],
                data,
            )
            if cst.BAND_CLASSIF not in coords:
                coords[cst.BAND_CLASSIF] = descriptions

        elif key == cst.EPI_COLOR:
            data = read_image_full(cloud[key], window=window, squeeze=False)
            descriptions = list(inputs.get_descriptions_bands(cloud[key]))
            attributes["color_type"] = color_type
            values[cst.EPI_COLOR] = ([cst.BAND_IM, cst.ROW, cst.COL], data)

            if cst.EPI_COLOR not in coords:
                coords[cst.BAND_IM] = descriptions

        elif key == cst.EPI_FILLING:
            data = read_image_full(cloud[key], window=window, squeeze=False)
            descriptions = inputs.get_descriptions_bands(cloud[key])
            values[cst.EPI_FILLING] = (
                [cst.BAND_FILLING, cst.ROW, cst.COL],
                data,
            )
            if cst.BAND_FILLING not in coords:
                coords[cst.BAND_FILLING] = descriptions
        else:
            data = read_image_full(cloud[key], window=window, squeeze=True)
            if data.shape == 2:
                values[key] = ([cst.ROW, cst.COL], data)
            else:
                logging.error(" {} data not managed".format(key))

    xr_cloud = xr.Dataset(values, coords=coords)
    xr_cloud.attrs = attributes

    return xr_cloud


def get_bounds(
    list_epipolar_points_cloud,
    epsg,
    roi_poly=None,
):
    """
    Get bounds of clouds

    :param list_epipolar_points_cloud: list of clouds
    :type list_epipolar_points_cloud: dict
    :param epsg: epsg of wanted roi
    :param roi_poly: crop with given roi

    :return bounds
    """
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []

    for _, point_cloud in list_epipolar_points_cloud.items():

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


def transform_input_pc(
    list_epipolar_points_cloud,
    epsg,
    roi_poly=None,
    epipolar_tile_size=600,
    orchestrator=None,
):
    """
    Transform point clouds from inputs into point cloud fusion application
    format.
    Create tiles, with x y min max informations.

    :param list_epipolar_points_cloud: list of epipolar point clouds
    :type list_epipolar_points_cloud: dict
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

    list_epipolar_points_cloud_by_tiles = []

    # For each stereo pair
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []
    for pair_key, items in list_epipolar_points_cloud.items():
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
        list_epipolar_points_cloud_by_tiles.append(epi_pc)

    # Breakpoint : compute
    # /!\ BE AWARE : this is not the conventionnal way
    # to parallelise tasks in CARS
    cars_orchestrator.breakpoint()

    # Get all local min and max
    for computed_epi_pc in list_epipolar_points_cloud_by_tiles:
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
        computed_epi_pc.attributes["xmin"] = min(pc_xmin_list)
        computed_epi_pc.attributes["ymin"] = min(pc_ymin_list)
        computed_epi_pc.attributes["xmax"] = max(pc_xmax_list)
        computed_epi_pc.attributes["ymax"] = max(pc_ymax_list)
        computed_epi_pc.attributes["epsg"] = epsg

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

    return (terrain_bbox, list_epipolar_points_cloud_by_tiles)


def compute_max_nb_point_clouds(list_epipolar_points_cloud_by_tiles):
    """
    Compute the maximum number of point clouds superposing.

    :param list_epipolar_points_cloud_by_tiles: list of tiled point clouds
    :type list_epipolar_points_cloud_by_tiles: list(CarsDataset)

    :return: max number of point clouds
    :rtype: int

    """

    # Create polygon for each CarsDataset

    list_pc_polygon = []
    for epi_pc_cars_ds in list_epipolar_points_cloud_by_tiles:
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


def compute_average_distance(list_epipolar_points_cloud_by_tiles):
    """
    Compute average distance between points


    :param list_epipolar_points_cloud_by_tiles: list of tiled point clouds
    :type list_epipolar_points_cloud_by_tiles: list(CarsDataset)

    :return: average distance between points
    :rtype: float

    """

    # Get average for each point
    list_average_dist = []
    for epi_pc_cars_ds in list_epipolar_points_cloud_by_tiles:
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
        meter_poly = projection.polygon_projection(poly, data_epsg, epsg_meter)

        # Compute perimeter in meter
        perimeter_meters = length(meter_poly)
        # Compute perimeter in pixel
        nb_row = np.max(epi_pc_cars_ds.tiling_grid[:, :, 1])
        nb_col = np.max(epi_pc_cars_ds.tiling_grid[:, :, 3])
        perimeter_pixels = 2 * nb_row + 2 * nb_col
        # Compute average distance
        list_average_dist.append(perimeter_meters / perimeter_pixels)

    return max(list_average_dist)


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
    x_y_min_max = get_min_max_band(
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
        cst.POINTS_CLOUD_CLR_KEY_ROOT: items[cst.POINTS_CLOUD_CLR_KEY_ROOT],
    }
    if cst.POINTS_CLOUD_MSK in items:
        data_dict[cst.POINTS_CLOUD_MSK] = items[cst.POINTS_CLOUD_MSK]
    if cst.POINTS_CLOUD_CLASSIF_KEY_ROOT in items:
        data_dict[cst.POINTS_CLOUD_CLASSIF_KEY_ROOT] = items[
            cst.POINTS_CLOUD_CLASSIF_KEY_ROOT
        ]
    if cst.POINTS_CLOUD_FILLING_KEY_ROOT in items:
        data_dict[cst.POINTS_CLOUD_FILLING_KEY_ROOT] = items[
            cst.POINTS_CLOUD_FILLING_KEY_ROOT
        ]
    if cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT in items:
        data_dict[cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT] = items[
            cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT
        ]

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
    list_epipolar_points_cloud_with_loc,
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
    :param list_epipolar_points_cloud_with_loc: list of left point clouds
    :type list_epipolar_points_cloud_with_loc: list(CarsDataset)
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
        len(list_epipolar_points_cloud_with_loc),
        len(list_epipolar_points_cloud_with_loc),
        1,
        len(list_epipolar_points_cloud_with_loc),
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
            list_epipolar_points_cloud_with_loc[row_fake_cars_ds],
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

            terrain_tile_polygon = convert_to_polygon(
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
                        point_cloud_tile_polygon = convert_to_polygon(
                            x_y_min_max
                        )

                        if intersect_polygons(
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
