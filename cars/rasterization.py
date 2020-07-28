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
This module contains all functions related to rasterization
"""

# Standard imports
from typing import List, Tuple, Union
import time
import logging
import warnings
import math

# Third party imports
import numpy as np
import pandas
from scipy.spatial import cKDTree
import xarray as xr
from numba import njit, float64, int64, boolean
from numba.core.errors import NumbaPerformanceWarning
import otbApplication
from osgeo import osr

# cars import
from cars import filtering, projection

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def get_utm_zone_as_epsg_code(lon, lat):
    """
    Returns the EPSG code of the UTM zone where the lat,lon point falls in

    :param lon: longitude of the point
    :type lon: float
    :param lat: lattitude of the point
    :type lat: float
    :returns: The EPSG code corresponding to the UTM zone
    :rtype: int
    """
    utm_app = otbApplication.Registry.CreateApplication(
        "ObtainUTMZoneFromGeoPoint")
    utm_app.SetParameterFloat("lon", float(lon))
    utm_app.SetParameterFloat("lat", float(lat))
    utm_app.Execute()
    zone = utm_app.GetParameterInt("utm")
    ns = 600 if lat >= 0 else 700
    return 32000 + ns + zone


def point_cloud_from_arrays(coords, mask, color, epsg_out):
    """

    :param coords:
    :param mask:
    :param color:
    :param epsg_out:
    :return:
    """
    dims = ('row', 'col')
    out_dset = xr.Dataset({
        'x': (dims, coords[..., 0]),
        'y': (dims, coords[..., 1]),
        'z': (dims, coords[..., 2]),
        'msk': (dims + ('i_mask',), mask)
    })

    if color is not None:
        color_dset = xr.Dataset({'im': (dims + ('band',), color)})
        out_dset = xr.merge((out_dset, color_dset))

    out_dset.attrs['epsg'] = epsg_out

    return out_dset


def compute_xy_starts_and_sizes(resolution: float, cloud: pandas.DataFrame) -> Tuple[float, float, int, int]:
    """
    Compute xstart, ystart, xsize and ysize of the rasterization grid from a set of points

    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units
    :param cloud: set of points as returned by the create_combined_cloud function
    :return: a tuple (xstart, ystart, xsize, ysize)
    """
    worker_logger = logging.getLogger('distributed.worker')

    # Derive xstart
    xmin = np.nanmin(cloud['x'].values)
    xmax = np.nanmax(cloud['x'].values)
    worker_logger.debug(
        "Points x coordinate range: [{},{}]".format(xmin, xmax))

    # Clamp to a regular grid
    x_start = np.floor(xmin / resolution) * resolution
    x_size = int(1 + np.floor((xmax - x_start) / resolution))

    # Derive ystart
    ymin = np.nanmin(cloud['y'].values)
    ymax = np.nanmax(cloud['y'].values)
    worker_logger.debug(
        "Points y coordinate range: [{},{}]".format(ymin, ymax))

    # Clamp to a regular grid
    y_start = np.ceil(ymax / resolution) * resolution
    y_size = int(1 + np.floor((y_start - ymin) / resolution))

    return x_start, y_start, x_size, y_size


def create_combined_cloud(cloud_list: List[xr.Dataset], dsm_epsg: int, color_list: List[xr.Dataset]=None,
                          resolution: float=None, xstart: float=None, ystart: float=None, xsize: int=None,
                          ysize: int=None, on_ground_margin: int=0, epipolar_border_margin: int=0, radius: float=1,
                          with_coords: bool=False) -> Tuple[pandas.DataFrame, int]:
    """
    Combine a list of clouds (and their colors) into a pandas dataframe structured with the following labels:

        * if no colors are set in input:
    >>> labels=['valid_data', 'x', 'y', 'z']

    The combined cloud has x, y, z columns along with 'valid data' one. The valid data is a mask set to True if the data
    are not on the epipolar image margin (epipolar_border_margin), otherwise it is set to False.

        * if colors are set in input:
    >>> labels=['valid_data', 'x', 'y', 'z', 'clr0', 'clr1', 'clr2']

    Color channels information are added to the dataframe.

        * if colors are set in input and the with_coords option is activated:
    >>> labels=['valid_data', 'x', 'y', 'z', 'clr0', 'clr1', 'clr2',
    >>>         'coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi']

    The pixel position of the xyz point in the original epipolar image (coord_epi_geom_i, coord_epi_geom_j) are added
    to the dataframe along with the index of its original cloud in the cloud_list input.

    :raise Exception if a color_list is set but does not have the same length as the cloud list

    :param cloud_list: list of cloud points to rasterize
    :param dsm_epsg: epsg code for the CRS of the final output raster
    :param color_list: Additional list of images with bands to rasterize (same size as cloud_list), or None
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units
    (if None, the whole clouds are combined)
    :param xstart: xstart of the rasterization grid (if None, the whole clouds are combined)
    :param ystart: ystart of the rasterization grid (if None, the whole clouds are combined)
    :param xsize: xsize of the rasterization grid (if None, the whole clouds are combined)
    :param ysize: ysize of the rasterization grid (if None, the whole clouds are combined)
    :param on_ground_margin: Margin added to the rasterization grid (default value: 0)
    :param epipolar_border_margin: Margin used to invalidate cells too close to epipolar border. (default value: 0)
    :param radius: Radius for hole filling (if None, the whole clouds are combined).
    :param with_coords: Option enabling the adding to the combined cloud of information of each point to retrieve their
    positions in the original epipolar images
    :return: Tuple formed with the combined clouds and color in a single pandas dataframe and the epsg code
    """
    worker_logger = logging.getLogger('distributed.worker')

    # check input data consistency
    if color_list is not None and len(cloud_list) != len(color_list):
        raise Exception("There shall be as many cloud elements as color ones")

    epsg = None
    for idx in range(len(cloud_list)):
        if epsg is None:
            epsg = int(cloud_list[idx].attrs['epsg'])
        elif int(cloud_list[idx].attrs['epsg']) != epsg:
            worker_logger.error('All points clouds do not have the same epsg code')

    # compute margin/roi and final number of data to add to the combined cloud
    roi = resolution is not None and xstart is not None and ystart is not None and \
          xsize is not None and ysize is not None
    if roi:
        total_margin = (on_ground_margin + radius + 1) * resolution
        xend = xstart + (xsize + 1) * resolution
        yend = ystart - (ysize + 1) * resolution

    nb_data = ['data_valid', 'x', 'y', 'z']

    nb_data_msk = 0
    for idx in range(len(cloud_list)):
        values_list = [key for key, _ in cloud_list[idx].items()]
        if 'msk' in values_list:
            nb_data.append('msk')
            nb_data_msk = 1
            break

    if color_list is not None:
        clr_im = color_list[0].im.values
        nb_band_clr = clr_im.shape[0]
        list_clr = ['clr{}'.format(i) for i in range(nb_band_clr)]
        nb_data.extend(list_clr)
    else:
        nb_band_clr = 0

    if with_coords:
        nb_data.extend(['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])
    print(nb_data)
    # iterate trough input clouds
    cloud = np.zeros((0, len(nb_data)), dtype=np.float64)
    nb_points = 0
    for idx in range(len(cloud_list)):
        full_x = cloud_list[idx].x.values
        full_y = cloud_list[idx].y.values
        full_z = cloud_list[idx].z.values

        # get mask of points inside the roi (plus margins)
        if roi:

            # if the points clouds are not in the same referential as the roi, it is converted using the dsm_epsg
            if epsg != dsm_epsg:
                full_x, full_y = projection.get_converted_xy_np_arrays_from_dataset(cloud_list[idx], dsm_epsg)

            msk_xstart = np.where(full_x > xstart - total_margin, True, False)
            msk_xend = np.where(full_x < xend + total_margin, True, False)
            msk_yend = np.where(full_y > yend - total_margin, True, False)
            msk_ystart = np.where(full_y < ystart + total_margin, True, False)
            terrain_tile_data_msk = np.logical_and(msk_xstart, np.logical_and(msk_xend, np.logical_and(msk_ystart, msk_yend)))
            terrain_tile_data_msk_pos = terrain_tile_data_msk.astype(np.int8).nonzero()

            # if the points clouds are not in the same referential as the roi, retreive the initial values
            if epsg != dsm_epsg:
                full_x = cloud_list[idx].x.values
                full_y = cloud_list[idx].y.values

            # if no point is found, continue
            if terrain_tile_data_msk_pos[0].shape[0] == 0:
                continue

            # get useful data bounding box
            bbox = [np.min(terrain_tile_data_msk_pos[0]),
                    np.min(terrain_tile_data_msk_pos[1]),
                    np.max(terrain_tile_data_msk_pos[0]),
                    np.max(terrain_tile_data_msk_pos[1])]
        else:
            bbox = [0,
                    0,
                    full_y.shape[0]-1,
                    full_y.shape[1]-1]

        # add (x, y, z) information to the current cloud
        c_x = full_x[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        c_y = full_y[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        c_z = full_z[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]

        c_cloud = np.zeros((len(nb_data), (bbox[2] - bbox[0] + 1)*(bbox[3] - bbox[1] + 1)))

        c_cloud[1, :] = np.ravel(c_x)
        c_cloud[2, :] = np.ravel(c_y)
        c_cloud[3, :] = np.ravel(c_z)

        values_list = [key for key, _ in cloud_list[idx].items()]
        print(values_list)
        if 'msk' in values_list:
            c_msk = cloud_list[idx].msk.values[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
            c_cloud[4, :] = np.ravel(c_msk)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(np.min(c_cloud[4, :]), np.max(c_cloud[4, :]))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        #
        # # add data valid mask (points that are not in the border of the epipolar image)
        # if epipolar_border_margin == 0:
        #     epipolar_margin_mask = \
        #         np.full((cloud_list[idx].x.values.shape[0], cloud_list[idx].x.values.shape[1]), True)
        # else:
        #     epipolar_margin_mask = \
        #         np.full((cloud_list[idx].x.values.shape[0], cloud_list[idx].x.values.shape[1]), False)
        #     epipolar_margin_mask[epipolar_border_margin:-epipolar_border_margin,
        #                          epipolar_border_margin:-epipolar_border_margin] = True
        #
        # c_epipolar_margin_mask = epipolar_margin_mask[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1]
        # c_cloud[0, :] = np.ravel(c_epipolar_margin_mask)

        # add the color information to the current cloud
        # if color_list is not None:
        #     c_color = color_list[idx].im.values[:, bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        #
        #     for band in range(nb_band_clr):
        #         c_cloud[4 + nb_data_msk + band, :] = np.ravel(c_color[band, :, :])
        #
        # # add the original image coordinates information to the current cloud
        # if with_coords:
        #     coords_line = np.linspace(bbox[0], bbox[2], bbox[2]-bbox[0]+1)
        #     coords_col = np.linspace(bbox[1], bbox[3], bbox[3]-bbox[1]+1)
        #     coords_col, coords_line = np.meshgrid(coords_col, coords_line)
        #
        #     c_cloud[4 + nb_data_msk + nb_band_clr, :] = np.ravel(coords_line)
        #     c_cloud[4 + nb_data_msk + nb_band_clr + 1, :] = np.ravel(coords_col)
        #     c_cloud[4 + nb_data_msk + nb_band_clr + 2, :] = idx

        # remove masked data (pandora + out of the terrain tile points)
        c_terrain_tile_data_msk = cloud_list[idx].pandora_msk.values[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1] == 255

        if roi:
            c_terrain_tile_data_msk = np.logical_and(c_terrain_tile_data_msk,
                                                     terrain_tile_data_msk[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1])
        c_terrain_tile_data_msk = np.ravel(c_terrain_tile_data_msk)

        c_terrain_tile_data_msk_pos = np.nonzero(~c_terrain_tile_data_msk)

        nb_points += c_cloud.shape[1]

        # c_cloud = np.delete(c_cloud.transpose(), c_terrain_tile_data_msk_pos[0], 0)
        c_cloud = c_cloud.transpose()

        # add current cloud to the combined one
        cloud = np.concatenate([cloud, c_cloud], axis=0)

    worker_logger.debug("Received {} points to rasterize".format(nb_points))
    worker_logger.debug("Keeping {}/{} points inside rasterization grid".format(cloud.shape[0], nb_points))

    pd_cloud = pandas.DataFrame(cloud, columns=nb_data)

    return pd_cloud, epsg


def simple_rasterization_dataset(cloud_list: List[xr.Dataset], resolution: float, epsg: int,
                                 color_list: List[xr.Dataset]=None, xstart: float=None, ystart: float=None,
                                 xsize: int=None, ysize: int=None, sigma: float=None, radius: int=1, margin: int=0,
                                 dsm_no_data: int=np.nan, color_no_data: int=np.nan, msk_no_data: int=np.nan,
                                 grid_points_division_factor: int=None,
                                 small_cpn_filter_params: Union[None, filtering.SmallComponentsFilterParams]=None,
                                 statistical_filter_params: Union[None, filtering.StatisticalFilterParams]=None,
                                 dump_filter_cloud:bool=False) \
        -> Union[xr.Dataset, Tuple[xr.Dataset, pandas.DataFrame]]:
    """
    Wrapper of simple_rasterization that has xarray.Dataset as inputs and outputs.

    :param cloud_list: list of cloud points to rasterize
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None
    :param epsg: epsg code for the CRS of the final raster
    :param color_list: Additional list of images with bands to rasterize (same size as cloud_list), or None
    :param xstart: xstart of the rasterization grid (if None, will be estimated by the function)
    :param ystart: ystart of the rasterization grid (if None, will be estimated by the function)
    :param xsize: xsize of the rasterization grid (if None, will be estimated by the function)
    :param ysize: ysize of the rasterization grid (if None, will be estimated by the function)
    :param sigma: sigma for gaussian interpolation. If None, set to resolution
    :param radius: Radius for hole filling.
    :param margin: Margin used to invalidate cells too close to epipolar border.
    Can only be used if input lists are of size 1.
    :param dsm_no_data: no data value to use in the final raster
    :param color_no_data: no data value to use in the final colored raster
    :param msk_no_data:
    :param grid_points_division_factor: number of blocs to use to divide the grid points (memory optimization, reduce
     the highest memory peak). If it is not set, the factor is automatically set to construct 700000 points blocs.
    :param small_cpn_filter_params: small component filtering parameters
    :param statistical_filter_params: statistical filtering parameters
    :param dump_filter_cloud: activate to dump filtered cloud alongside rasterized cloud and color
    :return: Rasterized cloud and Color (in a tuple with the filtered cloud if dump_filter_cloud is activated)
    """

    if small_cpn_filter_params is None:
        on_ground_margin = 0
    else:
        on_ground_margin = small_cpn_filter_params.on_ground_margin

    # combined clouds
    roi = resolution is not None and xstart is not None and ystart is not None \
          and xsize is not None and ysize is not None
    cloud, cloud_epsg = create_combined_cloud(cloud_list, epsg, resolution=resolution, xstart=xstart, ystart=ystart,
                                              xsize=xsize, ysize=ysize, color_list=color_list,
                                              on_ground_margin=on_ground_margin, epipolar_border_margin=margin,
                                              radius=radius, with_coords=True)

    # filter combined cloud
    if small_cpn_filter_params is not None:
        worker_logger = logging.getLogger('distributed.worker')

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(cloud_epsg)
        if spatial_ref.IsGeographic():
            worker_logger.warning('The points cloud to filter is not in a cartographic system. The filter\'s '
                                  'default parameters might not be adapted to this referential. Convert the points '
                                  'cloud to ECEF to ensure a proper filtering.')
        tic = time.process_time()
        cloud, filtered_elt_pos_infos = \
            filtering.small_components_filtering(cloud,
                                                 small_cpn_filter_params.connection_val,
                                                 small_cpn_filter_params.nb_pts_threshold,
                                                 small_cpn_filter_params.clusters_distance_threshold,
                                                 filtered_elt_pos=small_cpn_filter_params.filtered_elt_msk)
        toc = time.process_time()
        worker_logger.debug("Small components cloud filtering done in {} seconds".format(toc - tic))

        if small_cpn_filter_params.filtered_elt_msk:
            filtering.add_cloud_filtering_msk(cloud_list,
                                              filtered_elt_pos_infos,
                                              'filtered_elt_mask',
                                              small_cpn_filter_params.msk_value)

    if statistical_filter_params is not None:
        worker_logger = logging.getLogger('distributed.worker')

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(cloud_epsg)
        if spatial_ref.IsGeographic():
            worker_logger.warning('The points cloud to filter is not in a cartographic system. The filter\'s '
                                  'default parameters might not be adapted to this referential. Convert the points '
                                  'cloud to ECEF to ensure a proper filtering.')

        tic = time.process_time()
        cloud, filtered_elt_pos_infos = \
            filtering.statistical_outliers_filtering(cloud,
                                                     statistical_filter_params.k,
                                                     statistical_filter_params.std_dev_factor,
                                                     filtered_elt_pos=statistical_filter_params.filtered_elt_msk)
        toc = time.process_time()
        worker_logger.debug("Statistical cloud filtering done in {} seconds".format(toc - tic))

        if statistical_filter_params.filtered_elt_msk:
            filtering.add_cloud_filtering_msk(cloud_list,
                                              filtered_elt_pos_infos,
                                              'filtered_elt_mask',
                                              statistical_filter_params.msk_value)

    # If the points cloud is not in the right epsg referential, it is converted
    if cloud_epsg != epsg:
        projection.points_cloud_conversion_dataframe(cloud, cloud_epsg, epsg)

    # compute roi from the combined clouds if it is not set
    if not roi:
        xstart, ystart, xsize, ysize = compute_xy_starts_and_sizes(resolution, cloud)

    # rasterize clouds
    raster = rasterize(cloud, resolution, epsg, x_start=xstart, y_start=ystart, x_size=xsize,
                       y_size=ysize, sigma=sigma, radius=radius, hgt_no_data=dsm_no_data, color_no_data=color_no_data,
                       msk_no_data=msk_no_data, grid_points_division_factor=grid_points_division_factor)

    if dump_filter_cloud:
        return raster, cloud

    return raster


def compute_values_1d(x_start: float, y_start: float, x_size: int, y_size: int, resolution: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the x and y values as 1d arrays

    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :return: a tuple composed of the x and y 1d arrays
    """
    x_values_1d = np.linspace(x_start + 0.5 * resolution,
                              x_start + resolution * (x_size + 0.5), x_size,
                              endpoint=False)
    y_values_1d = np.linspace(y_start - 0.5 * resolution,
                              y_start - resolution * (y_size + 0.5), y_size,
                              endpoint=False)

    return x_values_1d, y_values_1d


def compute_grid_points(x_start: float, y_start: float, x_size: int, y_size: int, resolution: float) -> np.ndarray:
    """
    Compute the grid points

    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :return: Grid point as a numpy array
    """

    x_values_1d, y_values_1d = compute_values_1d(x_start, y_start, x_size, y_size, resolution)
    x_values_2d, y_values_2d = np.meshgrid(x_values_1d, y_values_1d)
    grid_points = np.stack((x_values_2d, y_values_2d), axis=2).reshape(-1, 2)

    return grid_points


def flatten_index_list(nd_list):
    """
    Converts neighbors indices jagged array into a linear 1d array and
    the number of neighbors for each grid point.

    :param nd_list: indices of each neighbor.
    :type nd_list: list of list of int.
    :return: the flattened neighbors ids list and the list of neighbors count for each grid point.
    :rtype: a tuple of 2 1d int64 numpy.ndarray.
    """
    lengths = np.array([len(l) for l in nd_list])  # number of neighbors
    list_1d = np.concatenate(nd_list).astype(int)

    return list_1d, lengths


def search_neighbors(grid_points: np.ndarray, cloud_tree: cKDTree, radius: int, resolution: float,
                     worker_logger: logging.Logger) -> List[List[int]]:
    """
    Search for neighbors of the grid points in the cloud kdTree

    :param grid_points: Grid points
    :param cloud_tree: Points cloud kdTree
    :param radius: Radius for hole filling.
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :param worker_logger: logger
    :return: The list of neighbors
    """
    # build a kD-tree with rasterization grid cells center coordinates
    tic = time.process_time()
    grid_tree = cKDTree(grid_points)
    toc = time.process_time()
    worker_logger.debug("Neighbors search: Grid point kD-tree built in {} seconds"
                        .format(toc - tic))

    # perform neighborhood query for all grid points
    tic = time.process_time()
    neighbors_list = grid_tree.query_ball_tree(cloud_tree, (radius + 0.5) * resolution)
    toc = time.process_time()
    worker_logger.debug("Neighbors search: Neighborhood query done in {} seconds"
                        .format(toc - tic))

    return neighbors_list


def get_flatten_neighbors(grid_points: np.ndarray, cloud: pandas.DataFrame, radius: int, resolution: float,
                          worker_logger: logging.Logger, grid_points_division_factor:int=None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the grid point neighbors of the cloud as flatten array. This is done by slicing the grid points by blocs in
    order to reduce the memory peak induced by the list of neighbors retrieve from the kdTree query done in the
    search_neighbors function.

    :param grid_points: Grid points
    :param cloud: combined cloud as returned by the create_combined_cloud function
    :param radius: Radius for hole filling.
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :param worker_logger: logger
    :param grid_points_division_factor: number of blocs to use to divide the grid points (memory optimization, reduce
     the highest memory peak). If it is not set, the factor is automatically set to construct 700000 points blocs.
    :return: the flattened neighbors ids list, the list start index for each
        grid point and the list of neighbors count for each grid point.
    """
    # Build a KDTree for with cloud points coordinates.
    tic = time.process_time()
    cloud_tree = cKDTree(cloud.loc[:, ['x', 'y']].values)
    toc = time.process_time()
    worker_logger.debug("Neighbors search: Point cloud kD-tree built in {} seconds"
                        .format(toc - tic))

    # compute blocs indexes (memory optimization)
    nb_grid_points = grid_points.shape[0]

    if grid_points_division_factor is None:
        default_bloc_size = 700000
        grid_points_division_factor = math.ceil(nb_grid_points/default_bloc_size)
        worker_logger.debug("The grid points will be divided in {} blocs"
                            .format(grid_points_division_factor))

    if nb_grid_points < grid_points_division_factor:
        grid_points_division_factor = 1
    index_division = np.linspace(0, nb_grid_points, grid_points_division_factor + 1)

    # compute neighbors per blocs
    neighbors_id = None
    n_count = None
    for i in range(grid_points_division_factor):
        sub_grid = grid_points[int(index_division[i]):int(index_division[i + 1]), :]
        neighbors_list = search_neighbors(sub_grid, cloud_tree, radius, resolution, worker_logger)

        # reorganize neighborhood query results with one as 1d arrays to be
        # compatible with numba.
        neighbors_id_cur, n_count_cur = flatten_index_list(neighbors_list)

        if neighbors_id is None:
            neighbors_id = neighbors_id_cur
        else:
            neighbors_id = np.concatenate([neighbors_id, neighbors_id_cur], axis=0)

        if n_count is None:
            n_count = n_count_cur
        else:
            n_count = np.concatenate([n_count, n_count_cur], axis=0)

    # compute starts indexes of each grid points
    start_ids = np.cumsum(np.concatenate(([0], n_count[:-1])))

    return neighbors_id, start_ids, n_count


def compute_vector_raster_and_stats(cloud: pandas.DataFrame, data_valid: np.ndarray, x_start: float, y_start: float,
                                    x_size: int, y_size: int, resolution: float, sigma: float, radius: int,
                                    worker_logger: logging.Logger, grid_points_division_factor: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[None, np.ndarray]]:
    """
    Compute vectorized raster and its statistics.

    :param cloud: combined cloud as returned by the create_combined_cloud function
    :param data_valid: mask of points which are not on the border of its original epipolar image.
    To compute a cell it has to have at least one data valid, for which case it is considered that no contributing
    points from other neighbor tiles are missing.
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :param sigma: sigma for gaussian interpolation. If None, set to resolution
    :param radius: Radius for hole filling.
    :param worker_logger: logger
    :param grid_points_division_factor: number of blocs to use to divide the grid points (memory optimization, reduce
     the highest memory peak). If it is not set, the factor is automatically set to construct 700000 points blocs.
    :return: a tuple with rasterization results and statistics.
    """
    # Build a grid of cell centers coordinates
    tic = time.process_time()
    grid_points = compute_grid_points(x_start, y_start, x_size, y_size, resolution)
    toc = time.process_time()
    worker_logger.debug("Cell centers array built in {} seconds"
                        .format(toc - tic))

    # Search for neighbors
    tic = time.process_time()
    neighbors_id, start_ids, n_count = get_flatten_neighbors(grid_points, cloud, radius, resolution, worker_logger,
                                                             grid_points_division_factor)
    toc = time.process_time()
    worker_logger.debug(
        "Total neighbors search done in {} seconds".format(toc - tic))

    # perform rasterization with gaussian interpolation
    tic = time.process_time()
    clr_bands = [band for band in cloud if str.find(band, 'clr') >= 0]
    cloud_band = ['x', 'y', 'z']
    cloud_band.extend(clr_bands)

    out, mean, stdev, n_pts, n_in_cell = gaussian_interp(
        cloud.loc[:, cloud_band].values, data_valid.astype(np.bool), neighbors_id, start_ids, n_count,
        grid_points, resolution, sigma
    )
    toc = time.process_time()
    worker_logger.debug(
        "Vectorized rasterization done in {} seconds".format(toc - tic))

    if 'msk' in cloud.columns:
        msk = mask_interp(
            cloud.loc[:, ['x', 'y', 'msk']].values, data_valid.astype(np.bool), neighbors_id, start_ids, n_count,
            grid_points, sigma, -1
        )
    else:
        msk = None
    msk = None

    return out, mean, stdev, n_pts, n_in_cell, msk


@njit((float64[:, :], boolean[:], int64, int64[:], int64[:], int64[:]), nogil=True, cache=True)
def get_neighbors(points: np.ndarray, data_valid: np.ndarray, i_grid: int, neighbors_id: np.ndarray,
                  neighbors_start: np.ndarray, neighbors_count: np.ndarray) -> Union[np.ndarray, None]:
    """
    Use the outputs of the get_flatten_neighbors function to get the neighbors of the i_grid point in the points
    numpy array.

    :param points: points numpy array (one line = one point)
    :param data_valid: valid data mask corresponding to the points
    :param i_grid: the index of the outputs of the get_flatten_neighbors function are used
    :param neighbors_id: the flattened neighbors ids list
    :param neighbors_start: the flattened neighbors start indexes
    :param neighbors_count: the flattened neighbors counts
    :return: a numpy array containing only the i_grid point neighbors
    or None if the point has no neighbors (or no valid neighbors)
    """
    n_neighbors = neighbors_count[i_grid]

    if n_neighbors == 0:
        return None

    n_start = neighbors_start[i_grid]
    neighbors = points[neighbors_id[n_start:n_start + n_neighbors]]
    n_valid = np.sum(
        data_valid[neighbors_id[n_start:n_start + n_neighbors]]
    )

    # discard if grid point has no valid neighbor in point cloud
    if n_valid == 0:
        return None

    return neighbors


@njit((float64[:, :], boolean[:], int64[:], int64[:], int64[:], float64[:, :], float64, int64), nogil=True, cache=True)
def mask_interp(mask_points: np.ndarray, data_valid: np.ndarray, neighbors_id: np.ndarray, neighbors_start: np.ndarray,
                neighbors_count: np.ndarray, grid_points: np.ndarray, sigma: float, undefined_val: int) -> np.ndarray:
    """
    Interpolates mask data at grid point locations.

    For a terrain cell each points contained into it have a weight depending on its distance to the cell center.
    For each classes, the weights are accumulated.
    The class which have the higher accumulated score is then used as the terrain cell's final value.

    :param mask_points: mask data, one point per row (first column is the x position, second is the y position,
    last column is the mask value).
    :param data_valid: flattened validity mask.
    :param neighbors_id: flattened neighboring cloud point indices.
    :param neighbors_start: flattened grid point neighbors start indices.
    :param neighbors_count: flattened grid point neighbor count.
    :param grid_points: grid point location, one per row.
    :param sigma: sigma parameter for weights computation.
    :param undefined_val: value in case of score equality.
    :return: The interpolated mask
    """
    # mask rasterization result
    result = np.full((neighbors_count.size, 1), np.nan, dtype=np.float32)
    print('//////////////////////////////////:')
    print(np.min(mask_points[:, 2]), np.max(mask_points[:, 2]))
    print('//////////////////////////////////:')
    for i_grid in range(neighbors_count.size):
        p_sample = grid_points[i_grid]

        neighbors = get_neighbors(mask_points, data_valid, i_grid, neighbors_id, neighbors_start, neighbors_count)
        if neighbors is None:
            continue

        # grid point to neighbors distance
        neighbors_vec = neighbors[:, :2] - p_sample
        distances = np.sqrt(np.sum(neighbors_vec * neighbors_vec, axis=1))

        # score computation
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

        val = []
        val_cum_weight = []
        for idx in range(len(neighbors)):
            msk_val = (neighbors[idx, 2:])

            if msk_val != 0: # only masked points are taken into account
                print('deal with it !')
                if msk_val in val:
                    msk_val_index = val.index(msk_val)
                    val_cum_weight[msk_val_index] += weights[idx]
                else:
                    val.append(msk_val)
                    val_cum_weight.append(weights[idx])

        # search for higher score
        if len(val) != 0:
            arr_val_cum_weight = np.asarray(val_cum_weight)
            ind_max_weight = np.argmax(arr_val_cum_weight)

            max_weight_values = [val[i] for i in range(len(val)) if val_cum_weight[i] == val_cum_weight[ind_max_weight]]
            if len(max_weight_values) == 1:
                result[i_grid] = val[ind_max_weight]
            else:
                result[i_grid] = undefined_val
        else: # no masked points in the terrain cell
            result[i_grid] = 0

    return result


@njit((float64[:, :], boolean[:], int64[:], int64[:], int64[:], float64[:, :],
       float64, float64), nogil=True, cache=True)
def gaussian_interp(cloud_points, data_valid, neighbors_id, neighbors_start,
                    neighbors_count, grid_points, resolution, sigma):
    """
    Interpolates point cloud data at grid point locations and produces
    quality statistics.

    :param cloud_points: point cloud data, one point per row.
    :type cloud_points: float64 numpy.ndarray.
    :param data_valid: flattened validity mask.
    :type data_valid: bool numpy.ndarray.
    :param neighbors_id: flattened neighboring cloud point indices.
    :type neighbors_id: int64 numpy.ndarray.
    :param neighbors_start: flattened grid point neighbors start indices.
    :type neighbors_start: int64 numpy.ndarray.
    :param neighbors_count: flattened grid point neighbor count.
    :type neighbors_count: int64 numpy.ndarray.
    :param grid_points: grid point location, one per row.
    :type grid_points: float64 numpy.ndarray.
    :param resolution: rasterization resolution.
    :type resolution: float.
    :param sigma: sigma parameter of gaussian interpolation.
    :type sigma: float
    :return: a tuple with rasterization results and statistics.
    """

    # rasterization result for both height and color(s)
    result = np.full((neighbors_count.size, cloud_points.shape[1] - 2), np.nan, dtype=np.float32)

    # statistics layers
    layer_mean = np.full((neighbors_count.size, cloud_points.shape[1] - 2),
                         np.nan, dtype=np.float32)
    layer_stdev = np.full((neighbors_count.size, cloud_points.shape[1] - 2),
                          np.nan, dtype=np.float32)
    n_pts = np.zeros(neighbors_count.size, np.uint16)
    n_pts_in_cell = np.zeros(neighbors_count.size, np.uint16)

    for i_grid in range(neighbors_count.size):

        p_sample = grid_points[i_grid]

        neighbors = get_neighbors(cloud_points, data_valid, i_grid, neighbors_id, neighbors_start, neighbors_count)
        if neighbors is None:
            continue

        # grid point to neighbors distance
        neighbors_vec = neighbors[:, :2] - p_sample
        distances = np.sqrt(np.sum(neighbors_vec * neighbors_vec, axis=1))

        # interpolation weights computation
        min_dist = np.amin(distances)
        weights = np.exp(-(distances - min_dist)**2 / (2 * sigma**2))
        total_weight = np.sum(weights)

        n_pts[i_grid] = neighbors_vec.shape[0]

        # interpolate point cloud data
        result[i_grid] = np.dot(weights, neighbors[:, 2:]) / total_weight

        # compute statistic for each layer
        for n_layer in range(2, cloud_points.shape[1]):
            layer_stdev[i_grid][n_layer - 2] = np.std(neighbors[:, n_layer])
            layer_mean[i_grid][n_layer - 2] = np.mean(neighbors[:, n_layer])

        n_pts_in_cell[i_grid] = np.sum(
            (np.abs(neighbors_vec[:, 0]) < 0.5 * resolution) &
            (np.abs(neighbors_vec[:, 1]) < 0.5 * resolution)
        )

    return result, layer_mean, layer_stdev, n_pts, n_pts_in_cell


def create_raster_dataset(raster: np.ndarray, x_start: float, y_start: float, x_size: int, y_size: int,
                          resolution: float, hgt_no_data: int, color_no_data: int, msk_no_data: int, epsg: int, mean: np.ndarray,
                          stdev: np.ndarray, n_pts: np.ndarray, n_in_cell: np.ndarray, msk:np.ndarray=None)\
        -> xr.Dataset:
    """
    Create final raster xarray dataset

    :param raster: height and colors
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :param hgt_no_data: no data value to use for height
    :param color_no_data: no data value to use for color
    :param msk_no_data:
    :param epsg: epsg code for the CRS of the final raster
    :param mean: mean of height and colors
    :param stdev: standard deviation of height and colors
    :param n_pts: number of points that are stricty in a cell
    :param n_in_cell: number of points which contribute to a cell
    :param msk: raster msk
    :return: the raster xarray dataset
    """
    raster_dims = ('y', 'x')
    n_layers = raster.shape[-1]
    x_values_1d, y_values_1d = compute_values_1d(x_start, y_start, x_size, y_size, resolution)
    raster_coords = {'x': x_values_1d, 'y': y_values_1d}
    hgt = np.nan_to_num(raster[..., 0], nan=hgt_no_data)
    raster_out = xr.Dataset({'hgt': (['y', 'x'], hgt)},
                            coords=raster_coords)

    if raster.shape[-1] > 1:  # rasterizer produced color output
        band = range(1, raster.shape[-1])
        # CAUTION: band/channel is set as the first dimension.
        clr = np.nan_to_num(np.rollaxis(raster[:, :, 1:], 2), nan=color_no_data)
        color_out = xr.Dataset(
            {'img': (['band', 'y', 'x'], clr)},
            coords={**raster_coords, 'band': band})
        # update raster output with color data
        raster_out = xr.merge((raster_out, color_out))

    raster_out.attrs['epsg'] = epsg
    raster_out.attrs['resolution'] = resolution

    # statics layer for height output
    raster_out['hgt_mean'] = xr.DataArray(mean[..., 0], coords=raster_coords,
                                          dims=raster_dims)
    raster_out['hgt_stdev'] = xr.DataArray(stdev[..., 0], coords=raster_coords,
                                           dims=raster_dims)
    # add each band statistics
    for i_layer in range(1, n_layers):
        raster_out['band_{}_mean'.format(i_layer)] = xr.DataArray(
            mean[..., i_layer], coords=raster_coords, dims=raster_dims
        )
        raster_out['band_{}_stdev'.format(i_layer)] = xr.DataArray(
            stdev[..., i_layer], coords=raster_coords, dims=raster_dims
        )

    raster_out['n_pts'] = xr.DataArray(n_pts, dims=raster_dims)
    raster_out['pts_in_cell'] = xr.DataArray(n_in_cell, dims=raster_dims)

    if msk is not None:
        msk = np.nan_to_num(msk, nan=msk_no_data)
        raster_out['left_msk'] = xr.DataArray(msk.astype(np.uint16), dims=raster_dims)

    return raster_out


def rasterize(cloud: pandas.DataFrame, resolution: float, epsg: int, x_start: float, y_start: float, x_size: int,
              y_size: int, sigma: float=None, radius: int=1, hgt_no_data: int=np.nan, color_no_data: int=np.nan,
              msk_no_data: int=np.nan, grid_points_division_factor: int=None)\
        -> Union[xr.Dataset, None]:
    """
    Rasterize a point cloud with its color bands to a Dataset that also contains quality statistics.

    :param cloud: combined cloud as returned by the create_combined_cloud function
    :param resolution: Resolution of rasterized cells, expressed in cloud CRS units or None.
    :param epsg: epsg code for the CRS of the final raster
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param sigma: sigma for gaussian interpolation. If None, set to resolution
    :param radius: Radius for hole filling.
    :param hgt_no_data: no data value to use for height
    :param color_no_data: no data value to use for color
    :param msk_no_data:
    :param grid_points_division_factor: number of blocs to use to divide the grid points (memory optimization, reduce
     the highest memory peak). If it is not set, the factor is automatically set to construct 700000 points blocs.
    :return: Rasterized cloud color and statistics.
    """
    worker_logger = logging.getLogger('distributed.worker')

    if sigma is None:
        sigma = resolution

    # generate validity mask from margins and all masks of cloud data.
    data_valid = cloud['data_valid'].values

    # If no valid points are found in cloud, return default values
    if np.size(data_valid) == 0:
        worker_logger.debug("No points to rasterize, returning None")
        return None

    worker_logger.debug(
        "Rasterization grid: start=[{},{}], size=[{},{}], resolution={}"
            .format(x_start, y_start, x_size, y_size, resolution))

    out, mean, stdev, n_pts, n_in_cell, msk = \
        compute_vector_raster_and_stats(cloud, data_valid, x_start, y_start, x_size, y_size, resolution,
                                        sigma, radius, worker_logger, grid_points_division_factor)

    # reshape data as a 2d grid.
    tic = time.process_time()
    shape_out = (y_size, x_size)
    out = out.reshape(shape_out + (-1,))
    mean = mean.reshape(shape_out + (-1,))
    stdev = stdev.reshape(shape_out + (-1,))
    n_pts = n_pts.reshape(shape_out)
    n_in_cell = n_in_cell.reshape(shape_out)

    if msk is not None:
        msk = msk.reshape(shape_out)

    toc = time.process_time()
    worker_logger.debug('Output reshaping done in {} seconds'
                        .format(toc - tic))

    # build output dataset
    tic = time.process_time()
    raster_out = create_raster_dataset(out, x_start, y_start, x_size, y_size, resolution, hgt_no_data, color_no_data,
                                       msk_no_data, epsg, mean, stdev, n_pts, n_in_cell, msk)

    toc = time.process_time()
    worker_logger.debug(
        "Final raster formatting into a xarray.Dataset done in {} seconds".format(toc - tic))

    return raster_out
