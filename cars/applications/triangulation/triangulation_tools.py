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
Preprocessing module:
contains functions used for triangulation
"""
import copy

# Third party imports
import logging
import os

# Standard imports
from typing import Dict

import numpy as np
import pandas
import xarray as xr
from scipy import interpolate
from shareloc.image import Image
from shareloc.proj_utils import transform_physical_point_to_index

from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import projection


def triangulate(
    geometry_plugin,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    disp_ref: xr.Dataset,
    disp_key: str = cst_disp.MAP,
) -> Dict[str, xr.Dataset]:
    """
    This function will perform triangulation from a disparity map

    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param sensor1: path to left sensor image
    :type sensor1: str
    :param sensor2: path to right sensor image
    :type sensor2: str
    :param geomodel1: path and attributes for left geomodel
    :type geomodel1: dict
    :param geomodel2: path and attributes for right geomodel
    :type geomodel2: dict
    :param grid1: dataset of the reference image grid file
    :type grid1: CarsDataset
    :param grid2: dataset of the secondary image grid file
    :type grid2: CarsDataset
    :param disp_ref: left to right disparity map dataset
    :param im_ref_msk_ds: reference image dataset (image and
                          mask (if indicated by the user) in epipolar geometry)
    :param disp_key: disparity key in the dataset\
            usually set to cst_disp.MAP, but can be a disparity interval bound
    :returns: point_cloud as a dictionary of dataset containing:

        - Array with shape (roi_size_x,roi_size_y,3), with last dimension \
          corresponding to longitude, latitude and elevation
        - Array with shape (roi_size_x,roi_size_y) with output mask
        - Array for color (optional): only if color1 is not None

    The dictionary keys are :

        - 'ref' to retrieve the dataset built from the left to \
           right disparity map
        - 'sec' to retrieve the dataset built from the right to \
           left disparity map (if provided in input)
    """

    if disp_key != cst_disp.MAP:
        # Switching the variable names so the desired disparity is named 'disp'
        # It does not modifies the dataset outside of this function
        disp_ref = disp_ref.rename_vars(
            {disp_key: cst_disp.MAP, cst_disp.MAP: disp_key}
        )

    point_clouds = {}
    point_clouds[cst.STEREO_REF] = compute_point_cloud(
        geometry_plugin,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid1,
        grid2,
        disp_ref,
        roi_key=cst.ROI,
    )

    return point_clouds


def triangulate_matches(
    geometry_plugin,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    matches,
):
    """
    This function will perform triangulation from sift matches

    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param sensor1: path to left sensor image
    :type sensor1: str
    :param sensor2: path to right sensor image
    :type sensor2: str
    :param geomodel1: path and attributes for left geomodel
    :type geomodel1: dict
    :param geomodel2: path and attributes for right geomodel
    :type geomodel2: dict
    :param grid1: dataset of the reference image grid file
    :type grid1: CarsDataset
    :param grid2: dataset of the secondary image grid file
    :type grid2: CarsDataset
    :param matches: numpy.array of matches of shape (nb_matches, 4)
    :type data: numpy.ndarray
    :returns: point_cloud as a panda DataFrame containing:

        - Array with shape (nb_matches,1,3), with last dimension \
        corresponding to longitude, latitude and elevation
        - Array with shape (nb_matches,1) with output mask
        - cst.X
        - cst.Y
        - cst.Z
        - corr_mask
        - lon
        - lat


    :rtype: pandas.DataFrame
    """
    llh = geometry_plugin.triangulate(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        cst.MATCHES_MODE,
        matches,
        grid1,
        grid2,
    )

    disparity = np.array([matches[:, 2] - matches[:, 0]])
    disparity = np.transpose(disparity)

    msk = np.full(llh.shape[0:2], 255, dtype=np.uint8)

    point_cloud_index = [
        cst.X,
        cst.Y,
        cst.Z,
        cst.DISPARITY,
        cst.POINT_CLOUD_CORR_MSK,
    ]
    point_cloud_array = np.zeros(
        (np.ravel(llh[:, :, 0]).size, len(point_cloud_index)), dtype=np.float64
    )
    point_cloud_array[:, 0] = np.ravel(llh[:, :, 0])
    point_cloud_array[:, 1] = np.ravel(llh[:, :, 1])
    point_cloud_array[:, 2] = np.ravel(llh[:, :, 2])
    point_cloud_array[:, 3] = np.ravel(disparity)
    point_cloud_array[:, 4] = np.ravel(msk)
    point_cloud = pandas.DataFrame(point_cloud_array, columns=point_cloud_index)
    point_cloud.attrs[cst.EPSG] = int(cst.EPSG_WSG84)
    return point_cloud


def triangulate_sparse_matches(
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    interpolated_grid_left,
    interpolated_grid_right,
    matches,
    geometry_plugin,
    epsg,
):
    """
    Triangulate matches in a metric system

    :param sensor_image_right: sensor image right
    :type sensor_image_right: CarsDataset
    :param sensor_image_left: sensor image left
    :type sensor_image_left: CarsDataset
    :param grid_left: grid left
    :type grid_left: CarsDataset CarsDataset
    :param grid_right: corrected grid right
    :type grid_right: CarsDataset
    :param interpolated_grid_left: rectification grid left
    :type interpolated_grid_left: shareloc.rectificationGrid
    :param interpolated_grid_right: rectification grid right
    :type interpolated_grid_right: shareloc.rectificationGrid
    :param matches: matches
    :type matches: np.ndarray
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: float
    :param pair_folder: folder used for current pair
    :type pair_folder: str
    :param epsg: ground epsg
    :type epsg: int

    :return: disp min and disp max
    :rtype: float, float
    """

    point_cloud = triangulate_matches(
        geometry_plugin,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        interpolated_grid_left,
        interpolated_grid_right,
        np.ascontiguousarray(matches),
    )

    # Project point cloud to UTM
    projection.point_cloud_conversion_dataset(point_cloud, epsg)

    # Convert point cloud to pandas format to allow statistical filtering
    labels = [cst.X, cst.Y, cst.Z, cst.DISPARITY, cst.POINT_CLOUD_CORR_MSK]
    cloud_array = []
    cloud_array.append(point_cloud[cst.X].values)
    cloud_array.append(point_cloud[cst.Y].values)
    cloud_array.append(point_cloud[cst.Z].values)
    cloud_array.append(point_cloud[cst.DISPARITY].values)
    cloud_array.append(point_cloud[cst.POINT_CLOUD_CORR_MSK].values)
    pd_cloud = pandas.DataFrame(
        np.transpose(np.array(cloud_array)), columns=labels
    )

    pd_cloud.attrs["epsg"] = epsg

    return pd_cloud


def compute_point_cloud(
    geometry_plugin,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    data: xr.Dataset,
    roi_key: str,
) -> xr.Dataset:
    # TODO detail a bit more what this method do
    """
    Compute point cloud

    :param geometry_plugin: geometry plugin to use
    :param sensor1: path to left sensor image
    :param sensor2: path to right sensor image
    :param geomodel1: path and attributes for left geomodel
    :param geomodel2: path and attributes for right geomodel
    :param grid1: dataset of the reference image grid file
    :param grid2: dataset of the secondary image grid file
    :param data: The reference to disparity map dataset
    :param roi_key: roi of the disparity map key
          ('roi' if cropped while calling create_disp_dataset,
          otherwise 'roi_with_margins')
    :return: the point cloud dataset
    """
    # Extract input paths from configuration
    llh = geometry_plugin.triangulate(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        cst.DISP_MODE,
        data,
        grid1,
        grid2,
        roi_key,
    )

    row = np.array(range(data.attrs[roi_key][1], data.attrs[roi_key][3]))
    col = np.array(range(data.attrs[roi_key][0], data.attrs[roi_key][2]))

    # apply no_data to X,Y and Z point cloud
    nodata_index = np.where(data[cst_disp.VALID].values == 0)
    llh[:, :, 0][nodata_index] = np.nan
    llh[:, :, 1][nodata_index] = np.nan
    llh[:, :, 2][nodata_index] = np.nan

    values = {
        cst.X: ([cst.ROW, cst.COL], llh[:, :, 0]),  # longitudes
        cst.Y: ([cst.ROW, cst.COL], llh[:, :, 1]),  # latitudes
        cst.Z: ([cst.ROW, cst.COL], llh[:, :, 2]),
        cst.POINT_CLOUD_CORR_MSK: (
            [cst.ROW, cst.COL],
            data[cst_disp.VALID].values,
        ),
    }

    # Copy all 2D attributes from disparity dataset to point cloud
    # except color and pandora validity mask (already copied in corr_msk)
    for key, val in data.items():
        if len(val.values.shape) == 2:
            if key not in (cst.EPI_COLOR, cst_disp.VALID):
                values[key] = ([cst.ROW, cst.COL], val.values)

    point_cloud = xr.Dataset(values, coords={cst.ROW: row, cst.COL: col})

    # add color and data type of image
    color_type = None
    if cst.EPI_COLOR in data:
        add_layer(data, cst.EPI_COLOR, cst.BAND_IM, point_cloud)
        color_type = data[cst.EPI_COLOR].attrs["color_type"]
    elif cst.EPI_IMAGE in data:
        color_type = data[cst.EPI_IMAGE].attrs["color_type"]
    if color_type:
        point_cloud.attrs["color_type"] = color_type

    # add classif
    if cst.EPI_CLASSIFICATION in data:
        add_layer(
            data,
            cst.EPI_CLASSIFICATION,
            cst.BAND_CLASSIF,
            point_cloud,
        )

    # add filling in data:
    if cst.EPI_FILLING in data:
        add_layer(
            data,
            cst.EPI_FILLING,
            cst.BAND_FILLING,
            point_cloud,
        )

    point_cloud.attrs[cst.ROI] = data.attrs[cst.ROI]
    if roi_key == cst.ROI_WITH_MARGINS:
        point_cloud.attrs[cst.ROI_WITH_MARGINS] = data.attrs[
            cst.ROI_WITH_MARGINS
        ]
    point_cloud.attrs[cst.EPI_FULL_SIZE] = data.attrs[cst.EPI_FULL_SIZE]
    point_cloud.attrs[cst.EPSG] = int(4326)

    return point_cloud


def add_layer(dataset, layer_name, layer_coords, point_cloud):
    """
    Add layer point cloud to point cloud dataset

    :param dataset: input disparity map dataset
    :param layer_name: layer key in disparity dataset
    :param layer_coords: layer axis name in disparity dataset
    :param point_cloud: output point cloud dataset
    """
    layers = dataset[layer_name].values
    band_layer = dataset.coords[layer_coords]

    if layer_coords not in point_cloud.dims:
        point_cloud.coords[layer_coords] = band_layer

    point_cloud[layer_name] = xr.DataArray(
        layers,
        dims=[layer_coords, cst.ROW, cst.COL],
    )


def interpolate_geoid_height(
    geoid_filename, positions, interpolation_method="linear"
):
    """
    terrain to index conversion
    retrieve geoid height above ellispoid
    This is a modified version of the Shareloc interpolate_geoid_height
    function that supports Nan positions (return Nan)

    :param geoid_filename: geoid_filename
    :type geoid_filename: str
    :param positions: geodetic coordinates
    :type positions: 2D numpy array: (number of points,[long coord, lat coord])
    :param interpolation_method: default is 'linear' (interpn parameter)
    :type interpolation_method: str
    :return: geoid height
    :rtype: 1 numpy array (number of points)
    """

    geoid_image = Image(geoid_filename, read_data=True)

    # Check longitude overlap is not present, rounding to handle egm2008 with
    # rounded pixel size
    if geoid_image.nb_columns * geoid_image.pixel_size_col - 360 < 10**-8:
        logging.debug("add one pixel overlap on longitudes")
        geoid_image.nb_columns += 1
        # Check if we can add a column
        geoid_image.data = np.column_stack(
            (geoid_image.data[:, :], geoid_image.data[:, 0])
        )

    # Prepare grid for interpolation
    row_indexes = np.arange(0, geoid_image.nb_rows, 1)
    col_indexes = np.arange(0, geoid_image.nb_columns, 1)
    points = (row_indexes, col_indexes)

    # add modulo lon/lat
    min_lon = geoid_image.origin_col + geoid_image.pixel_size_col / 2
    max_lon = (
        geoid_image.origin_col
        + geoid_image.nb_columns * geoid_image.pixel_size_col
        - geoid_image.pixel_size_col / 2
    )
    positions[:, 0] += ((positions[:, 0] + min_lon) < 0) * 360.0
    positions[:, 0] -= ((positions[:, 0] - max_lon) > 0) * 360.0
    if np.any(np.abs(positions[:, 1]) > 90.0):
        raise RuntimeError("Geoid cannot handle latitudes greater than 90 deg.")
    indexes_geoid = transform_physical_point_to_index(
        geoid_image.trans_inv, positions[:, 1], positions[:, 0]
    )
    return interpolate.interpn(
        points,
        geoid_image.data[:, :],
        indexes_geoid,
        bounds_error=False,
        method=interpolation_method,
    )


def geoid_offset(points, geoid_path):
    """
    Compute the point cloud height offset from geoid.

    :param points: point cloud data in lat/lon/alt WGS84 (EPSG 4326)
        coordinates.
    :type points: xarray.Dataset or pandas.DataFrame
    :param geoid_path: path to input geoid file on disk
    :type geoid_path: string
    :return: the same point cloud but using geoid as altimetric reference.
    :rtype: xarray.Dataset or pandas.DataFrame
    """

    # deep copy the given point cloud that will be used as output
    out_pc = points.copy(deep=True)

    # interpolate data
    if isinstance(out_pc, xr.Dataset):
        # Convert the dataset to a np array as expected by Shareloc
        pc_array = (
            out_pc[[cst.X, cst.Y]]
            .to_array()
            .to_numpy()
            .transpose((1, 2, 0))
            .reshape((out_pc.sizes["row"] * out_pc.sizes["col"], 2))
        )
        geoid_height_array = interpolate_geoid_height(
            geoid_path, pc_array
        ).reshape((out_pc.sizes["row"], out_pc.sizes["col"]))
    elif isinstance(out_pc, pandas.DataFrame):
        geoid_height_array = interpolate_geoid_height(
            geoid_path, out_pc[[cst.X, cst.Y]].to_numpy()
        )
    else:
        raise RuntimeError("Invalid point cloud type")

    # offset using geoid height
    out_pc[cst.Z] -= geoid_height_array

    return out_pc


def generate_point_cloud_file_names(
    csv_dir: str,
    laz_dir: str,
    row: int,
    col: int,
    index: dict = None,
    pair_key: str = "PAIR_0",
):
    """
    generate the point cloud CSV and LAZ filenames of a given tile from its
    corresponding row and col. Optionally update the index, if provided.

    :param csv_dir: target directory for csv files, If None no csv filenames
        will be generated
    :type csv_dir: str
    :param laz_dir: target directory for laz files, If None no laz filenames
        will be generated
    :type laz_dir: str
    :param row: row index of the tile
    :type row: int
    :param col: col index of the tile
    :type col: int
    :param index: product index to update with the filename
    :type index: dict
    :param pair_key: current product key (used in index), if a list is given
        a filename will be added to the index for each element of the list
    :type pair_key: str
    """

    file_name_root = str(col) + "_" + str(row)
    csv_pc_file_name = None
    if csv_dir is not None:
        csv_pc_file_name = os.path.join(csv_dir, file_name_root + ".csv")

    laz_pc_file_name = None
    if laz_dir is not None:
        laz_name = file_name_root + ".laz"
        laz_pc_file_name = os.path.join(laz_dir, laz_name)
        # add to index if the laz is saved to output product
        if index is not None:
            # index initialization, if it has not been done yet
            if "point_cloud" not in index:
                index["point_cloud"] = {}
            # case where merging=True and save_by_pair=False
            if pair_key is None:
                index["point_cloud"][file_name_root] = laz_name
            else:
                if isinstance(pair_key, str):
                    pair_key = [pair_key]
                for elem in pair_key:
                    if elem not in index["point_cloud"]:
                        index["point_cloud"][elem] = {}
                    index["point_cloud"][elem][file_name_root] = os.path.join(
                        elem, laz_name
                    )

    return csv_pc_file_name, laz_pc_file_name


def compute_performance_map(
    alti_ref, z_inf, z_sup, ambiguity_map=None, perf_ambiguity_threshold=None
):
    """
    Compute performance map

    :param alti_ref: z
    :type alti_ref: xarray Dataarray
    :param z_inf: z inf map
    :type z_inf: xarray Dataarray
    :param z_sup: z sup map
    :type z_sup: xarray Dataarray
    :param ambiguity_map: None or ambiguity map
    :type ambiguity_map: xarray Dataarray
    :param perf_ambiguity_threshold: ambiguity threshold to use
    :type perf_ambiguity_threshold: None or float

    """
    performance_map = copy.copy(alti_ref)

    performance_map_values = np.maximum(
        np.abs(alti_ref.values - z_inf.values),
        np.abs(z_sup.values - alti_ref.values),
    )

    if ambiguity_map is not None:
        ambiguity_map = 1 - ambiguity_map.values
        mask_ambi = ambiguity_map > perf_ambiguity_threshold
        w_ambi = ambiguity_map / perf_ambiguity_threshold
        w_ambi[mask_ambi] = 1
        performance_map_values *= w_ambi

    performance_map.values = performance_map_values

    return performance_map
