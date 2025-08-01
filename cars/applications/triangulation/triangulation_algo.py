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

# Standard imports
from typing import Dict

import numpy as np
import pandas
import xarray as xr

from cars.applications.triangulation import (
    triangulation_wrappers as triang_wrap,
)
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
        roi_key=cst.ROI_WITH_MARGINS,
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
            if key not in (cst.EPI_TEXTURE, cst_disp.VALID):
                values[key] = ([cst.ROW, cst.COL], val.values)

    point_cloud = xr.Dataset(values, coords={cst.ROW: row, cst.COL: col})

    # add color and data type of image
    color_type = None
    if cst.EPI_TEXTURE in data:
        triang_wrap.add_layer(data, cst.EPI_TEXTURE, cst.BAND_IM, point_cloud)
        color_type = data[cst.EPI_TEXTURE].attrs["color_type"]
    elif cst.EPI_IMAGE in data:
        color_type = data[cst.EPI_IMAGE].attrs["color_type"]
    if color_type:
        point_cloud.attrs["color_type"] = color_type

    # add classif
    if cst.EPI_CLASSIFICATION in data:
        triang_wrap.add_layer(
            data,
            cst.EPI_CLASSIFICATION,
            cst.BAND_CLASSIF,
            point_cloud,
        )

    # add filling in data:
    if cst.EPI_FILLING in data:
        triang_wrap.add_layer(
            data,
            cst.EPI_FILLING,
            cst.BAND_FILLING,
            point_cloud,
        )

    point_cloud.attrs[cst.ROI] = data.attrs[cst.ROI]
    point_cloud.attrs[cst.ROI_WITH_MARGINS] = data.attrs[cst.ROI_WITH_MARGINS]
    point_cloud.attrs[cst.EPI_MARGINS] = data.attrs[cst.EPI_MARGINS]
    point_cloud.attrs[cst.EPI_FULL_SIZE] = data.attrs[cst.EPI_FULL_SIZE]
    point_cloud.attrs[cst.EPSG] = int(4326)

    return point_cloud
