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
import logging
from typing import Dict

# Third party imports
import numpy as np
import pandas
import xarray as xr

from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp


def triangulate(
    geometry_plugin,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    disp_ref: xr.Dataset,
    im_ref_msk_ds: xr.Dataset = None,
) -> Dict[str, xr.Dataset]:
    """
    This function will perform triangulation from a disparity map

    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: str
    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param disp_ref: left to right disparity map dataset
    :param im_ref_msk_ds: reference image dataset (image and
                          mask (if indicated by the user) in epipolar geometry)
    :param snap_to_img1: If True, Lines of Sight of img2 are moved so as to
                         cross those of img1
    :param snap_to_img1: bool
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
    point_clouds = {}
    point_clouds[cst.STEREO_REF] = compute_points_cloud(
        geometry_plugin,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid1,
        grid2,
        disp_ref,
        roi_key=cst.ROI,
        dataset_msk=im_ref_msk_ds,
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

    :param TODO
    :type TODO
    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
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
        cst.POINTS_CLOUD_CORR_MSK,
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


def compute_points_cloud(
    geometry_plugin,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    data: xr.Dataset,
    roi_key: str,
    dataset_msk: xr.Dataset = None,
) -> xr.Dataset:
    # TODO detail a bit more what this method do
    """
    Compute points cloud

    :param geometry_plugin: geometry plugin to use
    :param data: The reference to disparity map dataset
    :param cars_conf: cars input configuration dictionary
    :param grid1: path to the reference image grid file
    :param grid2: path to the secondary image grid file
    :param roi_key: roi of the disparity map key
          ('roi' if cropped while calling create_disp_dataset,
          otherwise 'roi_with_margins')
    :param dataset_msk: dataset with mask information to use
    :return: the points cloud dataset
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
        cst.POINTS_CLOUD_CORR_MSK: (
            [cst.ROW, cst.COL],
            data[cst_disp.VALID].values,
        ),
    }
    if dataset_msk is not None:
        ds_values_list = [key for key, _ in dataset_msk.items()]

        if cst.EPI_MSK in ds_values_list:
            if roi_key == cst.ROI_WITH_MARGINS:
                ref_roi = [
                    0,
                    0,
                    int(dataset_msk.dims[cst.COL]),
                    int(dataset_msk.dims[cst.ROW]),
                ]
            else:
                ref_roi = [
                    int(-dataset_msk.attrs[cst.EPI_MARGINS][0]),
                    int(-dataset_msk.attrs[cst.EPI_MARGINS][1]),
                    int(
                        dataset_msk.dims[cst.COL]
                        - dataset_msk.attrs[cst.EPI_MARGINS][2]
                    ),
                    int(
                        dataset_msk.dims[cst.ROW]
                        - dataset_msk.attrs[cst.EPI_MARGINS][3]
                    ),
                ]

            # propagate all the data in the point cloud (except color)
            for key, val in dataset_msk.items():
                if len(val.values.shape) == 2:
                    values[key] = (
                        [cst.ROW, cst.COL],
                        val.values[
                            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
                        ],
                    )

            for key, val in data.items():
                if len(val.values.shape) == 2:
                    if "msk_" not in key and "color" not in key:
                        values[key] = ([cst.ROW, cst.COL], val.values)

        else:
            logging.warning("No mask is present in the image dataset")

    point_cloud = xr.Dataset(values, coords={cst.ROW: row, cst.COL: col})

    # add color
    if cst.EPI_COLOR in data:
        add_layer(data, cst.EPI_COLOR, cst.BAND_IM, point_cloud, nodata_index)

    # add classif
    if cst.EPI_CLASSIFICATION in data:
        add_layer(
            data,
            cst.EPI_CLASSIFICATION,
            cst.BAND_CLASSIF,
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


def add_layer(
    dataset, layer_name, layer_coords, point_cloud, nodata_index=None
):
    """
    Add layer point cloud to point cloud dataset

    :param data: layer point cloud dataset
    :param nodata_index: nodata index array
    :param point_cloud: point cloud dataset
    """
    layers = dataset[layer_name].values
    band_layer = dataset.coords[layer_coords]
    if nodata_index:
        nb_bands = layers.shape[0]
        for k in range(nb_bands):
            layers[k, :, :][nodata_index] = np.nan

    if layer_coords not in point_cloud.dims:
        point_cloud.coords[layer_coords] = band_layer

    point_cloud[layer_name] = xr.DataArray(
        layers,
        dims=[layer_coords, cst.ROW, cst.COL],
    )


def geoid_offset(points, geoid):
    """
    Compute the point cloud height offset from geoid.

    :param points: point cloud data in lat/lon/alt WGS84 (EPSG 4326)
        coordinates.
    :type points: xarray.Dataset
    :param geoid: geoid elevation data.
    :type geoid: xarray.Dataset
    :return: the same point cloud but using geoid as altimetric reference.
    :rtype: xarray.Dataset
    """

    # deep copy the given point cloud that will be used as output
    out_pc = points.copy(deep=True)

    # currently assumes that the OTB EGM96 geoid will be used with longitude
    # ranging from 0 to 360, so we must unwrap longitudes to this range.
    longitudes = np.copy(out_pc[cst.X].values)
    longitudes[longitudes < 0] += 360

    # perform interpolation using point cloud coordinates.
    if sum(longitudes.shape) != 0:
        if (
            not geoid.lat_min
            <= out_pc[cst.Y].min()
            <= out_pc[cst.Y].max()
            <= geoid.lat_max
            and geoid.lon_min
            <= np.min(longitudes)
            <= np.max(longitudes)
            <= geoid.lat_max
        ):
            raise RuntimeError(
                "Geoid does not fully cover the area spanned by"
                " the point cloud."
            )

        out_pc[cst.Z] = points[cst.Z]

        # interpolate data
        if isinstance(out_pc, xr.Dataset):
            ref_interp = geoid.interp(
                {
                    "lat": out_pc[cst.Y],
                    "lon": xr.DataArray(longitudes, dims=(cst.ROW, cst.COL)),
                }
            )

            ref_interp_hgt = ref_interp.hgt.values

        else:
            # one dimension is equal to 1, happens with matches tiangulation
            ref_interp = geoid.interp(
                {
                    "lat": out_pc[cst.Y].values,
                    "lon": longitudes,
                }
            )

            ref_interp_hgt = ref_interp.hgt.values.diagonal()

        # offset using geoid height
        out_pc[cst.Z] -= ref_interp_hgt

    return out_pc
