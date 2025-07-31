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

import numpy as np
import pandas
import xarray as xr
from scipy import interpolate
from shareloc.image import Image
from shareloc.proj_utils import transform_physical_point_to_index

from cars.core import constants as cst


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
        # ambiguity is already ambiguity, not confidence from ambiguity
        ambiguity_map = ambiguity_map.values
        mask_ambi = ambiguity_map > perf_ambiguity_threshold
        w_ambi = ambiguity_map / perf_ambiguity_threshold
        w_ambi[mask_ambi] = 1
        performance_map_values *= w_ambi

    performance_map.values = performance_map_values

    return performance_map
