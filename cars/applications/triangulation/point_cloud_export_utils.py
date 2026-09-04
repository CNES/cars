#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
Shared helpers for point cloud product exports.
"""

# pylint: disable=too-many-positional-arguments

import os

import numpy as np

import cars.orchestrator.orchestrator as ocht
from cars.applications.triangulation import pc_transform
from cars.core import constants as cst
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


def register_official_point_cloud_tif(
    orchestrator,
    output_dir,
    point_cloud,
    pair_key,
    cars_ds_name_suffix,
    invalidity_dtype,
):
    """
    Register official X/Y/Z/invalidity outputs and point-cloud index.
    """

    safe_makedirs(output_dir)

    orchestrator.add_to_save_lists(
        os.path.join(output_dir, "X.tif"),
        cst.X,
        point_cloud,
        cars_ds_name="depth_map_x_" + cars_ds_name_suffix,
        dtype=np.float64,
    )
    orchestrator.add_to_save_lists(
        os.path.join(output_dir, "Y.tif"),
        cst.Y,
        point_cloud,
        cars_ds_name="depth_map_y_" + cars_ds_name_suffix,
        dtype=np.float64,
    )
    orchestrator.add_to_save_lists(
        os.path.join(output_dir, "Z.tif"),
        cst.Z,
        point_cloud,
        cars_ds_name="depth_map_z_" + cars_ds_name_suffix,
        dtype=np.float64,
    )
    orchestrator.add_to_save_lists(
        os.path.join(output_dir, "invalidity_mask.tif"),
        cst.EPI_INVALIDITY_MASK,
        point_cloud,
        cars_ds_name="depth_map_invalidity_mask_" + cars_ds_name_suffix,
        dtype=invalidity_dtype,
        nodata=255,
        optional_data=True,
    )

    index = {
        cst.INDEX_DEPTH_MAP_X: os.path.join(pair_key, "X.tif"),
        cst.INDEX_DEPTH_MAP_Y: os.path.join(pair_key, "Y.tif"),
        cst.INDEX_DEPTH_MAP_Z: os.path.join(pair_key, "Z.tif"),
    }
    orchestrator.update_index({"point_cloud": {pair_key: index}})


def register_flatten_point_cloud_dataset(
    orchestrator,
    point_cloud,
    point_cloud_dir,
    dump_dir,
    point_cloud_format,
    save_intermediate_data,
    flattened_dataset_name,
    laz_cars_ds_name,
    csv_cars_ds_name,
):
    """
    Register flattened point-cloud dataset and output directories.
    """

    if isinstance(point_cloud_format, str):
        point_cloud_format = [point_cloud_format]

    save_point_cloud_as_csv = save_intermediate_data
    save_point_cloud_as_laz = (
        point_cloud_dir is not None and "laz" in point_cloud_format
    ) or save_intermediate_data

    flattened_point_cloud = cars_dataset.CarsDataset(
        "points", name=flattened_dataset_name
    )
    flattened_point_cloud.create_empty_copy(point_cloud)
    flattened_point_cloud.attributes = point_cloud.attributes.copy()

    laz_pc_dir_name = None
    if save_point_cloud_as_laz:
        if point_cloud_dir is not None:
            laz_pc_dir_name = os.path.join(point_cloud_dir, "laz")
        else:
            laz_pc_dir_name = os.path.join(dump_dir, "laz")
        safe_makedirs(laz_pc_dir_name)
        orchestrator.add_to_compute_lists(
            flattened_point_cloud,
            cars_ds_name=laz_cars_ds_name,
        )

    csv_pc_dir_name = None
    if save_point_cloud_as_csv:
        csv_pc_dir_name = os.path.join(dump_dir, "csv")
        safe_makedirs(csv_pc_dir_name)
        orchestrator.add_to_compute_lists(
            flattened_point_cloud,
            cars_ds_name=csv_cars_ds_name,
        )

    [saving_info_flatten] = orchestrator.get_saving_infos(
        [flattened_point_cloud]
    )

    return (
        flattened_point_cloud,
        laz_pc_dir_name,
        csv_pc_dir_name,
        saving_info_flatten,
    )


def build_flattened_dataframe(
    result,
    point_cloud_tile,
    saving_info_flatten,
):
    """
    Build flattened dataframe from a depth-map dataset tile.
    """

    flatten_result, cloud_epsg = pc_transform.depth_map_dataset_to_dataframe(
        result,
        result.attrs["epsg"],
    )
    attributes = {
        "epsg": cloud_epsg,
        "color_type": pc_transform.get_color_type([result]),
        cst.CROPPED_DISPARITY_RANGE: ocht.get_disparity_range_cropped(
            point_cloud_tile
        ),
    }
    cars_dataset.fill_dataframe(
        flatten_result,
        saving_info=saving_info_flatten,
        attributes=attributes,
    )
    return flatten_result


def save_flattened_dataframe(
    flatten_result,
    point_cloud_csv_file_name,
    point_cloud_laz_file_name,
):
    """
    Save flattened point cloud dataframe to csv/laz if requested.
    """

    if point_cloud_csv_file_name:
        cars_dataset.run_save_points(
            flatten_result,
            point_cloud_csv_file_name,
            overwrite=True,
            point_cloud_format="csv",
            overwrite_file_name=False,
        )
    if point_cloud_laz_file_name:
        cars_dataset.run_save_points(
            flatten_result,
            point_cloud_laz_file_name,
            overwrite=True,
            point_cloud_format="laz",
            overwrite_file_name=False,
        )
