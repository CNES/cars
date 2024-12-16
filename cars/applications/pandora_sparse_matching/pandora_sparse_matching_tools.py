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
"""
This module contains function for pandora sparse matching.
"""
import numpy as np
import pandas
import xarray as xr
from pandora.img_tools import add_global_disparity
from scipy.ndimage import zoom

import cars.applications.dense_matching.dense_matching_constants as dm_cst
from cars.applications.dense_matching import dense_matching_tools as dm_tools
from cars.applications.point_cloud_outlier_removal import outlier_removal_tools
from cars.core import constants as cst
from cars.data_structures import cars_dataset

# pylint: disable=too-many-lines


def downsample(tab, resolution):
    """
    Downsample the image dataset

    :param tab: the image dataset
    :type tab: cars dataset
    :param resolution: the resolution of the resampling
    :type resolution: float

    :return: the downsampled image
    :rtype: cars dataset

    """
    # Zoom is using round, that lead to some bugs,
    # so we had to redefine the resolution
    coords_row = np.ceil(resolution * tab["im"].shape[0])
    coords_col = np.ceil(resolution * tab["im"].shape[1])
    upscaled_factor = (
        coords_row / tab.im.shape[0],
        coords_col / tab.im.shape[1],
    )

    # downsample
    upsampled_raster = zoom(tab[cst.EPI_IMAGE], upscaled_factor, order=1)

    # Construct the new dataset
    upsampled_dataset = xr.Dataset(
        {cst.EPI_IMAGE: ([cst.ROW, cst.COL], upsampled_raster)},
        coords={
            cst.ROW: np.arange(0, upsampled_raster.shape[0]),
            cst.COL: np.arange(0, upsampled_raster.shape[1]),
        },
        attrs=tab.attrs,
    )

    cars_dataset.fill_dataset(
        upsampled_dataset,
        window=None,
        profile=None,
        overlaps=None,
    )

    if cst.EPI_MSK in tab:
        upsampled_msk = zoom(tab[cst.EPI_MSK], upscaled_factor, order=0)
        upsampled_dataset["msk"] = (["row", "col"], upsampled_msk)

    if cst.EPI_COLOR in tab:
        upsampled_color = zoom(tab[cst.EPI_MSK], upscaled_factor, order=0)
        upsampled_dataset["color"] = (["row", "col"], upsampled_color)

    # Change useful attributes
    transform = tab.transform * tab.transform.scale(
        (tab.im.shape[0] / upsampled_raster.shape[0]),
        (tab.im.shape[1] / upsampled_raster.shape[1]),
    )
    upsampled_dataset.attrs["transform"] = transform

    # roi_with_margins
    roi_with_margins = np.empty(4)
    roi_with_margins[0] = np.ceil(tab.roi_with_margins[0] * upscaled_factor[1])
    roi_with_margins[1] = np.ceil(tab.roi_with_margins[1] * upscaled_factor[0])
    roi_with_margins[2] = np.ceil(tab.roi_with_margins[2] * upscaled_factor[1])
    roi_with_margins[3] = np.ceil(tab.roi_with_margins[3] * upscaled_factor[0])
    upsampled_dataset.attrs["roi_with_margins"] = roi_with_margins.astype(int)

    # roi
    roi = np.empty(4)
    roi[0] = np.ceil(tab.roi[0] * upscaled_factor[1])
    roi[1] = np.ceil(tab.roi[1] * upscaled_factor[0])
    roi[2] = np.ceil(tab.roi[2] * upscaled_factor[1])
    roi[3] = np.ceil(tab.roi[3] * upscaled_factor[0])
    upsampled_dataset.attrs["roi"] = roi.astype(int)

    # margins
    margins = np.empty(4)
    margins[0] = -(roi[0] - roi_with_margins[0])
    margins[1] = -(roi[1] - roi_with_margins[1])
    margins[2] = roi_with_margins[2] - roi[2]
    margins[3] = roi_with_margins[3] - roi[3]
    upsampled_dataset.attrs["margins"] = margins

    return upsampled_dataset, upscaled_factor


def clustering_pandora_matches(
    triangulated_matches,
    connection_val=3.0,
    nb_pts_threshold=80,
):
    """
    Filter triangulated  matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param connection_val: distance to use
        to consider that two points are connected
    :param nb_pts_threshold: number of points to use
        to identify small clusters to filter

    :return: filtered_matches
    :rtype: pandas Dataframe
    """

    filtered_pandora_matches, _ = (
        outlier_removal_tools.small_component_filtering(
            triangulated_matches,
            connection_val=connection_val,
            nb_pts_threshold=nb_pts_threshold,
        )
    )

    filtered_pandora_matches_dataframe = pandas.DataFrame(
        filtered_pandora_matches
    )
    filtered_pandora_matches_dataframe.attrs["epsg"] = (
        triangulated_matches.attrs["epsg"]
    )

    return filtered_pandora_matches_dataframe


def filter_point_cloud_matches(
    pd_cloud,
    matches_filter_knn=25,
    matches_filter_dev_factor=3,
):
    """
    Filter triangulated  matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param matches_filter_knn: number of neighboors used to measure
                               isolation of matches
    :type matches_filter_knn: int
    :param matches_filter_dev_factor: factor of deviation in the
                                      formula to compute threshold of outliers
    :type matches_filter_dev_factor: float

    :return: disp min and disp max
    :rtype: float, float
    """

    # Statistical filtering
    filter_cloud, _ = outlier_removal_tools.statistical_outlier_filtering(
        pd_cloud,
        k=matches_filter_knn,
        dev_factor=matches_filter_dev_factor,
    )

    # filter nans
    filter_cloud.dropna(axis=0, inplace=True)

    return filter_cloud


def pandora_matches(
    left_image_object,
    right_image_object,
    corr_conf,
    resolution,
    disp_to_alt_ratio=None,
):
    """
    Calculate the pandora matches

    :param left_image_object: the left image dataset
    :type left_image_object: cars dataset
    :param right_image_object: the right image dataset
    :type right_image_object: cars dataset
    :param corr_conf: the pandora configuration
    :type corr_conf: dict
    :param resolution: the resolution of the resampling
    :type resolution: int
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float

    :return: matches and disparity_map
    :rtype: datasets

    """

    # Downsample the epipolar images
    epipolar_image_left_low_res, new_resolution = downsample(
        left_image_object, 1 / resolution
    )
    epipolar_image_right_low_res, _ = downsample(
        right_image_object, 1 / resolution
    )

    # Calculate the disparity grid
    roi_left = epipolar_image_left_low_res.roi_with_margins[0]
    roi_top = epipolar_image_left_low_res.roi_with_margins[1]
    roi_right = epipolar_image_left_low_res.roi_with_margins[2]
    roi_bottom = epipolar_image_left_low_res.roi_with_margins[3]

    # dmin & dmax
    dmin = -1000 / resolution
    dmax = 9000 / resolution

    # Create CarsDataset
    disp_range_grid = cars_dataset.CarsDataset(
        "arrays", name="grid_disp_range_unknown_pair"
    )
    # Only one tile
    disp_range_grid.tiling_grid = np.array(
        [[[roi_top, roi_bottom, roi_left, roi_right]]]
    )

    row_range = np.arange(roi_top, roi_bottom)
    col_range = np.arange(roi_left, roi_right)

    grid_attributes = {
        "row_range": row_range,
        "col_range": col_range,
    }
    disp_range_grid.attributes = grid_attributes.copy()

    grid_min = np.empty((len(row_range), len(col_range)))
    grid_max = np.empty((len(row_range), len(col_range)))

    grid_min[:, :] = dmin
    grid_max[:, :] = dmax

    disp_range_tile = xr.Dataset(
        data_vars={
            dm_cst.DISP_MIN_GRID: (["row", "col"], grid_min),
            dm_cst.DISP_MAX_GRID: (["row", "col"], grid_max),
        },
        coords={
            "row": np.arange(0, grid_min.shape[0]),
            "col": np.arange(0, grid_min.shape[1]),
        },
    )

    disp_range_grid[0, 0] = disp_range_tile

    (
        disp_min_grid,
        disp_max_grid,
    ) = dm_tools.compute_disparity_grid(
        disp_range_grid, epipolar_image_left_low_res
    )

    global_disp_min = np.floor(
        np.nanmin(disp_range_grid[0, 0]["disp_min_grid"].data)
    )
    global_disp_max = np.ceil(
        np.nanmax(disp_range_grid[0, 0]["disp_max_grid"].data)
    )

    # add global disparity in case of ambiguity normalization
    epipolar_image_left_low_res = add_global_disparity(
        epipolar_image_left_low_res, global_disp_min, global_disp_max
    )

    # Compute the disparity map
    epipolar_disparity_map = dm_tools.compute_disparity(
        epipolar_image_left_low_res,
        epipolar_image_right_low_res,
        corr_conf,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        disp_to_alt_ratio=disp_to_alt_ratio,
    )

    # get values
    mask = epipolar_disparity_map["disp_msk"].values
    disp_map = epipolar_disparity_map["disp"].values
    disp_map[mask == 0] = np.nan

    # Construct the matches using the disparity map
    rows = np.arange(
        epipolar_image_left_low_res.roi[1], epipolar_image_left_low_res.roi[3]
    )
    cols = np.arange(
        epipolar_image_left_low_res.roi[0], epipolar_image_left_low_res.roi[2]
    )

    cols_mesh, rows_mesh = np.meshgrid(cols, rows)
    left_points = np.column_stack(
        (cols_mesh.ravel(), rows_mesh.ravel())
    ).astype(float)

    right_points = np.copy(left_points)

    right_points[:, 0] += disp_map.ravel()

    matches = np.column_stack((left_points, right_points))

    matches = matches[~np.isnan(matches).any(axis=1)]
    matches_true_res = np.empty((len(matches), 4))

    matches_true_res[:, 0] = matches[:, 0] * 1 / new_resolution[1]
    matches_true_res[:, 1] = matches[:, 1] * 1 / new_resolution[0]
    matches_true_res[:, 2] = matches[:, 2] * 1 / new_resolution[1]
    matches_true_res[:, 3] = matches[:, 3] * 1 / new_resolution[0]

    order = np.argsort(matches_true_res[:, 0])
    matches_true_res = matches_true_res[order, :]

    return matches_true_res, epipolar_disparity_map
