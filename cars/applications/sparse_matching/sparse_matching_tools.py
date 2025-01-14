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
Sparse matching Sift module:
contains sift sparse matching method
"""

# Standard imports
from __future__ import absolute_import

import logging

# Third party imports
import numpy as np
import pandas
import xarray as xr
from scipy.ndimage import zoom
from vlsift.sift.sift import sift

import cars.applications.dense_matching.dense_matching_constants as dm_cst

# CARS imports
import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars.applications import application_constants
from cars.applications.dense_matching import dense_matching_tools as dm_tools
from cars.applications.point_cloud_outlier_removal import outlier_removal_tools
from cars.core import constants as cst
from cars.data_structures import cars_dataset


def euclidean_matrix_distance(descr1: np.array, descr2: np.array):
    """Compute a matrix containing cross euclidean distance
    :param descr1: first keypoints descriptor
    :type descr1: numpy.ndarray
    :param descr2: second keypoints descriptor
    :type descr2: numpy.ndarray
    :return euclidean matrix distance
    :rtype: float
    """
    sq_descr1 = np.sum(descr1**2, axis=1)[:, np.newaxis]
    sq_descr2 = np.sum(descr2**2, axis=1)
    dot_descr12 = np.dot(descr1, descr2.T)
    return np.sqrt(sq_descr1 + sq_descr2 - 2 * dot_descr12)


def compute_matches(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray = None,
    right_mask: np.ndarray = None,
    left_origin: [float, float] = None,
    right_origin: [float, float] = None,
    matching_threshold: float = 0.7,
    n_octave: int = 8,
    n_scale_per_octave: int = 3,
    peak_threshold: float = 4.0,
    edge_threshold: float = 10.0,
    magnification: float = 7.0,
    window_size: int = 2,
    backmatching: bool = True,
    disp_lower_bound=None,
    disp_upper_bound=None,
):
    """
    Compute matches between left and right
    Convention for masks: True is a valid pixel

    :param left: left image as numpy array
    :type left: np.ndarray
    :param right: right image as numpy array
    :type right: np.ndarray
    :param left_mask: left mask as numpy array
    :type left_mask: np.ndarray
    :param right_mask: right mask as numpy array
    :type right_mask: np.ndarray
    :param left_origin: left image origin in the full image
    :type left_origin: [float, float]
    :param right_origin: right image origin in the full image
    :type right_origin: [float, float]
    :param matching_threshold: threshold for the ratio to nearest second match
    :type matching_threshold: float
    :param n_octave: the number of octaves of the DoG scale space
    :type n_octave: int
    :param n_scale_per_octave: the nb of levels / octave of the DoG scale space
    :type n_scale_per_octave: int
    :param peak_threshold: the peak selection threshold
    :type peak_threshold: float
    :param edge_threshold: the edge selection threshold
    :type edge_threshold: float
    :param magnification: set the descriptor magnification factor
    :type magnification: float
    :param window_size: size of the window
    :type window_size: int
    :param backmatching: also check that right vs. left gives same match
    :type backmatching: bool

    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)

    """
    left_origin = [0, 0] if left_origin is None else left_origin
    right_origin = [0, 0] if right_origin is None else right_origin

    # compute keypoints + descriptors
    left_frames, left_descr = sift(
        left,
        n_octaves=n_octave,
        n_levels=n_scale_per_octave,
        first_octave=-1,
        peak_thresh=peak_threshold,
        edge_thresh=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        float_descriptors=True,
        compute_descriptor=True,
        verbose=False,
    )

    right_frames, right_descr = sift(
        right,
        n_octaves=n_octave,
        n_levels=n_scale_per_octave,
        first_octave=-1,
        peak_thresh=peak_threshold,
        edge_thresh=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        float_descriptors=True,
        compute_descriptor=True,
        verbose=False,
    )

    # Filter keypoints that falls out of the validity mask (0=valid)
    if left_mask is not None:
        pixel_indices = np.floor(left_frames[:, 0:2]).astype(int)
        valid_left_frames_mask = left_mask[
            pixel_indices[:, 0], pixel_indices[:, 1]
        ]
        left_frames = left_frames[valid_left_frames_mask]
        left_descr = left_descr[valid_left_frames_mask]

    if right_mask is not None:
        pixel_indices = np.floor(right_frames[:, 0:2]).astype(int)
        valid_right_frames_mask = right_mask[
            pixel_indices[:, 0], pixel_indices[:, 1]
        ]
        right_frames = right_frames[valid_right_frames_mask]
        right_descr = right_descr[valid_right_frames_mask]

    # Early return for empty frames
    # also if there are points to match
    # need minimum two right points to find the second nearest neighbor
    # (and two left points for backmatching)
    if left_frames.shape[0] < 2 or right_frames.shape[0] < 2:
        return np.empty((0, 4))

    # translate matches according image origin
    # revert origin due to frame convention: [Y, X, S, TH] X: 1, Y: 0)
    left_frames[:, 0:2] += left_origin[::-1]
    right_frames[:, 0:2] += right_origin[::-1]

    # sort frames (and descriptors) along X axis
    order = np.argsort(left_frames[:, 1])
    left_frames = left_frames[order]
    left_descr = left_descr[order]

    order = np.argsort(right_frames[:, 1])
    right_frames = right_frames[order]
    right_descr = right_descr[order]

    # compute best matches by blocks
    splits = np.arange(500, len(left_frames), 500)
    left_frames_splitted = np.split(left_frames, splits)
    left_descr_splitted = np.split(left_descr, splits)
    splits = np.insert(splits, 0, 0)

    matches_id = []

    for (
        left_id_offset,
        left_frames_block,
        left_descr_block,
    ) in zip(  # noqa: B905
        splits, left_frames_splitted, left_descr_splitted
    ):
        if disp_lower_bound is not None and disp_upper_bound is not None:
            # Find right block extremas
            right_x_min = np.min(left_frames_block[:, 1]) + disp_lower_bound
            right_x_max = np.max(left_frames_block[:, 1]) + disp_upper_bound
            if (
                np.max(right_frames[:, 1]) > right_x_min
                and np.min(right_frames[:, 1]) < right_x_max
            ):
                left_id = np.min(np.where(right_frames[:, 1] > right_x_min))
                right_id = np.max(np.where(right_frames[:, 1] < right_x_max))
                right_descr_block = right_descr[left_id:right_id]
                right_id_offset = left_id
            else:
                right_descr_block = []
        else:
            right_descr_block = right_descr
            right_id_offset = 0

        if len(left_descr_block) >= 2 and len(right_descr_block) >= 2:
            # compute euclidean matrix distance
            emd = euclidean_matrix_distance(left_descr_block, right_descr_block)

            # get nearest sift (regarding descriptors)
            id_nearest_dlr = (
                np.arange(np.shape(emd)[0]),
                np.nanargmin(emd, axis=1),
            )
            id_nearest_drl = (
                np.nanargmin(emd, axis=0),
                np.arange(np.shape(emd)[1]),
            )

            # get absolute distances
            dist_dlr = emd[id_nearest_dlr]
            dist_drl = emd[id_nearest_drl]

            # get relative distance (ratio to second nearest distance)
            second_dist_dlr = np.partition(emd, 1, axis=1)[:, 1]
            dist_dlr /= second_dist_dlr
            second_dist_drl = np.partition(emd, 1, axis=0)[1, :]
            dist_drl /= second_dist_drl

            # stack matches which its distance
            id_matches_dlr = np.column_stack((*id_nearest_dlr, dist_dlr))
            id_matches_drl = np.column_stack((*id_nearest_drl, dist_drl))

            # check backmatching
            if backmatching is True:
                back = (
                    id_matches_dlr[:, 0]
                    == id_matches_drl[id_matches_dlr[:, 1].astype(int)][:, 0]
                )
                id_matches_dlr = id_matches_dlr[back]

            # threshold matches
            id_matches_dlr = id_matches_dlr[
                id_matches_dlr[:, -1] < matching_threshold, :
            ][:, :-1]

            id_matches_dlr += (left_id_offset, right_id_offset)

            matches_id.append(id_matches_dlr)

    if matches_id:
        matches_id = np.concatenate(matches_id)
    else:
        matches_id = np.empty((0, 4))

    # retrieve points: [Y, X, S, TH] X: 1, Y: 0
    # fyi: ``S`` is the scale and ``TH`` is the orientation (in radians)
    left_points = left_frames[matches_id[:, 0].astype(int), 1::-1]
    right_points = right_frames[matches_id[:, 1].astype(int), 1::-1]
    matches = np.concatenate((left_points, right_points), axis=1)
    return matches


def dataset_matching(
    ds1,
    ds2,
    matching_threshold=0.7,
    n_octave=8,
    n_scale_per_octave=3,
    peak_threshold=4.0,
    edge_threshold=10.0,
    magnification=7.0,
    window_size=2,
    backmatching=True,
    disp_lower_bound=None,
    disp_upper_bound=None,
):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param matching_threshold: threshold for the ratio to nearest second match
    :type matching_threshold: float
    :param n_octave: the number of octaves of the DoG scale space
    :type n_octave: int
    :param n_scale_per_octave: the nb of levels / octave of the DoG scale space
    :type n_scale_per_octave: int
    :param peak_threshold: the peak selection threshold
    :type peak_threshold: int
    :param edge_threshold: the edge selection threshold.
    :param magnification: set the descriptor magnification factor
    :type magnification: float
    :param window_size: size of the window
    :type window_size: int
    :param backmatching: also check that right vs. left gives same match
    :type backmatching: bool

    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    # get input data from dataset
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]

    left = ds1.im.values
    right = ds2.im.values
    left_mask = ds1.msk.values == 0
    right_mask = ds2.msk.values == 0

    matches = compute_matches(
        left,
        right,
        left_mask=left_mask,
        right_mask=right_mask,
        left_origin=origin1,
        right_origin=origin2,
        matching_threshold=matching_threshold,
        n_octave=n_octave,
        n_scale_per_octave=n_scale_per_octave,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        magnification=magnification,
        window_size=window_size,
        backmatching=backmatching,
        disp_lower_bound=disp_lower_bound,
        disp_upper_bound=disp_upper_bound,
    )

    return matches


def remove_epipolar_outliers(matches, percent=0.1):
    # TODO used only in test functions to test compute_disparity_range
    # Refactor with sparse_matching
    """
    This function will filter the match vector
    according to a quantile of epipolar error
    used for testing compute_disparity_range sparse method

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema
    :type percent: float
    :return: the filtered match array
    :rtype: numpy array
    """
    epipolar_error_min = np.percentile(matches[:, 1] - matches[:, 3], percent)
    epipolar_error_max = np.percentile(
        matches[:, 1] - matches[:, 3], 100 - percent
    )
    logging.info(
        "Epipolar error range after outlier rejection: [{},{}]".format(
            epipolar_error_min, epipolar_error_max
        )
    )
    out = matches[(matches[:, 1] - matches[:, 3]) < epipolar_error_max]
    out = out[(out[:, 1] - out[:, 3]) > epipolar_error_min]

    return out


def compute_disparity_range(matches, percent=0.1):
    # TODO: Refactor with dense_matching to have only one API ?
    """
    This function will compute the disparity range
    from matches by filtering percent outliers

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema (in %)
    :type percent: float
    :return: the disparity range
    :rtype: float, float
    """
    disparity = matches[:, 2] - matches[:, 0]

    mindisp = np.percentile(disparity, percent)
    maxdisp = np.percentile(disparity, 100 - percent)

    return mindisp, maxdisp


def compute_disp_min_disp_max(
    pd_cloud,
    orchestrator,
    disp_margin=0.1,
    pair_key=None,
    disp_to_alt_ratio=None,
):
    """
    Compute disp min and disp max from triangulated and filtered matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param orchestrator: orchestrator used
    :type orchestrator: Orchestrator
    :param disp_margin: disparity margin
    :type disp_margin: float
    :param disp_to_alt_ratio: used for logging info
    :type disp_to_alt_ratio: float

    :return: disp min and disp max
    :rtype: float, float
    """

    # Obtain dmin dmax
    filt_disparity = np.array(pd_cloud.iloc[:, 3])
    dmax = np.nanmax(filt_disparity)
    dmin = np.nanmin(filt_disparity)

    margin = abs(dmax - dmin) * disp_margin
    dmin -= margin
    dmax += margin

    logging.info(
        "Disparity range with margin: [{:.3f} pix., {:.3f} pix.] "
        "(margin = {:.3f} pix.)".format(dmin, dmax, margin)
    )

    if disp_to_alt_ratio is not None:
        logging.info(
            "Equivalent range in meters: [{:.3f} m, {:.3f} m] "
            "(margin = {:.3f} m)".format(
                dmin * disp_to_alt_ratio,
                dmax * disp_to_alt_ratio,
                margin * disp_to_alt_ratio,
            )
        )

    # update orchestrator_out_json
    updating_infos = {
        application_constants.APPLICATION_TAG: {
            sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                pair_key: {
                    sm_cst.DISPARITY_MARGIN_PARAM_TAG: disp_margin,
                    sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                    sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                }
            }
        }
    }
    orchestrator.update_out_info(updating_infos)

    return dmin, dmax


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


def clustering_matches(
    triangulated_matches,
    connection_val=3.0,
    nb_pts_threshold=80,
    clusters_distance_threshold: float = None,
    filtered_elt_pos: bool = False,
):
    """
    Filter triangulated  matches

    :param pd_cloud: triangulated_matches
    :type pd_cloud: pandas Dataframe
    :param connection_val: distance to use
        to consider that two points are connected
    :param nb_pts_threshold: number of points to use
        to identify small clusters to filter
    :param clusters_distance_threshold: distance to use
        to consider if two points clusters are far from each other or not
        (set to None to deactivate this level of filtering)
    :param filtered_elt_pos: if filtered_elt_pos is set to True,
        the removed points positions in their original
        epipolar images are returned, otherwise it is set to None

    :return: filtered_matches
    :rtype: pandas Dataframe

    """

    filtered_pandora_matches, _ = (
        outlier_removal_tools.small_component_filtering(
            triangulated_matches,
            connection_val=connection_val,
            nb_pts_threshold=nb_pts_threshold,
            clusters_distance_threshold=clusters_distance_threshold,
            filtered_elt_pos=filtered_elt_pos,
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
    disp_upper_bound,
    disp_lower_bound,
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
    dmin = disp_lower_bound / resolution
    dmax = disp_upper_bound / resolution

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
