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
# pylint: disable=too-many-lines

"""
This module is responsible for the rasterization step:
- it contains all functions related to 3D representation on a 2D raster grid
TODO: refactor in several files and remove too-many-lines
"""


# Standard imports
import logging
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas

# cars-rasterize
import rasterize as crasterize  # pylint:disable=E0401
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.data_structures import cars_dataset


def compute_xy_starts_and_sizes(
    resolution: float, cloud: pandas.DataFrame
) -> Tuple[float, float, int, int]:
    """
    Compute xstart, ystart, xsize and ysize
    of the rasterization grid from a set of points

    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units
    :param cloud: set of points as returned
        by the create_combined_cloud function
    :return: a tuple (xstart, ystart, xsize, ysize)
    """

    # Derive xstart
    xmin = np.nanmin(cloud[cst.X].values)
    xmax = np.nanmax(cloud[cst.X].values)
    logging.debug("Points x coordinate range: [{},{}]".format(xmin, xmax))

    # Clamp to a regular grid
    x_start = np.floor(xmin / resolution) * resolution
    x_size = int(1 + np.floor((xmax - x_start) / resolution))

    # Derive ystart
    ymin = np.nanmin(cloud[cst.Y].values)
    ymax = np.nanmax(cloud[cst.Y].values)
    logging.debug("Points y coordinate range: [{},{}]".format(ymin, ymax))

    # Clamp to a regular grid
    y_start = np.ceil(ymax / resolution) * resolution
    y_size = int(1 + np.floor((y_start - ymin) / resolution))

    return x_start, y_start, x_size, y_size


def simple_rasterization_dataset_wrapper(
    cloud: pandas.DataFrame,
    resolution: float,
    epsg: int,
    xstart: float = None,
    ystart: float = None,
    xsize: int = None,
    ysize: int = None,
    sigma: float = None,
    radius: int = 1,
    dsm_no_data: int = np.nan,
    color_no_data: int = np.nan,
    msk_no_data: int = 255,
    list_computed_layers: List[str] = None,
    source_pc_names: List[str] = None,
) -> xr.Dataset:
    """
    Wrapper of simple_rasterization
    that has xarray.Dataset as inputs and outputs.

    :param cloud: cloud to rasterize
    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units or None
    :param epsg: epsg code for the CRS of the final raster
    :param color_list: Additional list of images
        with bands to rasterize (same size as cloud_list), or None
    :param xstart: xstart of the rasterization grid
        (if None, will be estimated by the function)
    :param ystart: ystart of the rasterization grid
        (if None, will be estimated by the function)
    :param xsize: xsize of the rasterization grid
        (if None, will be estimated by the function)
    :param ysize: ysize of the rasterization grid
        (if None, will be estimated by the function)
    :param sigma: sigma for gaussian interpolation.
        (If None, set to resolution)
    :param radius: Radius for hole filling.
    :param dsm_no_data: no data value to use in the final raster
    :param color_no_data: no data value to use in the final colored raster
    :param msk_no_data: no data value to use in the final mask image
    :param list_computed_layers: list of computed output data
    :param source_pc_names: list of names of points cloud before merging :
        name of sensors pair or name of point cloud file
    :return: Rasterized cloud
    """

    # combined clouds
    roi = (
        resolution is not None
        and xstart is not None
        and ystart is not None
        and xsize is not None
        and ysize is not None
    )

    # compute roi from the combined clouds if it is not set
    if not roi:
        (
            xstart,
            ystart,
            xsize,
            ysize,
        ) = compute_xy_starts_and_sizes(resolution, cloud)

    # rasterize clouds
    raster = rasterize(
        cloud,
        resolution,
        epsg,
        x_start=xstart,
        y_start=ystart,
        x_size=xsize,
        y_size=ysize,
        sigma=sigma,
        radius=radius,
        hgt_no_data=dsm_no_data,
        color_no_data=color_no_data,
        msk_no_data=msk_no_data,
        list_computed_layers=list_computed_layers,
        source_pc_names=source_pc_names,
    )

    return raster


def compute_values_1d(
    x_start: float, y_start: float, x_size: int, y_size: int, resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the x and y values as 1d arrays

    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells,
        in cloud CRS units or None.
    :return: a tuple composed of the x and y 1d arrays
    """
    x_values_1d = np.linspace(
        x_start + 0.5 * resolution,
        x_start + resolution * (x_size + 0.5),
        x_size,
        endpoint=False,
    )
    y_values_1d = np.linspace(
        y_start - 0.5 * resolution,
        y_start - resolution * (y_size + 0.5),
        y_size,
        endpoint=False,
    )

    return x_values_1d, y_values_1d


def substring_in_list(src_list, substring):
    """
    Check if the list contains substring
    """
    res = list(filter(lambda x: substring in x, src_list))
    return len(res) > 0


def find_indexes_in_point_cloud(
    cloud: pandas.DataFrame, tag: str, list_computed_layers: List[str] = None
) -> List[str]:
    """
    Find all indexes in point cloud that contains the key tag
    if it needs to be computed
    :param cloud: Combined cloud
    :param tag: substring of desired columns in cloud
    :param list_computed_layers: list of computed output data
    """
    indexes = []
    if list_computed_layers is None or substring_in_list(
        list_computed_layers, tag
    ):
        for key in cloud.columns:
            if tag in key:
                indexes.append(key)
    return indexes


def compute_vector_raster_and_stats(
    cloud: pandas.DataFrame,
    x_start: float,
    y_start: float,
    x_size: int,
    y_size: int,
    resolution: float,
    sigma: float,
    radius: int,
    list_computed_layers: List[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Union[None, np.ndarray, list, dict],
]:
    """
    Compute vectorized raster and its statistics.

    :param cloud: Combined cloud
        as returned by the create_combined_cloud function
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units or None.
    :param sigma: Sigma for gaussian interpolation. If None, set to resolution
    :param radius: Radius for hole filling.
    :param list_computed_layers: list of computed output data
    :return: a tuple with rasterization results and statistics.
    """
    # get points corresponding to (X, Y positions) + data_valid
    points = cloud.loc[:, [cst.X, cst.Y]].values.T
    nb_points = points.shape[1]
    valid = np.ones((1, nb_points))
    # create values: 1. altitudes and colors, 2. confidences, 3. masks
    # split_indexes allows to keep indexes separating values
    split_indexes = []

    # 1. altitudes and colors
    values_bands = [cst.Z] if cst.Z in cloud else []

    clr_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_CLR_KEY_ROOT
    )
    values_bands.extend(clr_indexes)
    split_indexes.append(len(values_bands))

    # 2. confidences
    confidences_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(confidences_indexes)
    split_indexes.append(len(confidences_indexes))

    # 3. confidence interval
    interval_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_INTERVALS_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(interval_indexes)
    split_indexes.append(len(interval_indexes))

    # 4. mask
    msk_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_MSK, list_computed_layers
    )
    values_bands.extend(msk_indexes)
    split_indexes.append(len(msk_indexes))

    # 5. classification
    classif_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_CLASSIF_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(classif_indexes)
    split_indexes.append(len(classif_indexes))

    # 6. source point cloud
    # Fill the dataframe with additional columns :
    # each column refers to a point cloud id
    number_of_pc = cars_dataset.get_attributes_dataframe(cloud)["number_of_pc"]
    if cst.POINTS_CLOUD_GLOBAL_ID in cloud.columns and (
        (list_computed_layers is None)
        or substring_in_list(
            list_computed_layers, cst.POINTS_CLOUD_SOURCE_KEY_ROOT
        )
    ):
        for pc_id in range(number_of_pc):
            # Create binary list that indicates from each point whether it comes
            # from point cloud number "pc_id"
            point_is_from_pc = list(
                map(int, cloud[cst.POINTS_CLOUD_GLOBAL_ID] == pc_id)
            )
            pc_key = "{}{}".format(cst.POINTS_CLOUD_SOURCE_KEY_ROOT, pc_id)
            cloud[pc_key] = point_is_from_pc

    source_pc_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_SOURCE_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(source_pc_indexes)
    split_indexes.append(len(source_pc_indexes))

    # 7. filling
    filling_indexes = find_indexes_in_point_cloud(
        cloud, cst.POINTS_CLOUD_FILLING_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(filling_indexes)

    values = (
        cloud.loc[:, values_bands].values.T
        if len(values_bands) > 0
        else np.empty((0, nb_points))
    )

    (out, weights_sum, mean, stdev, nb_pts_in_disc, nb_pts_in_cell) = (
        crasterize.pc_to_dsm(
            points,
            values,
            valid,
            x_start,
            y_start,
            x_size,
            y_size,
            resolution,
            radius,
            sigma,
        )
    )

    # pylint: disable=unbalanced-tuple-unpacking
    out, confidences, interval, msk, classif, source_pc, filling = np.split(
        out, np.cumsum(split_indexes), axis=-1
    )

    confidences_out = None
    if len(confidences_indexes) > 0:
        confidences_out = {}
        for k, key in enumerate(confidences_indexes):
            confidences_out[key] = confidences[..., k]

    interval_out = None
    interval_stat_index = None
    if len(interval_indexes) > 0:
        interval_out = interval
        interval_stat_index = [
            values_bands.index(int_ind) for int_ind in interval_indexes
        ]

    msk_out = None
    if len(msk_indexes) > 0:
        msk_out = np.ceil(msk)

    classif_out = None
    if len(classif_indexes) > 0:
        classif_out = np.ceil(classif)

    source_pc_out = None
    if len(source_pc_indexes) > 0:
        source_pc_out = np.ceil(source_pc)

    filling_out = None
    if len(filling_indexes) > 0:
        filling_out = np.ceil(filling)

    return (
        out,
        weights_sum,
        mean,
        stdev,
        nb_pts_in_disc,
        nb_pts_in_cell,
        msk_out,
        clr_indexes,
        classif_out,
        classif_indexes,
        confidences_out,
        interval_out,
        interval_stat_index,
        source_pc_out,
        filling_out,
        filling_indexes,
    )


def create_raster_dataset(
    raster: np.ndarray,
    weights_sum: np.ndarray,
    x_start: float,
    y_start: float,
    x_size: int,
    y_size: int,
    resolution: float,
    hgt_no_data: int,
    color_no_data: int,
    msk_no_data: int,
    epsg: int,
    mean: np.ndarray,
    stdev: np.ndarray,
    n_pts: np.ndarray,
    n_in_cell: np.ndarray,
    msk: np.ndarray = None,
    band_im: List[str] = None,
    classif: np.ndarray = None,
    band_classif: List[str] = None,
    confidences: np.ndarray = None,
    interval: np.ndarray = None,
    interval_stat_index: List[int] = None,
    source_pc: np.ndarray = None,
    source_pc_names: List[str] = None,
    filling: np.ndarray = None,
    band_filling: List[str] = None,
) -> xr.Dataset:
    """
    Create final raster xarray dataset

    :param raster: height and colors
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units or None.
    :param hgt_no_data: no data value to use for height
    :param color_no_data: no data value to use for color
    :param msk_no_data: no data value to use for mask and classif
    :param epsg: epsg code for the CRS of the final raster
    :param mean: mean of height and colors
    :param stdev: standard deviation of height and colors
    :param n_pts: number of points that are stricty in a cell
    :param n_in_cell: number of points which contribute to a cell
    :param msk: raster msk
    :param classif: raster classif
    :param confidences: raster containing the confidences
    :param interval: raster containing intervals inf and sup
    :param interval_stat_index: list containing index of
        intervals in mean and stdev rasters
    :param source_pc: binary raster with source point cloud information
    :param source_pc_names: list of names of points cloud before merging :
        name of sensors pair or name of point cloud file
    :return: the raster xarray dataset
    """
    raster_dims = (cst.Y, cst.X)
    n_layers = raster.shape[0]
    x_values_1d, y_values_1d = compute_values_1d(
        x_start, y_start, x_size, y_size, resolution
    )
    raster_coords = {cst.X: x_values_1d, cst.Y: y_values_1d}
    hgt = np.nan_to_num(raster[0], nan=hgt_no_data)
    raster_out = xr.Dataset(
        {
            cst.RASTER_HGT: ([cst.Y, cst.X], hgt),
            cst.RASTER_WEIGHTS_SUM: ([cst.Y, cst.X], weights_sum),
        },
        coords=raster_coords,
    )

    if raster.shape[0] > 1:  # rasterizer produced color output
        color = np.nan_to_num(raster[1:], nan=color_no_data)
        for idx, band_name in enumerate(band_im):
            band_im[idx] = band_name.replace(
                cst.POINTS_CLOUD_CLR_KEY_ROOT + "_", ""
            )
        color_out = xr.Dataset(
            {
                cst.RASTER_COLOR_IMG: (
                    [cst.BAND_IM, cst.Y, cst.X],
                    color,
                )
            },
            coords={**raster_coords, cst.BAND_IM: band_im},
        )
        # update raster output with classification data
        raster_out = xr.merge((raster_out, color_out))

    raster_out.attrs[cst.EPSG] = epsg
    raster_out.attrs[cst.RESOLUTION] = resolution

    # statistics layer for height output
    raster_out[cst.RASTER_HGT_MEAN] = xr.DataArray(
        mean[..., 0], coords=raster_coords, dims=raster_dims
    )
    raster_out[cst.RASTER_HGT_STD_DEV] = xr.DataArray(
        stdev[..., 0], coords=raster_coords, dims=raster_dims
    )

    # add each band statistics
    for i_layer in range(1, n_layers):
        raster_out["{}{}".format(cst.RASTER_BAND_MEAN, i_layer)] = xr.DataArray(
            mean[..., i_layer],
            coords=raster_coords,
            dims=raster_dims,
        )
        raster_out["{}{}".format(cst.RASTER_BAND_STD_DEV, i_layer)] = (
            xr.DataArray(
                stdev[..., i_layer],
                coords=raster_coords,
                dims=raster_dims,
            )
        )

    raster_out[cst.RASTER_NB_PTS] = xr.DataArray(n_pts, dims=raster_dims)
    raster_out[cst.RASTER_NB_PTS_IN_CELL] = xr.DataArray(
        n_in_cell, dims=raster_dims
    )

    if msk is not None:  # rasterizer produced mask output
        msk = np.nan_to_num(msk, nan=msk_no_data)
        raster_out[cst.RASTER_MSK] = xr.DataArray(msk, dims=raster_dims)

    if classif is not None:  # rasterizer produced classif output
        classif = np.nan_to_num(classif, nan=msk_no_data)
        for idx, band_name in enumerate(band_classif):
            band_classif[idx] = band_name.replace(
                cst.POINTS_CLOUD_CLASSIF_KEY_ROOT + "_", ""
            )
        classif_out = xr.Dataset(
            {
                cst.RASTER_CLASSIF: (
                    [cst.BAND_CLASSIF, cst.Y, cst.X],
                    classif,
                )
            },
            coords={**raster_coords, cst.BAND_CLASSIF: band_classif},
        )
        # update raster output with classification data
        raster_out = xr.merge((raster_out, classif_out))

    if confidences is not None:  # rasterizer produced color output
        for key in confidences:
            raster_out[key] = xr.DataArray(confidences[key], dims=raster_dims)

    if interval is not None:
        hgt_inf = np.nan_to_num(interval[0], nan=hgt_no_data)
        raster_out[cst.RASTER_HGT_INF] = xr.DataArray(
            hgt_inf, coords=raster_coords, dims=raster_dims
        )
        hgt_sup = np.nan_to_num(interval[1], nan=hgt_no_data)
        raster_out[cst.RASTER_HGT_SUP] = xr.DataArray(
            hgt_sup, coords=raster_coords, dims=raster_dims
        )
        hgt_inf_mean = np.nan_to_num(
            mean[..., interval_stat_index[0]], nan=hgt_no_data
        )
        raster_out[cst.RASTER_HGT_INF_MEAN] = xr.DataArray(
            hgt_inf_mean, coords=raster_coords, dims=raster_dims
        )
        hgt_sup_mean = np.nan_to_num(
            mean[..., interval_stat_index[1]], nan=hgt_no_data
        )
        raster_out[cst.RASTER_HGT_SUP_MEAN] = xr.DataArray(
            hgt_sup_mean, coords=raster_coords, dims=raster_dims
        )
        hgt_inf_stdev = np.nan_to_num(
            stdev[..., interval_stat_index[0]], nan=hgt_no_data
        )
        raster_out[cst.RASTER_HGT_INF_STD_DEV] = xr.DataArray(
            hgt_inf_stdev, coords=raster_coords, dims=raster_dims
        )
        hgt_sup_stdev = np.nan_to_num(
            stdev[..., interval_stat_index[1]], nan=hgt_no_data
        )
        raster_out[cst.RASTER_HGT_SUP_STD_DEV] = xr.DataArray(
            hgt_sup_stdev, coords=raster_coords, dims=raster_dims
        )

    if source_pc is not None and source_pc_names is not None:
        source_pc = np.nan_to_num(source_pc, nan=msk_no_data)
        source_pc_out = xr.Dataset(
            {
                cst.RASTER_SOURCE_PC: (
                    [cst.BAND_SOURCE_PC, cst.Y, cst.X],
                    source_pc,
                )
            },
            coords={**raster_coords, cst.BAND_SOURCE_PC: source_pc_names},
        )
        # update raster output with classification data
        raster_out = xr.merge((raster_out, source_pc_out))

    if filling is not None:  # rasterizer produced filling info output
        filling = np.nan_to_num(filling, nan=msk_no_data)
        for idx, band_name in enumerate(band_filling):
            band_filling[idx] = band_name.replace(
                cst.POINTS_CLOUD_FILLING_KEY_ROOT + "_", ""
            )
        filling_out = xr.Dataset(
            {
                cst.RASTER_FILLING: (
                    [cst.BAND_FILLING, cst.Y, cst.X],
                    filling,
                )
            },
            coords={**raster_coords, cst.BAND_FILLING: band_filling},
        )
        # update raster output with filling information
        raster_out = xr.merge((raster_out, filling_out))

    return raster_out


def rasterize(
    cloud: pandas.DataFrame,
    resolution: float,
    epsg: int,
    x_start: float,
    y_start: float,
    x_size: int,
    y_size: int,
    sigma: float = None,
    radius: int = 1,
    hgt_no_data: int = -32768,
    color_no_data: int = 0,
    msk_no_data: int = 255,
    list_computed_layers: List[str] = None,
    source_pc_names: List[str] = None,
) -> Union[xr.Dataset, None]:
    """
    Rasterize a point cloud with its color bands to a Dataset
    that also contains quality statistics.

    :param cloud: Combined cloud
        as returned by the create_combined_cloud function
    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units or None.
    :param epsg: epsg code for the CRS of the final raster
    :param x_start: x start of the rasterization grid
    :param y_start: y start of the rasterization grid
    :param x_size: x size of the rasterization grid
    :param y_size: y size of the rasterization grid
    :param sigma: sigma for gaussian interpolation. If None, set to resolution
    :param radius: Radius for hole filling.
    :param hgt_no_data: no data value to use for height
    :param color_no_data: no data value to use for color
    :param msk_no_data: no data value to use in the final mask image
    :param list_computed_layers: list of computed output data
    :return: Rasterized cloud color and statistics.
    """

    if sigma is None:
        sigma = resolution

    # If no valid points are found in cloud, return default values
    if cloud.size == 0:
        logging.debug("No points to rasterize, returning None")
        return None

    logging.debug(
        "Rasterization grid: start=[{},{}], size=[{},{}], resolution={}".format(
            x_start, y_start, x_size, y_size, resolution
        )
    )

    (
        out,
        weights_sum,
        mean,
        stdev,
        n_pts,
        n_in_cell,
        msk,
        clr_indexes,
        classif,
        classif_indexes,
        confidences,
        interval,
        interval_stat_index,
        source_pc,
        filling,
        filling_indexes,
    ) = compute_vector_raster_and_stats(
        cloud,
        x_start,
        y_start,
        x_size,
        y_size,
        resolution,
        sigma,
        radius,
        list_computed_layers,
    )

    # reshape data as a 2d grid.
    shape_out = (y_size, x_size)
    out = out.reshape(shape_out + (-1,))
    mean = mean.reshape(shape_out + (-1,))
    stdev = stdev.reshape(shape_out + (-1,))
    n_pts = n_pts.reshape(shape_out)
    n_in_cell = n_in_cell.reshape(shape_out)

    out = out.reshape(shape_out + (-1,))
    out = np.moveaxis(out, 2, 0)

    weights_sum = weights_sum.reshape(shape_out)

    if classif is not None:
        classif = classif.reshape(shape_out + (-1,))
        classif = np.moveaxis(classif, 2, 0)

    if msk is not None:
        msk = msk.reshape(shape_out)
    else:
        msk = np.isnan(out[0, :, :])

    if confidences is not None:
        for key in confidences:
            confidences[key] = confidences[key].reshape(shape_out)

    if interval is not None:
        interval = interval.reshape(shape_out + (-1,))
        interval = np.moveaxis(interval, 2, 0)

    if source_pc is not None:
        source_pc = source_pc.reshape(shape_out + (-1,))
        source_pc = np.moveaxis(source_pc, 2, 0)

    if filling is not None:
        filling = filling.reshape(shape_out + (-1,))
        filling = np.moveaxis(filling, 2, 0)

    # build output dataset
    raster_out = create_raster_dataset(
        out,
        weights_sum,
        x_start,
        y_start,
        x_size,
        y_size,
        resolution,
        hgt_no_data,
        color_no_data,
        msk_no_data,
        epsg,
        mean,
        stdev,
        n_pts,
        n_in_cell,
        msk,
        clr_indexes,
        classif,
        classif_indexes,
        confidences,
        interval,
        interval_stat_index,
        source_pc,
        source_pc_names,
        filling,
        filling_indexes,
    )

    return raster_out


def update_weights(old_weights, weights):
    """
    Update weights

    :param weights: current weights
    :param old_weights: old weights

    :return: updated weights
    """

    new_weights = weights
    if old_weights is not None:
        current_nan = weights == 0
        old_nan = old_weights == 0
        weights[current_nan] = 0
        old_weights[old_nan] = 0
        new_weights = old_weights + weights

    return new_weights


def update_data(
    old_data, current_data, weights, old_weights, nodata, method="basic"
):
    """
    Update current data with old data and weigths

    :param old_data: old data
    :param current_data: current data
    :param weights: current weights
    :param old_weights: old weights
    :param nodata: nodata associated to tag

    :return: updated current data
    """
    new_data = current_data
    if old_data is not None:
        old_data = np.squeeze(old_data)
        old_weights = np.squeeze(old_weights)
        shape = old_data.shape
        if len(old_data.shape) == 3:
            old_weights = np.repeat(
                np.expand_dims(old_weights, axis=0), old_data.shape[0], axis=0
            )

        current_data = np.squeeze(current_data)
        weights = np.squeeze(weights)
        if len(current_data.shape) == 3:
            weights = np.repeat(
                np.expand_dims(weights, axis=0), current_data.shape[0], axis=0
            )

        # compute masks
        current_valid = weights != 0
        old_valid = old_weights != 0

        both_valid = np.logical_and(current_valid, old_valid)
        total_weight = np.zeros(shape)
        total_weight[both_valid] = weights[both_valid] + old_weights[both_valid]

        # current factor
        current_factor = np.zeros(shape)
        current_factor[current_valid] = 1
        current_factor[both_valid] = (
            weights[both_valid] / total_weight[both_valid]
        )

        # old factor
        old_factor = np.zeros(shape)
        old_factor[old_valid] = 1
        old_factor[both_valid] = (
            old_weights[both_valid] / total_weight[both_valid]
        )

        # assign old weights
        new_data = np.zeros(shape)
        if method == "basic":
            new_data[old_valid] = old_data[old_valid] * old_factor[old_valid]
            new_data[current_valid] += (
                current_data[current_valid] * current_factor[current_valid]
            )
        elif method == "bool":
            new_data[old_valid] = old_data[old_valid]
            new_data[current_valid] = np.logical_or(
                current_data[current_valid], new_data[current_valid]
            )
        elif method == "sum":
            new_data[old_valid] = old_data[old_valid]
            new_data[current_valid] += current_data[current_valid]

        # set nodata
        all_nodata = (current_valid + old_valid) == 0
        new_data[all_nodata] = nodata
    return new_data
