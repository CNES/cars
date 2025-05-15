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
from typing import List, Tuple

# Third party imports
import numpy as np
import pandas

# cars-rasterize
import xarray as xr

# CARS imports
from cars.core import constants as cst


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

    # Derive ystart
    ymin = np.nanmin(cloud[cst.Y].values)
    ymax = np.nanmax(cloud[cst.Y].values)
    logging.debug("Points y coordinate range: [{},{}]".format(ymin, ymax))

    # Clamp to a regular grid
    y_start = np.ceil(ymax / resolution) * resolution

    x_size = int(1 + np.floor((xmax - x_start) / resolution))
    y_size = int(1 + np.floor((y_start - ymin) / resolution))

    return x_start, y_start, x_size, y_size


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


def phased_dsm(start: float, phase: float, resolution: float):
    """
    Phased the dsm

    :param start: start of the roi
    :param phase: the point for phasing
    :param resolution: resolution of the dsm
    """

    div = np.abs(start - phase) / resolution

    if phase > start:
        start = phase - resolution * np.floor(div)
    else:
        start = resolution * np.floor(div) + phase

    return start


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


def create_raster_dataset(  # noqa: C901
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
    layers_inf_sup: np.ndarray = None,
    layers_inf_sup_stat_index: List[int] = None,
    layer_inf_sup_indexes: List[str] = None,
    source_pc: np.ndarray = None,
    source_pc_names: List[str] = None,
    filling: np.ndarray = None,
    band_filling: List[str] = None,
    performance_map: np.ndarray = None,
    performance_map_classified: np.ndarray = None,
    performance_map_classified_index: list = None,
    band_performance_map: List[str] = None,
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
    :param layers_inf_sup: raster containing intervals inf and sup
    :param layers_inf_sup_stat_index: list containing index of
        intervals in mean and stdev rasters
    :param layer_inf_sup_indexes: list of band names
    :param source_pc: binary raster with source point cloud information
    :param source_pc_names: list of names of point cloud before merging :
        name of sensors pair or name of point cloud file
    :param performance_map: raster containing the raw performance map
    :param performance_map_classified: raster containing the classified
        performance map
    :param performance_map_classified_index: indexes of
        performance_map_classified
    :param band_performance_map: list of band names :
        max 2 bands: risk / interval
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
                cst.POINT_CLOUD_CLR_KEY_ROOT + "_", ""
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
                cst.POINT_CLOUD_CLASSIF_KEY_ROOT + "_", ""
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

    if layers_inf_sup is not None:
        # Get inf data
        hgt_layers_list_inf = []
        hgt_layers_list_sup = []
        hgt_mean_layers_list_inf = []
        hgt_mean_layers_list_sup = []
        hgt_stdev_layers_list_inf = []
        hgt_stdev_layers_list_sup = []
        bands_sup = []
        bands_inf = []
        # Get Data
        for current_layer, _ in enumerate(layers_inf_sup):
            if "inf" in layer_inf_sup_indexes[current_layer]:
                hgt_layers_list_inf.append(layers_inf_sup[current_layer])
                hgt_mean_layers_list_inf.append(
                    mean[..., layers_inf_sup_stat_index[current_layer]]
                )
                hgt_stdev_layers_list_inf.append(
                    stdev[..., layers_inf_sup_stat_index[current_layer]]
                )
                bands_inf.append(layer_inf_sup_indexes[current_layer])
            else:
                hgt_layers_list_sup.append(layers_inf_sup[current_layer])
                hgt_mean_layers_list_sup.append(
                    mean[..., layers_inf_sup_stat_index[current_layer]]
                )
                hgt_stdev_layers_list_sup.append(
                    stdev[..., layers_inf_sup_stat_index[current_layer]]
                )
                bands_sup.append(layer_inf_sup_indexes[current_layer])

        for (
            data_layer_list,
            dataset_key,
            band_key,
            bands_name,
        ) in zip(  # noqa: B905
            [
                hgt_layers_list_inf,
                hgt_mean_layers_list_inf,
                hgt_stdev_layers_list_inf,
                hgt_layers_list_sup,
                hgt_mean_layers_list_sup,
                hgt_stdev_layers_list_sup,
            ],
            [
                cst.RASTER_HGT_INF,
                cst.RASTER_HGT_INF_MEAN,
                cst.RASTER_HGT_INF_STD_DEV,
                cst.RASTER_HGT_SUP,
                cst.RASTER_HGT_SUP_MEAN,
                cst.RASTER_HGT_SUP_STD_DEV,
            ],
            [
                cst.BAND_LAYER_INF,
                cst.BAND_LAYER_INF,
                cst.BAND_LAYER_INF,
                cst.BAND_LAYER_SUP,
                cst.BAND_LAYER_SUP,
                cst.BAND_LAYER_SUP,
            ],
            [bands_inf, bands_inf, bands_inf, bands_sup, bands_sup, bands_sup],
        ):
            # Stack data
            data_layer = np.nan_to_num(
                np.stack(data_layer_list, axis=0), nan=hgt_no_data
            )
            # Add to datasets
            layer_out = xr.Dataset(
                {
                    dataset_key: (
                        [band_key, cst.Y, cst.X],
                        data_layer,
                    )
                },
                coords={**raster_coords, band_key: bands_name},
            )
            # update raster output with filling information
            raster_out = xr.merge((raster_out, layer_out))

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
                cst.POINT_CLOUD_FILLING_KEY_ROOT + "_", ""
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

    if performance_map is not None:
        performance_map = np.nan_to_num(performance_map, nan=msk_no_data)
        if len(performance_map.shape) == 3 and performance_map.shape[0] == 2:
            # Has both performance from risk and intervals
            perf_out = xr.Dataset(
                {
                    cst.RASTER_PERFORMANCE_MAP_RAW: (
                        [cst.BAND_PERFORMANCE_MAP, cst.Y, cst.X],
                        performance_map,
                    )
                },
                coords={
                    **raster_coords,
                    cst.BAND_PERFORMANCE_MAP: band_performance_map,
                },
            )
            # update raster output with performance information
            raster_out = xr.merge((raster_out, perf_out))

        else:
            # Only one performance map
            raster_out[cst.RASTER_PERFORMANCE_MAP_RAW] = xr.DataArray(
                performance_map[0, :, :], dims=raster_dims
            )
    if performance_map_classified is not None:
        if (
            len(performance_map_classified.shape) == 3
            and performance_map_classified.shape[0] == 2
        ):
            # Has both performance from risk and intervals
            perf_classified_out = xr.Dataset(
                {
                    cst.RASTER_PERFORMANCE_MAP: (
                        [cst.BAND_PERFORMANCE_MAP, cst.Y, cst.X],
                        performance_map_classified,
                    )
                },
                coords={
                    **raster_coords,
                    cst.BAND_PERFORMANCE_MAP: band_performance_map,
                },
            )
            # update raster output with performance information
            raster_out = xr.merge((raster_out, perf_classified_out))
        else:
            # Only one performance map
            raster_out[cst.RASTER_PERFORMANCE_MAP] = xr.DataArray(
                performance_map_classified[0, :, :], dims=raster_dims
            )
        raster_out.attrs[cst.RIO_TAG_PERFORMANCE_MAP_CLASSES] = (
            performance_map_classified_index
        )

    return raster_out


def classify_performance_map(
    performance_map_raw, performance_map_classes, msk_no_data
):
    """
    Classify performance map with given classes
    """
    if performance_map_classes[0] != 0:
        performance_map_classes = [0] + performance_map_classes
    if performance_map_classes[-1] != np.inf:
        performance_map_classes.append(np.inf)

    performance_map_classified_infos = {}

    performance_map_classified = msk_no_data * np.ones(
        performance_map_raw.shape, dtype=np.uint8
    )

    index_start, index_end = 0, 1
    value = 0
    while index_end < len(performance_map_classes):
        current_class = (
            performance_map_classes[index_start],
            performance_map_classes[index_end],
        )

        # update information
        performance_map_classified_infos[value] = current_class

        # create classified performance map
        performance_map_classified[
            np.logical_and(
                performance_map_raw >= current_class[0],
                performance_map_raw < current_class[1],
            )
        ] = value

        # next class
        index_start += 1
        index_end += 1
        value += 1

    return performance_map_classified, performance_map_classified_infos


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
    data = old_data
    if old_data is not None:
        old_data = np.squeeze(old_data)
        old_weights = np.squeeze(old_weights)
        shape = old_data.shape
        if len(data.shape) == 3 and data.shape[0] > 1:
            old_weights = np.repeat(
                np.expand_dims(old_weights, axis=0), old_data.shape[0], axis=0
            )

        current_data = np.squeeze(current_data)
        weights = np.squeeze(weights)
        if len(new_data.shape) == 3 and new_data.shape[0] > 1:
            weights = np.repeat(
                np.expand_dims(weights, axis=0), current_data.shape[0], axis=0
            )

        # compute masks
        current_valid = weights != 0
        old_valid = old_weights != 0

        both_valid = np.logical_and(current_valid, old_valid)

        total_weights = np.zeros(shape)

        total_weights[both_valid] = (
            weights[both_valid] + old_weights[both_valid]
        )

        # current factor
        current_factor = np.zeros(shape)
        current_factor[current_valid] = 1
        current_factor[both_valid] = (
            weights[both_valid] / total_weights[both_valid]
        )

        # old factor
        old_factor = np.zeros(shape)
        old_factor[old_valid] = 1
        old_factor[both_valid] = (
            old_weights[both_valid] / total_weights[both_valid]
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

        # round result if saved as integer
        if np.issubdtype(current_data.dtype, np.integer):
            new_data = np.round(new_data).astype(current_data.dtype)

        # set nodata
        all_nodata = (current_valid + old_valid) == 0
        new_data[all_nodata] = nodata
    return new_data
