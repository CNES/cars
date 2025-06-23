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

import numpy as np
import pandas

# cars-rasterize
import rasterize as crasterize  # pylint:disable=E0401
import xarray as xr

from cars.applications.rasterization import rasterization_wrappers as rast_wrap
from cars.core import constants as cst

# CARS imports
from cars.data_structures import cars_dataset


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
    texture_no_data: int = np.nan,
    msk_no_data: int = 255,
    list_computed_layers: List[str] = None,
    source_pc_names: List[str] = None,
    performance_map_classes: List[float] = None,
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
    :param texture_no_data: no data value to use in the final colored raster
    :param msk_no_data: no data value to use in the final mask image
    :param list_computed_layers: list of computed output data
    :param source_pc_names: list of names of point cloud before merging :
        name of sensors pair or name of point cloud file
    :param performance_map_classes: list for step defining border of class
    :type performance_map_classes: list or None
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
        ) = rast_wrap.compute_xy_starts_and_sizes(resolution, cloud)

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
        texture_no_data=texture_no_data,
        msk_no_data=msk_no_data,
        list_computed_layers=list_computed_layers,
        source_pc_names=source_pc_names,
        performance_map_classes=performance_map_classes,
    )

    return raster


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

    clr_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_CLR_KEY_ROOT
    )
    values_bands.extend(clr_indexes)
    split_indexes.append(len(values_bands))

    # 2. confidences
    if list_computed_layers is not None:
        if cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT not in list_computed_layers:
            confidences_indexes = rast_wrap.find_indexes_in_point_cloud(
                cloud, cst.POINT_CLOUD_AMBIGUITY_KEY_ROOT, list_computed_layers
            )
        else:
            confidences_indexes = rast_wrap.find_indexes_in_point_cloud(
                cloud, cst.POINT_CLOUD_CONFIDENCE_KEY_ROOT, list_computed_layers
            )
    else:
        confidences_indexes = []

    values_bands.extend(confidences_indexes)
    split_indexes.append(len(confidences_indexes))

    # 3. sup and inf layers interval
    layer_inf_sup_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_LAYER_SUP_OR_INF_ROOT, list_computed_layers
    )

    values_bands.extend(layer_inf_sup_indexes)
    split_indexes.append(len(layer_inf_sup_indexes))

    # 4. mask
    msk_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_MSK, list_computed_layers
    )
    values_bands.extend(msk_indexes)
    split_indexes.append(len(msk_indexes))

    # 5. classification
    classif_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_CLASSIF_KEY_ROOT, list_computed_layers
    )

    values_bands.extend(classif_indexes)
    split_indexes.append(len(classif_indexes))

    # 6. source point cloud
    # Fill the dataframe with additional columns :
    # each column refers to a point cloud id
    number_of_pc = cars_dataset.get_attributes(cloud)["number_of_pc"]
    if cst.POINT_CLOUD_GLOBAL_ID in cloud.columns and (
        (list_computed_layers is None)
        or rast_wrap.substring_in_list(
            list_computed_layers, cst.POINT_CLOUD_SOURCE_KEY_ROOT
        )
    ):
        for pc_id in range(number_of_pc):
            # Create binary list that indicates from each point whether it comes
            # from point cloud number "pc_id"
            point_is_from_pc = list(
                map(int, cloud[cst.POINT_CLOUD_GLOBAL_ID] == pc_id)
            )
            pc_key = "{}{}".format(cst.POINT_CLOUD_SOURCE_KEY_ROOT, pc_id)
            cloud[pc_key] = point_is_from_pc

    source_pc_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_SOURCE_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(source_pc_indexes)
    split_indexes.append(len(source_pc_indexes))

    # 7. filling
    filling_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_FILLING_KEY_ROOT, list_computed_layers
    )
    values_bands.extend(filling_indexes)
    split_indexes.append(len(filling_indexes))

    # 8. Performance map from risk and intervals
    performance_map_indexes = rast_wrap.find_indexes_in_point_cloud(
        cloud, cst.POINT_CLOUD_PERFORMANCE_MAP_ROOT, list_computed_layers
    )
    values_bands.extend(performance_map_indexes)

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
            float(radius),
            sigma,
        )
    )

    # pylint: disable=unbalanced-tuple-unpacking
    (
        out,
        confidences,
        interval,
        msk,
        classif,
        source_pc,
        filling,
        performance_map,
    ) = np.split(out, np.cumsum(split_indexes), axis=-1)

    confidences_out = None
    if len(confidences_indexes) > 0:
        confidences_out = {}
        for k, key in enumerate(confidences_indexes):
            confidences_out[key] = confidences[..., k]

    layers_inf_sup_out = None
    layers_inf_sup_stat_index = None
    if len(layer_inf_sup_indexes) > 0:
        layers_inf_sup_out = interval
        layers_inf_sup_stat_index = [
            values_bands.index(int_ind) for int_ind in layer_inf_sup_indexes
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

    if len(performance_map_indexes) == 0:
        performance_map = None

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
        layers_inf_sup_out,
        layers_inf_sup_stat_index,
        layer_inf_sup_indexes,
        source_pc_out,
        filling_out,
        filling_indexes,
        performance_map,
        performance_map_indexes,
    )


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
    texture_no_data: int = 0,
    msk_no_data: int = 255,
    list_computed_layers: List[str] = None,
    source_pc_names: List[str] = None,
    performance_map_classes: List[float] = None,
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
    :param texture_no_data: no data value to use for color
    :param msk_no_data: no data value to use in the final mask image
    :param list_computed_layers: list of computed output data
    :param source_pc_names: list of source pc names
    :param performance_map_classes: list for step defining border of class
    :type performance_map_classes: list or None
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
        layer_inf_sup,
        layer_inf_sup_stats_indexes,
        layer_inf_sup_indexes,
        source_pc,
        filling,
        filling_indexes,
        performance_map_raw,
        performance_map_raw_indexes,
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
        for key, value in confidences.items():
            confidences[key] = value.reshape(shape_out)

    if layer_inf_sup is not None:
        layer_inf_sup = layer_inf_sup.reshape(shape_out + (-1,))
        layer_inf_sup = np.moveaxis(layer_inf_sup, 2, 0)

    if source_pc is not None:
        source_pc = source_pc.reshape(shape_out + (-1,))
        source_pc = np.moveaxis(source_pc, 2, 0)

    if filling is not None:
        filling = filling.reshape(shape_out + (-1,))
        filling = np.moveaxis(filling, 2, 0)

    performance_map_classified = None
    performance_map_classified_indexes = None
    if performance_map_raw is not None:
        performance_map_raw = performance_map_raw.reshape(shape_out + (-1,))
        performance_map_raw = np.moveaxis(performance_map_raw, 2, 0)
        if performance_map_classes is not None:
            (performance_map_classified, performance_map_classified_indexes) = (
                rast_wrap.classify_performance_map(
                    performance_map_raw, performance_map_classes, msk_no_data
                )
            )

    # build output dataset
    raster_out = rast_wrap.create_raster_dataset(
        out,
        weights_sum,
        x_start,
        y_start,
        x_size,
        y_size,
        resolution,
        hgt_no_data,
        texture_no_data,
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
        layer_inf_sup,
        layer_inf_sup_stats_indexes,
        layer_inf_sup_indexes,
        source_pc,
        source_pc_names,
        filling,
        filling_indexes,
        performance_map_raw,
        performance_map_classified,
        performance_map_classified_indexes,
        performance_map_raw_indexes,
    )

    return raster_out
