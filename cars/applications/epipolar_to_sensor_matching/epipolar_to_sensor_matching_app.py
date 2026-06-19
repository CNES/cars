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
this module contains the epipolar grid correction application class.
"""
# Standard imports
from __future__ import absolute_import

import collections

# Standard imports
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import And, Checker
from scipy.interpolate import griddata
from shareloc import proj_utils

import cars.orchestrator.orchestrator as ocht

# CARS imports
# fmt: off
# isort: off
from cars.applications.epipolar_to_sensor_matching \
    .abstract_epipolar_to_sensor_matching_app import (
        EpipolarToSensorMatching,
    )
# isort: on
# fmt: on
from cars.core import constants as cst
from cars.core import inputs
from cars.core.geometry import abstract_geometry
from cars.data_structures import cars_dataset, cars_dataset_transformations
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


class EpipolarToSensorMatchingApp(
    EpipolarToSensorMatching, short_name="default"
):
    """
    EpipolarToSensorMatching
    """

    def __init__(self, conf=None):
        """
        Init function of EpipolarGridGeneration

        :param conf: configuration for grid generation
        :return: application_to_use object
        """

        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

        self.tile_size = self.used_config["tile_size"]

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # Init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "default")
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )
        overloaded_conf["tile_size"] = conf.get("tile_size", 600)

        epi_to_sensor_matches_schema = {
            "method": str,
            "save_intermediate_data": bool,
            "tile_size": And(int, lambda x: x > 0),
        }

        # Check conf
        checker = Checker(epi_to_sensor_matches_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # pylint: disable=R0917
        self,
        sensor_image_left,
        grid_left,
        grid_right,
        disparity_map,
        orchestrator=None,
        pair_folder=None,
        pair_key="pair_0",
    ):
        """
        Run EpipolarToSensorMatching application

        :param sensor_image_left: sensor image left
        :type sensor_image_left: CarsDataset
        :param grid_left: corrected grid left
        :type grid_left: dict
        :param grid_right: corrected grid right
        :type grid_right: dict
        :param disparity_map: disparity map
        :type disparity_map: CarsDataset
        :param orchestrator: orchestrator
        :param pair_folder: pair folder to save intermediate data,
            if save_intermediate_data is True
        :param pair_key: pair key to use
        :return: dense sensor matches from epipolar matches
        :rtype: CarsDataset
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            cars_orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            cars_orchestrator = orchestrator

        # Initialize CARS Dataset
        # Get profile
        with rio.open(
            sensor_image_left[sens_cst.INPUT_IMG]["bands"]["b0"]["path"]
        ) as src_left:
            width_left = src_left.width
            height_left = src_left.height
            transform_left = src_left.transform

        raster_profile_left = collections.OrderedDict(
            {
                "height": height_left,
                "width": width_left,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform_left,
                "tiled": True,
            }
        )

        sensor_matches_left = cars_dataset.CarsDataset(
            "arrays", name="sensor_matches_left" + pair_key
        )

        # update grid
        sensor_matches_left.create_grid(
            width_left, height_left, self.tile_size, self.tile_size, 0, 0
        )

        if self.save_intermediate_data:
            cars_orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "sensor_row_right.tif"),
                cst.SENSOR_RIGHT_ROW,
                sensor_matches_left,
                cars_ds_name="sensor_matches_left",
            )
            cars_orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "sensor_col_right.tif"),
                cst.SENSOR_RIGHT_COL,
                sensor_matches_left,
                cars_ds_name="sensor_matches_left",
            )

        # Get saving infos in order to save tiles when they are computed
        [saving_infos_sensor_left] = cars_orchestrator.get_saving_infos(
            [sensor_matches_left]
        )

        sensor_transform = inputs.rasterio_get_transform(
            sensor_image_left["image"]["bands"]["b0"]["path"], "north"
        )

        # left
        for col in range(sensor_matches_left.tiling_grid.shape[1]):
            for row in range(sensor_matches_left.tiling_grid.shape[0]):
                full_saving_info_left = ocht.update_saving_infos(
                    saving_infos_sensor_left, row=row, col=col
                )

                # Get corresponding epipolar tiles as CARS Dataset window
                sensor_pixel_window = sensor_matches_left.tiling_grid[row, col]

                sensor_pos_window = transform_pix_window_to_pos_window(
                    sensor_pixel_window, sensor_transform
                )
                disparity_map_tiles = get_cropped_disparity_map_tiles(
                    disparity_map, grid_left, sensor_pos_window
                )

                # Intepolate on epipolar
                (
                    sensor_matches_left[row, col]
                ) = cars_orchestrator.cluster.create_task(
                    sensor_matches_from_epi_wrapper, nout=1
                )(
                    sensor_image_left,
                    disparity_map_tiles,
                    grid_left,
                    grid_right,
                    full_saving_info_left,
                    sensor_matches_left.tiling_grid[row, col],
                    raster_profile=raster_profile_left,
                    window_dict=sensor_matches_left.get_window_as_dict(
                        row, col
                    ),
                )

        return sensor_matches_left


def transform_pix_window_to_pos_window(sensor_pixel_window, sensor_transform):
    """
    Transform pixel window to position window
    :param sensor_pixel_window:
    :param sensor_transform: transform
    :return:  position window corresponding to the pixel window
    """

    points = [
        (sensor_pixel_window[0], sensor_pixel_window[2]),
        (sensor_pixel_window[0], sensor_pixel_window[3]),
        (sensor_pixel_window[1], sensor_pixel_window[2]),
        (sensor_pixel_window[1], sensor_pixel_window[3]),
    ]

    sensor_position_points = []

    for point in points:
        sensor_position_points.append(
            proj_utils.transform_index_to_physical_point(
                sensor_transform, point[0], point[1]
            )
        )

    sensor_position_points = np.array(sensor_position_points)

    sensor_position_window = [
        np.nanmin(sensor_position_points[:, 1]),
        np.nanmax(sensor_position_points[:, 1]),
        np.nanmin(sensor_position_points[:, 0]),
        np.nanmax(sensor_position_points[:, 0]),
    ]

    return sensor_position_window


def get_cropped_disparity_map_tiles(disparity_map, grid_left, window):
    """
    Get cropped disparity map tiles as CARS Dataset windows
        corresponding to the sensor window

    :param disparity_map:
    :param grid_left:
    :param window:
    :return:
    """

    # Get corresponding epipolar window from sensor window
    with rio.open(grid_left["path"]) as grid_left_src:
        row_layer = grid_left_src.read(1)
        col_layer = grid_left_src.read(2)

    mask = (
        (row_layer >= window[0])
        & (row_layer <= window[1])
        & (col_layer >= window[2])
        & (col_layer <= window[3])
    )
    indexes = np.argwhere(mask)

    if indexes.size == 0:
        logging.warning(
            "No epipolar pixel found for sensor window {}".format(window)
        )
        return None

    row_min, col_min = indexes.min(axis=0)
    row_max, col_max = indexes.max(axis=0)

    # Get window for epipolar grid
    epipolar_grid_window = np.array(
        [row_min - 1, row_max + 2, col_min - 1, col_max + 2]
    )
    # Get window fot disparity map
    grid_margin = int(
        -grid_left["grid_origin"][0] / grid_left["epipolar_step"] - 0.5
    )

    epipolar_grid_window -= grid_margin

    epipolar_region = epipolar_grid_window * grid_left["epipolar_step"]

    # Generate corresponding CarsDataset
    cropped_disparity_map = cars_dataset_transformations.extract_cars_dataset(
        disparity_map, list(epipolar_region)
    )

    return cropped_disparity_map


def sensor_matches_from_epi_wrapper(  # pylint: disable=R0917
    sensor_image_left,
    disparity_map_cars_ds,
    grid_left,
    grid_right,
    full_saving_info_left,
    window,
    raster_profile=None,
    window_dict=None,
):
    """
    Wrapper for sensor_matches_from_epi function to be used in
    orchestrator cluster

    :param disparity_map_cars_ds: list of disparity map tiles
        as CARS Dataset windows
    :param grid_left: corrected grid left
    :param full_saving_info_left: full saving info for left sensor matches tile
    :param window: sensor window to compute
    :param raster_profile: raster profile to save left sensor matches tile
    :param window_dict: window dict to save left sensor matches tile
    :return: sensor matches from epipolar matches for the tile
    :rtype: np.ndarray
    """
    # reconstruct full array of sensor positions in epipolar structure
    points_list = []
    values_list = []
    for row in range(disparity_map_cars_ds.tiling_grid.shape[0]):
        for col in range(disparity_map_cars_ds.tiling_grid.shape[1]):
            tile_result = transform_epipolar_to_sensor_matches(
                disparity_map_cars_ds[row, col], grid_left, grid_right
            )
            pts = tile_result[:, :, 0:2].reshape(-1, 2)
            vals = tile_result[:, :, 2:4].reshape(-1, 2)
            valid = ~np.isnan(pts).any(axis=1)
            points_list.append(pts[valid])
            values_list.append(vals[valid])

    # points , values
    points = np.concatenate(points_list)
    values = np.concatenate(values_list)

    # sensor output
    rows = np.arange(window[0], window[1])
    cols = np.arange(window[2], window[3])

    # Transform to position
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")

    sensor_transform_left = inputs.rasterio_get_transform(
        sensor_image_left["image"]["bands"]["b0"]["path"], "north"
    )
    grid_col, grid_row = proj_utils.transform_index_to_physical_point(
        sensor_transform_left, grid_row, grid_col
    )

    grid_values = griddata(
        points, values, (grid_row, grid_col), method="linear"
    )

    left_row = grid_row
    left_col = grid_col
    right_row = grid_values[:, :, 0]
    right_col = grid_values[:, :, 1]

    values = {
        cst.SENSOR_RIGHT_ROW: (
            [
                cst.ROW,
                cst.COL,
            ],
            right_row,
        ),
        cst.SENSOR_RIGHT_COL: (
            [
                cst.ROW,
                cst.COL,
            ],
            right_col,
        ),
        cst.SENSOR_LEFT_ROW: (
            [
                cst.ROW,
                cst.COL,
            ],
            left_row,
        ),
        cst.SENSOR_LEFT_COL: (
            [
                cst.ROW,
                cst.COL,
            ],
            left_col,
        ),
    }
    outputs_dataset = xr.Dataset(
        values,
        coords={cst.ROW: rows, cst.COL: cols},
    )

    # Fill datasets based on target
    attributes = {}
    # Return results based on target
    cars_dataset.fill_dataset(
        outputs_dataset,
        saving_info=full_saving_info_left,
        window=window_dict,
        attributes=attributes,
        profile=raster_profile,
    )

    return outputs_dataset


def transform_epipolar_to_sensor_matches(
    disparity_map, epi_grid_left, epi_grid_right
):
    """
    Transform epipolar matches to sensor matches, in epipolar structure

    :param disparity_map: disparity map
    :param epi_grid_left: corrected grid left
    :param epi_grid_right: corrected grid right
    :return: sensor matches from epipolar matches for the tile
    :rtype: xarray.Dataset
    """

    # use cst.ROI , to remove overlap
    (
        matches_disp,
        mode,
        matches_msk,
        ul_matches_shift,
    ) = abstract_geometry.get_matches_to_sensor_coords_params(
        disparity_map, cst.DISP_MODE, cst.ROI
    )

    (
        sensor_pos_left,
        sensor_pos_right,
    ) = abstract_geometry.AbstractGeometry(  # pylint: disable=E0110
        "SharelocGeometry"
    ).matches_to_sensor_coords(
        epi_grid_left,
        epi_grid_right,
        matches_disp,
        mode,
        matches_msk,
        ul_matches_shift,
        interpolation_method=None,  # TODO parameter
    )

    # Fuse left and right sensor positions  in one dataset
    margins = disparity_map.attrs[cst.EPI_MARGINS]

    nb_rows, nb_cols = disparity_map[cst.DISP_MODE].shape

    return np.concatenate(
        (
            sensor_pos_left[
                -margins[1] : nb_rows - margins[3],
                -margins[0] : nb_cols - margins[2],
            ],
            sensor_pos_right[
                -margins[1] : nb_rows - margins[3],
                -margins[0] : nb_cols - margins[2],
            ],
        ),
        axis=-1,
    )
