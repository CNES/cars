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
this module contains the abstract geometry class to use in the
geometry plugins
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from scipy import interpolate

from cars.core import constants as cst


class AbstractGeometry(metaclass=ABCMeta):
    """
    AbstractGeometry
    """

    available_plugins: Dict = {}

    def __new__(cls, plugin_to_use):
        """
        Return the required plugin
        :raises:
         - KeyError when the required plugin is not registered

        :param plugin_to_use: plugin name to instantiate
        :return: a plugin_to_use object
        """

        if plugin_to_use not in cls.available_plugins.keys():
            logging.error(
                "No geometry plugin named {} registered".format(plugin_to_use)
            )
            raise KeyError(
                "No geometry plugin named {} registered".format(plugin_to_use)
            )

        logging.info(
            "[The AbstractGeometry {} plugin will be used".format(plugin_to_use)
        )

        return super(AbstractGeometry, cls).__new__(
            cls.available_plugins[plugin_to_use]
        )

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name
        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods
            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.available_plugins[short_name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def geo_conf_schema():
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

    @staticmethod
    @abstractmethod
    def check_products_consistency(geo_conf: Dict[str, str]) -> bool:
        """
        Test if the product is readable by the geometry loader

        :param: the geometry configuration as requested by the geometry loader
        schema
        :return: True if the products are readable, False otherwise
        """

    @staticmethod
    @abstractmethod
    def triangulate(
        mode: str,
        data: Union[xr.Dataset, np.ndarray],
        grid1: str,
        grid2: str,
        geo_conf: Dict[str, str],
        min_elev1: float = None,
        max_elev1: float = None,
        min_elev2: float = None,
        max_elev2: float = None,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param mode: triangulation mode
        (constants.DISP_MODE or constants.MATCHES)
        :param data: cars disparity dataset or matches as numpy array
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param geo_conf: dictionary with the fields requested in the schema
        given by the geo_conf_schema() method
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

    @staticmethod
    @abstractmethod
    def generate_epipolar_grids(
        left_img: str,
        right_img: str,
        dem: Union[None, str] = None,
        default_alt: Union[None, float] = None,
        epipolar_step: int = 30,
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param left_img: path to left image
        :param right_img: path to right image
        :param dem: path to the dem folder
        :param default_alt: default altitude to use in the missing dem regions
        :param epipolar_step: step to use to construct the epipolar grids
        :return: Tuple composed of :
            - the left epipolar grid as a numpy array
            - the right epipolar grid as a numpy array
            - the left grid origin as a list of float
            - the left grid spacing as a list of float
            - the epipolar image size as a list of int
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float
        """

    @staticmethod
    def matching_data_to_sensor_coords(grid1, grid2, data, data_type):
        """

        :param grid1:
        :param grid2:
        :param data:
        :param data_type:
        :return:
        """
        vec_epi_pos_left = None
        vec_epi_pos_right = None

        if data_type == cst.MATCHES_MODE:
            vec_epi_pos_left = data[:, 0:2]
            vec_epi_pos_right = data[:, 2:4]
        elif data_type == cst.DISP_MODE:
            epi_pos_left_x, epi_pos_left_y = np.meshgrid(data.col, data.row)
            epi_pos_left_x = epi_pos_left_x.astype(np.float64)
            epi_pos_left_y = epi_pos_left_y.astype(np.float64)
            disp_map = data[cst.DISP_MAP].values
            disp_msk = data[cst.DISP_MSK].values
            epi_pos_right_y = np.copy(epi_pos_left_y)
            epi_pos_right_x = np.copy(epi_pos_left_x)
            epi_pos_right_x[np.where(disp_msk == 255)] += disp_map[
                np.where(disp_msk == 255)
            ]
            vec_epi_pos_left = np.transpose(
                np.vstack([epi_pos_left_x.ravel(), epi_pos_left_y.ravel()])
            )
            vec_epi_pos_right = np.transpose(
                np.vstack([epi_pos_right_x.ravel(), epi_pos_right_y.ravel()])
            )

        sensor_pos_left = AbstractGeometry.sensor_position_from_grid(
            grid1, vec_epi_pos_left
        )
        sensor_pos_right = AbstractGeometry.sensor_position_from_grid(
            grid2, vec_epi_pos_right
        )

        if data_type == cst.DISP_MODE:
            disp_msk = data[cst.DISP_MSK].values

            sensor_pos_left_x = sensor_pos_left[:, 0].reshape(disp_msk.shape)
            sensor_pos_left_x[np.where(disp_msk != 255)] = np.nan
            sensor_pos_left_y = sensor_pos_left[:, 1].reshape(disp_msk.shape)
            sensor_pos_left_y[np.where(disp_msk != 255)] = np.nan

            sensor_pos_right_x = sensor_pos_right[:, 0].reshape(disp_msk.shape)
            sensor_pos_right_x[np.where(disp_msk != 255)] = np.nan
            sensor_pos_right_y = sensor_pos_right[:, 1].reshape(disp_msk.shape)
            sensor_pos_right_y[np.where(disp_msk != 255)] = np.nan

            sensor_pos_left = np.zeros(
                (disp_msk.shape[0], disp_msk.shape[1], 2)
            )
            sensor_pos_left[:, :, 0] = sensor_pos_left_x
            sensor_pos_left[:, :, 1] = sensor_pos_left_y
            sensor_pos_right = np.zeros(
                (disp_msk.shape[0], disp_msk.shape[1], 2)
            )
            sensor_pos_right[:, :, 0] = sensor_pos_right_x
            sensor_pos_right[:, :, 1] = sensor_pos_right_y

        return sensor_pos_left, sensor_pos_right

    @staticmethod
    def sensor_position_from_grid(grid, positions):
        """

        :param grid:
        :param positions:
        :return:
        """
        ds_grid = rio.open(grid)

        transform = ds_grid.transform
        step_col = transform[0]
        step_row = transform[4]

        # 0 or 0.5
        [ori_col, ori_row] = transform * (0.5, 0.5)  # positions au centre pixel

        last_col = ori_col + step_col * ds_grid.width
        last_row = ori_row + step_row * ds_grid.height

        # transform dep to positions
        row_dep = ds_grid.read(2).transpose()
        col_dep = ds_grid.read(1).transpose()

        cols = np.arange(ori_col, last_col, step_col)
        rows = np.arange(ori_row, last_row, step_row)

        points = (cols, rows)
        grid_row, grid_col = np.mgrid[
            ori_row:last_row:step_row, ori_col:last_col:step_col
        ]
        sensor_row_positions = row_dep + grid_row
        sensor_col_positions = col_dep + grid_col

        # interpolate sensor positions
        interp_row = interpolate.interpn(
            (cols, rows),
            sensor_row_positions,
            positions,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        interp_col = interpolate.interpn(
            points,
            sensor_col_positions,
            positions,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        sensor_positions = np.transpose(np.vstack([interp_col, interp_row]))
        return sensor_positions
