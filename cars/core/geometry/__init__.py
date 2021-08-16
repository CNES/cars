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

    available_loaders: Dict = {}

    def __new__(cls, loader_to_use):
        """
        Return the required loader
        :raises:
         - KeyError when the required loader is not registered

        :param loader_to_use: loader name to instantiate
        :return: a loader_to_use object
        """

        if loader_to_use not in cls.available_loaders.keys():
            logging.error(
                "No geometry loader named {} registered".format(loader_to_use)
            )
            raise KeyError(
                "No geometry loader named {} registered".format(loader_to_use)
            )

        logging.info(
            "[The AbstractGeometry {} loader will be used".format(loader_to_use)
        )

        return super(AbstractGeometry, cls).__new__(
            cls.available_loaders[loader_to_use]
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
            cls.available_loaders[short_name] = subclass
            return subclass

        return decorator

    @property
    @abstractmethod
    def conf_schema(self):
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

    @staticmethod
    @abstractmethod
    def check_products_consistency(cars_conf) -> bool:
        """
        Test if the product is readable by the geometry loader

        :param: cars_conf: cars input configuration
        :return: True if the products are readable, False otherwise
        """

    @staticmethod
    @abstractmethod
    def triangulate(
        mode: str,
        matches: Union[xr.Dataset, np.ndarray],
        grid1: str,
        grid2: str,
        cars_conf,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param mode: triangulation mode
        (constants.DISP_MODE or constants.MATCHES)
        :param matches: cars disparity dataset or matches as numpy array
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param cars_conf: cars input configuration
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

    @staticmethod
    @abstractmethod
    def generate_epipolar_grids(
        cars_conf,
        dem: Union[None, str] = None,
        default_alt: Union[None, float] = None,
        epipolar_step: int = 30,
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param cars_conf: cars input configuration
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
    def matches_to_sensor_coords(
        grid1: str,
        grid2: str,
        matches: Union[xr.Dataset, np.ndarray],
        matches_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert matches (sparse or dense matches) given in epipolar
        coordinates to sensor coordinates. This function is available for
        loaders if it requires matches in sensor coordinates to perform
        the triangulation.

        This function returns a tuple composed of the matches left and right
        sensor coordinates as numpy arrays. For each original image, the sensor
        coordinates are arranged as follows :
            * if the matches are a vector of matching points: a numpy array of
            size [number of matches, 2].
            The last index indicates the 'x' coordinate (last index set to 0) or
            the 'y' coordinate (last index set to 1).
            * if matches is a cars disparity dataset: a numpy array of size
            [nb_epipolar_line, nb_epipolar_col, 2]. Where
            [nb_epipolar_line, nb_epipolar_col] is the size of the disparity
            map. The last index indicates the 'x' coordinate (last index set
            to 0) or the 'y' coordinate (last index set to 1).

        :param grid1: path to epipolar grid of image 1
        :param grid2: path to epipolar grid of image 2
        :param matches: cars disparity dataset or matches as numpy array
        :param matches_type: matches type (cst.DISP_MODE or cst.MATCHES)
        :return: a tuple of numpy array. The first array corresponds to the
        left matches in sensor coordinates, the second one is the right
        matches in sensor coordinates.
        """
        vec_epi_pos_left = None
        vec_epi_pos_right = None

        if matches_type == cst.MATCHES_MODE:
            # retrieve left and right matches
            vec_epi_pos_left = matches[:, 0:2]
            vec_epi_pos_right = matches[:, 2:4]
        elif matches_type == cst.DISP_MODE:
            # convert disparity to matches
            epi_pos_left_x, epi_pos_left_y = np.meshgrid(
                matches.col, matches.row
            )
            epi_pos_left_x = epi_pos_left_x.astype(np.float64)
            epi_pos_left_y = epi_pos_left_y.astype(np.float64)
            disp_map = matches[cst.DISP_MAP].values
            disp_msk = matches[cst.DISP_MSK].values
            epi_pos_right_y = np.copy(epi_pos_left_y)
            epi_pos_right_x = np.copy(epi_pos_left_x)
            epi_pos_right_x[np.where(disp_msk == 255)] += disp_map[
                np.where(disp_msk == 255)
            ]

            # vectorize matches
            vec_epi_pos_left = np.transpose(
                np.vstack([epi_pos_left_x.ravel(), epi_pos_left_y.ravel()])
            )
            vec_epi_pos_right = np.transpose(
                np.vstack([epi_pos_right_x.ravel(), epi_pos_right_y.ravel()])
            )

        # convert epipolar matches to sensor coordinates
        sensor_pos_left = AbstractGeometry.sensor_position_from_grid(
            grid1, vec_epi_pos_left
        )
        sensor_pos_right = AbstractGeometry.sensor_position_from_grid(
            grid2, vec_epi_pos_right
        )

        if matches_type == cst.DISP_MODE:
            # rearrange matches in the original epipolar geometry
            disp_msk = matches[cst.DISP_MSK].values

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
    def sensor_position_from_grid(
        grid: str, positions: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the positions given as inputs using the grid

        :param grid: path to epipolar grid
        :param positions: epipolar positions to interpolate given as a numpy
        array of size [number of points, 2]. The last index indicates the 'x'
         coordinate (last index set to 0) or the 'y' coordinate
         (last index set to 1).
        :return: sensors positions as a numpy array of size
        [number of points, 2]. The last index indicates the 'x' coordinate
        (last index set to 0) or the 'y' coordinate (last index set to 1).
        """

        # open epipolar grid
        ds_grid = rio.open(grid)

        # retrieve grid step
        transform = ds_grid.transform
        step_col = transform[0]
        step_row = transform[4]

        # center-pixel positions
        [ori_col, ori_row] = transform * (0.5, 0.5)

        last_col = ori_col + step_col * ds_grid.width
        last_row = ori_row + step_row * ds_grid.height

        # transform dep to positions
        row_dep = ds_grid.read(2).transpose()
        col_dep = ds_grid.read(1).transpose()

        cols = np.arange(ori_col, last_col, step_col)
        rows = np.arange(ori_row, last_row, step_row)

        # create regular grid points positions
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

        # stack both coordinates
        sensor_positions = np.transpose(np.vstack([interp_col, interp_row]))
        return sensor_positions
