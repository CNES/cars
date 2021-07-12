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
from typing import Dict

import numpy as np
import xarray as xr


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
            raise KeyError

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
    def triangulate(
        data: xr.Dataset,
        roi_key: str,
        grid1: str,
        grid2: str,
        img1: str,
        img2: str,
        min_elev1: float,
        max_elev1: float,
        min_elev2: float,
        max_elev2: float,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity dataset

        :param data: cars disparity dataset
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param img1: path to image 1
        :param img2: path to image 2
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :return: the long/lat/height numpy array in output of the triangulation
        """

    @staticmethod
    @abstractmethod
    def triangulate_matches(
        matches: np.ndarray,
        grid1: str,
        grid2: str,
        img1: str,
        img2: str,
        min_elev1: float,
        max_elev1: float,
        min_elev2: float,
        max_elev2: float,
    ) -> np.ndarray:
        """
        Performs triangulation from matches

        :param matches: input matches to triangulate
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param img1: path to image 1
        :param img2: path to image 2
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :return: the long/lat/height numpy array in output of the triangulation
        """
