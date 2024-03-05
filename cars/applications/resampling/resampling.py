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
this module contains the abstract resampling application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("resampling")
class Resampling(ApplicationTemplate, metaclass=ABCMeta):
    """
    Resampling
    """

    available_applications: Dict = {}
    default_application = "bicubic"

    def __new__(cls, orchestrator=None, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param orchestrator: orchestrator used
        :param conf: configuration for resampling
        :return: an application_to_use object
        """

        resampling_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Resampling method not specified, default"
                " {} is used".format(resampling_method)
            )
        else:
            resampling_method = conf["method"]

        if resampling_method not in cls.available_applications:
            logging.error(
                "No resampling application named {} registered".format(
                    resampling_method
                )
            )
            raise KeyError(
                "No resampling application named {} registered".format(
                    resampling_method
                )
            )

        logging.info(
            "The Resampling({}) application will be used".format(
                resampling_method
            )
        )

        return super(Resampling, cls).__new__(
            cls.available_applications[resampling_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of Resampling

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        margins_fun=None,
        tile_width=None,
        tile_height=None,
        step=None,
        add_color=True,
        epipolar_roi=None,
    ):  # noqa: C901
        """
        Run resampling application.

        Creates left and right CarsDataset filled with xarray.Dataset,
        corresponding to sensor images resampled in epipolar geometry.

        :param sensor_images_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_left: CarsDataset
        :param sensor_images_right: tiled sensor right image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_right: CarsDataset
        :param grid_left: left epipolar grid
            Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio",\
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio",
        :type grid_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: directory to save files to
        :param pair_key: pair id
        :type pair_key: str
        :param margins: margins to use
        :type margins: xr.Dataset
        :param optimum_tile_size: optimum tile size to use
        :type optimum_tile_size: int
        :param add_color: add color image to dataset
        :type add_color: bool
        :param epipolar_roi: Epipolar roi to use if set.
            Set None tiles outsize roi
        :type epipolar_roi: list(int), [row_min, row_max,  col_min, col_max]

        :return: left epipolar image, right epipolar image. \
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data with keys : "im", "msk", "color", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max"\
                    "transform", "crs", "valid_pixels", "no_data_mask",
                    "no_data_img"
            - attributes containing: \
                "largest_epipolar_region","opt_epipolar_tile_size"

        :rtype: Tuple(CarsDataset, CarsDataset)
        """
