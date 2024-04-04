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
this module contains the abstract triangulation application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("triangulation")
class Triangulation(ApplicationTemplate, metaclass=ABCMeta):
    """
    Triangulation
    """

    available_applications: Dict = {}
    default_application = "line_of_sight_intersection"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for triangulation
        :return: a application_to_use object
        """

        triangulation_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Triangulation method not specified, default "
                " {} is used".format(triangulation_method)
            )
        else:
            triangulation_method = conf.get("method", cls.default_application)

        if triangulation_method not in cls.available_applications:
            logging.error(
                "No triangulation application named {} registered".format(
                    triangulation_method
                )
            )
            raise KeyError(
                "No triangulation application named {} registered".format(
                    triangulation_method
                )
            )

        logging.info(
            "The Triangulation({}) application will be used".format(
                triangulation_method
            )
        )

        return super(Triangulation, cls).__new__(
            cls.available_applications[triangulation_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of Triangulation

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(  # noqa: C901
        self,
        sensor_image_left,
        sensor_image_right,
        epipolar_image,
        grid_left,
        grid_right,
        epipolar_disparity_map,
        epsg,
        geometry_plugin,
        denoising_overload_fun=None,
        source_pc_names=None,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        uncorrected_grid_right=None,
        geoid_path=None,
        cloud_id=None,
        intervals=None,
    ):
        """
        Run Triangulation application.

        Created left and right CarsDataset filled with xarray.Dataset,
        corresponding to 3D points clouds, stored on epipolar geometry grid.

        :param sensor_image_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_left: CarsDataset
        :param sensor_image_right: tiled sensor right image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_right: CarsDataset
        :param epipolar_image: tiled epipolar left image
        :type epipolar_image: CarsDataset
        :param grid_left: left epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",\
                "epipolar_origin_y","epipolar_spacing_x",\
                "epipolar_spacing", "disp_to_alt_ratio",\
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",
                "epipolar_origin_y","epipolar_spacing_x",
                "epipolar_spacing", "disp_to_alt_ratio",
        :type grid_right: CarsDataset
        :param epipolar_disparity_map: tiled left disparity map or \
            sparse matches:

            - if CarsDataset is instance of "arrays", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "disp", "disp_msk"
                    - attrs with keys: profile, window, overlaps
                - attributes containing:"largest_epipolar_region"\
                  "opt_epipolar_tile_size"

            - if CarsDataset is instance of "points", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future pandas DataFrame containing:

                    - data : (L, 4) shape matches
                - attributes containing:"disp_lower_bound","disp_upper_bound",\
                    "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :type epipolar_disparity_map: CarsDataset
        :param denoising_overload_fun: function to overload dataset
        :type denoising_overload_fun: fun
        :param source_pc_names: source pc names
        :type source_pc_names: list[str]
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair key id
        :type pair_key: str
        :param uncorrected_grid_right: not corrected right epipolar grid
                used if self.snap_to_img1
        :type uncorrected_grid_right: CarsDataset
        :param geoid_path: geoid path
        :type geoid_path: str
        :param intervals: Either None or a List of 2 intervals indicators
        :type intervals: None or [str, str]

        :return: points cloud \
                The CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "color", "msk", "z_inf", "z_sup"
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :rtype: Tuple(CarsDataset, CarsDataset)
        """
