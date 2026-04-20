#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
this module contains the abstract matching application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ScalingApplicationTemplate
from cars.applications.dense_matching.methods import (
    abstract_dense_matching_method as adm,
)

AbstractDenseMatchingMethod = adm.AbstractDenseMatchingMethod


@Application.register("dense_matching")
class AbstractDenseMatchingApplication(
    ScalingApplicationTemplate, metaclass=ABCMeta
):
    """
    AbstractDenseMatchingApplication
    """

    available_applications: Dict = {}
    default_application = "basic"
    default_method = "pandora_auto"

    def __new__(cls, scaling_coeff, conf=None):

        matching_application = cls.default_application
        if bool(conf) is False or "application" not in conf:
            logging.info(
                "Dense Matching application not specified, "
                f"default {matching_application} is used"
            )
        else:
            matching_application = conf.get(
                "application", cls.default_application
            )

        if matching_application not in cls.available_applications:
            logging.error(
                f"No matching application named {matching_application} "
                "registered"
            )
            raise KeyError(
                f"No matching application named {matching_application} "
                "registered"
            )

        logging.info(
            f"The AbstractDenseMatching({matching_application}) application "
            "will be used"
        )

        return super().__new__(cls.available_applications[matching_application])

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_applications[name] = cls

    def __init__(self, scaling_coeff, conf=None):

        if conf is None:
            conf = {}

        # init the method before the application
        conf["method"] = conf.get("method", self.default_method)
        # pylint: disable=abstract-class-instantiated
        self.dense_matching_method = AbstractDenseMatchingMethod(conf)

        super().__init__(scaling_coeff, conf=conf)

    @abstractmethod
    def get_optimal_tile_size(self, disp_range_grid, max_ram_per_worker):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_range_grid: minimum and maximum disparity grid
        :param max_ram_per_worker: maximum ram per worker
        :return: optimal tile size

        """

    @abstractmethod
    def get_performance_map_parameters(self):
        """
        Get parameter linked to performance, that will be used in triangulation

        :return: parameters to use
        :type: dict
        """

    @abstractmethod
    def get_margins_fun(self, grid_left, disp_range_grid):
        """
        Get Margins function  that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :param disp_range_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi

        """

    # pylint: disable=too-many-positional-arguments
    @abstractmethod
    def generate_disparity_grids(  # noqa: C901
        self,
        sensor_image_right,
        grid_right,
        geom_plugin_with_dem_and_geoid,
        dmin=None,
        dmax=None,
        dem_min=None,
        dem_max=None,
        pair_folder=None,
        orchestrator=None,
    ):
        """
        Generate disparity grids min and max, with given step

        global mode: uses dmin and dmax
        local mode: uses dems


        :param sensor_image_right: sensor image
        :type sensor_image_right: dict
        :param grid_right: right epipolar grid
        :type grid_right: CarsDataset
        :param geom_plugin_with_dem_and_geoid: geometry plugin with dem mean
            used to generate epipolar grids
        :type geom_plugin_with_dem_and_geoid: GeometryPlugin
        :param dmin: minimum disparity
        :type dmin: float
        :param dmax: maximum disparity
        :type dmax: float
        :param dem_min: path to minimum dem
        :type dem_min: str
        :param dem_max: path to maximum dem
        :type dem_max: str
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator to perform inverse locs
        :type orchestrator: Orchestrator


        :return disparity grid range, containing grid min and max
        :rtype: CarsDataset
        """

    @abstractmethod
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_images_left,
        epipolar_images_right,
        local_tile_optimal_size_fun,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        disp_range_grid=None,
        compute_disparity_masks=False,
        margins_to_keep=0,
        texture_bands=None,
        classif_bands_to_mask=None,
    ):
        """
        Run Matching application.

        Create CarsDataset filled with xarray.Dataset, corresponding
        to epipolar disparities, on the same geometry than
        epipolar_images_left.

        :param epipolar_images_left: tiled left epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"
                        "transform", "crs", "valid_pixels", "no_data_mask",
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_right: CarsDataset
        :param local_tile_optimal_size_fun: function to compute local
            optimal tile size
        :type local_tile_optimal_size_fun: func
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param disp_to_alt_ratio: disp to alti ratio used for performance map
        :type disp_to_alt_ratio: float
        :param margins_to_keep: margin to keep after dense matching
        :type margins_to_keep: int
        :param texture_bands: indices of bands from epipolar_images_left
            used for output texture
        :type texture_bands: list

        :return: disparity map: \
            The CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                 "disp_min_tiling", "disp_max_tiling"

        :rtype: CarsDataset
        """
