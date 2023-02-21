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
this module contains the abstract dense matches filling application class.
"""
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.data_structures import cars_dataset


@Application.register("dense_matches_filling")
class DenseMatchingFilling(ApplicationTemplate, metaclass=ABCMeta):
    """
    DenseMatchingFilling
    """

    available_applications: Dict = {}
    default_application = "plane"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        fill_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "DenseMatchingFilling method not specified, "
                "default {} is used".format(fill_method)
            )
        else:
            fill_method = conf["method"]

        if fill_method not in cls.available_applications:
            logging.error(
                "No DenseMatchingFilling application "
                "named {} registered".format(fill_method)
            )
            raise KeyError(
                "No DenseMatchingFilling application"
                " named {} registered".format(fill_method)
            )

        logging.info(
            "[The DenseMatchingFilling {} application "
            "will be used".format(fill_method)
        )

        return super(DenseMatchingFilling, cls).__new__(
            cls.available_applications[fill_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        # init orchestrator
        cls.orchestrator = None
        for name in short_name:
            cls.available_applications[name] = cls

    @abstractmethod
    def get_is_activated(self):
        """
        Get the activated attribute

        :return: self.activated
        :rtype: bool
        """

    @abstractmethod
    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

    @abstractmethod
    def run(
        self,
        epipolar_disparity_map_left,
        epipolar_disparity_map_right,
        epipolar_images_left,
        **kwargs,
    ):
        """
        Run Refill application using plane method.

        :param epipolar_disparity_map_left:  left disparity
        :type epipolar_disparity_map_left: CarsDataset
        :param epipolar_disparity_map_right:  right disparity
        :type epipolar_disparity_map_right: CarsDataset
        :param epipolar_images_left: tiled left epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"
        :type epipolar_images_left: CarsDataset
        :param holes_bbox_left:  left holes
        :type holes_bbox_left: CarsDataset
        :param holes_bbox_right:  right holes
        :type holes_bbox_right: CarsDataset
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str

        :return: filled left disparity map, filled right disparity map: \
            Each CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"

        :rtype: Tuple(CarsDataset, CarsDataset)

        """

    def __register_dataset__(
        self,
        epipolar_disparity_map_left,
        epipolar_disparity_map_right,
        save_disparity_map,
        pair_folder,
    ):
        """
        Create dataset and registered the output in the orchestrator

        :param epipolar_disparity_map_left:  left disparity
        :type epipolar_disparity_map_left: CarsDataset
        :param epipolar_disparity_map_right:  right disparity
        :type epipolar_disparity_map_right: CarsDataset

        """
        # Create CarsDataset Epipolar_disparity
        new_epipolar_disparity_map_left = cars_dataset.CarsDataset("arrays")
        new_epipolar_disparity_map_left.create_empty_copy(
            epipolar_disparity_map_left
        )

        new_epipolar_disparity_map_right = cars_dataset.CarsDataset("arrays")
        new_epipolar_disparity_map_right.create_empty_copy(
            epipolar_disparity_map_right
        )

        # Update attributes to get epipolar info
        new_epipolar_disparity_map_left.attributes.update(
            epipolar_disparity_map_left.attributes
        )

        # Save disparity maps
        if save_disparity_map:
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_filled_left.tif"),
                cst_disp.MAP,
                new_epipolar_disparity_map_left,
                cars_ds_name="epi_disp_filled_left",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_filled_right.tif"),
                cst_disp.MAP,
                new_epipolar_disparity_map_right,
                cars_ds_name="epi_disp_filled_right",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_color_filled_left.tif"),
                cst.EPI_COLOR,
                new_epipolar_disparity_map_left,
                cars_ds_name="epi_disp_color_filled_left",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_color_filled_right.tif"),
                cst.EPI_COLOR,
                new_epipolar_disparity_map_right,
                cars_ds_name="epi_disp_color_filled_right",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_mask_filled_left.tif"),
                cst_disp.VALID,
                new_epipolar_disparity_map_left,
                cars_ds_name="epi_disp_mask_filled_left",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_disp_mask_filled_right.tif"),
                cst_disp.VALID,
                new_epipolar_disparity_map_right,
                cars_ds_name="epi_disp_mask_filled_right",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_ambiguity_filled_left.tif"),
                cst_disp.CONFIDENCE_FROM_AMBIGUITY,
                new_epipolar_disparity_map_left,
                cars_ds_name="epi_ambiguity_filled_left",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_ambiguity_filled_right.tif"),
                cst_disp.CONFIDENCE_FROM_AMBIGUITY,
                new_epipolar_disparity_map_right,
                cars_ds_name="epi_ambiguity_filled_right",
            )

            for _, item in enumerate(cst_disp.DISPARITY_CONFIDENCE):
                cards_ds_name_left = item + "_filled_left"
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_" + cards_ds_name_left + ".tif",
                    ),
                    item,
                    epipolar_disparity_map_left,
                    cars_ds_name=cards_ds_name_left,
                )
                cards_ds_name_right = item + "_filled_right"
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_" + cards_ds_name_right + ".tif",
                    ),
                    item,
                    epipolar_disparity_map_right,
                    cars_ds_name=cards_ds_name_right,
                )

        return new_epipolar_disparity_map_left, new_epipolar_disparity_map_right
