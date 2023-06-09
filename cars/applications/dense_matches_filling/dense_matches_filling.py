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

        # init classification
        cls.classification = None

        for name in short_name:
            cls.available_applications[name] = cls

    @abstractmethod
    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

    def get_classif(self):
        """
        Get classification band list
        :return: self.classification
        :rtype: list[str]
        """

        classif = []
        if self.classification is not None:
            classif = self.classification

        return classif

    @abstractmethod
    def run(
        self,
        epipolar_disparity_map,
        **kwargs,
    ):
        """
        Run Refill application using plane method.

        :param epipolar_disparity_map:  left disparity
        :type epipolar_disparity_map: CarsDataset
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

        :return: filled disparity map: \
            The CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"

        :rtype: CarsDataset

        """

    def __register_dataset__(
        self,
        epipolar_disparity_map,
        save_disparity_map,
        pair_folder,
        app_name=None,
    ):
        """
        Create dataset and registered the output in the orchestrator

        :param epipolar_disparity_map:  left disparity
        :type epipolar_disparity_map: CarsDataset
        :param save_disparity_map: true if save disparity map
        :type save_disparity_map: bool
        :param pair_folder: path to folder
        :type pair_folder: str
        :param app_name: application name for file names
        :type app_name: str

        """
        if app_name is None:
            app_name = ""

        # Create CarsDataset Epipolar_disparity
        new_epipolar_disparity_map = cars_dataset.CarsDataset("arrays")
        new_epipolar_disparity_map.create_empty_copy(epipolar_disparity_map)

        # Update attributes to get epipolar info
        new_epipolar_disparity_map.attributes.update(
            epipolar_disparity_map.attributes
        )

        # Save disparity maps
        if save_disparity_map:
            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder, "epi_disp_" + app_name + "_filled.tif"
                ),
                cst_disp.MAP,
                new_epipolar_disparity_map,
                cars_ds_name="epi_disp_" + app_name + "_filled",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder,
                    "epi_disp_color_" + app_name + "_filled.tif",
                ),
                cst.EPI_COLOR,
                new_epipolar_disparity_map,
                cars_ds_name="epi_disp_color_" + app_name + "_filled",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder,
                    "epi_disp_mask_" + app_name + "_filled.tif",
                ),
                cst_disp.VALID,
                new_epipolar_disparity_map,
                cars_ds_name="epi_disp_mask_" + app_name + "_filled",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder,
                    "epi_confidence_" + app_name + "_filled.tif",
                ),
                cst_disp.CONFIDENCE,
                new_epipolar_disparity_map,
                cars_ds_name="epi_ambiguity_" + app_name + "_filled",
            )

        return new_epipolar_disparity_map
