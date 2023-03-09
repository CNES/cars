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
this module contains the fill_disp application class.
"""


# Standard imports
import copy
import logging

# Third party imports
import numpy as np
from json_checker import Checker

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matches_filling import (
    fill_disp_constants as fd_cst,
)
from cars.applications.dense_matches_filling import fill_disp_tools as fd_tools
from cars.applications.dense_matches_filling.dense_matches_filling import (
    DenseMatchingFilling,
)
from cars.applications.dense_matching import dense_matching_tools
from cars.core import constants as cst
from cars.data_structures import cars_dataset


class ZerosPadding(
    DenseMatchingFilling, short_name=["zeros_padding"]
):  # pylint: disable=R0903
    """
    Fill invalid area in disparity map with zeros values
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of FillDisp

        :param conf: configuration for filling
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # get conf
        self.activated = self.used_config["activated"]
        self.used_method = self.used_config["method"]

        # Saving files
        self.save_disparity_map = self.used_config["save_disparity_map"]

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict
        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "plane")

        overloaded_conf["activated"] = conf.get("activated", False)

        # Saving files
        overloaded_conf["save_disparity_map"] = conf.get(
            "save_disparity_map", False
        )

        application_schema = {
            "method": str,
            "save_disparity_map": bool,
            "activated": bool,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_is_activated(self):
        """
        Get the activated attribute

        :return: self.activated
        :rtype: bool
        """

        return self.activated

    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

        return 0

    def run(
        self,
        epipolar_disparity_map_left,
        epipolar_disparity_map_right,
        epipolar_images_left,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using zeros_padding method.

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

        res = None, None

        if not self.activated:
            logging.info("Disparity holes filling was not activated")
            res = epipolar_disparity_map_left, epipolar_disparity_map_right

        else:
            # Default orchestrator
            if orchestrator is None:
                # Create defaut sequential orchestrator for current application
                # be awere, no out_json will be shared between orchestrators
                # No files saved
                self.orchestrator = ocht.Orchestrator(
                    orchestrator_conf={"mode": "sequential"}
                )
            else:
                self.orchestrator = orchestrator

            # Current application is not available in MP mode
            if self.orchestrator.cluster.checked_conf_cluster["mode"] == "mp":
                logging.error(
                    "DenseMatchingFilling and multiprocessing cluster "
                    "is currently not supported"
                )
                raise RuntimeError(
                    "DenseMatchingFilling and multiprocessing cluster "
                    "is currently not supported"
                )

            if epipolar_disparity_map_left.dataset_type == "arrays":
                # Create CarsDataset Epipolar_disparity
                # Save Disparity map
                (
                    new_epipolar_disparity_map_left,
                    new_epipolar_disparity_map_right,
                ) = self.__register_dataset__(
                    epipolar_disparity_map_left,
                    epipolar_disparity_map_right,
                    self.save_disparity_map,
                    pair_folder,
                    app_name="zeros_padding",
                )

                # Get saving infos in order to save tiles when they are computed
                [
                    saving_info_left,
                    saving_info_right,
                ] = self.orchestrator.get_saving_infos(
                    [
                        new_epipolar_disparity_map_left,
                        new_epipolar_disparity_map_right,
                    ]
                )
                # Add infos to orchestrator.out_json
                updating_dict = {
                    application_constants.APPLICATION_TAG: {
                        pair_key: {
                            fd_cst.FILL_DISP_WITH_ZEROS_RUN_TAG: {},
                        }
                    }
                }
                self.orchestrator.update_out_info(updating_dict)
                logging.info(
                    "Fill missing disparities with zeros values"
                    ": number tiles: {}".format(
                        epipolar_disparity_map_right.shape[1]
                        * epipolar_disparity_map_right.shape[0]
                    )
                )
                # Generate disparity maps
                for col in range(epipolar_disparity_map_right.shape[1]):
                    for row in range(epipolar_disparity_map_right.shape[0]):
                        # get tile window and overlap
                        window = new_epipolar_disparity_map_left.tiling_grid[
                            row, col
                        ]
                        overlap_left = new_epipolar_disparity_map_left.overlaps[
                            row, col
                        ]
                        overlap_right = (
                            new_epipolar_disparity_map_right.overlaps[row, col]
                        )
                        # TODO get classification index
                        classif_index = []
                        # copy dataset
                        (
                            new_epipolar_disparity_map_left[row, col],
                            new_epipolar_disparity_map_right[row, col],
                        ) = self.orchestrator.cluster.create_task(
                            wrapper_fill_disparity, nout=2
                        )(
                            epipolar_disparity_map_left[row, col],
                            epipolar_disparity_map_right[row, col],
                            epipolar_images_left,
                            window,
                            overlap_left,
                            overlap_right,
                            classif_index=classif_index,
                            saving_info_left=saving_info_left,
                            saving_info_right=saving_info_right,
                        )

                res = (
                    new_epipolar_disparity_map_left,
                    new_epipolar_disparity_map_right,
                )

            else:
                logging.error(
                    "FillDisp application doesn't support "
                    "this input data format"
                )
        return res


def wrapper_fill_disparity(
    left_disp,
    right_disp,
    left_epi_image,
    window,
    overlap_left,
    overlap_right,
    classif_index,
    saving_info_left=None,
    saving_info_right=None,
):
    """
    Wrapper to copy previous disparity

    :param left_disp: left disparity map
    :type left_disp: xr.Dataset
    :param right_disp: right disparity map
    :type right_disp: xr.Dataset
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap_left: left overlap [row min, row max, col min col max]
    :type overlap_left: list
    :param overlap_right: left overlap  [row min, row max, col min col max]
    :type overlap_right: list
    :param class_index: class index according to the classification tag
    :type class_index: list
    :param saving_info_left: saving infos left
    :type saving_info_left: dict
    :param saving_info_right: saving infos right
    :type saving_info_right: dict

    :return: left disp map, right disp map
    :rtype: xr.Dataset, xr.Dataset
    """

    for index in classif_index:
        fd_tools.fill_disp_using_zeros_padding(left_disp, index)
    # compute right color image from right-left disparity map
    color_sec = dense_matching_tools.estimate_color_from_disparity(
        right_disp,
        left_epi_image,
    )

    # check bands
    if len(left_epi_image[cst.EPI_COLOR].values.shape) > 2:
        nb_bands = left_epi_image[cst.EPI_COLOR].values.shape[0]
        if cst.BAND not in right_disp.dims:
            right_disp.assign_coords({cst.BAND: np.arange(nb_bands)})

    # merge colors
    right_disp[cst.EPI_COLOR] = color_sec[cst.EPI_IMAGE]

    result = (copy.copy(left_disp), copy.copy(right_disp))
    # Fill with attributes
    cars_dataset.fill_dataset(
        result[0],
        saving_info=saving_info_left,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap_left),
    )

    if result[1] is not None:
        cars_dataset.fill_dataset(
            result[1],
            saving_info=saving_info_right,
            window=cars_dataset.window_array_to_dict(window),
            profile=None,
            attributes=None,
            overlaps=cars_dataset.overlap_array_to_dict(overlap_right),
        )

    return result
