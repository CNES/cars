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
from json_checker import Checker, Or

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_match_filling import (
    abstract_dense_match_filling_app as abstract_dense_match_fill,
)
from cars.applications.dense_match_filling import fill_disp_algo as fd_algo
from cars.applications.dense_match_filling import fill_disp_constants as fd_cst
from cars.applications.dense_match_filling import (
    fill_disp_wrappers as fd_wrappers,
)
from cars.core import constants as cst
from cars.data_structures import cars_dataset


class ZerosPadding(
    abstract_dense_match_fill.DenseMatchFilling, short_name=["zero_padding"]
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
        self.used_method = self.used_config["method"]
        self.classification = self.used_config["classification"]
        self.fill_valid_pixels = self.used_config["fill_valid_pixels"]

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

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
        overloaded_conf["method"] = conf.get("method", "zero_padding")

        overloaded_conf["classification"] = conf.get("classification", None)
        if isinstance(overloaded_conf["classification"], str):
            overloaded_conf["classification"] = [
                overloaded_conf["classification"]
            ]
        overloaded_conf["fill_valid_pixels"] = conf.get(
            "fill_valid_pixels", True
        )
        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        application_schema = {
            "method": str,
            "save_intermediate_data": bool,
            "classification": Or(None, [str]),
            "fill_valid_pixels": bool,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

        return 0

    def run(
        self,
        epipolar_disparity_map,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using zero_padding method.

        :param epipolar_disparity_map:  left to right disparity
        :type epipolar_disparity_map: CarsDataset
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
        res = None

        if not self.classification:
            logging.info("Disparity holes filling was not activated")
            res = epipolar_disparity_map

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

            if epipolar_disparity_map.dataset_type == "arrays":
                # Create CarsDataset Epipolar_disparity
                # Save Disparity map
                (new_epipolar_disparity_map) = self.__register_dataset__(
                    epipolar_disparity_map,
                    self.save_intermediate_data,
                    pair_folder,
                    pair_key,
                    app_name="zero_padding",
                    nodata_epi_disp=-9999,
                )

                # Get saving infos in order to save tiles when they are computed
                [
                    saving_info,
                ] = self.orchestrator.get_saving_infos(
                    [new_epipolar_disparity_map]
                )
                # Add infos to orchestrator.out_json
                updating_dict = {
                    application_constants.APPLICATION_TAG: {
                        fd_cst.FILL_DISP_WITH_ZEROS_RUN_TAG: {
                            pair_key: {},
                        }
                    }
                }
                self.orchestrator.update_out_info(updating_dict)
                logging.info(
                    "Fill missing disparities with zeros values"
                    ": number tiles: {}".format(
                        epipolar_disparity_map.shape[1]
                        * epipolar_disparity_map.shape[0]
                    )
                )
                # Generate disparity maps
                for col in range(epipolar_disparity_map.shape[1]):
                    for row in range(epipolar_disparity_map.shape[0]):
                        if epipolar_disparity_map[row, col] is not None:
                            # get tile window and overlap
                            window = new_epipolar_disparity_map.tiling_grid[
                                row, col
                            ]
                            overlap = new_epipolar_disparity_map.overlaps[
                                row, col
                            ]
                            # update saving infos  for potential replacement
                            full_saving_info = ocht.update_saving_infos(
                                saving_info, row=row, col=col
                            )

                            # copy dataset
                            (
                                new_epipolar_disparity_map[row, col]
                            ) = self.orchestrator.cluster.create_task(
                                fill_disparity_zeros_wrapper
                            )(
                                epipolar_disparity_map[row, col],
                                window,
                                overlap,
                                classif=self.classification,
                                fill_valid_pixels=self.fill_valid_pixels,
                                saving_info=full_saving_info,
                            )

                res = new_epipolar_disparity_map

            else:
                logging.error(
                    "FillDisp application doesn't support "
                    "this input data format"
                )
        return res


# pylint: disable=too-many-positional-arguments
def fill_disparity_zeros_wrapper(
    disp,
    window,
    overlap,
    classif,
    fill_valid_pixels,
    saving_info=None,
):
    """
    Wrapper to copy previous disparity

    :param disp: left to right disparity map
    :type disp: xr.Dataset
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap: overlap [row min, row max, col min col max]
    :type overlap: list
    :param classif: classification tags
    :type classif: list
    :param fill_valid_pixels: option to fill valid pixels
    :type fill_valid_pixels: bool
    :param saving_info: saving infos
    :type saving_info: dict

    :return: disp map
    :rtype: xr.Dataset, xr.Dataset
    """
    # Add a band to disparity dataset to memorize which pixels are filled
    disp = fd_wrappers.add_empty_filling_band(disp, ["zeros_padding"])
    fd_algo.fill_disp_using_zero_padding(disp, classif, fill_valid_pixels)
    result = copy.copy(disp)

    # Fill with attributes
    attributes = {
        cst.CROPPED_DISPARITY_RANGE: (ocht.get_disparity_range_cropped(disp))
    }
    cars_dataset.fill_dataset(
        result,
        saving_info=saving_info,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=attributes,
        overlaps=cars_dataset.overlap_array_to_dict(overlap),
    )

    return result
