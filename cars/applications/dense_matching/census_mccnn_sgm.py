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
this module contains the dense_matching application class.
"""
import collections

# Standard imports
import logging
import math
import os
from typing import Dict, Tuple

# Third party imports
import numpy as np
import xarray as xr
from json_checker import Checker, Or

import cars.applications.dense_matching.dense_matching_constants as dm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matching import dense_matching_tools
from cars.applications.dense_matching.dense_matching import DenseMatching
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
)

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


class CensusMccnnSgm(
    DenseMatching, short_name=["census_sgm", "mccnn_sgm"]
):  # pylint: disable=R0903,disable=R0902
    """
    Census SGM & MCCNN SGM matching class
    """

    def __init__(self, conf=None):
        """
        Init function of DenseMatching

        :param conf: configuration for matching
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.min_epi_tile_size = self.used_config["min_epi_tile_size"]
        self.max_epi_tile_size = self.used_config["max_epi_tile_size"]
        self.epipolar_tile_margin_in_percent = self.used_config[
            "epipolar_tile_margin_in_percent"
        ]
        self.use_sec_disp = self.used_config["use_sec_disp"]
        self.min_elevation_offset = self.used_config["min_elevation_offset"]
        self.max_elevation_offset = self.used_config["max_elevation_offset"]
        # Saving files
        self.save_disparity_map = self.used_config["save_disparity_map"]

        # Get params from loader
        self.loader = self.used_config["loader"]
        self.corr_config = self.used_config["loader_conf"]
        # init orchestrator
        self.orchestrator = None

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
        overloaded_conf["method"] = conf.get(
            "method", "census_sgm"
        )  # change it if census_sgm is not default
        # method called in dense_matching.py
        overloaded_conf["min_epi_tile_size"] = conf.get(
            "min_epi_tile_size", 300
        )
        overloaded_conf["max_epi_tile_size"] = conf.get(
            "max_epi_tile_size", 1500
        )
        overloaded_conf["epipolar_tile_margin_in_percent"] = conf.get(
            "epipolar_tile_margin_in_percent", 60
        )
        overloaded_conf["use_sec_disp"] = conf.get("use_sec_disp", False)
        overloaded_conf["min_elevation_offset"] = conf.get(
            "min_elevation_offset", None
        )
        overloaded_conf["max_elevation_offset"] = conf.get(
            "max_elevation_offset", None
        )
        # Saving files
        overloaded_conf["save_disparity_map"] = conf.get(
            "save_disparity_map", False
        )

        # check loader
        loader_conf = conf.get("loader_conf", None)
        loader = conf.get("loader", "pandora")
        # TODO modify, use loader directly
        pandora_loader = PandoraLoader(
            conf=loader_conf, method_name=overloaded_conf["method"]
        )
        overloaded_conf["loader"] = loader
        overloaded_conf["loader_conf"] = collections.OrderedDict(
            pandora_loader.get_conf()
        )

        application_schema = {
            "method": str,
            "min_epi_tile_size": int,
            "max_epi_tile_size": int,
            "epipolar_tile_margin_in_percent": int,
            "use_sec_disp": bool,
            "min_elevation_offset": Or(None, int),
            "max_elevation_offset": Or(None, int),
            "save_disparity_map": bool,
            "loader_conf": dict,
            "loader": str,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_margins(self, grid_left, disp_min=None, disp_max=None):
        """
        Get Margins needed by matching method, to use during resampling

        :param grid_left: left epipolar grid
        :param disp_min: minimum disparity
        :param disp_max: maximum disparity
        :return: margins, updated disp_min, updated disp_max

        """

        # get disp_to_alt_ratio
        disp_to_alt_ratio = grid_left.attributes["disp_to_alt_ratio"]

        # Check if we need to override disp_min
        if self.min_elevation_offset is not None:
            user_disp_min = self.min_elevation_offset / disp_to_alt_ratio
            if user_disp_min > disp_min:
                logging.warning(
                    (
                        "Overridden disparity minimum "
                        "= {:.3f} pix. (= {:.3f} m.) "
                        "is greater than disparity minimum estimated "
                        "in prepare step = {:.3f} pix. (or {:.3f} m.) "
                        "for current pair"
                    ).format(
                        user_disp_min,
                        self.min_elevation_offset,
                        disp_min,
                        disp_min * disp_to_alt_ratio,
                    )
                )
            disp_min = user_disp_min

        # Check if we need to override disp_max
        if self.max_elevation_offset is not None:
            user_disp_max = self.max_elevation_offset / disp_to_alt_ratio
            if user_disp_max < disp_max:
                logging.warning(
                    (
                        "Overridden disparity maximum "
                        "= {:.3f} pix. (or {:.3f} m.) "
                        "is lower than disparity maximum estimated "
                        "in prepare step = {:.3f} pix. (or {:.3f} m.) "
                        "for current pair"
                    ).format(
                        user_disp_max,
                        self.max_elevation_offset,
                        disp_max,
                        disp_max * disp_to_alt_ratio,
                    )
                )
            disp_max = user_disp_max

        logging.info(
            "Disparity range for current pair: [{:.3f} pix., {:.3f} pix.] "
            "(or [{:.3f} m., {:.3f} m.])".format(
                disp_min,
                disp_max,
                disp_min * disp_to_alt_ratio,
                disp_max * disp_to_alt_ratio,
            )
        )

        # round disp min and max
        disp_min = int(math.floor(disp_min))
        disp_max = int(math.ceil(disp_max))

        # Compute margins for the correlator
        # TODO use loader correlators
        margins = dense_matching_tools.get_margins(
            disp_min, disp_max, self.corr_config
        )

        return margins, disp_min, disp_max

    def get_optimal_tile_size(self, disp_min, disp_max, max_ram_per_worker):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_min: minimum disparity
        :param disp_max: maximum disparity
        :param max_ram_per_worker: maximum ram per worker
        :return: optimal tile size

        """

        # Get tiling params from static conf

        opt_epipolar_tile_size = (
            dense_matching_tools.optimal_tile_size_pandora_plugin_libsgm(
                disp_min,
                disp_max,
                self.min_epi_tile_size,
                self.max_epi_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
            )
        )

        return opt_epipolar_tile_size

    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        disp_min=None,
        disp_max=None,
        compute_disparity_masks=False,
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with xarray.Dataset ,
        corresponding to epipolar disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        :param epipolar_images_left: tiled left epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"
                        "transform", "crs", "valid_pixels", "no_data_mask",
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int

        :return: left disparity map, right disparity map: \
            Each CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size"

        :rtype: Tuple(CarsDataset, CarsDataset)
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        if epipolar_images_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            epipolar_disparity_map_left = cars_dataset.CarsDataset("arrays")
            epipolar_disparity_map_left.create_empty_copy(epipolar_images_left)
            epipolar_disparity_map_left.overlaps *= 0

            epipolar_disparity_map_right = cars_dataset.CarsDataset("arrays")
            epipolar_disparity_map_right.create_empty_copy(
                epipolar_images_right
            )
            epipolar_disparity_map_right.overlaps *= 0

            # Update attributes to get epipolar info
            epipolar_disparity_map_left.attributes.update(
                epipolar_images_left.attributes
            )

            # Save disparity maps
            if self.save_disparity_map:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_left.tif"),
                    cst_disp.MAP,
                    epipolar_disparity_map_left,
                    cars_ds_name="epi_disp_left",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_right.tif"),
                    cst_disp.MAP,
                    epipolar_disparity_map_right,
                    cars_ds_name="epi_disp_right",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_color_left.tif"),
                    cst.EPI_COLOR,
                    epipolar_disparity_map_left,
                    cars_ds_name="epi_disp_color_left",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_color_right.tif"),
                    cst.EPI_COLOR,
                    epipolar_disparity_map_right,
                    cars_ds_name="epi_disp_color_right",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_mask_left.tif"),
                    cst_disp.VALID,
                    epipolar_disparity_map_left,
                    cars_ds_name="epi_disp_mask_left",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_mask_right.tif"),
                    cst_disp.VALID,
                    epipolar_disparity_map_right,
                    cars_ds_name="epi_disp_mask_right",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_confidence_left.tif",
                    ),
                    cst_disp.CONFIDENCE,
                    epipolar_disparity_map_left,
                    cars_ds_name="confidence",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_confidence_right.tif",
                    ),
                    cst_disp.CONFIDENCE,
                    epipolar_disparity_map_right,
                    cars_ds_name="confidence",
                )

            # Get saving infos in order to save tiles when they are computed
            [
                saving_info_left,
                saving_info_right,
            ] = self.orchestrator.get_saving_infos(
                [epipolar_disparity_map_left, epipolar_disparity_map_right]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    pair_key: {
                        dm_cst.DENSE_MATCHING_RUN_TAG: {},
                    }
                }
            }
            self.orchestrator.update_out_info(updating_dict)
            logging.info(
                "Compute disparity: number tiles: {}".format(
                    epipolar_disparity_map_right.shape[1]
                    * epipolar_disparity_map_right.shape[0]
                )
            )
            # Generate disparity maps
            for col in range(epipolar_disparity_map_right.shape[1]):
                for row in range(epipolar_disparity_map_right.shape[0]):
                    # Compute disparity
                    (
                        epipolar_disparity_map_left[row, col],
                        epipolar_disparity_map_right[row, col],
                    ) = self.orchestrator.cluster.create_task(
                        compute_disparity, nout=2
                    )(
                        epipolar_images_left[row, col],
                        epipolar_images_right[row, col],
                        self.corr_config,
                        disp_min=disp_min,
                        disp_max=disp_max,
                        use_sec_disp=self.use_sec_disp,
                        saving_info_left=saving_info_left,
                        saving_info_right=saving_info_right,
                        compute_disparity_masks=compute_disparity_masks,
                    )
        else:
            logging.error(
                "DenseMatching application doesn't "
                "support this input data format"
            )

        return epipolar_disparity_map_left, epipolar_disparity_map_right


def compute_disparity(
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    corr_cfg: dict,
    disp_min=None,
    disp_max=None,
    use_sec_disp=False,
    saving_info_left=None,
    saving_info_right=None,
    compute_disparity_masks=False,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute disparity maps from image objects.
    This function will be run as a delayed task.

    User must provide saving infos to save properly created datasets

    :param left_image_object: tiled Left image
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :type left_image_object: xr.Dataset
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :param right_image_object: tiled Right image
    :type right_image_object: xr.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int
    :param use_sec_disp: Boolean activating the use of the secondary \
                         disparity map
    :type use_sec_disp: bool
    :param compute_disparity_masks: Compute all the disparity \
                        pandora masks(disable by default)
    :type compute_disparity_masks: bool
    :return: Left disparity object, Right disparity object (if exists)

    Returned objects are composed of :
        - dataset (None for right object if use_sec_disp not activated) with :

            - cst_disp.MAP
            - cst_disp.VALID
            - cst.EPI_COLOR
    """

    # Compute disparity
    # TODO : remove overwriting of EPI_MSK
    disp = dense_matching_tools.compute_disparity(
        left_image_object,
        right_image_object,
        corr_cfg,
        disp_min,
        disp_max,
        use_sec_disp=use_sec_disp,
        compute_disparity_masks=compute_disparity_masks,
    )

    color_sec = None
    if cst.STEREO_SEC in disp:
        # compute right color image from right-left disparity map
        color_sec = dense_matching_tools.estimate_color_from_disparity(
            disp[cst.STEREO_SEC],
            left_image_object,
        )

        # check bands
        if len(left_image_object[cst.EPI_COLOR].values.shape) > 2:
            nb_bands = left_image_object[cst.EPI_COLOR].values.shape[0]
            if cst.BAND not in disp[cst.STEREO_SEC].dims:
                disp[cst.STEREO_SEC].assign_coords(
                    {cst.BAND: np.arange(nb_bands)}
                )

        # merge colors
        disp[cst.STEREO_SEC][cst.EPI_COLOR] = color_sec[cst.EPI_IMAGE]

    # Fill with attributes
    left_disp_dataset = disp[cst.STEREO_REF]
    cars_dataset.fill_dataset(
        left_disp_dataset,
        saving_info=saving_info_left,
        window=cars_dataset.get_window_dataset(left_image_object),
        profile=cars_dataset.get_profile_rasterio(left_image_object),
        attributes=None,
        overlaps=None,  # overlaps are removed
    )

    right_disp_dataset = None
    if cst.STEREO_SEC in disp:
        right_disp_dataset = disp[cst.STEREO_SEC]
        cars_dataset.fill_dataset(
            right_disp_dataset,
            saving_info=saving_info_right,
            window=cars_dataset.get_window_dataset(right_image_object),
            profile=cars_dataset.get_profile_rasterio(right_image_object),
            attributes=None,
            overlaps=None,
        )

    return left_disp_dataset, right_disp_dataset
