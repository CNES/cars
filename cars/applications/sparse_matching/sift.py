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

# pylint: disable= C0302

# Standard imports
import logging
import math
import os
from typing import Dict, Tuple

# Third party imports
import numpy as np
import pandas
import xarray as xr
from json_checker import And, Checker, Or

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants

# CARS imports
from cars.applications.sparse_matching import sparse_matching_tools
from cars.applications.sparse_matching.sparse_matching import SparseMatching
from cars.core import constants as cst
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


class Sift(SparseMatching, short_name="sift"):
    """
    SparseMatching
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of SparseMatching

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.disparity_margin = self.used_config["disparity_margin"]
        self.elevation_delta_lower_bound = self.used_config[
            "elevation_delta_lower_bound"
        ]
        self.elevation_delta_upper_bound = self.used_config[
            "elevation_delta_upper_bound"
        ]
        self.strip_margin = self.used_config["strip_margin"]
        self.epipolar_error_upper_bound = self.used_config[
            "epipolar_error_upper_bound"
        ]
        self.epipolar_error_maximum_bias = self.used_config[
            "epipolar_error_maximum_bias"
        ]

        # outlier filtering (used after application run, to filter matches)
        self.disparity_outliers_rejection_percent = self.used_config[
            "disparity_outliers_rejection_percent"
        ]

        # minimum number of matches to continue with
        self.minimum_nb_matches = self.used_config["minimum_nb_matches"]

        # sifts
        self.sift_matching_threshold = self.used_config[
            "sift_matching_threshold"
        ]
        self.sift_n_octave = self.used_config["sift_n_octave"]
        self.sift_n_scale_per_octave = self.used_config[
            "sift_n_scale_per_octave"
        ]
        self.sift_peak_threshold = self.used_config["sift_peak_threshold"]
        self.sift_edge_threshold = self.used_config["sift_edge_threshold"]
        self.sift_magnification = self.used_config["sift_magnification"]
        self.sift_back_matching = self.used_config["sift_back_matching"]

        # sifts filter
        self.matches_filter_knn = self.used_config["matches_filter_knn"]
        self.matches_filter_dev_factor = self.used_config[
            "matches_filter_dev_factor"
        ]

        # Saving files
        self.save_matches = self.used_config["save_matches"]

        # Init orchestrator
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
        overloaded_conf["method"] = conf.get("method", "sift")
        overloaded_conf["disparity_margin"] = conf.get("disparity_margin", 0.02)
        overloaded_conf["elevation_delta_lower_bound"] = conf.get(
            "elevation_delta_lower_bound", None
        )
        overloaded_conf["elevation_delta_upper_bound"] = conf.get(
            "elevation_delta_upper_bound", None
        )
        overloaded_conf["strip_margin"] = conf.get("strip_margin", 10)
        overloaded_conf["epipolar_error_upper_bound"] = conf.get(
            "epipolar_error_upper_bound", 10.0
        )
        overloaded_conf["epipolar_error_maximum_bias"] = conf.get(
            "epipolar_error_maximum_bias", 0.0
        )
        # outliers rejections used for matches filtering
        overloaded_conf["disparity_outliers_rejection_percent"] = conf.get(
            "disparity_outliers_rejection_percent", 0.1
        )

        # minimum number of matches to continue with
        overloaded_conf["minimum_nb_matches"] = conf.get(
            "minimum_nb_matches", 100
        )

        # sifts params
        overloaded_conf["sift_matching_threshold"] = conf.get(
            "sift_matching_threshold", 0.6
        )
        overloaded_conf["sift_n_octave"] = conf.get("sift_n_octave", 8)
        overloaded_conf["sift_n_scale_per_octave"] = conf.get(
            "sift_n_scale_per_octave", 3
        )
        overloaded_conf["sift_peak_threshold"] = conf.get(
            "sift_peak_threshold", None
        )
        overloaded_conf["sift_edge_threshold"] = conf.get(
            "sift_edge_threshold", 5.0
        )
        overloaded_conf["sift_magnification"] = conf.get(
            "sift_magnification", 2.0
        )
        overloaded_conf["sift_back_matching"] = conf.get(
            "sift_back_matching", True
        )

        # sifts filter params
        overloaded_conf["matches_filter_knn"] = conf.get(
            "matches_filter_knn", 25
        )
        overloaded_conf["matches_filter_dev_factor"] = conf.get(
            "matches_filter_dev_factor", 3.0
        )

        # Saving files
        overloaded_conf["save_matches"] = conf.get("save_matches", False)
        self.save_matches = overloaded_conf["save_matches"]

        sparse_matching_schema = {
            "method": str,
            "disparity_margin": float,
            "disparity_outliers_rejection_percent": And(
                float, lambda x: x >= 0, lambda x: x <= 1
            ),
            "minimum_nb_matches": And(int, lambda x: x > 0),
            "elevation_delta_lower_bound": Or(int, float, None),
            "elevation_delta_upper_bound": Or(int, float, None),
            "strip_margin": And(int, lambda x: x > 0),
            "epipolar_error_upper_bound": And(float, lambda x: x > 0),
            "epipolar_error_maximum_bias": And(float, lambda x: x >= 0),
            "sift_matching_threshold": And(float, lambda x: x > 0),
            "sift_n_octave": And(int, lambda x: x > 0),
            "sift_n_scale_per_octave": And(int, lambda x: x > 0),
            "sift_peak_threshold": Or(float, None),
            "sift_edge_threshold": float,
            "sift_magnification": And(float, lambda x: x > 0),
            "sift_back_matching": bool,
            "matches_filter_knn": int,
            "matches_filter_dev_factor": Or(int, float),
            "save_matches": bool,
        }

        # Check conf
        checker = Checker(sparse_matching_schema)
        checker.validate(overloaded_conf)

        # Check consistency between bounds for elevation delta
        elevation_delta_lower_bound = overloaded_conf[
            "elevation_delta_lower_bound"
        ]
        elevation_delta_upper_bound = overloaded_conf[
            "elevation_delta_upper_bound"
        ]
        if None not in (
            elevation_delta_lower_bound,
            elevation_delta_upper_bound,
        ):
            if elevation_delta_lower_bound > elevation_delta_upper_bound:
                raise ValueError(
                    "Upper bound must be bigger than "
                    "lower bound for expected elevation delta"
                )

        return overloaded_conf

    def get_save_matches(self):
        """
        Get save_matches parameter

        :return: true is save_matches activated
        :rtype: bool
        """

        return self.save_matches

    def get_disparity_margin(self):
        """
        Get disparity margin corresponding to sparse matches

        :return: margin in percent

        """
        return self.disparity_margin

    def get_matches_filter_knn(self):
        """
        Get matches_filter_knn :
        number of neighboors used to measure isolation of matches

        :return: matches_filter_knn

        """
        return self.matches_filter_knn

    def get_matches_filter_dev_factor(self):
        """
        Get matches_filter_dev_factor :
        factor of deviation in the formula
        to compute threshold of outliers

        :return: matches_filter_dev_factor

        """
        return self.matches_filter_dev_factor

    def get_margins_fun(self, disp_min=None, disp_max=None):
        """
        Get margins function to use in resampling

        :param disp_min: disp min for info
        :param disp_max: disp max for info

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

        # Compute margins
        corner = ["left", "up", "right", "down"]
        data = np.zeros(len(corner))
        col = np.arange(len(corner))
        margins = xr.Dataset(
            {"left_margin": (["col"], data)}, coords={"col": col}
        )
        margins["right_margin"] = xr.DataArray(data, dims=["col"])

        left_margin = self.strip_margin
        right_margin = left_margin + int(
            math.floor(
                self.epipolar_error_upper_bound
                + self.epipolar_error_maximum_bias
            )
        )

        # Compute margins for left region
        margins["left_margin"].data = [0, left_margin, 0, left_margin]

        # Compute margins for right region
        margins["right_margin"].data = [0, right_margin, 0, right_margin]

        # add disp range info
        margins.attrs["disp_min"] = disp_min
        margins.attrs["disp_max"] = disp_max

        logging.info(
            "Margins added to left region for matching: {}".format(
                margins["left_margin"].data
            )
        )

        logging.info(
            "Margins added to right region for matching: {}".format(
                margins["right_margin"].data
            )
        )

        def margins_wrapper(  # pylint: disable=unused-argument
            row_min, row_max, col_min, col_max
        ):
            """
            Generates margins Dataset used in resampling

            :param row_min: row min
            :param row_max: row max
            :param col_min: col min
            :param col_max: col max

            :return: margins
            :rtype: xr.Dataset
            """

            # Constant margins for all tiles
            return margins

        return margins_wrapper

    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        disp_to_alt_ratio,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with pandas.DataFrame ,
        corresponding to epipolar 2D disparities, on the same geometry
        that epipolar_images_left and epipolar_images_right.

        :param epipolar_images_left: tiled left epipolar. CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"
                        "transform", "crs", "valid_pixels", "no_data_mask",
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar.CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:"largest_epipolar_region", \
                  "opt_epipolar_tile_size"
        :type epipolar_images_right: CarsDataset
        :param disp_to_alt_ratio: disp to alti ratio
        :type disp_to_alt_ratio: float
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair key id
        :type pair_key: str

        :return left matches, right matches. Each CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future pandas DataFrame containing:
                - data : (L, 4) shape matches
            - attributes containing "disp_lower_bound",  "disp_upper_bound",\
                "elevation_delta_lower_bound","elevation_delta_upper_bound"

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
            epipolar_disparity_map_left = cars_dataset.CarsDataset(
                "points", name="sparse_matching_" + pair_key
            )
            epipolar_disparity_map_left.create_empty_copy(epipolar_images_left)

            # Update attributes to get epipolar info
            epipolar_disparity_map_left.attributes.update(
                epipolar_images_left.attributes
            )
            # check sift_peak_threshold with image type
            # only if sift_peak_threshold is None
            tmp_sift_peak_threshold = self.sift_peak_threshold
            if not self.sift_peak_threshold:
                logging.info("The sift_peak_threshold is set to auto-mode.")
                # sift_peak_threshold is None or not specified
                # check input type
                if np.issubdtype(
                    epipolar_disparity_map_left.attributes["image_type"],
                    np.uint8,
                ):
                    tmp_sift_peak_threshold = 1
                else:
                    tmp_sift_peak_threshold = 20
                logging.info(
                    "The sift_peak_threshold will be set to {}.".format(
                        tmp_sift_peak_threshold
                    )
                )
                self.sift_peak_threshold = tmp_sift_peak_threshold

            # Save disparity maps
            if self.save_matches:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_matches_left"),
                    None,
                    epipolar_disparity_map_left,
                    cars_ds_name="epi_matches_left",
                )

            # Compute disparity range
            if self.elevation_delta_lower_bound is None:
                disp_upper_bound = np.inf
            else:
                disp_upper_bound = (
                    -self.elevation_delta_lower_bound / disp_to_alt_ratio
                )
            if self.elevation_delta_upper_bound is None:
                disp_lower_bound = -np.inf
            else:
                disp_lower_bound = (
                    -self.elevation_delta_upper_bound / disp_to_alt_ratio
                )

            attributes = {
                "disp_lower_bound": disp_lower_bound,
                "disp_upper_bound": disp_upper_bound,
                "elevation_delta_lower_bound": self.elevation_delta_lower_bound,
                "elevation_delta_upper_bound": self.elevation_delta_upper_bound,
            }

            epipolar_disparity_map_left.attributes.update(attributes)

            # Get saving infos in order to save tiles when they are computed
            [saving_info_left] = self.orchestrator.get_saving_infos(
                [epipolar_disparity_map_left]
            )

            # Update orchestrator out_json
            updating_infos = {
                application_constants.APPLICATION_TAG: {
                    pair_key: {
                        sm_cst.SPARSE_MATCHING_RUN_TAG: {
                            sm_cst.DISP_LOWER_BOUND: disp_lower_bound,
                            sm_cst.DISP_UPPER_BOUND: disp_upper_bound,
                        },
                    }
                }
            }
            orchestrator.update_out_info(updating_infos)
            logging.info(
                "Generate disparity: Number tiles: {}".format(
                    epipolar_disparity_map_left.shape[1]
                    * epipolar_disparity_map_left.shape[0]
                )
            )

            # Add to replace list so tiles will be readable at the same time
            self.orchestrator.add_to_replace_lists(
                epipolar_disparity_map_left, cars_ds_name="epi_matches_left"
            )

            # Generate disparity maps
            for row in range(epipolar_disparity_map_left.shape[0]):
                # initialize list of matches
                full_saving_info_left = ocht.update_saving_infos(
                    saving_info_left, row=row, col=0
                )
                # Compute matches
                if type(None) not in (
                    type(epipolar_images_left[row, 0]),
                    type(epipolar_images_right[row, 0]),
                ):
                    (
                        epipolar_disparity_map_left[row, 0]
                    ) = self.orchestrator.cluster.create_task(
                        compute_matches_wrapper, nout=1
                    )(
                        epipolar_images_left[row, 0],
                        epipolar_images_right[row, 0],
                        matching_threshold=self.sift_matching_threshold,
                        n_octave=self.sift_n_octave,
                        n_scale_per_octave=self.sift_n_scale_per_octave,
                        peak_threshold=tmp_sift_peak_threshold,
                        edge_threshold=self.sift_edge_threshold,
                        magnification=self.sift_magnification,
                        backmatching=self.sift_back_matching,
                        disp_lower_bound=disp_lower_bound,
                        disp_upper_bound=disp_upper_bound,
                        saving_info_left=full_saving_info_left,
                    )

        else:
            logging.error(
                "SparseMatching application doesn't "
                "support this input data format"
            )

        return epipolar_disparity_map_left, None

    def filter_matches(
        self,
        epipolar_matches_left,
        grid_left,
        grid_right,
        orchestrator=None,
        pair_key="pair_0",
        pair_folder=None,
        save_matches=False,
    ):
        """
        Transform matches CarsDataset to numpy matches, and filters matches

        :param cars_orchestrator: orchestrator
        :param epipolar_matches_left: matches. CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future pandas DataFrame containing:

                - data : (L, 4) shape matches
            - attributes containing "disp_lower_bound",  "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"
        :type epipolar_matches_left: CarsDataset
        :param grid_left: left epipolar grid
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid
        :type grid_right: CarsDataset
        :param save_matches: true is matches needs to be saved
        :type save_matches: bool

        :return filtered matches
        :rtype: np.ndarray

        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            cars_orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            cars_orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(cars_orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        epipolar_error_upper_bound = self.epipolar_error_upper_bound
        epipolar_error_maximum_bias = self.epipolar_error_maximum_bias

        # Compute grid correction

        # Concatenated matches
        list_matches = []
        for row in range(epipolar_matches_left.shape[0]):
            for col in range(epipolar_matches_left.shape[1]):
                # CarsDataset containing Pandas DataFrame, not Delayed anymore
                if epipolar_matches_left[row, col] is not None:
                    epipolar_matches = epipolar_matches_left[
                        row, col
                    ].to_numpy()
                    sensor_matches = AbstractGeometry.matches_to_sensor_coords(
                        grid_left,
                        grid_right,
                        epipolar_matches,
                        cst.MATCHES_MODE,
                    )
                    sensor_matches = np.concatenate(sensor_matches, axis=1)
                    matches = np.concatenate(
                        [
                            epipolar_matches,
                            sensor_matches,
                        ],
                        axis=1,
                    )
                    list_matches.append(matches)

        matches = np.concatenate(list_matches)

        raw_nb_matches = matches.shape[0]

        logging.info(
            "Raw number of matches found: {} matches".format(raw_nb_matches)
        )

        # Export matches
        raw_matches_array_path = None
        if save_matches:
            logging.info("Writing raw matches file")
            raw_matches_array_path = os.path.join(
                pair_folder, "raw_matches.npy"
            )
            np.save(raw_matches_array_path, matches)

        # Filter matches that are out of margin
        if epipolar_error_maximum_bias == 0:
            epipolar_median_shift = 0
        else:
            epipolar_median_shift = np.median(matches[:, 3] - matches[:, 1])

        matches = matches[
            ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
            >= -epipolar_error_upper_bound
        ]
        matches = matches[
            ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
            <= epipolar_error_upper_bound
        ]

        matches_discarded_message = (
            "{} matches discarded because their epipolar error "
            "is greater than --epipolar_error_upper_bound = {} pix"
        ).format(raw_nb_matches - matches.shape[0], epipolar_error_upper_bound)

        if epipolar_error_maximum_bias != 0:
            matches_discarded_message += (
                " considering a shift of {} pix".format(epipolar_median_shift)
            )

        logging.info(matches_discarded_message)

        filtered_matches_array_path = None
        if save_matches:
            logging.info("Writing filtered matches file")
            filtered_matches_array_path = os.path.join(
                pair_folder, "filtered_matches.npy"
            )
            np.save(filtered_matches_array_path, matches)

        # Retrieve number of matches
        nb_matches = matches.shape[0]

        # Check if we have enough matches
        # TODO: we could also make it a warning and continue
        # with uncorrected grid
        # and default disparity range
        if nb_matches < self.minimum_nb_matches:
            error_message_matches = (
                "Insufficient amount of matches found ({} < {}), "
                "can not safely estimate epipolar error correction "
                " and disparity range".format(
                    nb_matches, self.minimum_nb_matches
                )
            )
            logging.error(error_message_matches)
            raise ValueError(error_message_matches)

        logging.info(
            "Number of matches kept for epipolar "
            "error correction: {} matches".format(nb_matches)
        )

        # Compute epipolar error
        epipolar_error = matches[:, 1] - matches[:, 3]
        epi_error_mean = np.mean(epipolar_error)
        epi_error_std = np.std(epipolar_error)
        epi_error_max = np.max(np.fabs(epipolar_error))
        logging.info(
            "Epipolar error before correction: mean = {:.3f} pix., "
            "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
                epi_error_mean,
                epi_error_std,
                epi_error_max,
            )
        )

        # Update orchestrator out_json
        raw_matches_infos = {
            application_constants.APPLICATION_TAG: {
                pair_key: {
                    sm_cst.MATCHES_FILTERING_TAG: {
                        sm_cst.RAW_MATCHES_TAG: raw_matches_array_path,
                        sm_cst.FILTERED_MATCHES_TAG: (
                            filtered_matches_array_path
                        ),
                        sm_cst.NUMBER_MATCHES_TAG: nb_matches,
                        sm_cst.RAW_NUMBER_MATCHES_TAG: raw_nb_matches,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MEAN: epi_error_mean,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_STD: epi_error_std,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MAX: epi_error_max,
                    }
                }
            }
        }
        cars_orchestrator.update_out_info(raw_matches_infos)

        return matches


def compute_matches_wrapper(
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    matching_threshold=None,
    n_octave=None,
    n_scale_per_octave=None,
    peak_threshold=None,
    edge_threshold=None,
    magnification=None,
    backmatching=None,
    disp_lower_bound=None,
    disp_upper_bound=None,
    saving_info_left=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute matches from image objects.
    This function will be run as a delayed task.

    User must provide saving infos to save properly created datasets

    :param left_image_object: tiled Left image dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :type left_image_object: xr.Dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :param right_image_object: tiled Right image
    :type right_image_object: xr.Dataset


    :return: Left matches object, Right matches object (if exists)

    Returned objects are composed of :

        - dataframe (None for right object) with :
            - TODO
    """

    # Create mask
    # TODO : remove overwriting of EPI_MSK
    saved_left_mask = np.copy(left_image_object[cst.EPI_MSK].values)
    saved_right_mask = np.copy(right_image_object[cst.EPI_MSK].values)

    # Compute matches
    matches = sparse_matching_tools.dataset_matching(
        left_image_object,
        right_image_object,
        matching_threshold=matching_threshold,
        n_octave=n_octave,
        n_scale_per_octave=n_scale_per_octave,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        magnification=magnification,
        backmatching=backmatching,
        disp_lower_bound=disp_lower_bound,
        disp_upper_bound=disp_upper_bound,
    )

    # Filter matches outside disparity range
    if disp_lower_bound is not None and disp_upper_bound is not None:
        filtered_nb_matches = matches.shape[0]

        matches = matches[matches[:, 2] - matches[:, 0] >= disp_lower_bound]
        matches = matches[matches[:, 2] - matches[:, 0] <= disp_upper_bound]

        logging.debug(
            "{} matches discarded because they fall outside of disparity range "
            "defined by --elevation_delta_lower_bound and "
            "--elevation_delta_upper_bound: [{} pix., {} pix.]".format(
                filtered_nb_matches - matches.shape[0],
                disp_lower_bound,
                disp_upper_bound,
            )
        )
    else:
        logging.debug("Matches outside disparity range were not filtered")

    # convert to Dataframe
    left_matches_dataframe = pandas.DataFrame(matches)

    # recover initial mask data in input images
    # TODO remove with proper dataset creation
    left_image_object[cst.EPI_MSK].values = saved_left_mask
    right_image_object[cst.EPI_MSK].values = saved_right_mask

    cars_dataset.fill_dataframe(
        left_matches_dataframe, saving_info=saving_info_left, attributes=None
    )

    return left_matches_dataframe
