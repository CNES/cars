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
This module contains the SIFT sparse matching method implementation.
"""

import logging

import numpy as np
import pandas
from json_checker import And, Checker, Or

from cars.applications.sparse_matching import sparse_matching_algo
from cars.applications.sparse_matching.methods import (
    abstract_sparse_matching_method as asmm,
)
from cars.core import constants as cst
from cars.data_structures import cars_dataset

AbstractSparseMatchingMethod = asmm.AbstractSparseMatchingMethod


def is_positive(value):
    """
    Check if a value is positive.

    :param value: value to check
    :type value: int or float
    :return: True if value is positive, False otherwise
    :rtype: bool
    """
    return value > 0


class SiftSparseMethod(AbstractSparseMatchingMethod, short_name=["sift"]):
    """
    Implementation of SIFT as a sparse matching method.
    """

    def __init__(self, conf):
        super().__init__(conf=conf)

        self.schema = {
            "method": str,
            "sift_matching_threshold": And(float, is_positive),
            "sift_n_octave": And(int, is_positive),
            "sift_n_scale_per_octave": And(int, is_positive),
            "sift_peak_threshold": Or(float, int),
            "sift_edge_threshold": float,
            "sift_magnification": And(float, is_positive),
            "sift_window_size": And(int, is_positive),
            "sift_back_matching": bool,
            "used_band": str,
        }

        self.used_config = self.check_conf(conf)

        self.method = self.used_config["method"]
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
        self.sift_window_size = self.used_config["sift_window_size"]
        self.sift_back_matching = self.used_config["sift_back_matching"]

        self.used_band = self.used_config["used_band"]

    def check_conf(self, conf):
        """
        Merge user configuration with default values and validate schema.
        Extra keys in conf are preserved and ignored during schema validation.
        """

        if conf is None:
            conf = {}

        default_conf = {
            "method": "sift",
            "sift_matching_threshold": 0.7,
            "sift_n_octave": 8,
            "sift_n_scale_per_octave": 3,
            "sift_peak_threshold": 4.0,
            "sift_edge_threshold": 10.0,
            "sift_magnification": 7.0,
            "sift_window_size": 2,
            "sift_back_matching": True,
            "used_band": "b0",
        }

        used_conf = default_conf.copy()
        used_conf.update(conf)

        # Validate only keys defined in schema
        conf_to_check = {k: used_conf[k] for k in self.schema if k in used_conf}

        checker = Checker(self.schema)
        checker.validate(conf_to_check)

        return conf_to_check

    def get_required_bands(self):
        """
        Get bands required by this method.

        :return: required bands for left and right image
        :rtype: dict
        """
        return {
            "left": [self.used_band],
            "right": [self.used_band],
        }

    def run(
        self,
        left_image_object,
        right_image_object,
        saving_info_left=None,
        disp_lower_bound=None,
        disp_upper_bound=None,
        classif_bands_to_mask=None,
    ):
        """
        Compute and filter sparse matches for one pair of epipolar tiles.
        """

        # TODO: remove overwriting of EPI_MSK
        saved_left_mask = np.copy(left_image_object[cst.EPI_MSK].values)
        saved_right_mask = np.copy(right_image_object[cst.EPI_MSK].values)

        matches = sparse_matching_algo.dataset_matching(
            left_image_object,
            right_image_object,
            self.used_band,
            matching_threshold=self.sift_matching_threshold,
            n_octave=self.sift_n_octave,
            n_scale_per_octave=self.sift_n_scale_per_octave,
            peak_threshold=self.sift_peak_threshold,
            edge_threshold=self.sift_edge_threshold,
            magnification=self.sift_magnification,
            window_size=self.sift_window_size,
            backmatching=self.sift_back_matching,
            disp_lower_bound=disp_lower_bound,
            disp_upper_bound=disp_upper_bound,
            classif_bands_to_mask=classif_bands_to_mask,
        )

        if disp_lower_bound is not None and disp_upper_bound is not None:
            filtered_nb_matches = matches.shape[0]

            matches = matches[matches[:, 2] - matches[:, 0] >= disp_lower_bound]
            matches = matches[matches[:, 2] - matches[:, 0] <= disp_upper_bound]

            logging.debug(
                "{} matches discarded because they fall outside of disparity "
                "range defined by --elevation_delta_lower_bound and "
                "--elevation_delta_upper_bound: [{} pix., {} pix.]".format(
                    filtered_nb_matches - matches.shape[0],
                    disp_lower_bound,
                    disp_upper_bound,
                )
            )
        else:
            logging.debug("Matches outside disparity range were not filtered")

        left_matches_dataframe = pandas.DataFrame(matches)

        # recover initial mask data in input images
        # TODO remove with proper dataset creation
        left_image_object[cst.EPI_MSK].values = saved_left_mask
        right_image_object[cst.EPI_MSK].values = saved_right_mask

        cars_dataset.fill_dataframe(
            left_matches_dataframe,
            saving_info=saving_info_left,
            attributes=None,
        )

        return left_matches_dataframe
