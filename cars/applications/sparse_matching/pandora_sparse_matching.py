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
this module contains the pandora_sparse_matching application class.
"""

# pylint: disable=too-many-lines
# pylint: disable= C0302


import collections
import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas
import xarray as xr
from json_checker import And, Checker, Or

import cars.orchestrator.orchestrator as ocht
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
)
from cars.applications.sparse_matching import (
    sparse_matching_tools as pandora_tools,
)
from cars.applications.sparse_matching.sparse_matching import SparseMatching
from cars.core import constants_disparity as cst_disp
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


class PandoraSparseMatching(
    SparseMatching, short_name=["pandora"]
):  # pylint: disable=R0903,disable=R0902
    """
    Pandora low resolution class
    """

    def __init__(self, conf=None):
        """
        Init function of PandoraSparseMatching

        :param conf: configuration for matching and resolution
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

        self.method = self.used_config["method"]
        self.resolution = self.used_config["resolution"]
        self.loader_conf = self.used_config["loader_conf"]
        self.strip_margin = self.used_config["strip_margin"]
        self.disparity_margin = self.used_config["disparity_margin"]
        self.epipolar_error_upper_bound = self.used_config[
            "epipolar_error_upper_bound"
        ]
        self.epipolar_error_maximum_bias = self.used_config[
            "epipolar_error_maximum_bias"
        ]

        self.elevation_delta_lower_bound = self.used_config[
            "elevation_delta_lower_bound"
        ]
        self.elevation_delta_upper_bound = self.used_config[
            "elevation_delta_upper_bound"
        ]

        # minimum number of matches to continue with
        self.minimum_nb_matches = self.used_config["minimum_nb_matches"]

        # filter
        self.connection_val = self.used_config["connection_val"]
        self.nb_pts_threshold = self.used_config["nb_pts_threshold"]
        self.clusters_distance_threshold = self.used_config[
            "clusters_distance_threshold"
        ]
        self.filtered_elt_pos = self.used_config["filtered_elt_pos"]
        self.match_filter_knn = self.used_config["match_filter_knn"]
        self.match_filter_constant = self.used_config["match_filter_constant"]
        self.match_filter_mean_factor = self.used_config[
            "match_filter_mean_factor"
        ]
        self.match_filter_dev_factor = self.used_config[
            "match_filter_dev_factor"
        ]
        self.confidence_filtering = self.used_config["confidence_filtering"]
        self.disparity_bounds_estimation = self.used_config[
            "disparity_bounds_estimation"
        ]

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

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
        overloaded_conf["method"] = conf.get("method", "pandora")
        overloaded_conf["connection_val"] = conf.get("connection_val", 3.0)
        overloaded_conf["nb_pts_threshold"] = conf.get("nb_pts_threshold", 80)
        overloaded_conf["resolution"] = conf.get("resolution", 4)
        overloaded_conf["strip_margin"] = conf.get("strip_margin", 30)
        overloaded_conf["disparity_margin"] = conf.get("disparity_margin", 0.02)

        # confidence filtering parameters
        overloaded_conf["confidence_filtering"] = conf.get(
            "confidence_filtering", {}
        )

        # confidence filtering parameters
        overloaded_conf["disparity_bounds_estimation"] = conf.get(
            "disparity_bounds_estimation", {}
        )

        overloaded_conf["epipolar_error_upper_bound"] = conf.get(
            "epipolar_error_upper_bound", 10.0
        )
        overloaded_conf["epipolar_error_maximum_bias"] = conf.get(
            "epipolar_error_maximum_bias", 0.0
        )

        # minimum number of matches to continue with
        overloaded_conf["minimum_nb_matches"] = conf.get(
            "minimum_nb_matches", 100
        )

        overloaded_conf["elevation_delta_lower_bound"] = conf.get(
            "elevation_delta_lower_bound", None
        )
        overloaded_conf["elevation_delta_upper_bound"] = conf.get(
            "elevation_delta_upper_bound", None
        )

        # filter params
        overloaded_conf["clusters_distance_threshold"] = conf.get(
            "clusters_distance_threshold", None
        )
        overloaded_conf["filtered_elt_pos"] = conf.get(
            "filtered_elt_pos", False
        )
        overloaded_conf["match_filter_knn"] = conf.get("match_filter_knn", 25)
        overloaded_conf["match_filter_constant"] = conf.get(
            "match_filter_constant", 0.0
        )

        overloaded_conf["match_filter_mean_factor"] = conf.get(
            "match_filter_mean_factor", 1.3
        )

        overloaded_conf["match_filter_dev_factor"] = conf.get(
            "match_filter_dev_factor", 3.0
        )

        # check loader
        loader_conf = conf.get("loader_conf", None)

        # TODO modify, use loader directly
        pandora_loader = PandoraLoader(
            conf=loader_conf,
            method_name="census_sgm_sparse",
            use_cross_validation=True,
        )

        # Get params from loader
        self.loader = pandora_loader
        self.corr_config = collections.OrderedDict(pandora_loader.get_conf())

        overloaded_conf["loader_conf"] = self.corr_config

        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        application_schema = {
            "method": str,
            "disparity_margin": float,
            "epipolar_error_upper_bound": And(float, lambda x: x > 0),
            "epipolar_error_maximum_bias": And(float, lambda x: x >= 0),
            "minimum_nb_matches": And(int, lambda x: x > 0),
            "elevation_delta_lower_bound": Or(int, float, None),
            "elevation_delta_upper_bound": Or(int, float, None),
            "resolution": Or(int, list),
            "loader_conf": Or(dict, collections.OrderedDict, str, None),
            "strip_margin": And(int, lambda x: x > 0),
            "connection_val": And(float, lambda x: x > 0),
            "nb_pts_threshold": And(int, lambda x: x > 0),
            "clusters_distance_threshold": Or(None, float),
            "filtered_elt_pos": bool,
            "match_filter_knn": int,
            "match_filter_constant": Or(int, float),
            "match_filter_mean_factor": Or(int, float),
            "match_filter_dev_factor": Or(int, float),
            "save_intermediate_data": bool,
            "confidence_filtering": dict,
            "disparity_bounds_estimation": dict,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        self.check_conf_confidence_filtering(overloaded_conf)
        self.check_conf_disparity_bounds_estimation(overloaded_conf)

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

    def check_conf_confidence_filtering(self, overloaded_conf):
        """
        Check the confidence filtering conf
        """
        overloaded_conf["confidence_filtering"]["activated"] = overloaded_conf[
            "confidence_filtering"
        ].get("activated", True)
        overloaded_conf["confidence_filtering"]["upper_bound"] = (
            overloaded_conf["confidence_filtering"].get("upper_bound", 5)
        )
        overloaded_conf["confidence_filtering"]["lower_bound"] = (
            overloaded_conf["confidence_filtering"].get("lower_bound", -20)
        )
        overloaded_conf["confidence_filtering"]["risk_max"] = overloaded_conf[
            "confidence_filtering"
        ].get("risk_max", 60)
        overloaded_conf["confidence_filtering"]["nan_threshold"] = (
            overloaded_conf["confidence_filtering"].get("nan_threshold", 0.1)
        )
        overloaded_conf["confidence_filtering"]["win_nanratio"] = (
            overloaded_conf["confidence_filtering"].get("win_nanratio", 20)
        )
        overloaded_conf["confidence_filtering"]["win_mean_risk_max"] = (
            overloaded_conf["confidence_filtering"].get("win_mean_risk_max", 7)
        )

        confidence_filtering_schema = {
            "activated": bool,
            "upper_bound": int,
            "lower_bound": int,
            "risk_max": int,
            "nan_threshold": float,
            "win_nanratio": int,
            "win_mean_risk_max": int,
        }

        checker_confidence_filtering_schema = Checker(
            confidence_filtering_schema
        )
        checker_confidence_filtering_schema.validate(
            overloaded_conf["confidence_filtering"]
        )

    def check_conf_disparity_bounds_estimation(self, overloaded_conf):
        """
        Check the disparity bounds estimation conf
        """
        overloaded_conf["disparity_bounds_estimation"]["activated"] = (
            overloaded_conf["disparity_bounds_estimation"].get(
                "activated", True
            )
        )
        overloaded_conf["disparity_bounds_estimation"]["percentile"] = (
            overloaded_conf["disparity_bounds_estimation"].get("percentile", 1)
        )
        overloaded_conf["disparity_bounds_estimation"]["lower_margin"] = (
            overloaded_conf["disparity_bounds_estimation"].get(
                "lower_margin", 500
            )
        )
        overloaded_conf["disparity_bounds_estimation"]["upper_margin"] = (
            overloaded_conf["disparity_bounds_estimation"].get(
                "upper_margin", 1000
            )
        )

        disparity_bounds_estimation_schema = {
            "activated": bool,
            "percentile": Or(int, float),
            "upper_margin": int,
            "lower_margin": int,
        }

        checker_disparity_bounds_estimation_schema = Checker(
            disparity_bounds_estimation_schema
        )
        checker_disparity_bounds_estimation_schema.validate(
            overloaded_conf["disparity_bounds_estimation"]
        )

    def get_save_matches(self):
        """
        Get save_matches parameter

        :return: true is save_matches activated
        :rtype: bool

        """

        return self.save_intermediate_data

    def get_disparity_margin(self):
        """
        Get disparity margin corresponding to sparse matches

        :return: margin in percent

        """
        return self.disparity_margin

    def get_connection_val(self):
        """
        Get connection_val :
        distance to use to consider that two points are connected

        :return: connection_val
        :rtype:

        """
        return self.connection_val

    def get_nb_pts_threshold(self):
        """
        Get nb_pts_threshold :
        number of points to use to identify small clusters to filter

        :return: nb_pts_threshold

        """
        return self.nb_pts_threshold

    def get_minimum_nb_matches(self):
        """
        Get minimum_nb_matches :
        get the minimum number of matches

        :return: minimum_nb_matches

        """

        return self.minimum_nb_matches

    def get_match_filter_knn(self):
        """
        Get match_filter_knn :
        number of neighboors used to measure isolation of matches

        :return: match_filter_knn

        """
        return self.match_filter_knn

    def get_match_filter_constant(self):
        """
        Get get_match_filter_constant :
        constant in the formula to compute threshold of outliers

        :return: match_filter_constant

        """
        return self.match_filter_constant

    def get_match_filter_mean_factor(self):
        """
        Get match_filter_mean_factor :
        factor of mean in the formula
        to compute threshold of outliers

        :return: match_filter_mean_factor

        """
        return self.match_filter_mean_factor

    def get_match_filter_dev_factor(self):
        """
        Get match_filter_dev_factor :
        factor of deviation in the formula
        to compute threshold of outliers

        :return: match_filter_dev_factor

        """
        return self.match_filter_dev_factor

    def get_filtered_elt_pos(self):
        """
        Get filtered_elt_pos :
        if filtered_elt_pos is set to True, \
        the removed points positions in their original \
        epipolar images are returned, otherwise it is set to None

        :return: filtered_elt_pos

        """
        return self.filtered_elt_pos

    def get_clusters_dist_thresh(self):
        """
        Get clusters_distance_threshold :
        distance to use to consider if two points clusters \
        are far from each other or not (set to None to deactivate \
        this level of filtering)

        :return: clusters_distance_threshold

        """
        return self.clusters_distance_threshold

    def get_strip_margin(self):
        """
        Get strip margin corresponding to sparse matches

        :return: margin in percent

        """
        return self.strip_margin

    def get_epipolar_error_upper_bound(self):
        """
        Get epipolar error upper bound corresponding to sparse matches

        :return: margin

        """

        return self.epipolar_error_upper_bound

    def get_epipolar_error_maximum_bias(self):
        """
        Get epipolar error maximum bias corresponding to sparse matches

        :return: margin

        """

        return self.epipolar_error_maximum_bias

    def run(
        self,
        epipolar_image_left,
        epipolar_image_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        disp_to_alt_ratio=None,
        disp_min=None,
        disp_max=None,
    ):
        """
        Run PandoraSparseMatching application.

        Get matches using pandora in low resolution

        :param epipolar_image_left: tiled left epipolar CarsDataset contains:
        :param epipolar_image_left: tiled left epipolar CarsDataset contains:

               - N x M Delayed tiles. \
                   Each tile will be a future xarray Dataset containing:

                   - data with keys : "im", "msk", "color"
                   - attrs with keys: "margins" with "disp_min" and "disp_max"\
                       "transform", "crs", "valid_pixels", "no_data_mask",\
                       "no_data_img"
               - attributes containing:
                   "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_image_left: CarsDataset
        :param epipolar_image_right: tiled right epipolar CarsDataset contains:

               - N x M Delayed tiles. \
                   Each tile will be a future xarray Dataset containing:

                   - data with keys : "im", "msk", "color"
                   - attrs with keys: "margins" with "disp_min" and "disp_max"
                       "transform", "crs", "valid_pixels", "no_data_mask",
                       "no_data_img"
               - attributes containing:
                   "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_image_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param disp_to_alt_ratio: disp to alti ratio used for performance map
        :type disp_to_alt_ratio: float

        :return: left matches, right matches. Each CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future pandas DataFrame containing:
                - data : (L, 4) shape matches
            - attributes containing "disp_lower_bound",  "disp_upper_bound",\
                "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :rtype: Tuple(CarsDataset, CarsDataset)

        """

        # Default orchestrator
        if orchestrator is None:
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")

        if epipolar_image_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            pandora_epipolar_matches = cars_dataset.CarsDataset(
                "points", name="pandora_sparse_matching_" + pair_key
            )
            pandora_epipolar_matches.create_empty_copy(epipolar_image_left)

            # Update attributes to get epipolar info
            pandora_epipolar_matches.attributes.update(
                epipolar_image_left.attributes
            )

            pandora_epipolar_disparity_map = cars_dataset.CarsDataset(
                "arrays", name="pandora_sparse_matching_" + pair_key
            )

            pandora_epipolar_disparity_map.create_empty_copy(
                epipolar_image_left
            )
            pandora_epipolar_disparity_map.overlaps *= 0
            # Update attributes to get epipolar info
            pandora_epipolar_disparity_map.attributes.update(
                epipolar_image_left.attributes
            )

            # Save disparity maps
            if self.save_intermediate_data:
                safe_makedirs(pair_folder)

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp.tif"),
                    cst_disp.MAP,
                    pandora_epipolar_disparity_map,
                    cars_ds_name="epi_disp",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_mask.tif"),
                    "disp_msk",
                    pandora_epipolar_disparity_map,
                    cars_ds_name="epi_disp_mask",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_confidence.tif",
                    ),
                    cst_disp.CONFIDENCE,
                    pandora_epipolar_disparity_map,
                    cars_ds_name="confidence",
                    optional_data=True,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pandora_matches_left"),
                    None,
                    pandora_epipolar_matches,
                    cars_ds_name="epi_pandora_matches_left",
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info_matches] = self.orchestrator.get_saving_infos(
                [pandora_epipolar_matches]
            )

            [saving_info_disparity_map] = self.orchestrator.get_saving_infos(
                [pandora_epipolar_disparity_map]
            )

            # Add to replace list so tiles will be readable at the same time
            self.orchestrator.add_to_replace_lists(
                pandora_epipolar_matches,
                cars_ds_name="epi_pandora_matches_left",
            )

            # Compute disparity range
            if self.elevation_delta_lower_bound is None:
                disp_upper_bound = np.inf
            else:
                disp_upper_bound = (
                    -self.elevation_delta_lower_bound / disp_to_alt_ratio
                )
            disp_upper_bound = min(disp_upper_bound, disp_max)
            if self.elevation_delta_upper_bound is None:
                disp_lower_bound = -np.inf
            else:
                disp_lower_bound = (
                    -self.elevation_delta_upper_bound / disp_to_alt_ratio
                )
            disp_lower_bound = max(disp_lower_bound, disp_min)

            # Get the max dim to register the disparity map correctly
            row_max = pandora_epipolar_matches.tiling_grid[-1, -1][1]
            col_max = pandora_epipolar_matches.tiling_grid[-1, -1][3]
            dim_max = [row_max, col_max]

            # Generate disparity maps
            for col in range(pandora_epipolar_matches.shape[1]):
                for row in range(pandora_epipolar_matches.shape[0]):
                    # initialize list of matches
                    full_saving_info_matches = ocht.update_saving_infos(
                        saving_info_matches, row=row, col=col
                    )
                    full_saving_info_disp_map = ocht.update_saving_infos(
                        saving_info_disparity_map, row=row, col=col
                    )
                    # Compute matches
                    if type(None) not in (
                        type(epipolar_image_left[row, col]),
                        type(epipolar_image_right[row, col]),
                    ):
                        (
                            pandora_epipolar_matches[row, col],
                            pandora_epipolar_disparity_map[row, col],
                        ) = self.orchestrator.cluster.create_task(
                            compute_pandora_matches_wrapper, nout=2
                        )(
                            epipolar_image_left[row, col],
                            epipolar_image_right[row, col],
                            self.corr_config,
                            self.confidence_filtering,
                            dim_max=dim_max,
                            disp_upper_bound=disp_upper_bound,
                            disp_lower_bound=disp_lower_bound,
                            resolution=self.resolution,
                            saving_info_matches=full_saving_info_matches,
                            saving_info_disparity_map=full_saving_info_disp_map,
                        )

        else:
            logging.error(
                "PandoraSparseMatching application doesn't "
                "support this input data format"
            )

        return pandora_epipolar_matches, None


def compute_pandora_matches_wrapper(
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    corr_conf,
    confidence_filtering,
    dim_max,
    disp_upper_bound,
    disp_lower_bound,
    resolution,
    saving_info_matches=None,
    saving_info_disparity_map=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute pandora matches from image objects.
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
    :param disp_upper_bound: upper bound of disparity range
    :type disp_upper_bound: float
    :param disp_lower_bound: lower bound of disparity range
    :type disp_lower_bound: float
    :param corr_conf: pandora conf
    :type corr_conf: dict
    :param confidence_filtering: True to filter the disp map
    :type confidence_filtering: dict
    :param dim_max: the maximum dimensions
    :type dim_max: list
    :param resolution: resolution for downsampling
    :type resolution: int or list

    :return: Left pandora matches object,\
    Right pandora matches object (if exists)

    """

    list_matches = None
    if isinstance(resolution, list):
        for res in resolution:
            if res == np.max(resolution):
                matches, disp_map_dataset, window, profile = (
                    pandora_tools.pandora_matches(
                        left_image_object,
                        right_image_object,
                        corr_conf,
                        dim_max,
                        confidence_filtering,
                        disp_upper_bound,
                        disp_lower_bound,
                        res,
                    )
                )
            else:
                matches, _, _, _ = pandora_tools.pandora_matches(
                    left_image_object,
                    right_image_object,
                    corr_conf,
                    dim_max,
                    confidence_filtering,
                    disp_upper_bound,
                    disp_lower_bound,
                    res,
                )

            if list_matches is None:
                list_matches = matches
            else:
                list_matches = np.row_stack((list_matches, matches))
    else:
        list_matches, disp_map_dataset, window, profile = (
            pandora_tools.pandora_matches(
                left_image_object,
                right_image_object,
                corr_conf,
                dim_max,
                confidence_filtering,
                disp_upper_bound,
                disp_lower_bound,
                resolution,
            )
        )

    # Resample the matches in full resolution
    left_pandora_matches_dataframe = pandas.DataFrame(list_matches)

    cars_dataset.fill_dataframe(
        left_pandora_matches_dataframe,
        saving_info=saving_info_matches,
        attributes=None,
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        disp_map_dataset,
        saving_info=saving_info_disparity_map,
        window=window,
        profile=profile,
        attributes=None,
        overlaps=None,  # overlaps are removed
    )
    return left_pandora_matches_dataframe, disp_map_dataset
