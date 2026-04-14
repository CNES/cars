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
this module contains the basic sparse matching application class.
"""

# pylint: disable= C0302

# Standard imports
import logging
import math
import os

# Third party imports
import numpy as np
import xarray as xr
from json_checker import And, Checker, Or
from shareloc.geofunctions.rectification_grid import RectificationGrid

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
import cars.applications.sparse_matching.sparse_matching_wrappers as sm_wrapper
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants

# CARS imports
from cars.applications.sparse_matching.abstract_sparse_matching_app import (
    SparseMatching,
)
from cars.core import constants as cst
from cars.core import inputs, tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


class BasicSparseMatchingApplication(
    SparseMatching,
    short_name=["basic"],
):
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

        self.used_config["application"] = "basic"

        # app-owned parameters
        self.elevation_delta_lower_bound = self.used_config[
            "elevation_delta_lower_bound"
        ]
        self.elevation_delta_upper_bound = self.used_config[
            "elevation_delta_upper_bound"
        ]
        self.tile_margin = self.used_config["tile_margin"]
        self.epipolar_error_upper_bound = self.used_config[
            "epipolar_error_upper_bound"
        ]
        self.epipolar_error_maximum_bias = self.used_config[
            "epipolar_error_maximum_bias"
        ]

        self.minimum_nb_matches = self.used_config["minimum_nb_matches"]
        self.decimation_factor = self.used_config["decimation_factor"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]
        self.disparity_bounds_estimation = self.used_config[
            "disparity_bounds_estimation"
        ]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Merge user configuration with default values and validate schema.
        Extra keys in conf are preserved and ignored during schema validation.

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """
        if conf is None:
            conf = {}

        self.schema = {
            "application": And(str, lambda x: x in self.available_applications),
            "decimation_factor": And(int, lambda x: x > 0),
            "elevation_delta_lower_bound": Or(int, float, None),
            "elevation_delta_upper_bound": Or(int, float, None),
            "tile_margin": And(int, lambda x: x > 0),
            "epipolar_error_upper_bound": Or(float, str),
            "epipolar_error_maximum_bias": Or(float, str),
            "minimum_nb_matches": And(int, lambda x: x > 0),
            "save_intermediate_data": bool,
            "disparity_bounds_estimation": dict,
        }

        default_conf = {
            "application": "basic",
            "decimation_factor": 30,
            "elevation_delta_lower_bound": None,
            "elevation_delta_upper_bound": None,
            "tile_margin": 10,
            "epipolar_error_upper_bound": "auto",
            "epipolar_error_maximum_bias": "auto",
            "minimum_nb_matches": 90,
            "save_intermediate_data": False,
            "disparity_bounds_estimation": {},
        }

        used_conf = default_conf.copy()
        used_conf.update(conf)
        used_conf.update(self.sparse_matching_method.used_config)

        complete_schema = self.schema.copy()
        complete_schema.update(self.sparse_matching_method.schema)

        checker = Checker(complete_schema)
        checker.validate(used_conf)

        self.check_conf_disparity_bounds_estimation(used_conf)

        if None not in (
            used_conf["elevation_delta_lower_bound"],
            used_conf["elevation_delta_upper_bound"],
        ) and (
            used_conf["elevation_delta_lower_bound"]
            > used_conf["elevation_delta_upper_bound"]
        ):
            raise ValueError(
                "Upper bound must be bigger than lower bound "
                "for expected elevation delta"
            )

        self.check_epipolar_error_values(used_conf)

        return used_conf

    def check_epipolar_error_values(self, used_conf):
        """check the epipolar_error values"""
        epipolar_error_upper_bound = used_conf["epipolar_error_upper_bound"]
        if isinstance(epipolar_error_upper_bound, str):
            if epipolar_error_upper_bound != "auto":
                raise RuntimeError(
                    "The value of epipolar_error_upper_bound "
                    "must be a float bigger than 0 or auto"
                )

        epipolar_error_maximum_bias = used_conf["epipolar_error_maximum_bias"]
        if isinstance(epipolar_error_maximum_bias, str):
            if epipolar_error_maximum_bias != "auto":
                raise RuntimeError(
                    "The value of epipolar_error_maximum_bias "
                    "must be a float bigger than 0 or auto"
                )

        if (
            isinstance(epipolar_error_upper_bound, float)
            and epipolar_error_upper_bound <= 0
        ):
            raise RuntimeError(
                "The value of epipolar_error_upper_bound "
                "must be a float bigger than 0"
            )

        if (
            isinstance(epipolar_error_maximum_bias, float)
            and epipolar_error_maximum_bias < 0
        ):
            raise RuntimeError(
                "The value of epipolar_error_maximum_bias "
                "must be a float bigger or equal to 0"
            )

    def get_required_bands(self):
        """
        Get bands required by this application

        :return: required bands for left and right image
        :rtype: dict
        """
        return self.sparse_matching_method.get_required_bands()

    def get_margins_strip_fun(
        self, disp_min=None, disp_max=None, method="sift"
    ):
        """
        Get margins function to use in resampling

        :param disp_min: disp min for info
        :param disp_max: disp max for info
        :param method: method for the margins

        :return: margins function
        :rtype: function generating  xr.Dataset

        """

        corner = ["left", "up", "right", "down"]
        data = np.zeros(len(corner))
        col = np.arange(len(corner))
        margins = xr.Dataset(
            {"left_margin": (["col"], data)}, coords={"col": col}
        )
        margins["right_margin"] = xr.DataArray(data, dims=["col"])

        left_margin = self.tile_margin

        if method == "sift":
            right_margin = self.tile_margin + int(
                math.floor(
                    self.epipolar_error_upper_bound
                    + self.epipolar_error_maximum_bias
                )
            )
        else:
            right_margin = left_margin

        margins["left_margin"].data = [0, left_margin, 0, left_margin]
        margins["right_margin"].data = [0, right_margin, 0, right_margin]

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
            """

            return margins

        return margins_wrapper

    def get_margins_tile_fun(self, grid_left, disp_range_grid, method="sift"):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :type grid_left: dict
        :param disp_range_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi

        """

        if method == "sift":
            right_margin = self.tile_margin + int(
                math.floor(
                    self.epipolar_error_upper_bound
                    + self.epipolar_error_maximum_bias
                )
            )
        else:
            right_margin = self.tile_margin

        disp_min_grid_arr, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_min_path"]
        )
        disp_max_grid_arr, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_max_path"]
        )
        step_row = disp_range_grid["step_row"]
        step_col = disp_range_grid["step_col"]
        row_range = disp_range_grid["row_range"]
        col_range = disp_range_grid["col_range"]

        disp_to_alt_ratio = grid_left["disp_to_alt_ratio"]

        disp_min_global = np.min(disp_min_grid_arr)
        disp_max_global = np.max(disp_max_grid_arr)

        logging.info(
            "Global Disparity range for current pair:  "
            "[{:.3f} pix., {:.3f} pix.] "
            "(or [{:.3f} m., {:.3f} m.])".format(
                disp_min_global,
                disp_max_global,
                disp_min_global * disp_to_alt_ratio,
                disp_max_global * disp_to_alt_ratio,
            )
        )

        def margins_wrapper(row_min, row_max, col_min, col_max):
            """
            Generates margins Dataset used in resampling
            """

            disp_min, disp_max = tiling.compute_local_disp_range_from_grids(
                row_min,
                row_max,
                col_min,
                col_max,
                disp_min_grid_arr,
                disp_max_grid_arr,
                step_row,
                step_col,
                row_range,
                col_range,
            )

            margins = sm_wrapper.get_margins(
                self.tile_margin,
                right_margin,
                disp_min,
                disp_max,
            )

            return margins

        return margins_wrapper

    def filter_matches(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_matches_left,
        grid_left,
        grid_right,
        geom_plugin,
        orchestrator=None,
        pair_key="pair_0",
        pair_folder=None,
        save_matches=False,
    ):
        """
        Transform matches CarsDataset to numpy matches, and filters matches
        """

        if orchestrator is None:
            cars_orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            cars_orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(cars_orchestrator.out_dir, "tmp")

        epipolar_error_upper_bound = self.epipolar_error_upper_bound
        epipolar_error_maximum_bias = self.epipolar_error_maximum_bias

        grid_left = RectificationGrid(
            grid_left["path"],
            interpolator=geom_plugin.interpolator,
        )
        grid_right = RectificationGrid(
            grid_right["path"],
            interpolator=geom_plugin.interpolator,
        )

        list_matches = []
        for row in range(epipolar_matches_left.shape[0]):
            for col in range(epipolar_matches_left.shape[1]):
                if epipolar_matches_left[row, col] is not None:
                    epipolar_matches = epipolar_matches_left[
                        row, col
                    ].to_numpy()

                    sensor_matches = geom_plugin.matches_to_sensor_coords(
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

        if save_matches:
            safe_makedirs(pair_folder)

            logging.info("Writing raw matches file")
            raw_matches_array_path = os.path.join(
                pair_folder, "raw_matches.npy"
            )
            np.save(raw_matches_array_path, matches)

        epipolar_median_shift = np.median(matches[:, 3] - matches[:, 1])

        if np.abs(epipolar_median_shift) > epipolar_error_maximum_bias:
            epipolar_median_shift = epipolar_error_maximum_bias * np.sign(
                epipolar_median_shift
            )

        # pylint: disable=invalid-unary-operand-type
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

        if save_matches:
            logging.info("Writing filtered matches file")
            filtered_matches_array_path = os.path.join(
                pair_folder, "filtered_matches.npy"
            )
            np.save(filtered_matches_array_path, matches)

        nb_matches = matches.shape[0]

        if nb_matches < self.minimum_nb_matches:
            error_message_matches = (
                "Insufficient amount of matches found ({} < {}), "
                "can not safely estimate epipolar error correction "
                " and disparity range".format(
                    nb_matches,
                    self.minimum_nb_matches,
                )
            )
            logging.warning(error_message_matches)

        logging.info(
            "Number of matches kept for epipolar "
            "error correction: {} matches".format(nb_matches)
        )

        if matches.shape[0] > 0:
            epipolar_error = matches[:, 1] - matches[:, 3]
            epi_error_mean = np.mean(epipolar_error)
            epi_error_std = np.std(epipolar_error)
            epi_error_max = np.max(np.fabs(epipolar_error))
        else:
            epi_error_mean = 0
            epi_error_std = 0
            epi_error_max = 0
            logging.info(
                "Epipolar error before correction: mean = {:.3f} pix., "
                "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
                    epi_error_mean,
                    epi_error_std,
                    epi_error_max,
                )
            )

        raw_matches_infos = {
            application_constants.APPLICATION_TAG: {
                sm_cst.MATCH_FILTERING_TAG: {
                    pair_key: {
                        sm_cst.NUMBER_MATCHES_TAG: nb_matches,
                        sm_cst.RAW_NUMBER_MATCHES_TAG: raw_nb_matches,
                        sm_cst.EPIPOLAR_ERROR_ESTIMATION: epi_error_mean,
                        sm_cst.EPIPOLAR_ERROR_MAXIMUM_BIAS: epi_error_std,
                        sm_cst.EPIPOLAR_ERROR_UPPER_BOUND: 2 * epi_error_std,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_STD: epi_error_std,
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MAX: epi_error_max,
                    }
                }
            }
        }
        cars_orchestrator.update_out_info(raw_matches_infos)

        return matches

    def check_conf_disparity_bounds_estimation(self, conf):
        """
        Validate and complete disparity bounds estimation parameters.
        """

        conf["disparity_bounds_estimation"]["activated"] = conf[
            "disparity_bounds_estimation"
        ].get("activated", True)
        conf["disparity_bounds_estimation"]["percentile"] = conf[
            "disparity_bounds_estimation"
        ].get("percentile", 1)
        conf["disparity_bounds_estimation"]["lower_margin"] = conf[
            "disparity_bounds_estimation"
        ].get("lower_margin", 500)
        conf["disparity_bounds_estimation"]["upper_margin"] = conf[
            "disparity_bounds_estimation"
        ].get("upper_margin", 1000)

        disparity_bounds_estimation_schema = {
            "activated": bool,
            "percentile": Or(int, float),
            "upper_margin": int,
            "lower_margin": int,
        }

        checker = Checker(disparity_bounds_estimation_schema)
        checker.validate(conf["disparity_bounds_estimation"])

    def run(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_image_left,
        epipolar_image_right,
        disp_to_alt_ratio,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        classif_bands_to_mask=None,
    ):
        """
        Run Matching application.

        Create left and right CarsDataset filled with pandas.DataFrame ,
        corresponding to epipolar 2D disparities, on the same geometry
        that epipolar_image_left and epipolar_image_right.

        :param epipolar_image_left: tiled left epipolar. CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"
                        "transform", "crs", "valid_pixels", "no_data_mask",
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_image_left: CarsDataset
        :param epipolar_image_right: tiled right epipolar.CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:"largest_epipolar_region", \
                  "opt_epipolar_tile_size"
        :type epipolar_image_right: CarsDataset
        :param disp_to_alt_ratio: disp to alti ratio
        :type disp_to_alt_ratio: float
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair key id
        :type pair_key: str
        :param classif_bands_to_mask: bands from classif to mask
        :type classif_bands_to_mask: list of str / int

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

        if epipolar_image_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            epipolar_disparity_map_left = cars_dataset.CarsDataset(
                "points", name="sparse_matching_" + pair_key
            )
            epipolar_disparity_map_left.create_empty_copy(epipolar_image_left)

            # Update attributes to get epipolar info
            epipolar_disparity_map_left.attributes.update(
                epipolar_image_left.attributes
            )

            # Save disparity maps
            if self.save_intermediate_data:
                safe_makedirs(pair_folder)

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
                    sm_cst.SPARSE_MATCHING_RUN_TAG: {
                        pair_key: {
                            sm_cst.DISP_LOWER_BOUND: disp_lower_bound,
                            sm_cst.DISP_UPPER_BOUND: disp_upper_bound,
                        },
                    }
                }
            }
            self.orchestrator.update_out_info(updating_infos)
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
            total_nb_band_sift = epipolar_disparity_map_left.shape[0]

            step = int(np.round(100 / self.decimation_factor))

            if total_nb_band_sift in (1, 2):
                step = 1
            elif total_nb_band_sift == 3:
                step = 2

            for row in range(0, total_nb_band_sift, step):
                for col in range(len(epipolar_image_left[row])):
                    # initialize list of matches
                    full_saving_info_left = ocht.update_saving_infos(
                        saving_info_left, row=row, col=col
                    )
                    # Compute matches
                    if type(None) not in (
                        type(epipolar_image_left[row, col]),
                        type(epipolar_image_right[row, col]),
                    ):
                        (
                            epipolar_disparity_map_left[row, col]
                        ) = self.orchestrator.cluster.create_task(
                            compute_matches_wrapper, nout=1
                        )(
                            self.sparse_matching_method,
                            epipolar_image_left[row, col],
                            epipolar_image_right[row, col],
                            disp_lower_bound=disp_lower_bound,
                            disp_upper_bound=disp_upper_bound,
                            saving_info_left=full_saving_info_left,
                            classif_bands_to_mask=classif_bands_to_mask,
                        )

        else:
            logging.error(
                "SparseMatching application doesn't "
                "support this input data format"
            )

        return epipolar_disparity_map_left, None


def compute_matches_wrapper(  # pylint: disable=too-many-positional-arguments
    sparse_matching_method,
    left_image_object,
    right_image_object,
    disp_lower_bound=None,
    disp_upper_bound=None,
    saving_info_left=None,
    classif_bands_to_mask=None,
):
    """
    Compute matches from image objects.
    This function will be run as a delayed task.

    User must provide saving infos to save properly created datasets

    :param left_image_object: tiled Left image dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_TEXTURE (for left, if given)
    :type left_image_object: xr.Dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_TEXTURE (for left, if given)
    :param right_image_object: tiled Right image
    :type right_image_object: xr.Dataset
    :param classif_bands_to_mask: bands from classif to mask
    :type classif_bands_to_mask: list of str / int


    :return: Left matches object, Right matches object (if exists)

    Returned objects are composed of :

        - dataframe (None for right object) with :
            - TODO
    """

    return sparse_matching_method.run(
        left_image_object,
        right_image_object,
        saving_info_left=saving_info_left,
        disp_lower_bound=disp_lower_bound,
        disp_upper_bound=disp_upper_bound,
        classif_bands_to_mask=classif_bands_to_mask,
    )
