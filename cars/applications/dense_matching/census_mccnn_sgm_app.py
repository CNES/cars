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
# pylint: disable=too-many-lines
import collections

# Standard imports
import copy
import logging
import math
import os
from typing import Dict, Tuple

# Third party imports
import numpy as np
import pandora
import xarray as xr
from json_checker import And, Checker, Or
from pandora.check_configuration import check_pipeline_section
from pandora.img_tools import add_global_disparity
from pandora.state_machine import PandoraMachine
from rasterio.profiles import DefaultGTiffProfile
from scipy.ndimage import maximum_filter, minimum_filter

import cars.applications.dense_matching.dense_matching_constants as dm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matching import dense_matching_algo as dm_algo
from cars.applications.dense_matching import (
    dense_matching_wrappers as dm_wrappers,
)
from cars.applications.dense_matching.abstract_dense_matching_app import (
    DenseMatching,
)
from cars.applications.dense_matching.disparity_grid_algo import (
    generate_disp_range_const_tile_wrapper,
    generate_disp_range_from_dem_wrapper,
)
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
)

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import inputs, tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, format_transformation
from cars.orchestrator.cluster.log_wrapper import cars_profile


class CensusMccnnSgm(
    DenseMatching,
    short_name=[
        "census_sgm_default",
        "census_sgm_urban",
        "census_sgm_shadow",
        "census_sgm_mountain_and_vegetation",
        "census_sgm_homogeneous",
        "mccnn_sgm",
        "auto",
    ],
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
        self.min_elevation_offset = self.used_config["min_elevation_offset"]
        self.max_elevation_offset = self.used_config["max_elevation_offset"]

        # Disparity threshold
        self.disp_min_threshold = self.used_config["disp_min_threshold"]
        self.disp_max_threshold = self.used_config["disp_max_threshold"]

        # Ambiguity
        self.generate_ambiguity = self.used_config["generate_ambiguity"]

        self.classification_fusion_margin = self.used_config[
            "classification_fusion_margin"
        ]

        # Margins computation parameters
        # Use local disp
        self.use_global_disp_range = self.used_config["use_global_disp_range"]
        self.local_disp_grid_step = self.used_config["local_disp_grid_step"]
        self.disp_range_propagation_filter_size = self.used_config[
            "disp_range_propagation_filter_size"
        ]
        self.epi_disp_grid_tile_size = self.used_config[
            "epi_disp_grid_tile_size"
        ]
        self.use_cross_validation = self.used_config["use_cross_validation"]
        self.denoise_disparity_map = self.used_config["denoise_disparity_map"]
        self.required_bands = self.used_config["required_bands"]
        self.used_band = self.used_config["used_band"]

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

        self.confidence_filtering = self.used_config["confidence_filtering"]
        self.threshold_disp_range_to_borders = self.used_config[
            "threshold_disp_range_to_borders"
        ]

        # init orchestrator
        self.orchestrator = None

    def get_performance_map_parameters(self):
        """
        Get parameter linked to performance, that will be used in triangulation

        :return: parameters to use
        :type: dict
        """

        return {
            "performance_map_method": self.used_config[
                "performance_map_method"
            ],
            "perf_ambiguity_threshold": self.used_config[
                "perf_ambiguity_threshold"
            ],
        }

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
            "method", "auto"
        )  # change it if census_sgm is not default
        # method called in abstract_dense_matching_app.py
        overloaded_conf["min_epi_tile_size"] = conf.get(
            "min_epi_tile_size", 300
        )
        overloaded_conf["max_epi_tile_size"] = conf.get(
            "max_epi_tile_size", 1500
        )
        overloaded_conf["epipolar_tile_margin_in_percent"] = conf.get(
            "epipolar_tile_margin_in_percent", 60
        )

        overloaded_conf["classification_fusion_margin"] = conf.get(
            "classification_fusion_margin", -1
        )
        overloaded_conf["min_elevation_offset"] = conf.get(
            "min_elevation_offset", None
        )
        overloaded_conf["max_elevation_offset"] = conf.get(
            "max_elevation_offset", None
        )
        overloaded_conf["denoise_disparity_map"] = conf.get(
            "denoise_disparity_map", False
        )

        # confidence filtering parameters
        overloaded_conf["confidence_filtering"] = conf.get(
            "confidence_filtering", {}
        )

        # Disparity threshold
        overloaded_conf["disp_min_threshold"] = conf.get(
            "disp_min_threshold", None
        )
        overloaded_conf["disp_max_threshold"] = conf.get(
            "disp_max_threshold", None
        )

        overloaded_conf["perf_eta_max_ambiguity"] = conf.get(
            "perf_eta_max_ambiguity", 0.99
        )
        overloaded_conf["perf_eta_max_risk"] = conf.get(
            "perf_eta_max_risk", 0.25
        )
        overloaded_conf["perf_eta_step"] = conf.get("perf_eta_step", 0.04)
        overloaded_conf["perf_ambiguity_threshold"] = conf.get(
            "perf_ambiguity_threshold", 0.6
        )
        overloaded_conf["use_cross_validation"] = conf.get(
            "use_cross_validation", True
        )
        # Margins computation parameters
        overloaded_conf["use_global_disp_range"] = conf.get(
            "use_global_disp_range", False
        )
        overloaded_conf["local_disp_grid_step"] = conf.get(
            "local_disp_grid_step", 10
        )
        overloaded_conf["disp_range_propagation_filter_size"] = conf.get(
            "disp_range_propagation_filter_size", 50
        )
        overloaded_conf["epi_disp_grid_tile_size"] = conf.get(
            "epi_disp_grid_tile_size", 800
        )
        overloaded_conf["required_bands"] = conf.get("required_bands", ["b0"])
        overloaded_conf["used_band"] = conf.get("used_band", "b0")
        overloaded_conf["threshold_disp_range_to_borders"] = conf.get(
            "threshold_disp_range_to_borders", False
        )

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        # Permormance map parameters
        overloaded_conf["generate_ambiguity"] = conf.get(
            "generate_ambiguity",
            overloaded_conf["save_intermediate_data"],
        )

        # Permormance map parameters
        default_perf_map_method = None
        if overloaded_conf["save_intermediate_data"]:
            default_perf_map_method = "risk"
        overloaded_conf["performance_map_method"] = conf.get(
            "performance_map_method",
            default_perf_map_method,
        )

        # Get performance map method
        perf_map_method = overloaded_conf["performance_map_method"]
        if isinstance(overloaded_conf["performance_map_method"], str):
            perf_map_method = [perf_map_method]
        elif perf_map_method is None:
            perf_map_method = []

        overloaded_conf["performance_map_method"] = perf_map_method

        # check loader
        loader_conf = conf.get("loader_conf", None)
        loader = conf.get("loader", "pandora")

        if overloaded_conf["use_cross_validation"] is True:
            overloaded_conf["use_cross_validation"] = "fast"

        # TODO modify, use loader directly
        logger = logging.getLogger("transitions.core")
        logger.addFilter(
            lambda record: "to model due to model override policy"
            not in record.getMessage()
        )
        pandora_loader = PandoraLoader(
            conf=loader_conf,
            method_name=overloaded_conf["method"],
            generate_performance_map_from_risk="risk" in perf_map_method,
            generate_performance_map_from_intervals="intervals"
            in perf_map_method,
            generate_ambiguity=overloaded_conf["generate_ambiguity"],
            perf_eta_max_ambiguity=overloaded_conf["perf_eta_max_ambiguity"],
            perf_eta_max_risk=overloaded_conf["perf_eta_max_risk"],
            perf_eta_step=overloaded_conf["perf_eta_step"],
            use_cross_validation=overloaded_conf["use_cross_validation"],
            denoise_disparity_map=overloaded_conf["denoise_disparity_map"],
            used_band=overloaded_conf["used_band"],
        )

        overloaded_conf["loader"] = loader

        # Get params from loader
        self.loader = pandora_loader
        self.corr_config = collections.OrderedDict(pandora_loader.get_conf())

        # Instantiate margins from pandora check conf
        # create the dataset
        fake_dataset = xr.Dataset(
            data_vars={},
            coords={
                "band_im": [overloaded_conf["used_band"]],
                "row": np.arange(10),
                "col": np.arange(10),
            },
            attrs={"disparity_source": [-1, 1]},
        )
        # Import plugins before checking configuration
        pandora.import_plugin()
        pandora_machine = PandoraMachine()
        corr_config_pipeline = {"pipeline": dict(self.corr_config["pipeline"])}

        saved_schema = copy.deepcopy(
            pandora.matching_cost.matching_cost.AbstractMatchingCost.schema
        )
        _ = check_pipeline_section(
            corr_config_pipeline, fake_dataset, fake_dataset, pandora_machine
        )
        # quick fix to remove when the problem is solved in pandora
        pandora.matching_cost.matching_cost.AbstractMatchingCost.schema = (
            saved_schema
        )
        self.margins = pandora_machine.margins.global_margins

        overloaded_conf["loader_conf"] = self.corr_config

        application_schema = {
            "method": str,
            "min_epi_tile_size": And(int, lambda x: x > 0),
            "max_epi_tile_size": And(int, lambda x: x > 0),
            "epipolar_tile_margin_in_percent": int,
            "min_elevation_offset": Or(None, int),
            "max_elevation_offset": Or(None, int),
            "disp_min_threshold": Or(None, int),
            "disp_max_threshold": Or(None, int),
            "save_intermediate_data": bool,
            "generate_ambiguity": bool,
            "performance_map_method": And(
                list,
                lambda x: all(y in ["risk", "intervals"] for y in x),
            ),
            "perf_eta_max_ambiguity": float,
            "perf_eta_max_risk": float,
            "perf_eta_step": float,
            "classification_fusion_margin": int,
            "perf_ambiguity_threshold": float,
            "use_cross_validation": Or(bool, str),
            "denoise_disparity_map": bool,
            "use_global_disp_range": bool,
            "local_disp_grid_step": int,
            "disp_range_propagation_filter_size": And(
                Or(int, float), lambda x: x >= 0
            ),
            "epi_disp_grid_tile_size": int,
            "required_bands": [str],
            "used_band": str,
            "loader_conf": Or(dict, collections.OrderedDict, str, None),
            "loader": str,
            "confidence_filtering": dict,
            "threshold_disp_range_to_borders": bool,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        self.check_conf_confidence_filtering(overloaded_conf)

        # Check consistency between bounds for optimal tile size search
        min_epi_tile_size = overloaded_conf["min_epi_tile_size"]
        max_epi_tile_size = overloaded_conf["max_epi_tile_size"]
        if min_epi_tile_size > max_epi_tile_size:
            raise ValueError(
                "Maximal tile size should be bigger than "
                "minimal tile size for optimal tile size search"
            )

        # Check consistency between bounds for elevation offset
        min_elevation_offset = overloaded_conf["min_elevation_offset"]
        max_elevation_offset = overloaded_conf["max_elevation_offset"]
        if (
            min_elevation_offset is not None
            and max_elevation_offset is not None
            and min_elevation_offset > max_elevation_offset
        ):
            raise ValueError(
                "Maximal elevation should be bigger than "
                "minimal elevation for dense matching"
            )

        disp_min_threshold = overloaded_conf["disp_min_threshold"]
        disp_max_threshold = overloaded_conf["disp_max_threshold"]
        if (
            disp_min_threshold is not None
            and disp_max_threshold is not None
            and disp_min_threshold > disp_max_threshold
        ):
            raise ValueError(
                "Maximal disparity should be bigger than "
                "minimal disparity for dense matching"
            )

        return overloaded_conf

    def check_conf_confidence_filtering(self, overloaded_conf):
        """
        Check the confidence filtering conf
        """
        overloaded_conf["confidence_filtering"]["activated"] = overloaded_conf[
            "confidence_filtering"
        ].get("activated", True)
        overloaded_conf["confidence_filtering"]["bounds_ratio_threshold"] = (
            overloaded_conf["confidence_filtering"].get(
                "bounds_ratio_threshold", 0.2
            )
        )
        overloaded_conf["confidence_filtering"]["risk_ratio_threshold"] = (
            overloaded_conf["confidence_filtering"].get(
                "risk_ratio_threshold", 0.8
            )
        )
        overloaded_conf["confidence_filtering"]["bounds_range_threshold"] = (
            overloaded_conf["confidence_filtering"].get(
                "bounds_range_threshold", 4
            )
        )
        overloaded_conf["confidence_filtering"]["risk_range_threshold"] = (
            overloaded_conf["confidence_filtering"].get(
                "risk_range_threshold", 12
            )
        )
        overloaded_conf["confidence_filtering"]["nan_threshold"] = (
            overloaded_conf["confidence_filtering"].get("nan_threshold", 0.2)
        )
        overloaded_conf["confidence_filtering"]["win_nanratio"] = (
            overloaded_conf["confidence_filtering"].get("win_nanratio", 20)
        )

        confidence_filtering_schema = {
            "activated": bool,
            "bounds_ratio_threshold": float,
            "risk_ratio_threshold": float,
            "bounds_range_threshold": int,
            "risk_range_threshold": int,
            "nan_threshold": float,
            "win_nanratio": int,
        }

        checker_confidence_filtering_schema = Checker(
            confidence_filtering_schema
        )
        checker_confidence_filtering_schema.validate(
            overloaded_conf["confidence_filtering"]
        )

    @cars_profile(name="Get margin fun")
    def get_margins_fun(self, grid_left, disp_range_grid):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :type grid_left: dict
        :param disp_range_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi

        """

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

        # get disp_to_alt_ratio
        disp_to_alt_ratio = grid_left["disp_to_alt_ratio"]

        # Check if we need to override disp_min
        if self.min_elevation_offset is not None:
            user_disp_min = self.min_elevation_offset / disp_to_alt_ratio
            if np.any(disp_min_grid_arr < user_disp_min):
                logging.warning(
                    (
                        "Overridden disparity minimum "
                        "= {:.3f} pix. (= {:.3f} m.) "
                        "is greater than disparity minimum estimated "
                        "in prepare step "
                        "for current pair"
                    ).format(
                        user_disp_min,
                        self.min_elevation_offset,
                    )
                )
            disp_min_grid_arr[:, :] = user_disp_min

        # Check if we need to override disp_max
        if self.max_elevation_offset is not None:
            user_disp_max = self.max_elevation_offset / disp_to_alt_ratio
            if np.any(disp_max_grid_arr > user_disp_max):
                logging.warning(
                    (
                        "Overridden disparity maximum "
                        "= {:.3f} pix. (or {:.3f} m.) "
                        "is lower than disparity maximum estimated "
                        "in prepare step "
                        "for current pair"
                    ).format(
                        user_disp_max,
                        self.max_elevation_offset,
                    )
                )
            disp_max_grid_arr[:, :] = user_disp_max

        # Compute global range of logging
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

            :param row_min: row min
            :param row_max: row max
            :param col_min: col min
            :param col_max: col max

            :return: margins
            :rtype: xr.Dataset
            """

            assert row_min < row_max
            assert col_min < col_max

            # Get region in grid

            grid_row_min = max(0, int(np.floor((row_min - 1) / step_row)) - 1)
            grid_row_max = min(
                len(row_range), int(np.ceil((row_max + 1) / step_row) + 1)
            )
            grid_col_min = max(0, int(np.floor((col_min - 1) / step_col)) - 1)
            grid_col_max = min(
                len(col_range), int(np.ceil((col_max + 1) / step_col)) + 1
            )

            # Compute disp min and max in row
            disp_min = np.min(
                disp_min_grid_arr[
                    grid_row_min:grid_row_max, grid_col_min:grid_col_max
                ]
            )
            disp_max = np.max(
                disp_max_grid_arr[
                    grid_row_min:grid_row_max, grid_col_min:grid_col_max
                ]
            )
            # round disp min and max
            disp_min = int(math.floor(disp_min))
            disp_max = int(math.ceil(disp_max))

            # Compute margins for the correlator
            margins = dm_wrappers.get_margins(self.margins, disp_min, disp_max)

            return margins

        return margins_wrapper

    def get_method(self):
        """
        return the method that will be used in dense_matching
        """

        return self.used_method

    @cars_profile(name="Optimal size estimation")
    def get_optimal_tile_size(self, disp_range_grid, max_ram_per_worker):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_range_grid: minimum and maximum disparity grid
        :param max_ram_per_worker: maximum ram per worker
        :return: optimal tile size

        """

        disp_min_grids, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_min_path"]
        )
        disp_max_grids, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_max_path"]
        )

        # use max tile size as overlap for min and max:
        # max Point to point diff is less than diff of tile

        # use filter of size max_epi_tile_size
        overlap = 3 * int(self.max_epi_tile_size / self.local_disp_grid_step)

        disp_min_grids = minimum_filter(
            disp_min_grids, size=[overlap, overlap], mode="nearest"
        )
        disp_max_grids = maximum_filter(
            disp_max_grids, size=[overlap, overlap], mode="nearest"
        )

        # Worst cases scenario:
        # 1: [global max - max diff, global max]
        # 2: [global min, global min  max diff]

        max_diff = np.round(np.nanmax(disp_max_grids - disp_min_grids)) + 1
        global_min = np.floor(np.nanmin(disp_min_grids))
        global_max = np.ceil(np.nanmax(disp_max_grids))

        # Get tiling param
        opt_epipolar_tile_size_1 = (
            dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                global_min,
                global_min + max_diff,
                self.min_epi_tile_size,
                self.max_epi_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
            )
        )
        opt_epipolar_tile_size_2 = (
            dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                global_max - max_diff,
                global_max,
                self.min_epi_tile_size,
                self.max_epi_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
            )
        )

        # return worst case
        opt_epipolar_tile_size = min(
            opt_epipolar_tile_size_1, opt_epipolar_tile_size_2
        )

        # Define function to compute local optimal size for each tile
        def local_tile_optimal_size_fun(local_disp_min, local_disp_max):
            """
            Compute optimal tile size for tile

            :return: local tile size, global optimal tile sizes

            """
            local_opt_tile_size = (
                dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                    local_disp_min,
                    local_disp_max,
                    0,
                    20000,  # arbitrary
                    max_ram_per_worker,
                    margin=self.epipolar_tile_margin_in_percent,
                )
            )

            # Get max range to use with current optimal size
            max_range = dm_wrappers.get_max_disp_from_opt_tile_size(
                opt_epipolar_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
                used_disparity_range=(local_disp_max - local_disp_min),
            )

            return local_opt_tile_size, opt_epipolar_tile_size, max_range

        return opt_epipolar_tile_size, local_tile_optimal_size_fun

    def get_required_bands(self):
        """
        Get bands required by this application

        :return: required bands for left and right image
        :rtype: dict
        """
        required_bands = {}
        required_bands["left"] = self.required_bands
        required_bands["right"] = self.required_bands
        return required_bands

    @cars_profile(name="Disp Grid Generation")
    def generate_disparity_grids(  # noqa: C901
        self,
        sensor_image_right,
        grid_right,
        geom_plugin_with_dem_and_geoid,
        dmin=None,
        dmax=None,
        dem_median=None,
        dem_min=None,
        dem_max=None,
        pair_folder=None,
        orchestrator=None,
    ):
        """
        Generate disparity grids min and max, with given step

        global mode: uses dmin and dmax
        local mode: uses dems


        :param sensor_image_right: sensor image right
        :type sensor_image_right: dict
        :param grid_right: right epipolar grid
        :type grid_right: dict
        :param geom_plugin_with_dem_and_geoid: geometry plugin with dem mean
            used to generate epipolar grids
        :type geom_plugin_with_dem_and_geoid: GeometryPlugin
        :param dmin: minimum disparity
        :type dmin: float
        :param dmax: maximum disparity
        :type dmax: float
        :param dem_median: path to median dem
        :type dem_median: str
        :param dem_min: path to minimum dem
        :type dem_min: str
        :param dem_max: path to maximum dem
        :type dem_max: str
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator
        :type orchestrator: Orchestrator


        :return disparity grid range, containing grid min and max
        :rtype: CarsDataset
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be aware, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        epi_size_row = grid_right["epipolar_size_y"]
        epi_size_col = grid_right["epipolar_size_x"]
        disp_to_alt_ratio = grid_right["disp_to_alt_ratio"]

        # Generate grid array
        nb_rows = int(epi_size_row / self.local_disp_grid_step) + 1
        nb_cols = int(epi_size_col / self.local_disp_grid_step) + 1
        row_range, step_row = np.linspace(
            0, epi_size_row, nb_rows, retstep=True
        )
        col_range, step_col = np.linspace(
            0, epi_size_col, nb_cols, retstep=True
        )

        # Create CarsDataset
        grid_disp_range = cars_dataset.CarsDataset(
            "arrays", name="grid_disp_range_unknown_pair"
        )
        global_infos_cars_ds = cars_dataset.CarsDataset("dict")

        # Generate profile
        raster_profile = DefaultGTiffProfile(count=1)

        # saving infos
        # disp grids

        if self.save_intermediate_data:
            grid_min_path = os.path.join(pair_folder, "disp_min_grid.tif")
            grid_max_path = os.path.join(pair_folder, "disp_max_grid.tif")
            safe_makedirs(pair_folder)
        else:
            if pair_folder is None:
                tmp_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            else:
                tmp_folder = os.path.join(pair_folder, "tmp")
            safe_makedirs(tmp_folder)
            self.orchestrator.add_to_clean(tmp_folder)
            grid_min_path = os.path.join(tmp_folder, "disp_min_grid.tif")
            grid_max_path = os.path.join(tmp_folder, "disp_max_grid.tif")

        if None not in (dmin, dmax):
            # use global disparity range
            if None not in (dem_min, dem_max):
                raise RuntimeError("Mix between local and global mode")

            # Only one tile
            grid_disp_range.tiling_grid = np.array([[[0, nb_rows, 0, nb_cols]]])

        elif None not in (dem_min, dem_max, dem_median):

            # Generate multiple tiles
            grid_tile_size = self.epi_disp_grid_tile_size
            grid_disp_range.tiling_grid = tiling.generate_tiling_grid(
                0,
                0,
                nb_rows,
                nb_cols,
                grid_tile_size,
                grid_tile_size,
            )

        # add tiling of  global_infos_cars_ds
        global_infos_cars_ds.tiling_grid = grid_disp_range.tiling_grid
        self.orchestrator.add_to_replace_lists(
            global_infos_cars_ds,
            cars_ds_name="global infos",
        )

        self.orchestrator.add_to_save_lists(
            grid_min_path,
            dm_cst.DISP_MIN_GRID,
            grid_disp_range,
            dtype=np.float32,
            nodata=0,
            cars_ds_name="disp_min_grid",
        )

        self.orchestrator.add_to_save_lists(
            grid_max_path,
            dm_cst.DISP_MAX_GRID,
            grid_disp_range,
            dtype=np.float32,
            nodata=0,
            cars_ds_name="disp_max_grid",
        )
        [saving_info] = (  # pylint: disable=unbalanced-tuple-unpacking
            self.orchestrator.get_saving_infos([grid_disp_range])
        )
        [saving_info_global_infos] = self.orchestrator.get_saving_infos(
            [global_infos_cars_ds]
        )

        # Generate grids on dict format
        grid_disp_range_dict = {
            "grid_min_path": grid_min_path,
            "grid_max_path": grid_max_path,
            "global_min": None,
            "global_max": None,
            "step_row": step_row,
            "step_col": step_col,
            "row_range": row_range,
            "col_range": col_range,
        }

        if None not in (dmin, dmax):
            # use global disparity range
            if None not in (dem_min, dem_max):
                raise RuntimeError("Mix between local and global mode")

            saving_info_global_infos_full = ocht.update_saving_infos(
                saving_info_global_infos, row=0, col=0
            )
            saving_info_full = ocht.update_saving_infos(
                saving_info, row=0, col=0
            )

            (
                grid_disp_range[0, 0],
                global_infos_cars_ds[0, 0],
            ) = self.orchestrator.cluster.create_task(
                generate_disp_range_const_tile_wrapper, nout=2
            )(
                row_range,
                col_range,
                dmin,
                dmax,
                raster_profile,
                saving_info_full,
                saving_info_global_infos_full,
            )

        elif None not in (dem_min, dem_max, dem_median):

            # use filter to propagate min and max
            filter_overlap = (
                2
                * int(
                    self.disp_range_propagation_filter_size
                    / self.local_disp_grid_step
                )
                + 1
            )

            for col in range(grid_disp_range.shape[1]):
                for row in range(grid_disp_range.shape[0]):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    saving_info_global_infos_full = ocht.update_saving_infos(
                        saving_info_global_infos, row=row, col=col
                    )
                    array_window = grid_disp_range.get_window_as_dict(row, col)
                    (
                        grid_disp_range[row, col],
                        global_infos_cars_ds[row, col],
                    ) = self.orchestrator.cluster.create_task(
                        generate_disp_range_from_dem_wrapper, nout=2
                    )(
                        array_window,
                        row_range,
                        col_range,
                        sensor_image_right,
                        grid_right,
                        geom_plugin_with_dem_and_geoid,
                        dem_median,
                        dem_min,
                        dem_max,
                        raster_profile,
                        full_saving_info,
                        saving_info_global_infos_full,
                        filter_overlap,
                        disp_to_alt_ratio,
                        disp_min_threshold=self.disp_min_threshold,
                        disp_max_threshold=self.disp_max_threshold,
                    )

        # Compute grid
        self.orchestrator.breakpoint()

        # Fill global infos
        mins, maxs = [], []
        for row in range(global_infos_cars_ds.shape[0]):
            for col in range(global_infos_cars_ds.shape[1]):
                try:
                    dict_data = global_infos_cars_ds[row, col].data
                    mins.append(dict_data["global_min"])
                    maxs.append(dict_data["global_max"])
                except Exception:
                    logging.info(
                        "Tile {} {} not computed in epi disp range"
                        " generation".format(row, col)
                    )
        grid_disp_range_dict["global_min"] = np.floor(np.nanmin(mins))
        grid_disp_range_dict["global_max"] = np.ceil(np.nanmax(maxs))

        return grid_disp_range_dict

    def run(
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
        :type disp_range_grid: CarsDataset
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

        if epipolar_images_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            epipolar_disparity_map = cars_dataset.CarsDataset(
                "arrays", name="dense_matching_" + pair_key
            )
            epipolar_disparity_map.create_empty_copy(epipolar_images_left)
            # Modify overlaps
            epipolar_disparity_map.overlaps = (
                format_transformation.reduce_overlap(
                    epipolar_disparity_map.overlaps, margins_to_keep
                )
            )

            # Update attributes to get epipolar info
            epipolar_disparity_map.attributes.update(
                epipolar_images_left.attributes
            )

            # Save disparity maps
            if self.save_intermediate_data:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp.tif"),
                    cst_disp.MAP,
                    epipolar_disparity_map,
                    cars_ds_name="epi_disp",
                    nodata=-9999,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_texture.tif"),
                    cst.EPI_TEXTURE,
                    epipolar_disparity_map,
                    cars_ds_name="epi_disp_color",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_mask.tif"),
                    cst_disp.VALID,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_mask",
                    optional_data=True,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_classif.tif"),
                    cst.EPI_CLASSIFICATION,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_classif",
                    optional_data=True,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_confidence.tif",
                    ),
                    cst_disp.CONFIDENCE,
                    epipolar_disparity_map,
                    cars_ds_name="confidence",
                    optional_data=True,
                )

                # disparity grids
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_min.tif",
                    ),
                    cst_disp.EPI_DISP_MIN_GRID,
                    epipolar_disparity_map,
                    cars_ds_name="disp_min",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_max.tif",
                    ),
                    cst_disp.EPI_DISP_MAX_GRID,
                    epipolar_disparity_map,
                    cars_ds_name="disp_max",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_filling.tif",
                    ),
                    cst_disp.FILLING,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_filling",
                    nodata=255,
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [epipolar_disparity_map]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    dm_cst.DENSE_MATCHING_RUN_TAG: {
                        pair_key: {
                            "global_disp_min": disp_range_grid["global_min"],
                            "global_disp_max": disp_range_grid["global_max"],
                        },
                    },
                }
            }
            self.orchestrator.update_out_info(updating_dict)
            logging.info(
                "Compute disparity: number tiles: {}".format(
                    epipolar_disparity_map.shape[1]
                    * epipolar_disparity_map.shape[0]
                )
            )

            nb_total_tiles_roi = 0

            # broadcast grids
            broadcasted_disp_range_grid = self.orchestrator.cluster.scatter(
                disp_range_grid
            )

            # Generate disparity maps
            for col in range(epipolar_disparity_map.shape[1]):
                for row in range(epipolar_disparity_map.shape[0]):
                    use_tile = False
                    if type(None) not in (
                        type(epipolar_images_left[row, col]),
                        type(epipolar_images_right[row, col]),
                    ):
                        use_tile = True
                        nb_total_tiles_roi += 1

                        # Compute optimal tile size for tile
                        (
                            _,
                            _,
                            crop_with_range,
                        ) = local_tile_optimal_size_fun(
                            np.array(
                                epipolar_images_left.attributes[
                                    "disp_min_tiling"
                                ]
                            )[row, col],
                            np.array(
                                epipolar_images_left.attributes[
                                    "disp_max_tiling"
                                ]
                            )[row, col],
                        )

                    if use_tile:
                        # update saving infos  for potential replacement
                        full_saving_info = ocht.update_saving_infos(
                            saving_info, row=row, col=col
                        )
                        # Compute disparity
                        (
                            epipolar_disparity_map[row, col]
                        ) = self.orchestrator.cluster.create_task(
                            compute_disparity_wrapper
                        )(
                            epipolar_images_left[row, col],
                            epipolar_images_right[row, col],
                            self.corr_config,
                            self.used_band,
                            broadcasted_disp_range_grid,
                            saving_info=full_saving_info,
                            compute_disparity_masks=compute_disparity_masks,
                            crop_with_range=crop_with_range,
                            left_overlaps=cars_dataset.overlap_array_to_dict(
                                epipolar_disparity_map.overlaps[row, col]
                            ),
                            margins_to_keep=margins_to_keep,
                            classification_fusion_margin=(
                                self.classification_fusion_margin
                            ),
                            texture_bands=texture_bands,
                            conf_filtering=self.confidence_filtering,
                            threshold_disp_range_to_borders=(
                                self.threshold_disp_range_to_borders
                            ),
                        )

        else:
            logging.error(
                "DenseMatching application doesn't "
                "support this input data format"
            )
        return epipolar_disparity_map


def compute_disparity_wrapper(
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    corr_cfg: dict,
    used_band: str,
    disp_range_grid,
    saving_info=None,
    compute_disparity_masks=False,
    crop_with_range=None,
    left_overlaps=None,
    margins_to_keep=0,
    classification_fusion_margin=-1,
    texture_bands=None,
    conf_filtering=None,
    threshold_disp_range_to_borders=False,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute disparity maps from image objects.
    This function will be run as a delayed task.

    User must provide saving infos to save properly created datasets

    :param left_image_object: tiled Left image
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_TEXTURE (for left, if given)
    :type left_image_object: xr.Dataset
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_TEXTURE (for left, if given)
    :param right_image_object: tiled Right image
    :type right_image_object: xr.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param used_band: name of band used for correlation
    :type used_band: str
    :param disp_range_grid: minimum and maximum disparity grid
    :type disp_range_grid: np.ndarray
    :param compute_disparity_masks: Compute all the disparity \
                        pandora masks(disable by default)
    :type compute_disparity_masks: bool
    :param generate_performance_map: True if generate performance map
    :type generate_performance_map: bool
    :param perf_ambiguity_threshold: ambiguity threshold used for
         performance map
    :type perf_ambiguity_threshold: float
    :param disp_to_alt_ratio: disp to alti ratio used for performance map
    :type disp_to_alt_ratio: float
    :param crop_with_range: range length to crop disparity range with
    :type crop_with_range: float
    :param left_overlaps: left overlap
    :type: left_overlaps: dict
    :param margins_to_keep: margin to keep after dense matching
    :type margins_to_keep: int
    :param classification_fusion_margin: the margin to add for the fusion
    :type classification_fusion_margin: int


    :return: Left to right disparity dataset
        Returned dataset is composed of :

        - cst_disp.MAP
        - cst_disp.VALID
        - cst.EPI_TEXTURE

    """
    # Generate disparity grids
    (
        disp_min_grid,
        disp_max_grid,
    ) = dm_algo.compute_disparity_grid(
        disp_range_grid,
        left_image_object,
        right_image_object,
        used_band,
        threshold_disp_range_to_borders,
    )

    global_disp_min = disp_range_grid["global_min"]
    global_disp_max = disp_range_grid["global_max"]

    # add global disparity in case of ambiguity normalization
    left_image_object = add_global_disparity(
        left_image_object, global_disp_min, global_disp_max
    )

    # Crop interval if needed
    mask_crop = np.zeros(disp_min_grid.shape, dtype=int)
    is_cropped = False
    if crop_with_range is not None:
        current_min = np.min(disp_min_grid)
        current_max = np.max(disp_max_grid)
        if (current_max - current_min) > crop_with_range:
            is_cropped = True
            logging.warning("disparity range for current tile is cropped")
            # crop
            new_min = (
                current_min * crop_with_range / (current_max - current_min)
            )
            new_max = (
                current_max * crop_with_range / (current_max - current_min)
            )

            mask_crop = np.logical_or(
                disp_min_grid < new_min, disp_max_grid > new_max
            )
            mask_crop = mask_crop.astype(bool)
            disp_min_grid[mask_crop] = new_min
            disp_max_grid[mask_crop] = new_max

    # Compute disparity
    # TODO : remove overwriting of EPI_MSK
    disp_dataset = dm_algo.compute_disparity(
        left_image_object,
        right_image_object,
        corr_cfg,
        used_band,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        compute_disparity_masks=compute_disparity_masks,
        cropped_range=mask_crop,
        margins_to_keep=margins_to_keep,
        classification_fusion_margin=classification_fusion_margin,
        texture_bands=texture_bands,
    )

    mask = disp_dataset["disp_msk"].values
    disp_map = disp_dataset["disp"].values
    disp_map[mask == 0] = np.nan

    # Filtering by using the confidence
    requested_confidence = [
        "confidence_from_risk_min.cars_2",
        "confidence_from_risk_max.cars_2",
        "confidence_from_interval_bounds_inf.cars_3",
        "confidence_from_interval_bounds_sup.cars_3",
    ]

    if (
        all(key in disp_dataset for key in requested_confidence)
        and conf_filtering["activated"] is True
    ):
        dm_wrappers.confidence_filtering(
            disp_dataset,
            disp_map,
            requested_confidence,
            conf_filtering,
        )

    # Fill with attributes
    cars_dataset.fill_dataset(
        disp_dataset,
        saving_info=saving_info,
        window=cars_dataset.get_window_dataset(left_image_object),
        profile=cars_dataset.get_profile_rasterio(left_image_object),
        attributes={cst.CROPPED_DISPARITY_RANGE: is_cropped},
        overlaps=left_overlaps,
    )

    return disp_dataset
