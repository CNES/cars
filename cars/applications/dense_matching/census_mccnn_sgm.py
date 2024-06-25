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
import logging
import math
import os
from typing import Dict, Tuple

# Third party imports
import affine
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from json_checker import And, Checker, Or
from pandora.img_tools import add_global_disparity
from scipy.ndimage import generic_filter

import cars.applications.dense_matching.dense_matching_constants as dm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matching import dense_matching_tools as dm_tools
from cars.applications.dense_matching.dense_matching import DenseMatching
from cars.applications.dense_matching.dense_matching_tools import (
    LinearInterpNearestExtrap,
)
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
)

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import inputs, projection
from cars.core.projection import points_cloud_conversion
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile


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
        self.min_elevation_offset = self.used_config["min_elevation_offset"]
        self.max_elevation_offset = self.used_config["max_elevation_offset"]

        # Disparity threshold
        self.disp_min_threshold = self.used_config["disp_min_threshold"]
        self.disp_max_threshold = self.used_config["disp_max_threshold"]

        # Performance map
        self.generate_performance_map = self.used_config[
            "generate_performance_map"
        ]
        self.perf_ambiguity_threshold = self.used_config[
            "perf_ambiguity_threshold"
        ]

        # Margins computation parameters
        # Use local disp
        self.use_global_disp_range = self.used_config["use_global_disp_range"]
        self.local_disp_grid_step = self.used_config["local_disp_grid_step"]
        self.disp_range_propagation_filter_size = self.used_config[
            "disp_range_propagation_filter_size"
        ]
        # Saving files
        self.save_disparity_map = self.used_config["save_disparity_map"]

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
        overloaded_conf["min_elevation_offset"] = conf.get(
            "min_elevation_offset", None
        )
        overloaded_conf["max_elevation_offset"] = conf.get(
            "max_elevation_offset", None
        )

        # Disparity threshold
        overloaded_conf["disp_min_threshold"] = conf.get(
            "disp_min_threshold", None
        )
        overloaded_conf["disp_max_threshold"] = conf.get(
            "disp_max_threshold", None
        )

        # Permormance map parameters
        overloaded_conf["generate_performance_map"] = conf.get(
            "generate_performance_map", False
        )
        overloaded_conf["generate_confidence_intervals"] = conf.get(
            "generate_confidence_intervals", False
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
        # Margins computation parameters
        overloaded_conf["use_global_disp_range"] = conf.get(
            "use_global_disp_range", False
        )
        overloaded_conf["local_disp_grid_step"] = conf.get(
            "local_disp_grid_step", 30
        )
        overloaded_conf["disp_range_propagation_filter_size"] = conf.get(
            "disp_range_propagation_filter_size", 300
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
            conf=loader_conf,
            method_name=overloaded_conf["method"],
            generate_performance_map=overloaded_conf[
                "generate_performance_map"
            ],
            generate_confidence_intervals=overloaded_conf[
                "generate_confidence_intervals"
            ],
            perf_eta_max_ambiguity=overloaded_conf["perf_eta_max_ambiguity"],
            perf_eta_max_risk=overloaded_conf["perf_eta_max_risk"],
            perf_eta_step=overloaded_conf["perf_eta_step"],
        )
        overloaded_conf["loader"] = loader
        overloaded_conf["loader_conf"] = loader_conf

        # Get params from loader
        self.loader = pandora_loader
        self.corr_config = collections.OrderedDict(pandora_loader.get_conf())

        application_schema = {
            "method": str,
            "min_epi_tile_size": And(int, lambda x: x > 0),
            "max_epi_tile_size": And(int, lambda x: x > 0),
            "epipolar_tile_margin_in_percent": int,
            "min_elevation_offset": Or(None, int),
            "max_elevation_offset": Or(None, int),
            "disp_min_threshold": Or(None, int),
            "disp_max_threshold": Or(None, int),
            "save_disparity_map": bool,
            "generate_performance_map": bool,
            "generate_confidence_intervals": bool,
            "perf_eta_max_ambiguity": float,
            "perf_eta_max_risk": float,
            "perf_eta_step": float,
            "perf_ambiguity_threshold": float,
            "use_global_disp_range": bool,
            "local_disp_grid_step": int,
            "disp_range_propagation_filter_size": And(
                Or(int, float), lambda x: x >= 0
            ),
            "loader_conf": Or(dict, collections.OrderedDict, str, None),
            "loader": str,
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

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

    def get_margins_fun(self, grid_left, disp_range_grid):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :param disp_min_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi

        """

        disp_min_grid_arr = disp_range_grid[0, 0]["disp_min_grid"].values
        disp_max_grid_arr = disp_range_grid[0, 0]["disp_max_grid"].values
        step_row = disp_range_grid.attributes["step_row"]
        step_col = disp_range_grid.attributes["step_col"]
        row_range = disp_range_grid.attributes["row_range"]
        col_range = disp_range_grid.attributes["col_range"]

        # get disp_to_alt_ratio
        disp_to_alt_ratio = grid_left.attributes["disp_to_alt_ratio"]

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
            # TODO use loader correlators
            margins = dm_tools.get_margins(disp_min, disp_max, self.corr_config)
            return margins

        return margins_wrapper

    def get_optimal_tile_size(self, disp_range_grid, max_ram_per_worker):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_range_grid: minimum and maximum disparity grid
        :param max_ram_per_worker: maximum ram per worker
        :return: optimal tile size

        """

        disp_min_grids = disp_range_grid[0, 0][dm_cst.DISP_MIN_GRID].values
        disp_max_grids = disp_range_grid[0, 0][dm_cst.DISP_MAX_GRID].values

        # use max tile size as overlap for min and max:
        # max Point to point diff is less than diff of tile

        # use filter of size max_epi_tile_size
        overlap = 3 * int(self.max_epi_tile_size / self.local_disp_grid_step)
        disp_min_grids = generic_filter(
            disp_min_grids, np.nanmin, [overlap, overlap], mode="nearest"
        )
        disp_max_grids = generic_filter(
            disp_max_grids, np.nanmax, [overlap, overlap], mode="nearest"
        )

        # Worst cases scenario:
        # 1: [global max - max diff, global max]
        # 2: [global min, global min  max diff]

        max_diff = np.round(np.nanmax(disp_max_grids - disp_min_grids)) + 1
        global_min = np.floor(np.nanmin(disp_min_grids))
        global_max = np.ceil(np.nanmax(disp_max_grids))

        # Get tiling param
        opt_epipolar_tile_size_1 = (
            dm_tools.optimal_tile_size_pandora_plugin_libsgm(
                global_min,
                global_min + max_diff,
                self.min_epi_tile_size,
                self.max_epi_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
            )
        )
        opt_epipolar_tile_size_2 = (
            dm_tools.optimal_tile_size_pandora_plugin_libsgm(
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
                dm_tools.optimal_tile_size_pandora_plugin_libsgm(
                    local_disp_min,
                    local_disp_max,
                    0,
                    20000,  # arbitrary
                    max_ram_per_worker,
                    margin=self.epipolar_tile_margin_in_percent,
                )
            )

            # Get max range to use with current optimal size
            max_range = dm_tools.get_max_disp_from_opt_tile_size(
                opt_epipolar_tile_size,
                max_ram_per_worker,
                margin=self.epipolar_tile_margin_in_percent,
                used_disparity_range=(local_disp_max - local_disp_min),
            )

            return local_opt_tile_size, opt_epipolar_tile_size, max_range

        return opt_epipolar_tile_size, local_tile_optimal_size_fun

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
    ):
        """
        Generate disparity grids min and max, with given step

        global mode: uses dmin and dmax
        local mode: uses dems


        :param sensor_image_right: sensor image right
        :type sensor_image_right: dict
        :param grid_right: right epipolar grid
        :type grid_right: CarsDataset
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


        :return disparity grid range, containing grid min and max
        :rtype: CarsDataset
        """

        # Create sequential orchestrator for savings
        grid_orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )

        epi_size_row = grid_right.attributes["epipolar_size_y"]
        epi_size_col = grid_right.attributes["epipolar_size_x"]
        disp_to_alt_ratio = grid_right.attributes["disp_to_alt_ratio"]

        # Generate grid array
        nb_rows = int(epi_size_row / self.local_disp_grid_step) + 1
        nb_cols = int(epi_size_col / self.local_disp_grid_step) + 1
        row_range, step_row = np.linspace(
            0, epi_size_row, nb_rows, retstep=True
        )
        col_range, step_col = np.linspace(
            0, epi_size_col, nb_cols, retstep=True
        )

        grid_min = np.empty((len(row_range), len(col_range)))
        grid_max = np.empty((len(row_range), len(col_range)))

        # Create CarsDataset
        grid_disp_range = cars_dataset.CarsDataset(
            "arrays", name="grid_disp_range_unknown_pair"
        )
        # Only one tile
        grid_disp_range.tiling_grid = np.array(
            [[[0, epi_size_row, 0, epi_size_col]]]
        )

        grid_attributes = {
            "step_row": step_row,
            "step_col": step_col,
            "row_range": row_range,
            "col_range": col_range,
        }
        grid_disp_range.attributes = grid_attributes.copy()

        # saving infos
        # disp grids
        if self.save_disparity_map:
            grid_min_path = os.path.join(pair_folder, "disp_min_grid.tif")
            grid_orchestrator.add_to_save_lists(
                grid_min_path,
                dm_cst.DISP_MIN_GRID,
                grid_disp_range,
                dtype=np.float32,
                cars_ds_name="disp_min_grid",
            )
            grid_max_path = os.path.join(pair_folder, "disp_max_grid.tif")
            grid_orchestrator.add_to_save_lists(
                grid_max_path,
                dm_cst.DISP_MAX_GRID,
                grid_disp_range,
                dtype=np.float32,
                cars_ds_name="disp_max_grid",
            )

        if None not in (dmin, dmax):
            # use global disparity range
            if None not in (dem_min, dem_max):
                raise RuntimeError("Mix between local and global mode")

            grid_min[:, :] = dmin
            grid_max[:, :] = dmax

        elif None not in (dem_min, dem_max, dem_median):
            # use local disparity
            if None not in (dmin, dmax):
                raise RuntimeError("Mix between local and global mode")

            # dem min max are the same shape
            # dem median is not , hence we crop it

            # Get associated alti mean / min / max values
            dem_median_shape = inputs.rasterio_get_size(dem_median)
            dem_median_width, dem_median_height = dem_median_shape

            # get epsg
            terrain_epsg = inputs.rasterio_get_epsg(dem_median)

            # Get epipolar position of all dem mean
            transform = inputs.rasterio_get_transform(dem_median)

            # get terrain bounds dem min
            dem_min_bounds = inputs.rasterio_get_bounds(dem_min)

            # find roi in dem_mean
            roi_points = np.array(
                [
                    [dem_min_bounds[0], dem_min_bounds[1]],
                    [dem_min_bounds[0], dem_min_bounds[3]],
                    [dem_min_bounds[2], dem_min_bounds[1]],
                    [dem_min_bounds[2], dem_min_bounds[3]],
                ]
            )

            # Transform points to terrain_epsg (dem min is in 4326)
            roi_points_terrain = points_cloud_conversion(
                roi_points,
                4326,
                terrain_epsg,
            )

            # Get pixel roi in dem mean
            pixel_roi_dem_mean = inputs.rasterio_get_pixel_points(
                dem_median, roi_points_terrain
            )
            roi_lower_row = np.floor(np.min(pixel_roi_dem_mean[:, 0]))
            roi_upper_row = np.ceil(np.max(pixel_roi_dem_mean[:, 0]))
            roi_lower_col = np.floor(np.min(pixel_roi_dem_mean[:, 1]))
            roi_upper_col = np.ceil(np.max(pixel_roi_dem_mean[:, 1]))

            min_row = int(max(0, roi_lower_row))
            max_row = int(
                min(
                    dem_median_height,  # number of rows
                    roi_upper_row,
                )
            )
            min_col = int(max(0, roi_lower_col))
            max_col = int(
                min(
                    dem_median_width,  # number of columns
                    roi_upper_col,
                )
            )

            # compute terrain positions to use (all dem min and max)
            row_indexes = range(min_row, max_row)
            col_indexes = range(min_col, max_col)
            transformer = rasterio.transform.AffineTransformer(transform)

            terrain_positions = []
            for row in row_indexes:
                for col in col_indexes:
                    x_geo, y_geo = transformer.xy(row, col)
                    terrain_positions.append((x_geo, y_geo))
            terrain_positions = np.array(terrain_positions)

            # dem mean in terrain_epsg
            x_mean = terrain_positions[:, 0]
            y_mean = terrain_positions[:, 1]

            dem_median_list = inputs.rasterio_get_values(
                dem_median, x_mean, y_mean, points_cloud_conversion
            )

            # transform to lon lat
            terrain_position_lon_lat = projection.points_cloud_conversion(
                terrain_positions, terrain_epsg, 4326
            )
            lon_mean = terrain_position_lon_lat[:, 0]
            lat_mean = terrain_position_lon_lat[:, 1]

            # dem min and max are in 4326
            dem_min_list = inputs.rasterio_get_values(
                dem_min, lon_mean, lat_mean, points_cloud_conversion
            )
            dem_max_list = inputs.rasterio_get_values(
                dem_max, lon_mean, lat_mean, points_cloud_conversion
            )

            # sensors positions as index
            (
                ind_cols_sensor,
                ind_rows_sensor,
                _,
            ) = geom_plugin_with_dem_and_geoid.inverse_loc(
                sensor_image_right["image"],
                sensor_image_right["geomodel"],
                lat_mean,
                lon_mean,
                z_coord=dem_median_list,
            )

            # Generate epipolar disp grids
            # Get epipolar positions
            (epipolar_positions_row, epipolar_positions_col) = np.meshgrid(
                col_range, row_range
            )
            epipolar_positions = np.stack(
                [epipolar_positions_row, epipolar_positions_col], axis=2
            )

            # Get sensor position
            sensors_positions = (
                geom_plugin_with_dem_and_geoid.sensor_position_from_grid(
                    grid_right,
                    np.reshape(
                        epipolar_positions,
                        (
                            epipolar_positions.shape[0]
                            * epipolar_positions.shape[1],
                            2,
                        ),
                    ),
                )
            )

            # compute reverse matrix
            transform_sensor = inputs.rasterio_get_transform(
                sensor_image_right["image"]
            )

            trans_inv = ~transform_sensor
            # Transform to positive values
            trans_inv = np.array(trans_inv)
            trans_inv = np.reshape(trans_inv, (3, 3))
            if trans_inv[0, 0] < 0:
                trans_inv[0, :] *= -1
            if trans_inv[1, 1] < 0:
                trans_inv[1, :] *= -1
            trans_inv = affine.Affine(*list(trans_inv.flatten()))

            # Transform physical position to index
            index_positions = np.empty(sensors_positions.shape)
            for row_point in range(index_positions.shape[0]):
                row_geo, col_geo = sensors_positions[row_point, :]
                col, row = trans_inv * (row_geo, col_geo)
                index_positions[row_point, :] = (row, col)

            ind_rows_sensor_grid = index_positions[:, 0] - 0.5
            ind_cols_sensor_grid = index_positions[:, 1] - 0.5

            # Interpolate disparity
            disp_min_points = (
                -(dem_max_list - dem_median_list) / disp_to_alt_ratio
            )
            disp_max_points = (
                -(dem_min_list - dem_median_list) / disp_to_alt_ratio
            )

            interp_min_linear = LinearInterpNearestExtrap(
                list(zip(ind_rows_sensor, ind_cols_sensor)),  # noqa: B905
                disp_min_points,
            )
            interp_max_linear = LinearInterpNearestExtrap(
                list(zip(ind_rows_sensor, ind_cols_sensor)),  # noqa: B905
                disp_max_points,
            )

            grid_min = np.reshape(
                interp_min_linear(ind_rows_sensor_grid, ind_cols_sensor_grid),
                (
                    epipolar_positions.shape[0],
                    epipolar_positions.shape[1],
                ),
            )

            grid_max = np.reshape(
                interp_max_linear(ind_rows_sensor_grid, ind_cols_sensor_grid),
                (
                    epipolar_positions.shape[0],
                    epipolar_positions.shape[1],
                ),
            )

        else:
            raise RuntimeError(
                "Not a global or local mode for disparity range estimation"
            )

        # Add margin
        diff = grid_max - grid_min
        logging.info("Max grid max - grid min : {} disp ".format(np.max(diff)))

        if self.disp_min_threshold is not None:
            if np.any(grid_min < self.disp_min_threshold):
                logging.warning(
                    "Override disp_min  with disp_min_threshold {}".format(
                        self.disp_min_threshold
                    )
                )
                grid_min[grid_min < self.disp_min_threshold] = (
                    self.disp_min_threshold
                )
        if self.disp_max_threshold is not None:
            if np.any(grid_max > self.disp_max_threshold):
                logging.warning(
                    "Override disp_max with disp_max_threshold {}".format(
                        self.disp_max_threshold
                    )
                )
                grid_max[grid_max > self.disp_max_threshold] = (
                    self.disp_max_threshold
                )

        # use filter to propagate min and max
        overlap = (
            2
            * int(
                self.disp_range_propagation_filter_size
                / self.local_disp_grid_step
            )
            + 1
        )

        grid_min = generic_filter(
            grid_min, np.min, [overlap, overlap], mode="nearest"
        )
        grid_max = generic_filter(
            grid_max, np.max, [overlap, overlap], mode="nearest"
        )

        # Generate dataset
        # min and max are reversed
        disp_range_tile = xr.Dataset(
            data_vars={
                dm_cst.DISP_MIN_GRID: (["row", "col"], grid_min),
                dm_cst.DISP_MAX_GRID: (["row", "col"], grid_max),
            },
            coords={
                "row": np.arange(0, grid_min.shape[0]),
                "col": np.arange(0, grid_min.shape[1]),
            },
        )

        # Save
        [saving_info] = (  # pylint: disable=unbalanced-tuple-unpacking
            grid_orchestrator.get_saving_infos([grid_disp_range])
        )
        saving_info = ocht.update_saving_infos(saving_info, row=0, col=0)
        # Generate profile
        # Generate profile
        geotransform = (
            epi_size_row,
            step_row,
            0.0,
            epi_size_col,
            0.0,
            step_col,
        )

        transform = Affine.from_gdal(*geotransform)
        raster_profile = collections.OrderedDict(
            {
                "height": nb_rows,
                "width": nb_cols,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform,
                "tiled": True,
            }
        )
        cars_dataset.fill_dataset(
            disp_range_tile,
            saving_info=saving_info,
            window=None,
            profile=raster_profile,
            attributes=None,
            overlaps=None,
        )
        grid_disp_range[0, 0] = disp_range_tile

        if self.save_disparity_map:
            grid_orchestrator.breakpoint()

        if np.any(diff < 0):
            logging.error("grid min > grid max in {}".format(pair_folder))
            raise RuntimeError("grid min > grid max in {}".format(pair_folder))

        return grid_disp_range

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
        disp_to_alt_ratio=None,
    ):
        """
        Run Matching application.

        Create CarsDataset filled with xarray.Dataset, corresponding
        to epipolar disparities, on the same geometry than
        epipolar_images_left.

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
        :param disp_to_alt_ratio: disp to alti ratio used for performance map
        :type disp_to_alt_ratio: float

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

        # crash if generate performance and disp_to_alt_ratio not set
        if disp_to_alt_ratio is None and self.generate_performance_map:
            raise RuntimeError(
                "User wants to generate performance map without "
                "providing disp_to_alt_ratio"
            )

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        if epipolar_images_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            epipolar_disparity_map = cars_dataset.CarsDataset(
                "arrays", name="dense_matching_" + pair_key
            )
            epipolar_disparity_map.create_empty_copy(epipolar_images_left)
            epipolar_disparity_map.overlaps *= 0

            # Update attributes to get epipolar info
            epipolar_disparity_map.attributes.update(
                epipolar_images_left.attributes
            )

            # Save disparity maps
            if self.save_disparity_map:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp.tif"),
                    cst_disp.MAP,
                    epipolar_disparity_map,
                    cars_ds_name="epi_disp",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_color.tif"),
                    cst.EPI_COLOR,
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
                    dm_cst.DENSE_MATCHING_PARAM_TAG: {
                        "pandora_config": self.corr_config
                    },
                    pair_key: {
                        dm_cst.DENSE_MATCHING_RUN_TAG: {
                            "global_disp_min": np.nanmin(
                                disp_range_grid[0, 0][
                                    dm_cst.DISP_MIN_GRID
                                ].values
                            ),
                            "global_disp_max": np.nanmax(
                                disp_range_grid[0, 0][
                                    dm_cst.DISP_MAX_GRID
                                ].values
                            ),
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
                            broadcasted_disp_range_grid,
                            saving_info=full_saving_info,
                            compute_disparity_masks=compute_disparity_masks,
                            generate_performance_map=(
                                self.generate_performance_map
                            ),
                            perf_ambiguity_threshold=(
                                self.perf_ambiguity_threshold
                            ),
                            disp_to_alt_ratio=disp_to_alt_ratio,
                            crop_with_range=crop_with_range,
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
    disp_range_grid,
    saving_info=None,
    compute_disparity_masks=False,
    generate_performance_map=False,
    perf_ambiguity_threshold=0.6,
    disp_to_alt_ratio=None,
    crop_with_range=None,
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
    :return: Left to right disparity dataset
        Returned dataset is composed of :

        - cst_disp.MAP
        - cst_disp.VALID
        - cst.EPI_COLOR

    """
    # Generate disparity grids
    (
        disp_min_grid,
        disp_max_grid,
    ) = dm_tools.compute_disparity_grid(disp_range_grid, left_image_object)

    global_disp_min = np.floor(
        np.nanmin(disp_range_grid[0, 0]["disp_min_grid"].data)
    )
    global_disp_max = np.ceil(
        np.nanmax(disp_range_grid[0, 0]["disp_max_grid"].data)
    )

    # add global disparity in case of ambiguity normalization
    left_image_object = add_global_disparity(
        left_image_object, global_disp_min, global_disp_max
    )

    # Crop interval if needed
    mask_crop = np.zeros(disp_min_grid.shape, dtype=int)
    if crop_with_range is not None:
        current_min = np.min(disp_min_grid)
        current_max = np.max(disp_max_grid)
        if (current_max - current_min) > crop_with_range:
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
    disp_dataset = dm_tools.compute_disparity(
        left_image_object,
        right_image_object,
        corr_cfg,
        disp_min_grid=disp_min_grid,
        disp_max_grid=disp_max_grid,
        compute_disparity_masks=compute_disparity_masks,
        generate_performance_map=generate_performance_map,
        perf_ambiguity_threshold=perf_ambiguity_threshold,
        disp_to_alt_ratio=disp_to_alt_ratio,
        cropped_range=mask_crop,
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        disp_dataset,
        saving_info=saving_info,
        window=cars_dataset.get_window_dataset(left_image_object),
        profile=cars_dataset.get_profile_rasterio(left_image_object),
        attributes=None,
        overlaps=None,  # overlaps are removed
    )

    return disp_dataset
