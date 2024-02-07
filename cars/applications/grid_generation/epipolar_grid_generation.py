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
this module contains the epipolar grid generation application class.
"""

# Standard imports
import logging
import os

# Third party imports
import numpy as np
from json_checker import And, Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_generation import grid_constants, grids
from cars.applications.grid_generation.grid_generation import GridGeneration
from cars.core import projection

# CARS imports
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)


class EpipolarGridGeneration(GridGeneration, short_name="epipolar"):
    """
    EpipolarGridGeneration
    """

    def __init__(self, conf=None):
        """
        Init function of EpipolarGridGeneration

        :param conf: configuration for grid generation
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.epi_step = self.used_config["epi_step"]
        # Saving files
        # TODO not implemented, future work
        self.save_grids = self.used_config["save_grids"]

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

        # Init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "epipolar")
        overloaded_conf["epi_step"] = conf.get("epi_step", 30)
        overloaded_conf["save_grids"] = conf.get("save_grids", False)

        grid_generation_schema = {
            "method": str,
            "epi_step": And(int, lambda x: x > 0),
            "save_grids": bool,
        }

        # Check conf
        checker = Checker(grid_generation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="Epi Grid Generation")
    def run(
        self,
        image_left,
        image_right,
        geometry_plugin,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run EpipolarGridGeneration application

        Create left and right grid CarsDataset filled with xarray.Dataset ,
        corresponding to left and right epipolar grids.

        :param image_left: left image. Dict Must contain keys : \
         "image", "color", "geomodel","no_data", "mask". Paths must be absolutes
        :type image_left: dict
        :param image_right: right image. Dict Must contain keys :\
         "image", "color", "geomodel","no_data", "mask". Paths must be absolutes
        :type image_right: dict
        :param geometry_plugin: geometry plugin to use
        :type geometry_plugin: AbstractGeometry
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator used
        :param pair_key: pair configuration id
        :type pair_key: str

        :return: left grid, right grid. Each grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape\
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x", \
                "epipolar_origin_y","epipolar_spacing_x", \
                "epipolar_spacing", "disp_to_alt_ratio",

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

        sensor1 = image_left[sens_cst.INPUT_IMG]
        sensor2 = image_right[sens_cst.INPUT_IMG]
        geomodel1 = image_left[sens_cst.INPUT_GEO_MODEL]
        geomodel2 = image_right[sens_cst.INPUT_GEO_MODEL]

        # Get satellites angles from ground: Azimuth to north, Elevation angle
        (
            left_az,
            left_elev_angle,
            right_az,
            right_elev_angle,
            convergence_angle,
        ) = projection.get_ground_angles(
            sensor1, sensor2, geomodel1, geomodel2, geometry_plugin
        )

        logging.info(
            "Left satellite acquisition angles: Azimuth angle: {:.1f}°, "
            "Elevation angle: {:.1f}°".format(left_az, left_elev_angle)
        )

        logging.info(
            "Right satellite acquisition angles: Azimuth angle: {:.1f}°, "
            "Elevation angle: {:.1f}°".format(right_az, right_elev_angle)
        )

        logging.info(
            "Stereo satellite convergence angle from ground: {:.1f}°".format(
                convergence_angle
            )
        )

        # Generate rectification grids
        (
            grid1,
            grid2,
            grid_origin,
            grid_spacing,
            epipolar_size,
            disp_to_alt_ratio,
        ) = grids.generate_epipolar_grids(
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            geometry_plugin,
            self.epi_step,
        )

        # Create CarsDataset
        grid_left = cars_dataset.CarsDataset(
            "arrays", name="grid_left_" + pair_key
        )
        grid_right = cars_dataset.CarsDataset(
            "arrays", name="grid_right_" + pair_key
        )

        # Compute tiling grid
        # Only one tile
        grid_left.tiling_grid = np.array(
            [[[0, epipolar_size[0], 0, epipolar_size[1]]]]
        )
        grid_right.tiling_grid = grid_left.tiling_grid

        # Fill tile
        grid_left[0, 0] = grid1
        grid_right[0, 0] = grid2

        # Fill attributes
        grid_attributes = {
            "grid_spacing": grid_spacing,
            "grid_origin": grid_origin,
            "epipolar_size_x": epipolar_size[0],
            "epipolar_size_y": epipolar_size[1],
            "epipolar_origin_x": grid_origin[0],
            "epipolar_origin_y": grid_origin[1],
            "epipolar_spacing_x": grid_spacing[0],
            "epipolar_spacing": grid_spacing[1],
            "disp_to_alt_ratio": disp_to_alt_ratio,
            "epipolar_step": self.epi_step,
            "path": None,
        }
        grid_left.attributes = grid_attributes.copy()
        grid_right.attributes = grid_attributes.copy()

        grid_origin = grid_left.attributes["grid_origin"]
        grid_spacing = grid_left.attributes["grid_spacing"]

        if self.save_grids:
            left_grid_path = os.path.join(pair_folder, "left_epi_grid.tif")
            right_grid_path = os.path.join(pair_folder, "right_epi_grid.tif")
        else:
            if pair_folder is None:
                tmp_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            else:
                tmp_folder = os.path.join(pair_folder, "tmp")
            safe_makedirs(tmp_folder)
            left_grid_path = os.path.join(tmp_folder, "left_epi_grid.tif")
            right_grid_path = os.path.join(tmp_folder, "right_epi_grid.tif")

        grids.write_grid(
            grid_left[0, 0], left_grid_path, grid_origin, grid_spacing
        )
        grids.write_grid(
            grid_right[0, 0], right_grid_path, grid_origin, grid_spacing
        )

        grid_left.attributes["path"] = left_grid_path
        grid_right.attributes["path"] = right_grid_path

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pair_key: {
                    grid_constants.GRID_GENERATION_RUN_TAG: {
                        grid_constants.EPIPOLAR_SIZE_X_TAG: epipolar_size[0],
                        grid_constants.EPIPOLAR_SIZE_Y_TAG: epipolar_size[1],
                        grid_constants.EPIPOLAR_ORIGIN_X_TAG: grid_origin[0],
                        grid_constants.EPIPOLAR_ORIGIN_Y_TAG: grid_origin[1],
                        grid_constants.EPIPOLAR_SPACING_X_TAG: grid_spacing[0],
                        grid_constants.EPIPOLAR_SPACING_Y_TAG: grid_spacing[1],
                        grid_constants.DISP_TO_ALT_RATIO_TAG: disp_to_alt_ratio,
                        grid_constants.LEFT_AZIMUTH_ANGLE_TAG: left_az,
                        grid_constants.LEFT_ELEVATION_ANGLE_TAG: (
                            left_elev_angle
                        ),
                        grid_constants.RIGHT_AZIMUTH_ANGLE_TAG: right_az,
                        grid_constants.RIGHT_ELEVATION_ANGLE_TAG: (
                            right_elev_angle
                        ),
                        grid_constants.CONVERGENCE_ANGLE_TAG: convergence_angle,
                    },
                }
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        logging.info(
            "Size of epipolar images: {}x{} pixels".format(
                epipolar_size[0], epipolar_size[1]
            )
        )
        logging.info(
            "Disparity to altitude factor: {} m/pixel".format(disp_to_alt_ratio)
        )

        return grid_left, grid_right
