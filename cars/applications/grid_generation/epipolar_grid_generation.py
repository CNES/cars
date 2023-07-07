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
import importlib.util
import logging
import os

# Third party imports
import numpy as np
from json_checker import And, Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_generation import grid_constants, grids
from cars.applications.grid_generation.grid_generation import GridGeneration
from cars.core import preprocessing, projection

# CARS imports
from cars.core.geometry import AbstractGeometry
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


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

        # check loader
        # TODO
        self.geometry_loader = self.used_config["geometry_loader"]
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            self.geometry_loader
        )

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

        # check geometry tool availability
        geometry = "OTBGeometry"

        # 1/ check otbApplication python module
        otb_app = importlib.util.find_spec("otbApplication")
        # 2/ check remote modules
        if otb_app is not None:
            otb_geometry = (
                AbstractGeometry(  # pylint: disable=abstract-class-instantiated
                    "OTBGeometry"
                )
            )
            missing_remote = otb_geometry.check_otb_remote_modules()

        if otb_app is None or len(missing_remote) > 0:
            geometry = "SharelocGeometry"

        # Overloader loader
        overloaded_conf["geometry_loader"] = conf.get(
            "geometry_loader", geometry
        )

        grid_generation_schema = {
            "method": str,
            "epi_step": And(int, lambda x: x > 0),
            "geometry_loader": str,
            "save_grids": bool,
        }

        # Check conf
        checker = Checker(grid_generation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        image_left,
        image_right,
        orchestrator=None,
        pair_folder=None,
        srtm_dir=None,
        default_alt=None,
        geoid_path=None,
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
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator used
        :param srtm_dir: srtm directory
        :type srtm_dir: str
        :param default_alt: default altitude
        :type default_alt: float
        :param geoid_path: geoid path
        :type geoid_path: str
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

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        # TODO save grid

        # Create config from left and right inputs
        # TODO change it, modify geometry loader inputs
        config = preprocessing.create_former_cars_conf(
            image_left, image_right, srtm_dir=srtm_dir, default_alt=default_alt
        )

        # Get satellites angles from ground: Azimuth to north, Elevation angle
        (
            left_az,
            left_elev_angle,
            right_az,
            right_elev_angle,
            convergence_angle,
        ) = projection.get_ground_angles(config, self.geometry_loader)

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
            config,
            self.geometry_loader,
            dem=srtm_dir,
            default_alt=default_alt,
            epipolar_step=self.epi_step,
            geoid=geoid_path,
        )

        # Create CarsDataset
        grid_left = cars_dataset.CarsDataset("arrays")
        grid_right = cars_dataset.CarsDataset("arrays")

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
        }
        grid_left.attributes = grid_attributes
        grid_right.attributes = grid_attributes

        # add logs

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pair_key: {
                    grid_constants.GRID_GENERATION_RUN_TAG: {
                        grid_constants.EPIPOLAR_SIZE_X_TAG: (epipolar_size[0]),
                        grid_constants.EPIPOLAR_SIZE_Y_TAG: (epipolar_size[1]),
                        grid_constants.EPIPOLAR_ORIGIN_X_TAG: (grid_origin[0]),
                        grid_constants.EPIPOLAR_ORIGIN_Y_TAG: (grid_origin[1]),
                        grid_constants.EPIPOLAR_SPACING_X_TAG: (
                            grid_spacing[0]
                        ),
                        grid_constants.EPIPOLAR_SPACING_Y_TAG: (
                            grid_spacing[1]
                        ),
                        grid_constants.DISP_TO_ALT_RATIO_TAG: (
                            disp_to_alt_ratio
                        ),
                        grid_constants.LEFT_AZIMUTH_ANGLE_TAG: (left_az),
                        grid_constants.LEFT_ELEVATION_ANGLE_TAG: (
                            left_elev_angle
                        ),
                        grid_constants.RIGHT_AZIMUTH_ANGLE_TAG: (right_az),
                        grid_constants.RIGHT_ELEVATION_ANGLE_TAG: (
                            right_elev_angle
                        ),
                        grid_constants.CONVERGENCE_ANGLE_TAG: (
                            convergence_angle
                        ),
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
