#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains the abstract direct_localization application class.
"""
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr

# Standard imports
from json_checker import And, Checker

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications.ground_truth_reprojection import (
    ground_truth_reprojection,
    ground_truth_reprojection_tools,
)

# CARS imports
from cars.core import constants as cst
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


class DirectLocalization(
    ground_truth_reprojection.GroundTruthReprojection, short_name="direct_loc"
):
    """
    DirectLocalization

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of DirectLocalization
        :param conf: configuration for resampling
        :return: an application_to_use object

        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.target = self.used_config["target"]

        # Saving bools
        self.save_intermediate_data = True

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
        overloaded_conf["method"] = conf.get("method", "direct_loc")
        overloaded_conf["target"] = conf.get("target", "epipolar")
        overloaded_conf["save_intermediate_data"] = True

        ground_truth_reprojection_schema = {
            "method": str,
            "target": And(
                str, lambda input: input in ["epipolar", "sensor", "all"]
            ),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(ground_truth_reprojection_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa: C901
        self,
        dem,
        sensor_left,
        grid_left,
        geom_left,
        geom_plugin,
        disp_to_alt_ratio,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run direct localization for ground truth disparity

        :param dem: path to reference dem
        :type dem: str
        :param sensor_left: Tiled sensor left image.
            Dict must contain keys: "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolute.
        :type sensor_left: CarsDataset
        :param grid_left: Grid left.
        :type grid_left: CarsDataset
        :param geom_left: Path and attributes for left geomodel.
        :type geom_left: dict
        :param geom_plugin: Geometry plugin with user's DSM used to
            generate epipolar grids.
        :type geom_plugin: GeometryPlugin
        :param disp_to_alt_ratio: Disp to altitude ratio used
            for performance map.
        :type disp_to_alt_ratio: float
        :param orchestrator: orchestrator used
        :type orchestrator: orchestrator
        :param pair_folder: Folder used for current pair.
        :type pair_folder: str
        :param pair_key: Pair ID.
        :type pair_key: str
        """

        logging.info("Starting ground truth reprojection application")

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

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
        else:
            safe_makedirs(pair_folder)

        if self.used_config["target"] in ["all", "epipolar"]:

            epi_disparity_ground_truth = cars_dataset.CarsDataset(
                "arrays", name="epipolar_disparity_ground_truth" + pair_key
            )

            epi_disparity_ground_truth.create_grid(
                grid_left.attributes["epipolar_size_x"],
                grid_left.attributes["epipolar_size_y"],
                2500,
                2500,
                0,
                0,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epipolar_disp_ground_truth.tif"),
                cst.EPI_GROUND_TRUTH,
                epi_disparity_ground_truth,
                cars_ds_name="epipolar_disparity_ground_truth",
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_infos_epi] = self.orchestrator.get_saving_infos(
                [epi_disparity_ground_truth]
            )

            for col in range(epi_disparity_ground_truth.tiling_grid.shape[1]):
                for row in range(
                    epi_disparity_ground_truth.tiling_grid.shape[0]
                ):
                    full_saving_info = ocht.update_saving_infos(
                        saving_infos_epi, row=row, col=col
                    )
                    (
                        epi_disparity_ground_truth[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        dem,
                        sensor_left,
                        grid_left,
                        geom_left,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "epipolar",
                        full_saving_info,
                        epi_disparity_ground_truth.tiling_grid[row, col],
                    )

        if self.used_config["target"] in ["all", "sensor"]:

            sensor_dsm_ground_truth = cars_dataset.CarsDataset(
                "arrays", name="sensor_dsm_ground_truth" + pair_key
            )

            with rio.open(sensor_left[sens_cst.INPUT_IMG]) as src:
                width = src.width
                height = src.height

            sensor_dsm_ground_truth.create_grid(width, height, 2500, 2500, 0, 0)

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "sensor_dsm_ground_truth.tif"),
                cst.SENSOR_GROUND_TRUTH,
                sensor_dsm_ground_truth,
                cars_ds_name="sensor_dsm_ground_truth",
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_infos_sensor] = self.orchestrator.get_saving_infos(
                [sensor_dsm_ground_truth]
            )

            for col in range(sensor_dsm_ground_truth.tiling_grid.shape[1]):
                for row in range(sensor_dsm_ground_truth.tiling_grid.shape[0]):
                    full_saving_info = ocht.update_saving_infos(
                        saving_infos_sensor, row=row, col=col
                    )
                    (
                        sensor_dsm_ground_truth[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        dem,
                        sensor_left,
                        grid_left,
                        geom_left,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "sensor",
                        full_saving_info,
                        sensor_dsm_ground_truth.tiling_grid[row, col],
                    )


def maps_generation_wrapper(
    dem,
    sensor_left,
    grid_left,
    geom_left,
    geom_plugin,
    disp_to_alt_ratio,
    target,
    saving_infos,
    window,
):
    """
    Computes ground truth epipolar disparity map and sensor geometry.

    :param dem: path to reference dem
    :type dem: str
    :param sensor_left: Tiled sensor left image.
        Dict must contain keys: "image", "color", "geomodel",
        "no_data", "mask". Paths must be absolute.
    :type sensor_left: CarsDataset
    :param grid_left: Grid left.
    :type grid_left: CarsDataset
    :param geom_left: Path and attributes for left geomodel.
    :type geom_left: dict
    :param geom_plugin: Geometry plugin with user's DSM used to
        generate epipolar grids.
    :type geom_plugin: GeometryPlugin
    :param disp_to_alt_ratio: Disp to altitude ratio used for performance map.
    :type disp_to_alt_ratio: float
    :param target: "epipolar", "sensor", or both ("all") geometry.
    :type target: str
    :param saving_infos: Information about CarsDataset ID.
    :type saving_infos: dict
    :param window: size of tile
    :type window: np.ndarray
    """

    ground_truth = ground_truth_reprojection_tools.get_ground_truth(
        dem,
        geom_plugin,
        grid_left,
        sensor_left[sens_cst.INPUT_IMG],
        geom_left,
        disp_to_alt_ratio,
        target,
        window,
    )

    constant_for_dataset = cst.EPI_GROUND_TRUTH
    if target == "sensor":
        constant_for_dataset = cst.SENSOR_GROUND_TRUTH

    rows = np.arange(window[0], window[1])
    cols = np.arange(window[2], window[3])

    values = {constant_for_dataset: ([cst.COL, cst.ROW], ground_truth)}
    outputs_dataset = xr.Dataset(
        values,
        coords={cst.COL: cols, cst.ROW: rows},
    )

    # Fill datasets based on target
    attributes = {}
    # Return results based on target
    cars_dataset.fill_dataset(
        outputs_dataset,
        saving_info=saving_infos,
        attributes=attributes,
    )

    return outputs_dataset
