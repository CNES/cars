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

# Standard imports
import collections
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import And, Checker

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications.ground_truth_reprojection import (
    ground_truth_reprojection,
)
from cars.applications.ground_truth_reprojection import (
    ground_truth_reprojection_tools as gnd_truth_tools,
)

# CARS imports
from cars.core import constants as cst
from cars.core import inputs
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
        self.tile_size = self.used_config["tile_size"]

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
        overloaded_conf["tile_size"] = conf.get("tile_size", 2500)
        # save_intermediate_data not used in this app, but overiden
        overloaded_conf["save_intermediate_data"] = True

        ground_truth_reprojection_schema = {
            "method": str,
            "target": And(
                str, lambda input: input in ["epipolar", "sensor", "all"]
            ),
            "tile_size": And(int, lambda size: size > 0),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(ground_truth_reprojection_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa: C901
        self,
        sensor_left,
        sensor_right,
        grid_left,
        grid_right,
        geom_plugin,
        geom_plugin_dem_median,
        disp_to_alt_ratio,
        auxiliary_values,
        auxiliary_interp,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run direct localization for ground truth disparity

        :param sensor_left: Tiled sensor left image.
            Dict must contain keys: "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolute.
        :type sensor_left: CarsDataset
        :param sensor_right: Tiled sensor right image.
            Dict must contain keys: "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolute.
        :type sensor_right: CarsDataset
        :param grid_left: Grid left.
        :type grid_left: CarsDataset
        :param grid_right: Grid right.
        :type grid_right: CarsDataset
        :param geom_plugin_dem_median: Geometry plugin with dem median
        :type geom_plugin_dem_median: geometry_plugin
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

        # Get profile
        with rio.open(sensor_left[sens_cst.INPUT_IMG]) as src_left:
            width_left = src_left.width
            height_left = src_left.height
            transform_left = src_left.transform

        with rio.open(sensor_right[sens_cst.INPUT_IMG]) as src_right:
            width_right = src_right.width
            height_right = src_right.height
            transform_right = src_right.transform

        raster_profile_left = collections.OrderedDict(
            {
                "height": height_left,
                "width": width_left,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform_left,
                "tiled": True,
            }
        )

        raster_profile_right = collections.OrderedDict(
            {
                "height": height_right,
                "width": width_right,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform_right,
                "tiled": True,
            }
        )

        if self.used_config["target"] in ["all", "epipolar"]:

            # Create cars datasets
            epi_disparity_ground_truth_left = cars_dataset.CarsDataset(
                "arrays", name="epipolar_disparity_ground_truth_left" + pair_key
            )
            epi_disp_ground_truth_right = cars_dataset.CarsDataset(
                "arrays",
                name="epipolar_disparity_ground_truth_right" + pair_key,
            )

            epi_disparity_ground_truth_left.create_grid(
                grid_left.attributes["epipolar_size_x"],
                grid_left.attributes["epipolar_size_y"],
                self.tile_size,
                self.tile_size,
                0,
                0,
            )
            epi_disp_ground_truth_right.tiling_grid = (
                epi_disparity_ground_truth_left.tiling_grid
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder, "epipolar_disp_ground_truth_left.tif"
                ),
                cst.EPI_GROUND_TRUTH,
                epi_disparity_ground_truth_left,
                cars_ds_name="epipolar_disparity_ground_truth",
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(
                    pair_folder, "epipolar_disp_ground_truth_right.tif"
                ),
                cst.EPI_GROUND_TRUTH,
                epi_disp_ground_truth_right,
                cars_ds_name="epipolar_disparity_ground_truth",
            )

            # Save all file that are in inputs
            if auxiliary_values is not None:
                for key in auxiliary_values.keys():
                    if key in (cst.DSM_COLOR, cst.DSM_WEIGHTS_SUM):
                        option = False
                    else:
                        option = True

                    out_file_left_name = os.path.join(
                        pair_folder, key + "_left_epipolar.tif"
                    )

                    orchestrator.add_to_save_lists(
                        out_file_left_name,
                        key,
                        epi_disparity_ground_truth_left,
                        dtype=inputs.rasterio_get_dtype(auxiliary_values[key]),
                        nodata=inputs.rasterio_get_nodata(
                            auxiliary_values[key]
                        ),
                        cars_ds_name=key,
                        optional_data=option,
                    )

                    out_file_right_name = os.path.join(
                        pair_folder, key + "_right_epipolar.tif"
                    )

                    orchestrator.add_to_save_lists(
                        out_file_right_name,
                        key,
                        epi_disp_ground_truth_right,
                        dtype=inputs.rasterio_get_dtype(auxiliary_values[key]),
                        nodata=inputs.rasterio_get_nodata(
                            auxiliary_values[key]
                        ),
                        cars_ds_name=key,
                        optional_data=option,
                    )

            # Get saving infos in order to save tiles when they are computed
            [saving_infos_epi_left] = self.orchestrator.get_saving_infos(
                [epi_disparity_ground_truth_left]
            )
            [saving_infos_epi_right] = self.orchestrator.get_saving_infos(
                [epi_disp_ground_truth_right]
            )

            for col in range(
                epi_disparity_ground_truth_left.tiling_grid.shape[1]
            ):
                for row in range(
                    epi_disparity_ground_truth_left.tiling_grid.shape[0]
                ):
                    # update saving infos with row col
                    full_saving_info_left = ocht.update_saving_infos(
                        saving_infos_epi_left, row=row, col=col
                    )
                    full_saving_info_right = ocht.update_saving_infos(
                        saving_infos_epi_right, row=row, col=col
                    )

                    # generate ground truth
                    (
                        epi_disparity_ground_truth_left[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        sensor_left,
                        grid_left,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "epipolar",
                        full_saving_info_left,
                        epi_disparity_ground_truth_left.tiling_grid[row, col],
                        auxiliary_values,
                        auxiliary_interp,
                        geom_plugin_dem_median=geom_plugin_dem_median,
                        window_dict=(
                            epi_disparity_ground_truth_left.get_window_as_dict(
                                row, col
                            )
                        ),
                    )

                    (
                        epi_disp_ground_truth_right[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        sensor_right,
                        grid_right,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "epipolar",
                        full_saving_info_right,
                        epi_disp_ground_truth_right.tiling_grid[row, col],
                        auxiliary_values,
                        auxiliary_interp,
                        geom_plugin_dem_median=geom_plugin_dem_median,
                        reverse=True,
                        window_dict=(
                            epi_disp_ground_truth_right.get_window_as_dict(
                                row, col
                            )
                        ),
                    )

        if self.used_config["target"] in ["all", "sensor"]:

            sensor_dsm_gt_left = cars_dataset.CarsDataset(
                "arrays", name="sensor_dsm_ground_truth_left" + pair_key
            )
            sensor_dsm_gt_right = cars_dataset.CarsDataset(
                "arrays", name="sensor_dsm_ground_truth_right" + pair_key
            )

            # update grid
            sensor_dsm_gt_left.create_grid(
                width_left, height_left, self.tile_size, self.tile_size, 0, 0
            )
            sensor_dsm_gt_right.create_grid(
                width_right, height_right, self.tile_size, self.tile_size, 0, 0
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "sensor_dsm_ground_truth_left.tif"),
                cst.SENSOR_GROUND_TRUTH,
                sensor_dsm_gt_left,
                cars_ds_name="sensor_dsm_ground_truth_left",
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "sensor_dsm_ground_truth_right.tif"),
                cst.SENSOR_GROUND_TRUTH,
                sensor_dsm_gt_right,
                cars_ds_name="sensor_dsm_ground_truth_right",
            )

            # Save all file that are in inputs
            if auxiliary_values is not None:
                for key in auxiliary_values.keys():
                    if key in (cst.DSM_COLOR, cst.DSM_WEIGHTS_SUM):
                        option = False
                    else:
                        option = True

                    out_file_left_name = os.path.join(
                        pair_folder, key + "_left_sensor.tif"
                    )

                    orchestrator.add_to_save_lists(
                        out_file_left_name,
                        key,
                        sensor_dsm_gt_left,
                        dtype=inputs.rasterio_get_dtype(auxiliary_values[key]),
                        nodata=inputs.rasterio_get_nodata(
                            auxiliary_values[key]
                        ),
                        cars_ds_name=key,
                        optional_data=option,
                    )

                    out_file_right_name = os.path.join(
                        pair_folder, key + "_right_sensor.tif"
                    )

                    orchestrator.add_to_save_lists(
                        out_file_right_name,
                        key,
                        sensor_dsm_gt_right,
                        dtype=inputs.rasterio_get_dtype(auxiliary_values[key]),
                        nodata=inputs.rasterio_get_nodata(
                            auxiliary_values[key]
                        ),
                        cars_ds_name=key,
                        optional_data=option,
                    )

            # Get saving infos in order to save tiles when they are computed
            [saving_infos_sensor_left] = self.orchestrator.get_saving_infos(
                [sensor_dsm_gt_left]
            )
            [saving_infos_sensor_right] = self.orchestrator.get_saving_infos(
                [sensor_dsm_gt_right]
            )

            # left
            for col in range(sensor_dsm_gt_left.tiling_grid.shape[1]):
                for row in range(sensor_dsm_gt_left.tiling_grid.shape[0]):
                    full_saving_info_left = ocht.update_saving_infos(
                        saving_infos_sensor_left, row=row, col=col
                    )
                    (
                        sensor_dsm_gt_left[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        sensor_left,
                        grid_left,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "sensor",
                        full_saving_info_left,
                        sensor_dsm_gt_left.tiling_grid[row, col],
                        auxiliary_values,
                        auxiliary_interp,
                        raster_profile=raster_profile_left,
                        window_dict=sensor_dsm_gt_left.get_window_as_dict(
                            row, col
                        ),
                    )

            # right
            for col in range(sensor_dsm_gt_right.tiling_grid.shape[1]):
                for row in range(sensor_dsm_gt_right.tiling_grid.shape[0]):
                    full_saving_info_right = ocht.update_saving_infos(
                        saving_infos_sensor_right, row=row, col=col
                    )
                    (
                        sensor_dsm_gt_right[row, col]
                    ) = self.orchestrator.cluster.create_task(
                        maps_generation_wrapper, nout=1
                    )(
                        sensor_right,
                        grid_right,
                        geom_plugin,
                        disp_to_alt_ratio,
                        "sensor",
                        full_saving_info_right,
                        sensor_dsm_gt_right.tiling_grid[row, col],
                        auxiliary_values,
                        auxiliary_interp,
                        raster_profile=raster_profile_right,
                        window_dict=sensor_dsm_gt_right.get_window_as_dict(
                            row, col
                        ),
                    )


def maps_generation_wrapper(
    sensor_left,
    grid_left,
    geom_plugin,
    disp_to_alt_ratio,
    target,
    saving_infos,
    window,
    auxiliary_values,
    auxiliary_interp,
    raster_profile=None,
    geom_plugin_dem_median=None,
    reverse=False,
    window_dict=None,
):
    """
    Computes ground truth epipolar disparity map and sensor geometry.

    :param sensor_left: sensor data
        Dict must contain keys: "image", "color", "geomodel",
        "no_data", "mask". Paths must be absolute.
    :type sensor_left: dict
    :param grid_left: Grid left.
    :type grid_left: CarsDataset
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
    :param raster_profile: dictionnary containing dataset information
    :type raster_profile: dict
    :param geom_plugin_dem_median: Geometry plugin with dem median
    :type geom_plugin_dem_median: geometry_plugin
    :param reverse: true if right-> left
    :type reverse: bool
    :param window_dict: window as dict
    """

    ground_truth, direct_loc = gnd_truth_tools.get_ground_truth(
        geom_plugin,
        grid_left,
        sensor_left,
        disp_to_alt_ratio,
        target,
        window,
        geom_plugin_dem_median,
        reverse=reverse,
    )

    constant_for_dataset = cst.EPI_GROUND_TRUTH
    if target == "sensor":
        constant_for_dataset = cst.SENSOR_GROUND_TRUTH

    rows = np.arange(window[0], window[1])
    cols = np.arange(window[2], window[3])

    values = {
        constant_for_dataset: (
            [
                cst.ROW,
                cst.COL,
            ],
            ground_truth,
        )
    }
    outputs_dataset = xr.Dataset(
        values,
        coords={cst.ROW: rows, cst.COL: cols},
    )

    if auxiliary_values is not None:
        for key in auxiliary_values.keys():
            if auxiliary_interp is not None and key in auxiliary_interp:
                interpolation = auxiliary_interp[key]
            else:
                interpolation = "nearest"

            band_description = inputs.get_descriptions_bands(
                auxiliary_values[key]
            )

            keep_band = False
            if band_description[0] is not None or len(band_description) > 1:
                if len(band_description) == 1:
                    band_description = np.array([band_description[0]])
                else:
                    band_description = list(band_description)

                band_description = [
                    "band_" + str(i + 1) if v is None else v
                    for i, v in enumerate(band_description)
                ]

                outputs_dataset.coords[cst.BAND_IM] = (
                    key,
                    band_description,
                )
                dim = [key, cst.Y, cst.X]
                keep_band = True
            else:
                dim = [cst.Y, cst.X]

            interp_value = gnd_truth_tools.resample_auxiliary_values(
                direct_loc,
                auxiliary_values[key],
                window,
                interpolation,
                keep_band,
            )

            outputs_dataset[key] = (dim, interp_value)

    # Fill datasets based on target
    attributes = {}
    # Return results based on target
    cars_dataset.fill_dataset(
        outputs_dataset,
        saving_info=saving_infos,
        window=window_dict,
        attributes=attributes,
        profile=raster_profile,
    )

    return outputs_dataset
