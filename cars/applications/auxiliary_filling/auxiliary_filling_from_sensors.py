# !/usr/bin/env python
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
this module contains the AuxiliaryFillingFromSensors application class.
"""

import os

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications.auxiliary_filling import auxiliary_filling_tools

# CARS imports
from cars.applications.auxiliary_filling.auxiliary_filling import (
    AuxiliaryFilling,
)
from cars.core import constants as cst
from cars.core import inputs, projection, tiling
from cars.data_structures import cars_dataset, format_transformation


class AuxiliaryFillingFromSensors(
    AuxiliaryFilling, short_name="auxiliary_filling_from_sensors"
):
    """
    AuxiliaryFillingFromSensors Application
    """

    def __init__(self, conf=None):
        """
        Init function of AuxiliaryFillingFromSensors

        :param conf: configuration for AuxiliaryFillingFromSensors
        :return: an application_to_use object
        """

        self.orchestrator = None

        super().__init__(conf=conf)

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
            "method", "auxiliary_filling_from_sensors"
        )

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        auxiliary_filling_schema = {
            "method": str,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(auxiliary_filling_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        dsm_file,
        color_file,
        classif_file,
        out_dir,
        sensor_inputs,
        geom_plugin,
        orchestrator=None,
    ):
        """
        run AuxiliaryFillingFromSensors
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

        print(f"bounds {inputs.rasterio_get_bounds(dsm_file)}")

        print(f"bounds {inputs.rasterio_get_size(dsm_file)}")

        # Create CarsDataset
        # output color
        aux_filled_image = cars_dataset.CarsDataset(
            "arrays", name="aux_filled_image"
        )

        ground_image_width, ground_image_height = inputs.rasterio_get_size(
            dsm_file
        )

        reference_transform = inputs.rasterio_get_transform(dsm_file)
        reference_epsg = inputs.rasterio_get_epsg(dsm_file)

        print(f"reference_transform {reference_transform}")
        print(f"reference_epsg {reference_epsg}")

        region_grid = tiling.generate_tiling_grid(
            0,
            0,
            ground_image_height,
            ground_image_width,
            300,  # TODO tiling
            300,  # TODO tiling
        )

        # Compute tiling grid
        aux_filled_image.tiling_grid = region_grid

        self.orchestrator.add_to_save_lists(
            os.path.join(out_dir, "filled_color.tif"),
            cst.RASTER_COLOR_IMG,
            aux_filled_image,
            cars_ds_name="filled_color",
        )

        self.orchestrator.add_to_save_lists(
            os.path.join(out_dir, "ground_row_values.tif"),
            "row_values_2d",
            aux_filled_image,
            cars_ds_name="ground_row_values",
        )
        self.orchestrator.add_to_save_lists(
            os.path.join(out_dir, "ground_col_values.tif"),
            "col_values_2d",
            aux_filled_image,
            cars_ds_name="ground_col_values",
        )
        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos([aux_filled_image])

        for row in range(aux_filled_image.shape[0]):
            for col in range(aux_filled_image.shape[1]):
                print(f"colrow {col} {row}")

                # TODO?
                (
                    pc_row,
                    pc_col,
                ) = format_transformation.get_corresponding_indexes(row, col)
                print(f"pc_row pc_col {pc_row} {pc_col}")

                # Get window
                window = cars_dataset.window_array_to_dict(
                    aux_filled_image.tiling_grid[row, col]
                )
                print(f"window {window}")

                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )
                aux_filled_image[
                    row, col
                ] = self.orchestrator.cluster.create_task(
                    filling_from_sensor_wrapper, nout=1
                )(
                    row,
                    col,
                    dsm_file,
                    color_file,
                    sensor_inputs,
                    window,
                    reference_transform,
                    reference_epsg,
                    full_saving_info,
                    geom_plugin,
                )

        self.orchestrator.breakpoint()
        return aux_filled_image


def filling_from_sensor_wrapper(
    row,
    col,
    dsm_file,
    color_file,
    sensor_inputs,
    window,
    transform,
    epsg,
    saving_info,
    geom_plugin,
):
    """
    filling from sensor for a terrain tile
    """

    cols = np.arange(
        window["col_min"] * transform[0] + transform[2],
        window["col_max"] * transform[0] + transform[2],
        transform[0],
    )
    rows = np.arange(
        window["row_min"] * transform[4] + transform[5],
        window["row_max"] * transform[4] + transform[5],
        transform[4],
    )

    cols_values_2d, rows_values_2d = np.meshgrid(cols, rows)

    stacked_values = np.vstack([cols_values_2d.ravel(), rows_values_2d.ravel()])

    lon_lat = projection.point_cloud_conversion(
        stacked_values.transpose(), epsg, 4326
    )

    rio_window = rio.windows.Window.from_slices(
        (
            window["row_min"],
            window["row_max"],
        ),
        (
            window["col_min"],
            window["col_max"],
        ),
    )

    mode = "fill_nan"

    with rio.open(dsm_file) as dsm_image:
        alt_values = dsm_image.read(1, window=rio_window)
        target_mask = dsm_image.read_masks(1, window=rio_window)

    with rio.open(color_file) as color_image:
        img_crs = color_image.profile["crs"]
        img_transform = color_image.profile["transform"]
        profile = color_image.profile

        color_values = color_image.read(window=rio_window)

        if mode == "fill_nan":
            # TODO different no data in bands ??
            target_mask = target_mask & ~color_image.read_masks(
                1, window=rio_window
            )

    index_1d = target_mask.flatten().nonzero()[0]

    color_values_filled = auxiliary_filling_tools.fill_auxiliary(
        sensor_inputs,
        lon_lat[:, 0],
        lon_lat[:, 1],
        alt_values.ravel(),
        geom_plugin,
    )

    color_values_filled_v2 = auxiliary_filling_tools.fill_auxiliary(
        sensor_inputs,
        lon_lat[index_1d, 0],
        lon_lat[index_1d, 1],
        alt_values.ravel()[index_1d],
        geom_plugin,
    )

    color_values_filled = color_values_filled.reshape(color_values.shape)

    np.put(color_values[0, :, :], index_1d, color_values_filled_v2[0, :])
    np.put(color_values[1, :, :], index_1d, color_values_filled_v2[1, :])
    np.put(color_values[2, :, :], index_1d, color_values_filled_v2[2, :])
    np.put(color_values[3, :, :], index_1d, color_values_filled_v2[3, :])

    values = {
        cst.RASTER_COLOR_IMG: ([cst.BAND_IM, cst.ROW, cst.COL], color_values)
    }

    values["row_values_2d"] = ([cst.ROW, cst.COL], rows_values_2d)
    values["col_values_2d"] = ([cst.ROW, cst.COL], cols_values_2d)

    row_arr = np.array(range(window["row_min"], window["row_max"]))
    col_arr = np.array(range(window["col_min"], window["col_max"]))

    print(f"inside filling_from_sensor_wrapper {row} {col} ")

    print(f"window {window} ")

    print(f"rio_window {rio_window}")
    print(f"color_values.shape {color_values.shape}")

    attributes = {}

    dataset = xr.Dataset(
        values,
        coords={
            cst.BAND_IM: ["R", "G", "B", "NIR"],
            cst.ROW: row_arr,
            cst.COL: col_arr,
        },
    )

    attributes[cst.EPI_CRS] = img_crs
    attributes[cst.EPI_TRANSFORM] = img_transform

    print(f"attributes {attributes}")

    print(f"profile {profile}")
    cars_dataset.fill_dataset(
        dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        attributes=attributes,
        overlaps=None,
    )

    return dataset
