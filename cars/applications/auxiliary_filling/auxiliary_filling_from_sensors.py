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
from cars.data_structures import cars_dataset


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

        overloaded_conf["mode"] = conf.get("mode", "fill_nan")

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        auxiliary_filling_schema = {
            "method": str,
            "mode": str,
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

        region_grid = tiling.generate_tiling_grid(
            0,
            0,
            ground_image_height,
            ground_image_width,
            1000,  # TODO tiling
            1000,  # TODO tiling
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
            os.path.join(out_dir, "filled_classification.tif"),
            cst.RASTER_CLASSIF,
            aux_filled_image,
            cars_ds_name="filled_classif",
        )

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos([aux_filled_image])

        for row in range(aux_filled_image.shape[0]):
            for col in range(aux_filled_image.shape[1]):

                # TODO?
                # (
                #     pc_row,
                #     pc_col,
                # ) = format_transformation.get_corresponding_indexes(row, col)

                # Get window
                window = cars_dataset.window_array_to_dict(
                    aux_filled_image.tiling_grid[row, col]
                )

                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )
                aux_filled_image[
                    row, col
                ] = self.orchestrator.cluster.create_task(
                    filling_from_sensor_wrapper, nout=1
                )(
                    dsm_file,
                    color_file,
                    classif_file,
                    sensor_inputs,
                    window,
                    reference_transform,
                    reference_epsg,
                    full_saving_info,
                    geom_plugin,
                    self.used_config["mode"],
                )

        #
        if orchestrator is None:
            self.orchestrator.breakpoint()

        return aux_filled_image


def filling_from_sensor_wrapper(
    dsm_file,
    color_file,
    classification_file,
    sensor_inputs,
    window,
    transform,
    epsg,
    saving_info,
    geom_plugin,
    mode,
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

    with rio.open(dsm_file) as dsm_image:
        alt_values = dsm_image.read(1, window=rio_window)
        target_mask = dsm_image.read_masks(1, window=rio_window)

    with rio.open(color_file) as color_image:
        profile = color_image.profile

        number_of_color_bands = color_image.count

        color_values = color_image.read(window=rio_window)

        if mode == "fill_nan":
            target_mask = target_mask & ~color_image.read_masks(
                1, window=rio_window
            )

    with rio.open(classification_file) as classification_image:

        number_of_classification_bands = classification_image.count

        classification_values = classification_image.read(window=rio_window)
        classification_band_names = list(classification_image.descriptions)

    index_1d = target_mask.flatten().nonzero()[0]

    if len(index_1d) > 0:
        color_values_filled, classification_values_filled = (
            auxiliary_filling_tools.fill_auxiliary(
                sensor_inputs,
                lon_lat[index_1d, 0],
                lon_lat[index_1d, 1],
                alt_values.ravel()[index_1d],
                geom_plugin,
                number_of_color_bands,
                number_of_classification_bands,
            )
        )

        for band_index in range(number_of_color_bands):
            np.put(
                color_values[band_index, :, :],
                index_1d,
                color_values_filled[band_index, :],
            )

        for band_index in range(number_of_classification_bands):
            np.put(
                classification_values[band_index, :, :],
                index_1d,
                classification_values_filled[band_index, :],
            )

    values = {}

    values[cst.RASTER_COLOR_IMG] = (
        [cst.BAND_IM, cst.ROW, cst.COL],
        color_values,
    )
    values[cst.RASTER_CLASSIF] = (
        [cst.BAND_CLASSIF, cst.ROW, cst.COL],
        classification_values,
    )

    row_arr = np.array(range(window["row_min"], window["row_max"]))
    col_arr = np.array(range(window["col_min"], window["col_max"]))

    attributes = {}

    band_names = ["R", "G", "B", "NIR"]

    dataset = xr.Dataset(
        values,
        coords={
            cst.BAND_CLASSIF: classification_band_names,
            cst.BAND_IM: band_names[:number_of_color_bands],
            cst.ROW: row_arr,
            cst.COL: col_arr,
        },
    )
    cars_dataset.fill_dataset(
        dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        attributes=attributes,
        overlaps=None,
    )

    return dataset
