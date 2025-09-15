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

import logging
import os
import shutil

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import Checker
from pyproj import CRS
from shapely.geometry import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.applications.auxiliary_filling import (
    auxiliary_filling_algo,
    auxiliary_filling_wrappers,
)

# CARS imports
from cars.applications.auxiliary_filling.abstract_auxiliary_filling_app import (
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

        if overloaded_conf["mode"] not in ["fill_nan", "full"]:
            raise RuntimeError(
                f"Invalid mode {overloaded_conf['mode']} for "
                "AuxiliaryFilling, supported modes are fill_nan "
                "and full"
            )

        overloaded_conf["texture_interpolator"] = conf.get(
            "texture_interpolator", "linear"
        )
        overloaded_conf["activated"] = conf.get("activated", False)
        overloaded_conf["use_mask"] = conf.get("use_mask", True)

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        auxiliary_filling_schema = {
            "method": str,
            "activated": bool,
            "mode": str,
            "use_mask": bool,
            "texture_interpolator": str,
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
        dump_dir,
        sensor_inputs,
        pairing,
        geom_plugin,
        texture_bands,
        orchestrator=None,
    ):
        """
        run AuxiliaryFillingFromSensors

        :param dsm_file: path to the filled dsm file
        :type dsm_file: str
        :param color_file: path to the color file
        :type color_file: str
        :param classification_file: path to the classification file
        :type classification_file: str
        :param dump_dir: output dump directory
        :type dump_dir: str
        :param sensor_inputs: dictionary containing paths to input images and
            models
        :type sensor_inputs: dict
        :param pairing: pairing between input images
        :type pairing: list
        :param geom_plugin: geometry plugin used for inverse locations
        :type geom_plugin: AbstractGeometry
        :param texture_bands: list of band names used for output texture
        :type texture_bands: list
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        """

        if not self.used_config["activated"]:
            return None
        if sensor_inputs is None:
            logging.error(
                "No sensor inputs were provided, "
                "auxiliary_filling will not run."
            )
            return None

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

        dump_dir = os.path.join(dump_dir, "auxiliary_filling")

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        color_not_filled_file = os.path.join(dump_dir, "texture_not_filled.tif")
        if os.path.exists(color_file):
            shutil.move(color_file, color_not_filled_file)

        classification_not_filled_file = None
        # classif_file could be defined without data attached
        if classif_file is not None and not os.path.exists(classif_file):
            classif_file = None

        if classif_file is not None:
            classification_not_filled_file = os.path.join(
                dump_dir, "classification_not_filled.tif"
            )
            shutil.move(classif_file, classification_not_filled_file)

        # Clean dump_dir at the end of processing if required
        if not self.used_config["save_intermediate_data"]:
            self.orchestrator.add_to_clean(dump_dir)

        # Create output CarsDataset
        aux_filled_image = cars_dataset.CarsDataset(
            "arrays", name="aux_filled_image"
        )

        # Create tiling grid
        ground_image_width, ground_image_height = inputs.rasterio_get_size(
            dsm_file
        )

        region_grid = tiling.generate_tiling_grid(
            0,
            0,
            ground_image_height,
            ground_image_width,
            1000,
            1000,
        )

        aux_filled_image.tiling_grid = region_grid

        # Initialize no data value
        classification_no_data_value = 0
        texture_no_data_value = 0
        color_dtype = np.float32
        classif_dtype = np.uint8

        if os.path.exists(color_not_filled_file):
            with rio.open(color_not_filled_file, "r") as descriptor:
                texture_no_data_value = descriptor.nodata
                color_dtype = descriptor.profile.get("dtype", np.float32)

        self.orchestrator.add_to_save_lists(
            os.path.join(dump_dir, color_file),
            cst.RASTER_COLOR_IMG,
            aux_filled_image,
            nodata=texture_no_data_value,
            dtype=color_dtype,
            cars_ds_name="filled_texture",
        )

        if classif_file is not None:
            if os.path.exists(classification_not_filled_file):
                with rio.open(
                    classification_not_filled_file, "r"
                ) as descriptor:
                    classification_no_data_value = descriptor.nodata
                    classif_dtype = descriptor.profile.get("dtype", np.uint8)

            self.orchestrator.add_to_save_lists(
                os.path.join(dump_dir, classif_file),
                cst.RASTER_CLASSIF,
                aux_filled_image,
                dtype=classif_dtype,
                nodata=classification_no_data_value,
                cars_ds_name="filled_classification",
            )

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos([aux_filled_image])

        reference_transform = inputs.rasterio_get_transform(dsm_file)
        reference_crs = inputs.rasterio_get_crs(dsm_file)

        # Pre-compute sensor bounds of all sensors to filter sensors that do
        # not intersect with tile in tasks
        sensor_bounds = auxiliary_filling_wrappers.compute_sensor_bounds(
            sensor_inputs, geom_plugin, reference_crs
        )

        for row in range(aux_filled_image.shape[0]):
            for col in range(aux_filled_image.shape[1]):

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
                    color_not_filled_file,
                    classification_not_filled_file,
                    sensor_inputs,
                    sensor_bounds,
                    pairing,
                    window,
                    reference_transform,
                    reference_crs,
                    full_saving_info,
                    geom_plugin,
                    texture_bands,
                    mode=self.used_config["mode"],
                    texture_interpolator=self.used_config[
                        "texture_interpolator"
                    ],
                    use_mask=self.used_config["use_mask"],
                )

        # Run tasks if an internal orchestrator is used, in order to save output
        # files
        if orchestrator is None:
            self.orchestrator.breakpoint()

        return aux_filled_image


def filling_from_sensor_wrapper(
    dsm_file,
    color_file,
    classification_file,
    sensor_inputs,
    sensor_bounds,
    pairing,
    window,
    transform,
    crs,
    saving_info,
    geom_plugin,
    texture_bands,
    mode,
    texture_interpolator,
    use_mask,
):
    """
    Fill color and classification from sensor information for a terrain tile

    :param dsm_file: path to the filled dsm file
    :type dsm_file: str
    :param color_file: path to the color file
    :type color_file: str
    :param classification_file: path to the classification file
    :type classification_file: str
    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param sensor_bounds: dictionary containing bounds of input sensors
    :type sensor_bounds: dict
    :param pairing: pairing between input images
    :type pairing: list
    :param window: window of the current tile
    :type window: dict
    :param transform: input geo transform
    :type transform: tuple
    :param crs: input crs
    :type crs: CRS
    :param saving_info: saving info for cars orchestrator
    :type saving_info: dict
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param texture_bands: list of band names used for output texture
    :type texture_bands: list
    :param mode: geometry plugin used for inverse locations
    :type mode: str
    :param texture_interpolator: scipy interpolator use to interpolate color
        values
    :type texture_interpolator: str
    :param use_mask: use mask information from sensors in color computation
    :type use_mask: bool

    """

    col_min = window["col_min"]
    col_max = window["col_max"]
    row_min = window["row_min"]
    row_max = window["row_max"]

    col_min_ground = col_min * transform[0] + transform[2]
    col_max_ground = col_max * transform[0] + transform[2]
    row_min_ground = row_min * transform[4] + transform[5]
    row_max_ground = row_max * transform[4] + transform[5]

    ground_polygon = Polygon(
        [
            (col_min_ground, row_min_ground),
            (col_min_ground, row_max_ground),
            (col_max_ground, row_max_ground),
            (col_max_ground, row_min_ground),
            (col_min_ground, row_min_ground),
        ]
    )

    cols = (
        np.linspace(col_min, col_max, col_max - col_min) * transform[0]
        + transform[2]
    )
    rows = (
        np.linspace(row_min, row_max, row_max - row_min) * transform[4]
        + transform[5]
    )

    cols_values_2d, rows_values_2d = np.meshgrid(cols, rows)

    stacked_values = np.vstack([cols_values_2d.ravel(), rows_values_2d.ravel()])

    lon_lat = projection.point_cloud_conversion_crs(
        stacked_values.transpose(), crs, CRS(4326)
    )

    rio_window = rio.windows.Window.from_slices(
        (
            row_min,
            row_max,
        ),
        (
            col_min,
            col_max,
        ),
    )

    # From input DSM read altitudes for localisation and no-data mask.
    # if fill_nan mode is chosed, all values valid in dsm and invalid in color
    # will be filled
    # if not, all values valid in dsm will be filled
    # Note that the same pixels are filled for color and classification
    with rio.open(dsm_file) as dsm_image:
        alt_values = dsm_image.read(1, window=rio_window)
        target_mask = dsm_image.read_masks(1, window=rio_window)
        dsm_profile = dsm_image.profile

    nodata_color = None
    nodata_classif = None

    if os.path.exists(color_file):
        with rio.open(color_file) as color_image:
            profile = color_image.profile
            nodata_color = color_image.nodata

            number_of_color_bands = color_image.count
            color_band_names = list(color_image.descriptions)

            color_values = color_image.read(window=rio_window)

            if mode == "fill_nan":
                target_mask = target_mask & ~color_image.read_masks(
                    1, window=rio_window
                )
    else:
        profile = dsm_profile
        number_of_color_bands = inputs.rasterio_get_nb_bands(
            sensor_inputs[list(sensor_inputs.keys())[0]].get("texture", None)
        )
        color_values = np.full(
            (number_of_color_bands, *target_mask.shape), np.nan
        )
        color_band_names = inputs.get_descriptions_bands(
            sensor_inputs[list(sensor_inputs.keys())[0]].get("texture", None)
        )
        # update profile
        profile.update(count=number_of_color_bands)

    number_of_classification_bands = 0
    classification_values = None
    classification_band_names = None

    if classification_file is not None:
        if os.path.exists(classification_file):
            with rio.open(classification_file) as classification_image:
                nodata_classif = None

                number_of_classification_bands = classification_image.count

                classification_values = classification_image.read(
                    window=rio_window
                )
                classification_band_names = list(
                    classification_image.descriptions
                )
        else:
            profile = dsm_profile
            number_of_classification_bands = inputs.rasterio_get_nb_bands(
                sensor_inputs[list(sensor_inputs.keys())[0]].get(
                    "classification", None
                )
            )
            classification_values = np.full(
                (number_of_classification_bands, *target_mask.shape), np.nan
            )
            classification_band_names = inputs.get_descriptions_bands(
                sensor_inputs[list(sensor_inputs.keys())[0]].get(
                    "classification", None
                )
            )
            # update profile
            profile.update(count=number_of_classification_bands)

    # 1D index list from target mask
    index_1d = target_mask.flatten().nonzero()[0]

    # Remove sensor that don't intesects with current tile
    filtered_sensor_inputs = auxiliary_filling_wrappers.filter_sensor_inputs(
        sensor_inputs, sensor_bounds, ground_polygon
    )

    if len(index_1d) > 0:
        # Fill required pixels
        color_values_filled, classification_values_filled = (
            auxiliary_filling_algo.fill_auxiliary(
                filtered_sensor_inputs,
                pairing,
                lon_lat[index_1d, 0],
                lon_lat[index_1d, 1],
                alt_values.ravel()[index_1d],
                geom_plugin,
                number_of_color_bands,
                number_of_classification_bands,
                texture_bands,
                texture_interpolator=texture_interpolator,
                use_mask=use_mask,
            )
        )

        # Change nan to nodata
        if nodata_color is not None:
            color_values_filled[np.isnan(color_values_filled)] = nodata_color
        if nodata_classif is not None:
            classification_values_filled[
                np.isnan(classification_values_filled)
            ] = nodata_classif

        # forward filled values in the output buffer
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

    row_arr = np.array(range(row_min, row_max))
    col_arr = np.array(range(col_min, col_max))

    values = {}
    coords = {cst.ROW: row_arr, cst.COL: col_arr}

    if len(color_band_names) == 0 or None in color_band_names:
        color_band_names = [
            str(current_band) for current_band in range(number_of_color_bands)
        ]

    values[cst.RASTER_COLOR_IMG] = (
        [cst.BAND_IM, cst.ROW, cst.COL],
        color_values,
    )
    coords[cst.BAND_IM] = list(color_band_names)

    if classification_values is not None:
        values[cst.RASTER_CLASSIF] = (
            [cst.BAND_CLASSIF, cst.ROW, cst.COL],
            classification_values,
        )
        coords[cst.BAND_CLASSIF] = list(classification_band_names)

    attributes = {}

    dataset = xr.Dataset(
        values,
        coords=coords,
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
