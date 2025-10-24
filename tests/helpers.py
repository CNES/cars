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
Helpers shared testing generic module:
contains global shared generic functions for tests/*.py
TODO: Try to put the most part in cars source code (if pertinent) and
organized functionnally.
TODO: add conftest.py general tests conf with tests refactor.
"""

import json
import logging

# Standard imports
import os
import tempfile

# Third party imports
from shutil import copy2

import numpy as np
import pandas
import pandora
import rasterio as rio
import xarray as xr
from pandora.check_configuration import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
)
from pandora.state_machine import PandoraMachine
from pytest_check import check
from scipy.ndimage import zoom

from cars.applications.dense_matching.loaders.pandora_loader import (
    check_input_section_custom_cars,
    get_config_input_custom_cars,
)

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core.datasets import get_color_bands
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.pipelines.parameters import sensor_inputs

# Specific values
# 0 = valid pixels
# 255 = value used as no data during the resampling in the epipolar geometry
PROTECTED_VALUES = [255]


def cars_path():
    """
    Return root of cars source directory
    One level down from tests
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def cars_copy2(in_data, out_data):
    """
    Copy raster file from in_data to out_data,
    adding overviews to out_data
    """

    with rio.open(in_data, "r+") as src:
        data = src.read()
        nodata = src.nodata

        nb_bands = data.shape[0]
        if nb_bands == 1:
            selected_data = np.concatenate([data, data, data], axis=0)
        elif nb_bands == 2:
            selected_data = np.concatenate([data, data[1:2]], axis=0)
        else:
            selected_data = data[:3]

        # Create mask for valid data
        if nodata is not None:
            valid_mask = selected_data != nodata
        else:
            valid_mask = ~np.isnan(selected_data)

        # Only process valid data for normalization
        valid_data = selected_data[valid_mask]

        if len(valid_data) > 0:
            # Normalize only valid data to uint8 range
            data_min = np.nanmin(valid_data)
            data_max = np.nanmax(valid_data)

            if data_max > data_min:
                data_uint8 = np.full_like(selected_data, 0, dtype="uint8")
                # Normalize valid data
                normalized = (
                    (selected_data - data_min) / (data_max - data_min) * 254
                ).astype("uint8")
                data_uint8[valid_mask] = normalized[valid_mask]
                # Set nodata pixels to 255 (valid range for uint8)
                data_uint8[~valid_mask] = 255
            else:
                # All valid pixels have the same value
                data_uint8 = np.full_like(selected_data, 127, dtype="uint8")
                data_uint8[~valid_mask] = 255
        else:
            # No valid data
            data_uint8 = np.full_like(selected_data, 255, dtype="uint8")

    # downsample for overview
    zoom_factor = (1, 0.25, 0.25)
    data_uint8 = zoom(data_uint8, zoom_factor, order=1)

    # Save as PNG overview
    base_name = os.path.splitext(out_data)[0]
    png_path = f"{base_name}_overview.png"

    png_profile = {
        "driver": "PNG",
        "height": src.height,
        "width": src.width,
        "count": data_uint8.shape[0],
        "dtype": "uint8",
    }

    with rio.open(png_path, "w", **png_profile) as png_dst:
        png_dst.write(data_uint8)

    # Copy file
    copy2(in_data, out_data)


def generate_input_json(
    input_json,
    output_directory,
    orchestrator_mode,
    orchestrator_parameters=None,
    geometry_plugin_name=None,
):
    """
    Load a partially filled input.json, fill it with output directory
    and orchestrator mode, and transform relative path to
     absolute paths. Generates a new json dumped in output directory

    :param input_json: input json
    :type input_json: str
    :param output_directory: absolute path out directory
    :type output_directory: str
    :param orchestrator_mode: orchestrator mode
    :type orchestrator_mode: str
    :param orchestrator_parameters: advanced orchestrator params
    :type orchestrator_parameters: dict

    :return: path of generated json, dict input config
    :rtype: str, dict
    """
    # Load dict
    json_dir_path = os.path.dirname(input_json)
    with open(input_json, "r", encoding="utf8") as fstream:
        config = json.load(fstream)
    # Overload orchestrator
    config["orchestrator"] = {"mode": orchestrator_mode}
    if orchestrator_mode in ("mp", "multiprocessing"):
        config["orchestrator"]["per_job_timeout"] = 120
    if orchestrator_parameters is not None:
        config["orchestrator"].update(orchestrator_parameters)
    # Overload output directory
    config["output"] = {"directory": os.path.join(output_directory, "output")}

    # set installed (Shareloc) geometry plugin if not specified
    if geometry_plugin_name is None:
        geometry_plugin_name = get_geometry_plugin().plugin_name

    config["advanced"] = {
        "geometry_plugin": geometry_plugin_name,
        "pipeline": "default",
    }

    # Create keys
    if "applications" not in config:
        config["applications"] = {}

    if "advanced" not in config:
        config["advanced"] = {}

    # transform paths
    new_config = config.copy()

    new_config["inputs"] = sensor_inputs.sensors_check_inputs(
        new_config["inputs"], config_dir=json_dir_path
    )

    # dump json
    new_json_path = os.path.join(output_directory, "new_input.json")
    with open(new_json_path, "w", encoding="utf8") as fstream:
        json.dump(new_config, fstream, indent=2)

    return new_json_path, new_config


def absolute_data_path(data_path):
    """
    Return a full absolute path to test data
    environment variable.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_folder, data_path)


def get_geoid_path():
    return os.path.join(cars_path(), "cars/conf/geoid/egm96.grd")


def get_geometry_plugin(
    conf="SharelocGeometry", dem=None, default_alt=None
) -> AbstractGeometry:
    """
    returns the default Shareloc geometry plugin for test

    :param conf: conf to use
    :param dem: if defined, dem to use in AbstractGeometry object returned
    :param default_alt: default alt optional used in Abstractgeometry returned
    :return: AbstractGeometry object to use in tests
    """

    return AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        conf,
        dem=dem,
        geoid=get_geoid_path(),
        default_alt=default_alt,
    )


def temporary_dir():
    """
    Returns path to temporary dir from CARS_TEST_TEMPORARY_DIR environment
    variable. Defaults to default temporary directory
    (/tmp or TMPDIR environment variable)
    """
    if "CARS_TEST_TEMPORARY_DIR" not in os.environ:
        # return default tmp dir
        logging.info(
            "CARS_TEST_TEMPORARY_DIR is not set, "
            "cars will use default temporary directory instead"
        )
        return tempfile.gettempdir()
    # return env defined tmp dir
    return os.environ["CARS_TEST_TEMPORARY_DIR"]


@check.check_func  # pylint: disable=no-member
def assert_same_images(actual, expected, rtol=0, atol=0):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            assert rio_actual.transform == rio_expected.transform
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            data1 = rio_actual.read()
            data2 = rio_expected.read()
            data1[data1 == rio_actual.nodata] = 0
            data1[np.isnan(data1)] = 0
            data2[data2 == rio_expected.nodata] = 0
            data2[np.isnan(data2)] = 0
            np.testing.assert_allclose(data1, data2, rtol=rtol, atol=atol)


def assert_same_carsdatasets(actual, expected, rtol=0, atol=0):
    """
    Compare two Carsdatasets:
    """
    assert (
        list(actual.attributes.keys()).sort()
        == list(expected.attributes.keys()).sort()
    )
    for key in expected.attributes.keys():
        if isinstance(expected.attributes[key], np.ndarray):
            np.testing.assert_allclose(
                actual.attributes[key], expected.attributes[key]
            )
        elif key == "path":
            # quick fix for path (only shareloc) in cars_dataset, do nothing
            # tmp dir differ in each run, so test break.
            pass
        else:
            assert actual.attributes[key] == expected.attributes[key]

    assert actual.shape == expected.shape
    assert actual.tiling_grid.size == expected.tiling_grid.size
    assert actual.overlaps.size == expected.overlaps.size
    assert list(actual.tiling_grid).sort() == list(expected.tiling_grid).sort()
    assert list(actual.overlaps).sort() == list(expected.overlaps).sort()
    for idx, actual_tiles in enumerate(actual.tiles):
        assert len(actual_tiles) == len(expected.tiles[idx])
        for idx_tile, actual_tile in enumerate(actual_tiles):
            if isinstance(actual_tile, np.ndarray):
                np.testing.assert_allclose(
                    actual_tile,
                    expected.tiles[idx][idx_tile],
                    rtol,
                    atol,
                )
            elif isinstance(actual_tile, xr.Dataset):
                assert_same_datasets(
                    actual_tile, expected.tiles[idx][idx_tile], rtol, atol
                )
            elif isinstance(actual_tile, xr.DataArray):
                assert_same_datasets(
                    actual_tile, expected.tiles[idx][idx_tile], rtol, atol
                )
            elif isinstance(actual_tile, pandas.DataFrame):
                assert_same_dataframes(
                    actual_tile, expected.tiles[idx][idx_tile], rtol, atol
                )
            elif isinstance(actual_tile, dict):
                assert all(
                    (
                        expected.tiles[idx][idx_tile].get(k) == v
                        for k, v in actual_tile.items()
                    )
                )
            else:
                logging.error(
                    "the tile format unsupported by helper.py: {}".format(
                        type(actual_tile)
                    )
                )


def assert_same_datasets(actual, expected, rtol=0, atol=0):
    """
    Compare two datasets:
    """
    assert (
        list(actual.attrs.keys()).sort() == list(expected.attrs.keys()).sort()
    )
    for key in expected.attrs.keys():
        if isinstance(expected.attrs[key], np.ndarray):
            np.testing.assert_allclose(actual.attrs[key], expected.attrs[key])
        else:
            assert actual.attrs[key] == expected.attrs[key]
    assert actual.dims == expected.dims
    assert (
        list(actual.coords.keys()).sort() == list(expected.coords.keys()).sort()
    )
    for key in expected.coords.keys():
        np.array_equal(actual.coords[key].values, expected.coords[key].values)
    assert (
        list(actual.data_vars.keys()).sort()
        == list(expected.data_vars.keys()).sort()
    )
    for key in expected.data_vars.keys():
        np.testing.assert_allclose(
            actual[key].values, expected[key].values, rtol=rtol, atol=atol
        )


def assert_same_dataframes(actual, expected, rtol=0, atol=0):
    """
    Compare two dataframes:
    """
    assert (
        list(actual.attrs.keys()).sort() == list(expected.attrs.keys()).sort()
    )
    for key in expected.attrs.keys():
        if isinstance(expected.attrs[key], np.ndarray):
            np.testing.assert_allclose(actual.attrs[key], expected.attrs[key])
        else:
            assert actual.attrs[key] == expected.attrs[key]
    assert list(actual.keys()).sort() == list(expected.keys()).sort()
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=rtol, atol=atol
    )


def add_color(dataset, color_array, margin=None):
    """ " Add color array to xarray dataset"""

    new_dataset = dataset.copy(deep=True)

    if margin is None:
        margin = [0, 0, 0, 0]

    if cst.EPI_IMAGE in dataset:
        nb_row = dataset[cst.EPI_IMAGE].values.shape[0]
        nb_col = dataset[cst.EPI_IMAGE].values.shape[1]
    elif cst_disp.MAP in dataset:
        nb_row = dataset[cst_disp.MAP].values.shape[0]
        nb_col = dataset[cst_disp.MAP].values.shape[1]
    elif cst.X in dataset:
        nb_row = dataset[cst.X].values.shape[0]
        nb_col = dataset[cst.X].values.shape[1]
    else:
        logging.error("nb_row and nb_col not set")
        nb_row = color_array.shape[-2] + margin[1] + margin[3]
        nb_col = color_array.shape[-1] + margin[0] + margin[2]

    # add color
    if len(color_array.shape) > 2:
        nb_band = color_array.shape[0]
        if margin is None:
            new_color_array = color_array
        else:
            new_color_array = np.zeros([nb_band, nb_row, nb_col])
            new_color_array[
                :,
                margin[1] : nb_row - margin[3],
                margin[0] : nb_col - margin[2],
            ] = color_array
        # multiple bands
        if cst.BAND_IM not in new_dataset.dims:
            if cst.EPI_TEXTURE in new_dataset:
                band_im = get_color_bands(new_dataset)
            else:
                default_band = ["R", "G", "B", "N"]
                band_im = default_band[:nb_band]
            new_dataset.coords[cst.BAND_IM] = band_im

        new_dataset[cst.EPI_TEXTURE] = xr.DataArray(
            new_color_array,
            dims=[cst.BAND_IM, cst.ROW, cst.COL],
        )
    else:
        if margin is None:
            new_color_array = color_array
        else:
            new_color_array = np.zeros([nb_row, nb_col])
            new_color_array[
                margin[1] : nb_row - margin[3], margin[0] : nb_col - margin[2]
            ] = color_array
        new_dataset[cst.EPI_TEXTURE] = xr.DataArray(
            new_color_array,
            dims=[cst.ROW, cst.COL],
        )

    return new_dataset


def create_corr_conf(user_cfg, left_input, right_input, used_band="b0"):
    """
    Create correlator configuration for stereo testing
    TODO: put in CARS source code ? (external?)
    """

    # Import plugins before checking configuration
    pandora.import_plugin()
    # Check configuration and update the configuration with default values
    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()
    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)

    left_input = left_input.copy()
    right_input = right_input.copy()
    left_input.attrs["disparity_source"] = None
    right_input.attrs["disparity_source"] = None

    if used_band:
        user_cfg_pipeline["pipeline"]["matching_cost"]["band"] = used_band

    cfg_pipeline = check_pipeline_section(
        user_cfg_pipeline, left_input, right_input, pandora_machine
    )
    # check a part of input section
    user_cfg_input = get_config_input_custom_cars(user_cfg, 0, 0)
    cfg_input = check_input_section_custom_cars(user_cfg_input)
    # concatenate updated config
    cfg = concat_conf([cfg_input, cfg_pipeline])
    return cfg


def corr_conf_defaut():
    """
    Provide pandora default configuration for test
    """
    user_cfg = {}
    with open(
        absolute_data_path(os.path.join("conf_pandora", "conf_default.json")),
        "r",
        encoding="utf8",
    ) as fstream:
        user_cfg = json.load(fstream)
    return user_cfg


def corr_conf_with_confidence():
    """
    Provide pandora configuration with confidence option
    """
    user_cfg = {}
    with open(
        absolute_data_path(
            os.path.join("conf_pandora", "conf_with_all_confidences.json")
        ),
        "r",
        encoding="utf8",
    ) as fstream:
        user_cfg = json.load(fstream)
    return user_cfg
