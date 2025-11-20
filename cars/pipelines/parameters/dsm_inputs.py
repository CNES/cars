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
# pylint: disable=too-many-lines
"""
CARS dsm inputs
"""

import collections
import logging
import os

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from json_checker import Checker, Or
from rasterio.windows import from_bounds

# CARS imports
import cars.orchestrator.orchestrator as ocht
import cars.pipelines.parameters.dsm_inputs_constants as dsm_cst
from cars.applications.rasterization.rasterization_wrappers import (
    update_data,
    update_weights,
)
from cars.core import constants as cst
from cars.core import inputs, preprocessing, tiling
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import make_relative_path_absolute, safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.parameters import sensor_inputs as sens_inp
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def check_dsm_inputs(conf, config_dir=None):
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_dir: directory of used json/yaml, if
        user filled paths with relative paths
    :type config_dir: str

    :return: overloader inputs
    :rtype: dict
    """

    overloaded_conf = {}

    # Overload some optional parameters
    overloaded_conf[dsm_cst.DSMS] = {}

    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)

    overloaded_conf[sens_cst.INITIAL_ELEVATION] = (
        sens_inp.get_initial_elevation(
            conf.get(sens_cst.INITIAL_ELEVATION, None)
        )
    )

    overloaded_conf[sens_cst.SENSORS] = conf.get(sens_cst.SENSORS, None)

    overloaded_conf[sens_cst.PAIRING] = conf.get(sens_cst.PAIRING, None)

    # Validate inputs
    inputs_schema = {
        dsm_cst.DSMS: dict,
        sens_cst.ROI: Or(str, dict, None),
        sens_cst.INITIAL_ELEVATION: Or(dict, None),
        sens_cst.SENSORS: Or(dict, None),
        sens_cst.PAIRING: Or([[str]], None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate depth maps

    dsm_schema = {
        cst.DSM_CLASSIF: Or(str, None),
        cst.DSM_ALT: Or(str, None),
        cst.DSM_ALT_INF: Or(str, None),
        cst.DSM_ALT_SUP: Or(str, None),
        cst.DSM_WEIGHTS_SUM: Or(str, None),
        cst.DSM_MSK: Or(str, None),
        cst.DSM_NB_PTS: Or(str, None),
        cst.DSM_NB_PTS_IN_CELL: Or(str, None),
        cst.DSM_MEAN: Or(str, None),
        cst.DSM_STD_DEV: Or(str, None),
        cst.DSM_INF_MEAN: Or(str, None),
        cst.DSM_INF_STD: Or(str, None),
        cst.DSM_SUP_MEAN: Or(str, None),
        cst.DSM_SUP_STD: Or(str, None),
        cst.DSM_AMBIGUITY: Or(str, None),
        cst.DSM_PERFORMANCE_MAP: Or(str, None),
        cst.DSM_SOURCE_PC: Or(str, None),
        cst.DSM_FILLING: Or(str, None),
        cst.DSM_COLOR: Or(str, None),
    }

    checker_pc = Checker(dsm_schema)
    for dsm_key in conf[dsm_cst.DSMS]:
        # Get depth maps with default
        overloaded_conf[dsm_cst.DSMS][dsm_key] = {}
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_CLASSIF] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("classification", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_COLOR] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("image", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_MSK] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("mask", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT_INF] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_ALT_SUP] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_WEIGHTS_SUM] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("weights", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_NB_PTS] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_n_pts", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_NB_PTS_IN_CELL] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_pts_in_cell", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_STD_DEV] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_INF_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_INF_STD] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_inf_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SUP_MEAN] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup_mean", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SUP_STD] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_sup_std", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_AMBIGUITY] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("ambiguity", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_PERFORMANCE_MAP] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("performance_map", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SOURCE_PC] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("contributing_pair", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_FILLING] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("filling", None)

        # validate
        checker_pc.validate(overloaded_conf[dsm_cst.DSMS][dsm_key])

    # Modify to absolute path
    if config_dir is not None:
        modify_to_absolute_path(config_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    for dsm_key in conf[dsm_cst.DSMS]:
        # check sizes
        check_input_size(
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_ALT],
            overloaded_conf[dsm_cst.DSMS][dsm_key][
                cst.INDEX_DSM_CLASSIFICATION
            ],
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_COLOR],
            overloaded_conf[dsm_cst.DSMS][dsm_key][cst.INDEX_DSM_MASK],
        )

    # Check srtm dir
    sens_inp.check_srtm(
        overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
    )

    check_phasing(conf[dsm_cst.DSMS])

    overloaded_conf[sens_cst.LOADERS] = sens_inp.check_loaders(
        conf.get(sens_cst.LOADERS, {})
    )

    classif_loader = overloaded_conf[sens_cst.LOADERS][
        sens_cst.INPUT_CLASSIFICATION
    ]

    overloaded_conf[sens_cst.FILLING] = sens_inp.check_filling(
        conf.get(sens_cst.FILLING, {}), classif_loader
    )

    if sens_cst.SENSORS in conf and conf[sens_cst.SENSORS] is not None:
        sens_inp.check_sensors(conf, overloaded_conf, config_dir)

    return overloaded_conf


def check_geometry_plugin(conf_inputs, conf_geom_plugin):
    """
    Check the geometry plugin with inputs
    :param conf_geom_plugin: name of geometry plugin
    :type conf_geom_plugin: str
    :param conf_inputs: checked configuration of inputs
    :type conf_inputs: type

    :return: geometry plugin with dem
    """
    if conf_geom_plugin is None:
        conf_geom_plugin = "SharelocGeometry"

    dem_path = conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]

    if dem_path is None:
        return conf_geom_plugin, None

    # Initialize a geometry plugin with elevation information
    geom_plugin_with_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            dem=dem_path,
            geoid=conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
            default_alt=sens_cst.CARS_DEFAULT_ALT,
        )
    )

    return conf_geom_plugin, geom_plugin_with_dem_and_geoid


def check_input_size(dsm, classif, color, mask):
    """
    Check dsm, mask, color, classif given

    Images must have same size

    :param dsm: phased dsm path
    :type dsm: str
    :param classif: classif path
    :type classif: str
    :param color: color path
    :type color: str
    :param mask: mask path
    :type mask: str
    """

    if inputs.rasterio_get_nb_bands(dsm) != 1:
        raise RuntimeError("{} is not mono-band image".format(dsm))

    for path in [mask, color, classif]:
        if path is not None:
            if inputs.rasterio_get_size(dsm) != inputs.rasterio_get_size(path):
                raise RuntimeError(
                    "The image {} and {} "
                    "do not have the same size".format(dsm, path)
                )


def modify_to_absolute_path(config_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_dir: directory of the json configuration
    :type config_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for dsm_key in overloaded_conf[dsm_cst.DSMS]:
        dsms = overloaded_conf[dsm_cst.DSMS][dsm_key]
        for tag in [
            cst.INDEX_DSM_ALT,
            cst.INDEX_DSM_CLASSIFICATION,
            cst.INDEX_DSM_COLOR,
            cst.INDEX_DSM_MASK,
        ]:
            if dsms[tag] is not None:
                dsms[tag] = make_relative_path_absolute(dsms[tag], config_dir)

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_dir
            )


def check_phasing(dsm_dict):
    """
    Check if the dsm are phased, and if resolution and epsg code are equivalent

    :param dsm_dict: list of phased dsm
    :type dsm_dict: dict
    """

    ref_key = next(iter(dsm_dict))
    ref_epsg = inputs.rasterio_get_epsg_code(dsm_dict[ref_key]["dsm"])
    ref_profile = inputs.rasterio_get_profile(dsm_dict[ref_key]["dsm"])
    ref_transform = list(ref_profile["transform"])
    ref_res_x = ref_transform[0]
    ref_res_y = ref_transform[4]
    ref_bounds = inputs.rasterio_get_bounds(dsm_dict[ref_key]["dsm"])

    for dsm_key in dsm_dict:
        if dsm_key == ref_key:
            continue

        epsg = inputs.rasterio_get_epsg_code(dsm_dict[dsm_key]["dsm"])
        profile = inputs.rasterio_get_profile(dsm_dict[ref_key]["dsm"])
        transform = list(profile["transform"])
        res_x = transform[0]
        res_y = transform[4]
        bounds = inputs.rasterio_get_bounds(dsm_dict[dsm_key]["dsm"])

        if epsg != ref_epsg:
            raise RuntimeError(
                f"EPSG mismatch: DSM {dsm_key} has EPSG {epsg}, "
                f"expected {ref_epsg}."
            )

        if ref_res_x != res_x or ref_res_y != res_y:
            raise RuntimeError(
                f"Resolution mismatch: DSM {dsm_key} has resolution "
                f"{(res_x, res_y)}, expected {(ref_res_x, ref_res_y)}."
            )

        # Compare the left_bottom corner
        diff = ref_bounds[0:2] - bounds[0:2]
        resolution = np.array([ref_res_x, -ref_res_y])
        res_ratio = diff / resolution

        if ~np.all(np.equal(res_ratio, res_ratio.astype(int))) and ~np.all(
            np.equal(1 / res_ratio, (1 / res_ratio).astype(int))
        ):
            raise RuntimeError(f"DSM {dsm_key} and {ref_key} are not phased")


# pylint: disable=too-many-positional-arguments
def merge_dsm_infos(  # noqa: C901 function is too complex
    dict_path,
    orchestrator,
    roi_poly,
    terrain_tile_size,
    dump_dir=None,
    dsm_file_name=None,
    color_file_name=None,
    classif_file_name=None,
    filling_file_name=None,
    performance_map_file_name=None,
    ambiguity_file_name=None,
    contributing_pair_file_name=None,
):
    """
    Merge all the dsms

    :param dict_path: path of all variables from all dsms
    :type dict_path: dict
    :param orchestrator: orchestrator used
    :param terrain_tile_size: tile size to use
    :type terrain_tile_size: int
    :param dump_dir: output path
    :type dump_dir: str
    :param dsm_file_name: name of the dsm output file
    :type dsm_file_name: str
    :param color_file_name: name of the color output file
    :type color_file_name: str
    :param classif_file_name: name of the classif output file
    :type classif_file_name: str
    :param filling_file_name: name of the filling output file
    :type filling_file_name: str
    :param performance_map_file_name: name of the performance_map output file
    :type performance_map_file_name: str
    :param ambiguity_file_name: name of the ambiguity output file
    :type ambiguity_file_name: str
    :param contributing_pair_file_name: name of contributing_pair output file
    :type contributing_pair_file_name: str

     :return: raster DSM. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "hgt", "img", "raster_msk",optional : \
                  "n_pts", "pts_in_cell", "hgt_mean", "hgt_stdev",\
                  "hgt_inf", "hgt_sup"
                - attrs with keys: "epsg"
            - attributes containing: None

    :rtype : CarsDataset filled with xr.Dataset
    """

    # Create CarsDataset
    terrain_raster = cars_dataset.CarsDataset("arrays", name="rasterization")

    # find the global bounds of the dataset
    dsm_nodata = None
    epsg = None
    resolution = None
    for index, path in enumerate(dict_path["dsm"]):
        with rasterio.open(path) as src:
            if index == 0:
                bounds = src.bounds
                global_bounds = bounds
                profile = src.profile
                transform = list(profile["transform"])
                res_x = transform[0]
                res_y = transform[4]
                resolution = (res_y, res_x)

                epsg = src.crs

                dsm_nodata = src.nodata
            else:
                bounds = src.bounds
                global_bounds = (
                    min(bounds[0], global_bounds[0]),  # xmin
                    min(bounds[1], global_bounds[1]),  # ymin
                    max(bounds[2], global_bounds[2]),  # xmax
                    max(bounds[3], global_bounds[3]),  # ymax
                )

    if roi_poly is not None:
        global_bounds = preprocessing.crop_terrain_bounds_with_roi(
            roi_poly,
            global_bounds[0],
            global_bounds[1],
            global_bounds[2],
            global_bounds[3],
        )

    # Tiling of the dataset
    [xmin, ymin, xmax, ymax] = global_bounds

    terrain_raster.tiling_grid = tiling.generate_tiling_grid(
        xmin,
        ymin,
        xmax,
        ymax,
        terrain_tile_size,
        terrain_tile_size,
    )

    xsize, ysize = tiling.roi_to_start_and_size(global_bounds, resolution[1])[
        2:
    ]

    # build the tranform of the dataset
    # Generate profile
    geotransform = (
        global_bounds[0],
        resolution[1],
        0.0,
        global_bounds[3],
        0.0,
        -resolution[1],
    )

    transform = Affine.from_gdal(*geotransform)
    raster_profile = collections.OrderedDict(
        {
            "height": ysize,
            "width": xsize,
            "driver": "GTiff",
            "dtype": "float32",
            "transform": transform,
            "crs": "EPSG:{}".format(epsg),
            "tiled": True,
            "no_data": dsm_nodata,
        }
    )

    # Get sources pc
    full_sources_band_descriptions = None
    if cst.DSM_SOURCE_PC in list(dict_path.keys()):
        full_sources_band_descriptions = []
        for source_pc in dict_path[cst.DSM_SOURCE_PC]:
            full_sources_band_descriptions += list(
                inputs.get_descriptions_bands(source_pc)
            )

        # remove copies
        full_sources_band_descriptions = list(
            dict.fromkeys(full_sources_band_descriptions)
        )

    # Setup dump directory
    if dump_dir is not None:
        out_dump_dir = dump_dir
        safe_makedirs(out_dump_dir)
    else:
        out_dump_dir = orchestrator.out_dir

    if dsm_file_name is not None:
        safe_makedirs(os.path.dirname(dsm_file_name))

    # Save all file that are in inputs
    for key in dict_path.keys():
        if key in (cst.DSM_ALT, cst.DSM_COLOR, cst.DSM_WEIGHTS_SUM):
            option = False
        else:
            option = True

        if key == cst.DSM_ALT and dsm_file_name is not None:
            out_file_name = dsm_file_name
        elif key == cst.DSM_COLOR and color_file_name is not None:
            out_file_name = color_file_name
        elif key == cst.DSM_CLASSIF and classif_file_name is not None:
            out_file_name = classif_file_name
        elif key == cst.DSM_FILLING and filling_file_name is not None:
            out_file_name = filling_file_name
        elif (
            key == cst.DSM_PERFORMANCE_MAP
            and performance_map_file_name is not None
        ):
            out_file_name = performance_map_file_name
        elif key == cst.DSM_AMBIGUITY and ambiguity_file_name is not None:
            out_file_name = ambiguity_file_name
        elif (
            key == cst.DSM_SOURCE_PC and contributing_pair_file_name is not None
        ):
            out_file_name = contributing_pair_file_name
        else:
            out_file_name = os.path.join(out_dump_dir, key + ".tif")

        orchestrator.add_to_save_lists(
            out_file_name,
            key,
            terrain_raster,
            dtype=inputs.rasterio_get_dtype(dict_path[key][0]),
            nodata=inputs.rasterio_get_nodata(dict_path[key][0]),
            cars_ds_name=key,
            optional_data=option,
        )

    [saving_info] = orchestrator.get_saving_infos([terrain_raster])
    logging.info(
        "Merge DSM info in {} x {} tiles".format(
            terrain_raster.shape[0], terrain_raster.shape[1]
        )
    )
    for col in range(terrain_raster.shape[1]):
        for row in range(terrain_raster.shape[0]):
            # update saving infos for potential replacement
            full_saving_info = ocht.update_saving_infos(
                saving_info, row=row, col=col
            )

            # Delayed call to dsm merging operations using all
            terrain_raster[row, col] = orchestrator.cluster.create_task(
                dsm_merging_wrapper, nout=1
            )(
                dict_path,
                terrain_raster.tiling_grid[row, col],
                resolution,
                raster_profile,
                full_saving_info,
                full_sources_band_descriptions,
            )

    return terrain_raster


def dsm_merging_wrapper(  # pylint: disable=too-many-positional-arguments # noqa C901
    dict_path,
    tile_bounds,
    resolution,
    profile,
    saving_info=None,
    full_sources_band_descriptions=None,
):
    """
    Merge all the variables

    :param dict_path: path of all variables from all dsms
    :type dict_path: dict
    :param tile_bounds: list of tiles coordinates
    :type tile_bounds: list
    :param resolution: resolution of the dsms
    :type resolution: list
    :param profile: profile of the global dsm
    :type profile: OrderedDict
    :saving_info: the saving infos
    """

    # create the tile dataset
    x_value = np.arange(tile_bounds[0], tile_bounds[1], resolution[1])
    y_value = np.arange(tile_bounds[2], tile_bounds[3], resolution[1])
    height = len(y_value)
    width = len(x_value)

    dataset = xr.Dataset(
        data_vars={},
        coords={
            "y": y_value,
            "x": x_value,
        },
    )

    # calculate the bounds intersection between each path
    list_intersection = []

    for path in dict_path["dsm"]:
        with rasterio.open(path) as src:
            intersect_bounds = (
                max(tile_bounds[0], src.bounds.left),  # xmin
                max(tile_bounds[2], src.bounds.bottom),  # ymin
                min(tile_bounds[1], src.bounds.right),  # xmax
                min(tile_bounds[3], src.bounds.top),  # ymax
            )

            if (
                intersect_bounds[0] < intersect_bounds[2]
                and intersect_bounds[1] < intersect_bounds[3]
            ):
                list_intersection.append(intersect_bounds)
            else:
                list_intersection.append("no intersection")

    # Update the data
    for key in dict_path.keys():
        # Choose the method regarding the variable
        if key in [cst.DSM_NB_PTS, cst.DSM_NB_PTS_IN_CELL]:
            method = "sum"
        elif key in [
            cst.DSM_FILLING,
            cst.DSM_CLASSIF,
            cst.DSM_SOURCE_PC,
        ]:
            method = "bool"
        else:
            method = "basic"

        # take band description information
        band_descriptions = list(
            inputs.get_descriptions_bands(dict_path[key][0])
        )
        nb_bands = inputs.rasterio_get_nb_bands(dict_path[key][0])
        if len(band_descriptions) == 0:
            band_descriptions = []
        elif (
            key
            in [
                cst.DSM_COLOR,
                cst.DSM_SOURCE_PC,
                cst.DSM_CLASSIF,
                cst.DSM_FILLING,
            ]
            and None in band_descriptions
        ):
            band_descriptions = [
                str(current_band) for current_band in range(nb_bands)
            ]

        # Define the dimension of the data in the dataset
        if key == cst.DSM_COLOR:
            dataset.coords[cst.BAND_IM] = (cst.BAND_IM, band_descriptions)
            dim = [cst.BAND_IM, cst.Y, cst.X]
        elif key == cst.DSM_SOURCE_PC:
            dataset.coords[cst.BAND_SOURCE_PC] = (
                cst.BAND_SOURCE_PC,
                full_sources_band_descriptions,
            )
            dim = [cst.BAND_SOURCE_PC, cst.Y, cst.X]
        elif key == cst.DSM_CLASSIF:
            dataset.coords[cst.BAND_CLASSIF] = (
                cst.BAND_CLASSIF,
                band_descriptions,
            )
            dim = [cst.BAND_CLASSIF, cst.Y, cst.X]
        elif key == cst.DSM_FILLING:
            dataset.coords[cst.BAND_FILLING] = (
                cst.BAND_FILLING,
                band_descriptions,
            )
            dim = [cst.BAND_FILLING, cst.Y, cst.X]
        else:
            dim = [cst.Y, cst.X]

        # Update data
        if key == cst.DSM_ALT:
            # Update dsm_value and weights once
            value, weights = assemblage(
                dict_path[key],
                dict_path[cst.DSM_WEIGHTS_SUM],
                method,
                list_intersection,
                tile_bounds,
                height,
                width,
                band_descriptions,
            )

            dataset[key] = (dim, value)
            dataset[cst.DSM_WEIGHTS_SUM] = (dim, weights)
        elif key == cst.DSM_SOURCE_PC:
            value, _ = assemblage(
                dict_path[key],
                dict_path[cst.DSM_WEIGHTS_SUM],
                method,
                list_intersection,
                tile_bounds,
                height,
                width,
                full_sources_band_descriptions,
                merge_sources=True,
            )
            dataset[key] = (dim, value)
        elif key != cst.DSM_WEIGHTS_SUM:
            # Update other variables
            value, _ = assemblage(
                dict_path[key],
                dict_path[cst.DSM_WEIGHTS_SUM],
                method,
                list_intersection,
                tile_bounds,
                height,
                width,
                band_descriptions,
            )

            dataset[key] = (dim, value)

        # add performance map classes
        if key == cst.DSM_PERFORMANCE_MAP:
            perf_map_classes = inputs.rasterio_get_tags(dict_path[key][0])[
                "CLASSES"
            ]
            dataset.attrs[cst.RIO_TAG_PERFORMANCE_MAP_CLASSES] = (
                perf_map_classes
            )

    # Define the tile transform
    bounds = [tile_bounds[0], tile_bounds[2], tile_bounds[1], tile_bounds[3]]
    xstart, ystart, xsize, ysize = tiling.roi_to_start_and_size(
        bounds, resolution[1]
    )
    transform = rasterio.Affine(*profile["transform"][0:6])

    row_pix_pos, col_pix_pos = rasterio.transform.AffineTransformer(
        transform
    ).rowcol(xstart, ystart)
    window = [
        row_pix_pos,
        row_pix_pos + ysize,
        col_pix_pos,
        col_pix_pos + xsize,
    ]

    window = cars_dataset.window_array_to_dict(window)

    # Fill dataset
    cars_dataset.fill_dataset(
        dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        overlaps=None,
    )

    return dataset


def assemblage(  # pylint: disable=too-many-positional-arguments
    out,
    current_weights,
    method,
    intersect_bounds,
    tile_bounds,
    height,
    width,
    band_descriptions=None,
    merge_sources=False,
):
    """
    Update data

    :param out: the data to update
    :type out: list of path
    :param current_weights: the current weights of the data
    :type current_weights: list of path
    :param method: the method used to update the data
    :type method: str
    :param intersect_bounds: the bounds intersection
    :type intersect_bounds: list of bounds
    :param height: the height of the tile
    :type height: int
    :param width: the width of the tile
    :type width: int
    :param band_descriptions: the band description of the data
    :type band_descriptions: str of list
    :param merge_sources: merge source pc, using full band_description
    :type merge_sources: bool

    """
    # Initialize the tile
    if merge_sources:
        nb_bands = len(band_descriptions)
    else:
        nb_bands = inputs.rasterio_get_nb_bands(out[0])

    dtype = inputs.rasterio_get_dtype(out[0])
    nodata = inputs.rasterio_get_nodata(out[0])

    if band_descriptions[0] is not None:
        tile = np.full((nb_bands, height, width), nodata, dtype=dtype)
    else:
        tile = np.full((height, width), nodata, dtype=dtype)

    # Initialize the weights
    weights = np.full((height, width), 0, dtype=dtype)

    for idx, path in enumerate(out):
        with (
            rasterio.open(path) as src,
            rasterio.open(current_weights[idx]) as drt,
        ):
            if intersect_bounds[idx] != "no intersection":
                # Build the window
                window = from_bounds(
                    *intersect_bounds[idx], transform=src.transform
                )

                # Extract the data
                current_nb_bands = src.count
                if current_nb_bands > 1:
                    data = src.read(window=window)
                    _, rows, cols = data.shape
                else:
                    data = src.read(1, window=window)
                    rows, cols = data.shape

                indexes = list(range(current_nb_bands))
                if merge_sources:
                    # Extract current band description
                    current_band_descriptions = list(src.descriptions)
                    # Get position
                    indexes = []
                    for current_band in current_band_descriptions:
                        indexes.append(band_descriptions.index(current_band))

                current_weights_window = drt.read(1, window=window)

                # Calculate the x and y offset because the current_data
                # doesn't equal to the entire tile
                x_offset = int(
                    (intersect_bounds[idx][0] - tile_bounds[0])
                    / np.abs(src.res[0])
                )
                y_offset = int(
                    (tile_bounds[3] - intersect_bounds[idx][3])
                    / np.abs(src.res[1])
                )

                if cols > 0 and rows > 0:
                    tab_x = np.arange(x_offset, x_offset + cols)

                    tab_y = np.arange(y_offset, y_offset + rows)

                    # Update data
                    if band_descriptions[0] is not None:

                        tile[np.ix_(indexes, tab_y, tab_x)] = np.reshape(
                            update_data(
                                tile[np.ix_(indexes, tab_y, tab_x)],
                                data,
                                current_weights_window,
                                weights[np.ix_(tab_y, tab_x)],
                                nodata,
                                method=method,
                            ),
                            tile[np.ix_(indexes, tab_y, tab_x)].shape,
                        )
                    else:
                        tile[np.ix_(tab_y, tab_x)] = np.reshape(
                            update_data(
                                tile[np.ix_(tab_y, tab_x)],
                                data,
                                current_weights_window,
                                weights[np.ix_(tab_y, tab_x)],
                                nodata,
                                method=method,
                            ),
                            tile[np.ix_(tab_y, tab_x)].shape,
                        )

                    # Update weights
                    weights[np.ix_(tab_y, tab_x)] = update_weights(
                        weights[np.ix_(tab_y, tab_x)], current_weights_window
                    )

    return tile, weights
