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
from cars.applications.rasterization.rasterization_tools import (
    update_data,
    update_weights,
)
from cars.core import constants as cst
from cars.core import inputs, tiling
from cars.core.utils import make_relative_path_absolute, safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.parameters import sensor_inputs as sens_inp
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


def check_dsm_inputs(conf, config_json_dir=None):
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_json_dir: directory of used json, if
        user filled paths with relative paths
    :type config_json_dir: str

    :return: overloader inputs
    :rtype: dict
    """

    overloaded_conf = {}

    # Overload some optional parameters
    overloaded_conf[dsm_cst.DSMS] = {}

    overloaded_conf[sens_cst.INITIAL_ELEVATION] = (
        sens_inp.get_initial_elevation(
            conf.get(sens_cst.INITIAL_ELEVATION, None)
        )
    )

    # Validate inputs
    inputs_schema = {
        dsm_cst.DSMS: dict,
        sens_cst.INITIAL_ELEVATION: Or(dict, None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate depth maps

    pc_schema = {
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
        cst.DSM_CONFIDENCE: Or(str, None),
        cst.DSM_PERFORMANCE_MAP: Or(str, None),
        cst.DSM_SOURCE_PC: Or(str, None),
        cst.DSM_FILLING: Or(str, None),
        cst.DSM_COLOR: Or(str, None),
    }

    checker_pc = Checker(pc_schema)
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
        ][dsm_key].get("color", None)
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
        ][dsm_key].get("dsm_nb_pts", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_NB_PTS_IN_CELL] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("dsm_nb_pts_in_cell", None)
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
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_CONFIDENCE] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("confidence_from_ambiguity", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_PERFORMANCE_MAP] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("performance_map", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_SOURCE_PC] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("source_pc", None)
        overloaded_conf[dsm_cst.DSMS][dsm_key][cst.DSM_FILLING] = conf[
            dsm_cst.DSMS
        ][dsm_key].get("filling", None)

        # validate
        checker_pc.validate(overloaded_conf[dsm_cst.DSMS][dsm_key])

    # Modify to absolute path
    if config_json_dir is not None:
        modify_to_absolute_path(config_json_dir, overloaded_conf)
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

    return overloaded_conf


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


def modify_to_absolute_path(config_json_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_json_dir: directory of the json configuration
    :type config_json_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for dsm_key in overloaded_conf[dsm_cst.DSMS]:
        depth_map = overloaded_conf[dsm_cst.DSMS][dsm_key]
        for tag in [
            cst.INDEX_DSM_ALT,
            cst.INDEX_DSM_CLASSIFICATION,
            cst.INDEX_DSM_COLOR,
            cst.INDEX_DSM_MASK,
        ]:
            if depth_map[tag] is not None:
                depth_map[tag] = make_relative_path_absolute(
                    depth_map[tag], config_json_dir
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


def merge_dsm_infos(dict_path, orchestrator, dsm_file_name=None):
    """
    Merge all the dsms

    :param dict_path: path of all variables from all dsms
    :type dict_path: dict
    :param orchestrator: orchestrator used
    :param dsm_file_name: name of the dsm output file
    :type dsm_file_name: str

    """

    # Create CarsDataset
    terrain_raster = cars_dataset.CarsDataset("arrays", name="rasterization")

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

    [xmin, ymin, xmax, ymax] = global_bounds
    optimal_terrain_tile_width = 500
    optimal_terrain_tile_height = 500

    terrain_raster.tiling_grid = tiling.generate_tiling_grid(
        xmin,
        ymin,
        xmax,
        ymax,
        optimal_terrain_tile_height,
        optimal_terrain_tile_width,
    )

    xsize, ysize = tiling.roi_to_start_and_size(global_bounds, resolution[1])[
        2:
    ]

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

    if dsm_file_name is not None:
        safe_makedirs(os.path.dirname(dsm_file_name))

    out_dsm_file_name = dsm_file_name

    if out_dsm_file_name is not None:
        orchestrator.add_to_save_lists(
            out_dsm_file_name,
            cst.DSM_ALT,
            terrain_raster,
            dtype=np.float32,
            nodata=dsm_nodata,
            cars_ds_name="dsm",
        )

    orchestrator.add_to_replace_lists(terrain_raster, cars_ds_name="dsm")

    [saving_info] = orchestrator.get_saving_infos([terrain_raster])
    for col in range(terrain_raster.shape[1]):
        for row in range(terrain_raster.shape[0]):
            # update saving infos for potential replacement
            full_saving_info = ocht.update_saving_infos(
                saving_info, row=row, col=col
            )

            # Delayed call to dsm merging operations using all
            # required dsms
            terrain_raster[row, col] = orchestrator.cluster.create_task(
                dsm_merging_wrapper, nout=1
            )(
                dict_path,
                terrain_raster.tiling_grid[row, col],
                resolution,
                raster_profile,
                dsm_nodata,
                full_saving_info,
            )


def dsm_merging_wrapper(
    dict_path,
    tile_bounds,
    resolution,
    profile,
    dsm_nodata,
    saving_info=None,
):
    """
    Merge the dsms

    :param dict_path: path of all variables from all dsms
    :type dict_path: dict
    :param tile_bounds: list of tiles coordinates
    :type tile_bounds: list
    :param resolution: resolution of the dsms
    :type resolution: list
    :param profile: profile of the global dsm
    :type profile: OrderedDict
    :param dsm_nodata: the nodata of the dsms
    :type dsm_nodata: float
    :saving_info: the saving infos
    """

    x = np.arange(tile_bounds[0], tile_bounds[1], resolution[1])
    y = np.arange(tile_bounds[2], tile_bounds[3], resolution[1])
    height = len(y)
    width = len(x)

    dataset = xr.Dataset(
        data_vars={},
        coords={
            "y": y,
            "x": x,
            "band_im": ["r", "g", "b"],
        },
    )

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

    for key in dict_path.keys():
        if key == "dsm":
            dataset[key] = (
                [cst.Y, cst.X],
                assemblage(
                    dict_path[key],
                    dict_path["weights"],
                    "basic",
                    list_intersection,
                    tile_bounds,
                    height,
                    width,
                    dsm_nodata,
                ),
            )

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

    cars_dataset.fill_dataset(
        dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        overlaps=None,
    )

    return dataset


def assemblage(
    out,
    weights,
    method,
    intersect_bounds,
    tile_bounds,
    height,
    width,
    dsm_nodata,
):
    tile = np.full((height, width), dsm_nodata, dtype="float32")
    old_weights = np.full((height, width), 0, dtype="float32")

    for idx, path in enumerate(out):
        with rasterio.open(path) as src, rasterio.open(weights[idx]) as drt:
            window = from_bounds(
                *intersect_bounds[idx], transform=src.transform
            )

            data = src.read(1, window=window)
            weights_window = drt.read(1, window=window)

            x_offset = int(
                (intersect_bounds[idx][0] - tile_bounds[0]) * np.abs(src.res[0])
            )
            y_offset = int(
                (tile_bounds[3] - intersect_bounds[idx][3]) * np.abs(src.res[1])
            )

            rows, cols = data.shape

            ind_x = np.arange(x_offset, x_offset + cols)
            ind_y = np.arange(y_offset, y_offset + rows)

            tile[np.ix_(ind_y, ind_x)] = update_data(
                tile[np.ix_(ind_y, ind_x)],
                data,
                weights_window,
                old_weights[np.ix_(ind_y, ind_x)],
                dsm_nodata,
                method=method,
            )

            old_weights[np.ix_(ind_y, ind_x)] = update_weights(
                old_weights[np.ix_(ind_y, ind_x)], weights_window
            )

    return tile
