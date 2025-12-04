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
from json_checker import Checker
from rasterio.windows import from_bounds

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications.rasterization.rasterization_wrappers import (
    update_data,
    update_weights,
)
from cars.core import constants as cst
from cars.core import inputs, preprocessing, tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset

from .abstract_dsm_merging_app import DsmMerging


class WeightedFusion(DsmMerging, short_name="weighted_fusion"):
    """
    DSM merging app
    """

    def __init__(self, conf=None):
        """
        Init function of BulldozerFilling

        :param conf: configuration for BulldozerFilling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.tile_size = self.used_config["tile_size"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "weighted_fusion")
        overloaded_conf["tile_size"] = conf.get("tile_size", 4000)
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "tile_size": int,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    # pylint: disable=too-many-positional-arguments
    def run(  # noqa: C901
        self,
        dict_path,
        orchestrator,
        roi_poly,
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
        :param performance_map_file_name: name of the performance_map file
        :type performance_map_file_name: str
        :param ambiguity_file_name: name of the ambiguity output file
        :type ambiguity_file_name: str
        :param contributing_pair_file_name: name of contributing_pair file
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
        terrain_raster = cars_dataset.CarsDataset(
            "arrays", name="rasterization"
        )

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
            self.tile_size,
            self.tile_size,
        )

        xsize, ysize = tiling.roi_to_start_and_size(
            global_bounds, resolution[1]
        )[2:]

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
                key == cst.DSM_SOURCE_PC
                and contributing_pair_file_name is not None
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
