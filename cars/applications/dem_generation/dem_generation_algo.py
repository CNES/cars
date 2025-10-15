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
this module contains tools for the dem generation
"""

import contextlib
import logging

# Standard imports
import os

# Third party imports
import numpy as np
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm


def launch_bulldozer(
    input_dem,
    output_dir,
    cars_orchestrator,
    max_object_size,
):
    """
    Launch bulldozer on a DEM to smooth it

    :param input_dem: path of DEM to reverse
    :type input_dem: str
    :param output_dir: directory where bulldozer output is dumped
    :type output_dir: str
    :param cars_orchestrator: orchestrator of the whole pipeline
                              (used to get number of workers)
    :type cars_orchestrator: Orchestrator
    :param max_object_size: bulldozer parameter "max_object_size"
    :type max_object_size: int
    """
    bull_conf_path = os.path.join(
        os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
    )
    with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
        bull_conf = yaml.safe_load(bull_conf_file)

    bull_conf["dsm_path"] = input_dem
    bull_conf["output_dir"] = output_dir
    if cars_orchestrator is not None:
        if (
            cars_orchestrator.get_conf()["mode"] == "multiprocessing"
            or cars_orchestrator.get_conf()["mode"] == "local_dask"
        ):
            bull_conf["nb_max_workers"] = cars_orchestrator.get_conf()[
                "nb_workers"
            ]
    else:
        bull_conf["nb_max_workers"] = 4
    bull_conf["max_object_size"] = max_object_size

    os.makedirs(output_dir, exist_ok=True)
    bull_conf_path = os.path.join(output_dir, "bulldozer_config.yaml")
    with open(bull_conf_path, "w", encoding="utf8") as bull_conf_file:
        yaml.dump(bull_conf, bull_conf_file)

    output_dem = os.path.join(bull_conf["output_dir"], "dtm.tif")

    try:
        try:
            # suppress prints in bulldozer by redirecting stdout&stderr
            with open(os.devnull, "w", encoding="utf8") as devnull:
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    dsm_to_dtm(bull_conf_path)
        except Exception:
            logging.info("Bulldozer failed on its first execution. Retrying")
            # suppress prints in bulldozer by redirecting stdout&stderr
            with open(os.devnull, "w", encoding="utf8") as devnull:
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    dsm_to_dtm(bull_conf_path)
    except Exception:
        logging.error(
            "Bulldozer failed on its second execution."
            + " The DSM could not be smoothed."
        )
        output_dem = None

    return output_dem


def multi_res_rec(  # pylint: disable=too-many-positional-arguments
    pd_pc,
    list_fun,
    x_grid,
    y_grid,
    list_z_grid,
    row_min,
    row_max,
    col_min,
    col_max,
    min_number_matches,
    overlap,
):
    """
    Recursive function to fill grid with results of given functions

    :param pd_pc: point cloud
    :type pd_pc: Pandas Dataframe
    :param list_fun: list of functions
    :type list_fun: list(function)
    :param x_grid: x grid
    :type x_grid: numpy array
    :param y_grid: y grid
    :type y_grid: numpy array
    :param list_z_grid: list of z grid computed with functions
    :type list_z_grid: list(numpy array)
    :param row_min: row min
    :type row_min: int
    :param row_max: row max
    :type row_max: int
    :param col_min: col min
    :type col_min: int
    :param col_max: col max
    :type col_max: int
    :param min_number_matches: minimum of matches: stop condition
    :type min_number_matches: int
    :param overlap: overlap to use for include condition
    :type overlap: float

    """

    if pd_pc.shape[0] < min_number_matches:
        raise RuntimeError("Not enough matches")

    if len(list_fun) != len(list_z_grid):
        raise RuntimeError(
            "Number of functions must match the number of z layers"
        )

    x_values = x_grid[row_min:row_max, col_min:col_max]
    y_values = y_grid[row_min:row_max, col_min:col_max]
    xmin = np.nanmin(x_values)
    ymin = np.nanmin(y_values)
    xmax = np.nanmax(x_values)
    ymax = np.nanmax(y_values)
    xcenter = (xmax + xmin) / 2
    ycenter = (ymax + ymin) / 2

    # find points
    tile_pc = pd_pc.loc[
        (pd_pc["x"] >= xmin - overlap)
        & (pd_pc["x"] < xmax + overlap)
        & (pd_pc["y"] >= ymin - overlap)
        & (pd_pc["y"] < ymax + overlap)
    ]

    nb_matches = tile_pc.shape[0]

    if (
        nb_matches > min_number_matches
        and (row_max - row_min > 0)
        and (col_max - col_min > 0)
    ):
        # apply global value
        for fun, z_grid in zip(list_fun, list_z_grid):  # noqa: B905
            if (
                np.abs(xcenter - np.median(tile_pc["x"])) < overlap
                and np.abs(ycenter - np.median(tile_pc["y"])) < overlap
            ):
                if isinstance(fun, tuple):
                    # percentile
                    z_grid[row_min:row_max, col_min:col_max] = fun[0](
                        tile_pc["z"], fun[1]
                    )
                else:
                    z_grid[row_min:row_max, col_min:col_max] = fun(tile_pc["z"])
            else:
                z_grid[row_min:row_max, col_min:col_max] = np.nan

        list_row = []
        if row_max - row_min >= 2:
            med = int((row_max + row_min) / 2)
            list_row.append((row_min, med))
            list_row.append((med, row_max))
        else:
            list_row.append((row_min, row_max))

        list_col = []
        if col_max - col_min >= 2:
            med = int((col_max + col_min) / 2)
            list_col.append((col_min, med))
            list_col.append((med, col_max))
        else:
            list_col.append((col_min, col_max))

        # if not ( len(list_row) == 1 and len(list_col) == 1):
        if len(list_row) + len(list_col) > 2:
            for row_min_tile, row_max_tile in list_row:
                for col_min_tile, col_max_tile in list_col:
                    multi_res_rec(
                        tile_pc,
                        list_fun,
                        x_grid,
                        y_grid,
                        list_z_grid,
                        row_min_tile,
                        row_max_tile,
                        col_min_tile,
                        col_max_tile,
                        min_number_matches,
                        overlap,
                    )
