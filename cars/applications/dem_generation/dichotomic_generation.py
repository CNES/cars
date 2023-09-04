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
this module contains the dichotomic dem generation application class.
"""


# Standard imports
import collections
import os

# Third party imports
import numpy as np
import pandas
import xarray as xr
from affine import Affine
from json_checker import And, Checker, Or

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation.dem_generation import DemGeneration
from cars.core import projection
from cars.data_structures import cars_dataset


class DichotomicGeneration(DemGeneration, short_name="dichotomic"):
    """
    DichotomicGeneration
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of DichotomicGeneration

        :param conf: configuration for DichotomicGeneration
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.resolution = self.used_config["resolution"]
        # Saving bools
        self.margin = self.used_config["margin"]
        self.percentile = self.used_config["percentile"]
        self.min_number_matches = self.used_config["min_number_matches"]

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
        overloaded_conf["method"] = conf.get("method", "dichotomic")
        overloaded_conf["resolution"] = conf.get("resolution", 90)
        # default margin: (z max - zmin) * tan(teta)
        # with z max = 9000, z min = 0, teta = 30 degrees
        overloaded_conf["margin"] = conf.get("margin", 6000)
        overloaded_conf["percentile"] = conf.get("percentile", 10)
        overloaded_conf["min_number_matches"] = conf.get(
            "min_number_matches", 30
        )

        rectification_schema = {
            "method": str,
            "resolution": And(Or(float, int), lambda x: x > 0),
            "margin": And(Or(float, int), lambda x: x > 0),
            "percentile": And(int, lambda x: x > 0),
            "min_number_matches": And(int, lambda x: x > 0),
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(self, triangulated_matches_list, output_dir):
        """
        Run dichotomic dem generation using matches

        :param triangulated_matches_list: list of triangulated matches
            positions must be in a metric system
        :type triangulated_matches_list: list(pandas.Dataframe)
        :param output_dir: directory to save dem
        :type output_dir: str

        :return: dem data computed with mean, min and max.
            dem is also saved in disk, and paths are available in attributes.
            (DEM_MEAN_PATH, DEM_MIN_PATH, DEM_MAX_PATH)
        :rtype: CarsDataset
        """

        # Create sequential orchestrator for savings
        self.orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )

        # Generate point cloud
        epsg = None

        for pair_pc in triangulated_matches_list:
            if epsg is None:
                # epsg must be a metric system
                epsg = pair_pc.attrs["epsg"]

            if pair_pc.attrs["epsg"] != epsg:
                projection.points_cloud_conversion_dataset(pair_pc, epsg)

        merged_point_cloud = pandas.concat(
            triangulated_matches_list,
            ignore_index=True,
            sort=False,
        )

        # Get borders

        # Get min max with margin
        mins = merged_point_cloud.min(skipna=True)
        maxs = merged_point_cloud.max(skipna=True)
        xmin = mins["x"] - self.margin
        ymin = mins["y"] - self.margin
        xmax = maxs["x"] + self.margin
        ymax = maxs["y"] + self.margin

        # Generate regular grid
        xnew, ynew = generate_grid(
            merged_point_cloud,
            self.resolution,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

        # define functions to use
        funcs = [
            np.mean,
            (np.percentile, self.percentile),
            (np.percentile, 100 - self.percentile),
        ]

        list_z_grid = []
        for _ in range(len(funcs)):
            list_z_grid.append(np.full_like(xnew, np.nan))

        row_min = 0
        col_min = 0
        row_max = list_z_grid[0].shape[0]
        col_max = list_z_grid[0].shape[1]

        # Modify output grids
        multi_res_rec(
            merged_point_cloud,
            funcs,
            xnew,
            ynew,
            list_z_grid,
            row_min,
            row_max,
            col_min,
            col_max,
            self.min_number_matches,
        )

        mnt_mean = list_z_grid[0]
        mnt_min = list_z_grid[1]
        mnt_max = list_z_grid[2]

        # Generate CarsDataset

        dem = cars_dataset.CarsDataset("arrays")

        # Compute tiling grid
        # Only one tile
        dem.tiling_grid = np.array([[[0, row_max, 0, col_max]]])

        # saving infos
        # dem mean
        dem_mean_path = os.path.join(output_dir, "dem_mean.tif")
        self.orchestrator.add_to_save_lists(
            dem_mean_path,
            dem_gen_cst.DEM_MEAN,
            dem,
            dtype=np.float32,
            cars_ds_name="dem_mean",
        )
        dem.attributes[dem_gen_cst.DEM_MEAN_PATH] = dem_mean_path
        # dem min
        dem_min_path = os.path.join(output_dir, "dem_min.tif")
        self.orchestrator.add_to_save_lists(
            dem_min_path,
            dem_gen_cst.DEM_MIN,
            dem,
            dtype=np.float32,
            cars_ds_name="dem_min",
        )
        dem.attributes[dem_gen_cst.DEM_MIN_PATH] = dem_min_path
        # dem max
        dem_max_path = os.path.join(output_dir, "dem_max.tif")
        self.orchestrator.add_to_save_lists(
            dem_max_path,
            dem_gen_cst.DEM_MAX,
            dem,
            dtype=np.float32,
            cars_ds_name="dem_max",
        )
        dem.attributes[dem_gen_cst.DEM_MAX_PATH] = dem_max_path

        bounds = [xmin, ymin, xmax, ymax]

        # Generate profile
        geotransform = (
            bounds[0],
            self.resolution,
            0.0,
            bounds[3],
            0.0,
            -self.resolution,
        )

        transform = Affine.from_gdal(*geotransform)
        raster_profile = collections.OrderedDict(
            {
                "height": row_max,
                "width": col_max,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform,
                "crs": "EPSG:{}".format(epsg),
                "tiled": True,
            }
        )

        # Generate dataset
        dem_tile = xr.Dataset(
            data_vars={
                "dem_mean": (["row", "col"], mnt_mean),
                "dem_min": (["row", "col"], mnt_min),
                "dem_max": (["row", "col"], mnt_max),
            },
            coords={
                "row": np.arange(0, row_max),
                "col": np.arange(0, col_max),
            },
        )

        [  # pylint: disable=unbalanced-tuple-unpacking
            saving_info
        ] = self.orchestrator.get_saving_infos([dem])
        saving_info = ocht.update_saving_infos(saving_info, row=0, col=0)
        window = cars_dataset.window_array_to_dict(dem.tiling_grid[0, 0])
        cars_dataset.fill_dataset(
            dem_tile,
            saving_info=saving_info,
            window=window,
            profile=raster_profile,
            attributes=None,
            overlaps=None,
        )

        dem[0, 0] = dem_tile

        # Save
        self.orchestrator.breakpoint()

        return dem


def generate_grid(
    pd_pc, resolution, xmin=None, xmax=None, ymin=None, ymax=None
):
    """
    Generate regular grid

    :param pd_pc: point cloud
    :type pd_pc: Pandas Dataframe
    :param resolution: resolution in meter
    :type resolution: float
    :param xmin: x min position in metric system
    :type xmin: float
    :param xmax: x max position in metric system
    :type xmax: float
    :param ymin: y min position in metric system
    :type ymin: float
    :param ymax: y max position in metric system
    :type ymax: float

    :return: regular grid
    :rtype: numpy array

    """

    if None in (xmin, xmax, ymin, ymax):
        mins = pd_pc.min(skipna=True)
        maxs = pd_pc.max(skipna=True)
        xmin = mins["x"]
        ymin = mins["y"]
        xmax = maxs["x"]
        ymax = maxs["y"]

    nb_x = int((xmax - xmin) / resolution)
    x_range = np.linspace(xmin + 0.5, xmax, nb_x)
    nb_y = int((ymax - ymin) / resolution)

    y_range = np.linspace(ymin + 0.5, ymax, nb_y)
    x_grid, y_grid = np.meshgrid(x_range, y_range)  # 2D grid for interpolation

    return x_grid, y_grid


def multi_res_rec(
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

    # find points
    tile_pc = pd_pc.loc[
        (pd_pc["x"] >= xmin)
        & (pd_pc["x"] < xmax)
        & (pd_pc["y"] >= ymin)
        & (pd_pc["y"] < ymax)
    ]

    nb_matches = tile_pc.shape[0]

    if (
        nb_matches > min_number_matches
        and (row_max - row_min > 0)
        and (col_max - col_min > 0)
    ):
        # apply global value
        for fun, z_grid in zip(list_fun, list_z_grid):  # noqa: B905
            if isinstance(fun, tuple):
                # percentile
                z_grid[row_min:row_max, col_min:col_max] = fun[0](
                    tile_pc["z"], fun[1]
                )
            else:
                z_grid[row_min:row_max, col_min:col_max] = fun(tile_pc["z"])

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
                    )
