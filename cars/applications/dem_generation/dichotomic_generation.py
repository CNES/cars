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
import logging
import os

# Third party imports
import numpy as np
import pandas
import rasterio
import xarray as xr
from json_checker import And, Checker, Or

import cars.orchestrator.orchestrator as ocht
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation.dem_generation import DemGeneration
from cars.applications.triangulation import triangulation_tools

# CARS imports
from cars.core import constants as cst
from cars.core import preprocessing, projection
from cars.core.geometry.abstract_geometry import read_geoid_file
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile


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
        height_margin = self.used_config["height_margin"]
        if isinstance(height_margin, list):
            self.min_height_margin = height_margin[0]
            self.max_height_margin = height_margin[1]
        else:
            self.min_height_margin = height_margin
            self.max_height_margin = height_margin
        self.percentile = self.used_config["percentile"]
        self.min_number_matches = self.used_config["min_number_matches"]
        self.fillnodata_max_search_distance = self.used_config[
            "fillnodata_max_search_distance"
        ]

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
        overloaded_conf["resolution"] = conf.get("resolution", 200)
        # default margin: (z max - zmin) * tan(teta)
        # with z max = 9000, z min = 0, teta = 30 degrees
        overloaded_conf["margin"] = conf.get("margin", 6000)
        overloaded_conf["height_margin"] = conf.get("height_margin", 20)
        overloaded_conf["percentile"] = conf.get("percentile", 3)
        overloaded_conf["min_number_matches"] = conf.get(
            "min_number_matches", 100
        )

        overloaded_conf["fillnodata_max_search_distance"] = conf.get(
            "fillnodata_max_search_distance", 3
        )

        rectification_schema = {
            "method": str,
            "resolution": And(Or(float, int), lambda x: x > 0),
            "margin": And(Or(float, int), lambda x: x > 0),
            "height_margin": Or(list, int),
            "percentile": And(Or(int, float), lambda x: x >= 0),
            "min_number_matches": And(int, lambda x: x > 0),
            "fillnodata_max_search_distance": And(int, lambda x: x > 0),
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="DEM Generation")
    def run(
        self,
        triangulated_matches_list,
        output_dir,
        geoid_path,
        dem_roi_to_use=None,
    ):
        """
        Run dichotomic dem generation using matches

        :param triangulated_matches_list: list of triangulated matches
            positions must be in a metric system
        :type triangulated_matches_list: list(pandas.Dataframe)
        :param output_dir: directory to save dem
        :type output_dir: str
        :param geoid_path: geoid path
        :param dem_roi_to_use: dem roi polygon to use as roi

        :return: dem data computed with mean, min and max.
            dem is also saved in disk, and paths are available in attributes.
            (DEM_MEDIAN_PATH, DEM_MIN_PATH, DEM_MAX_PATH)
        :rtype: CarsDataset
        """

        # Create sequential orchestrator for savings
        self.orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )

        # Generate point cloud
        epsg = 4326

        for pair_pc in triangulated_matches_list:
            if epsg is None:
                # epsg must be a metric system
                epsg = pair_pc.attrs["epsg"]

            # convert to degrees for geoid offset
            if pair_pc.attrs["epsg"] != epsg:
                projection.points_cloud_conversion_dataset(pair_pc, epsg)

        merged_point_cloud = pandas.concat(
            triangulated_matches_list,
            ignore_index=True,
            sort=False,
        )
        merged_point_cloud.attrs["epsg"] = epsg

        # Get bounds
        if dem_roi_to_use is not None:
            bounds_poly = dem_roi_to_use.bounds
            xmin = min(bounds_poly[0], bounds_poly[2])
            xmax = max(bounds_poly[0], bounds_poly[2])
            ymin = min(bounds_poly[1], bounds_poly[3])
            ymax = max(bounds_poly[1], bounds_poly[3])
        else:
            # Get min max with margin
            mins = merged_point_cloud.min(skipna=True)
            maxs = merged_point_cloud.max(skipna=True)
            xmin = mins["x"]
            ymin = mins["y"]
            xmax = maxs["x"]
            ymax = maxs["y"]

        bounds_cloud = [xmin, ymin, xmax, ymax]

        # Convert resolution and margin to degrees
        utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
        conversion_factor = preprocessing.get_conversion_factor(
            bounds_cloud, epsg, utm_epsg
        )
        self.margin *= conversion_factor
        self.resolution *= conversion_factor

        # Get borders, adding margin
        xmin = xmin - self.margin
        ymin = ymin - self.margin
        xmax = xmax + self.margin
        ymax = ymax + self.margin

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
            np.median,
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

        # use 100% overlap for dem
        overlap = 1 * self.resolution

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
            overlap,
        )

        # Remove geoid
        # add offset
        geoid_data = read_geoid_file(geoid_path)

        # Generate dense dataset with z = 0
        alti_zeros_dataset = xr.Dataset(
            {
                cst.X: (["row", "col"], xnew),
                cst.Y: (["row", "col"], ynew),
                cst.Z: (["row", "col"], np.zeros(xnew.shape)),
            },
            coords={
                "row": np.arange(xnew.shape[0]),
                "col": np.arange(xnew.shape[1]),
            },
        )
        alti_zeros_dataset.attrs[cst.EPSG] = epsg
        # Transform to lon lat
        projection.points_cloud_conversion_dataset(alti_zeros_dataset, 4326)

        geoid_offset = triangulation_tools.geoid_offset(
            alti_zeros_dataset, geoid_data
        )

        # fillnodata
        valid = np.isfinite(list_z_grid[0])
        msd = self.fillnodata_max_search_distance
        for idx in range(3):
            list_z_grid[idx] = rasterio.fill.fillnodata(
                list_z_grid[idx], mask=valid, max_search_distance=msd
            )
            list_z_grid[idx] += geoid_offset[cst.Z].values
            list_z_grid[idx] = np.nan_to_num(list_z_grid[idx])

        dem_median = list_z_grid[0]
        dem_min = list_z_grid[1]
        dem_max = list_z_grid[2]

        if np.any((dem_max - dem_min) < 0):
            logging.error("dem min > dem max")
            raise RuntimeError("dem min > dem max")

        # apply height margin
        dem_min -= self.min_height_margin
        dem_max += self.max_height_margin

        # Convert to int
        dem_median = dem_median.astype(int)
        dem_min = np.floor(dem_min).astype(int)
        dem_max = np.ceil(dem_max).astype(int)

        # Generate CarsDataset

        dem = cars_dataset.CarsDataset("arrays", name="dem_generation")

        # Compute tiling grid
        # Only one tile
        dem.tiling_grid = np.array([[[0, row_max, 0, col_max]]])

        # saving infos
        # dem mean
        dem_median_path = os.path.join(output_dir, "dem_median.tif")
        self.orchestrator.add_to_save_lists(
            dem_median_path,
            dem_gen_cst.DEM_MEDIAN,
            dem,
            dtype=np.int32,
            nodata=-32768,
            cars_ds_name="dem_median",
        )
        dem.attributes[dem_gen_cst.DEM_MEDIAN_PATH] = dem_median_path
        # dem min
        dem_min_path = os.path.join(output_dir, "dem_min.tif")
        self.orchestrator.add_to_save_lists(
            dem_min_path,
            dem_gen_cst.DEM_MIN,
            dem,
            dtype=np.int32,
            nodata=-32768,
            cars_ds_name="dem_min",
        )
        dem.attributes[dem_gen_cst.DEM_MIN_PATH] = dem_min_path
        # dem max
        dem_max_path = os.path.join(output_dir, "dem_max.tif")
        self.orchestrator.add_to_save_lists(
            dem_max_path,
            dem_gen_cst.DEM_MAX,
            dem,
            dtype=np.int32,
            nodata=-32768,
            cars_ds_name="dem_max",
        )
        dem.attributes[dem_gen_cst.DEM_MAX_PATH] = dem_max_path

        bounds = [xmin, ymin, xmax, ymax]

        # Generate profile
        geotransform = (
            self.resolution,
            0,
            bounds[0],
            0,
            -self.resolution,
            bounds[3],
        )

        transform = rasterio.Affine(*geotransform)
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
                "dem_median": (["row", "col"], np.flip(dem_median, axis=0)),
                "dem_min": (["row", "col"], np.flip(dem_min, axis=0)),
                "dem_max": (["row", "col"], np.flip(dem_max, axis=0)),
            },
            coords={
                "row": np.arange(0, row_max),
                "col": np.arange(0, col_max),
            },
        )

        [saving_info] = (  # pylint: disable=unbalanced-tuple-unpacking
            self.orchestrator.get_saving_infos([dem])
        )
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
    x_range = np.linspace(xmin, xmax, nb_x)
    nb_y = int((ymax - ymin) / resolution)

    y_range = np.linspace(ymin, ymax, nb_y)
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
