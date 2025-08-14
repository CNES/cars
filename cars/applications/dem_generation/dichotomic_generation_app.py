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
from cars.applications import application_constants
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation.abstract_dem_generation_app import (
    DemGeneration,
)
from cars.applications.dem_generation.dem_generation_algo import multi_res_rec
from cars.applications.dem_generation.dem_generation_wrappers import (
    fit_initial_elevation_on_dem_median,
    generate_grid,
)
from cars.applications.triangulation import triangulation_wrappers

# CARS imports
from cars.core import constants as cst
from cars.core import preprocessing, projection
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
        self.min_dem = self.used_config["min_dem"]
        self.max_dem = self.used_config["max_dem"]
        self.coregistration = self.used_config["coregistration"]
        self.coregistration_max_shift = self.used_config[
            "coregistration_max_shift"
        ]
        self.save_intermediate_data = self.used_config[
            application_constants.SAVE_INTERMEDIATE_DATA
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
        overloaded_conf["resolution"] = conf.get("resolution", 90)
        # default margin: (z max - zmin) * tan(teta)
        # with z max = 9000, z min = 0, teta = 30 degrees
        overloaded_conf["margin"] = conf.get("margin", 6000)
        overloaded_conf["height_margin"] = conf.get("height_margin", 20)
        overloaded_conf["percentile"] = conf.get("percentile", 1)
        overloaded_conf["min_number_matches"] = conf.get(
            "min_number_matches", 100
        )
        overloaded_conf["min_dem"] = conf.get("min_dem", -500)
        overloaded_conf["max_dem"] = conf.get("max_dem", 1000)

        overloaded_conf["fillnodata_max_search_distance"] = conf.get(
            "fillnodata_max_search_distance", 5
        )

        overloaded_conf["coregistration"] = conf.get("coregistration", True)
        overloaded_conf["coregistration_max_shift"] = conf.get(
            "coregistration_max_shift", 180
        )

        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )

        rectification_schema = {
            "method": str,
            "resolution": And(Or(float, int), lambda x: x > 0),
            "margin": And(Or(float, int), lambda x: x > 0),
            "height_margin": Or(list, int),
            "percentile": And(Or(int, float), lambda x: x >= 0),
            "min_number_matches": And(int, lambda x: x > 0),
            "min_dem": And(Or(int, float), lambda x: x < 0),
            "max_dem": And(Or(int, float), lambda x: x > 0),
            "fillnodata_max_search_distance": And(int, lambda x: x > 0),
            "coregistration": bool,
            "coregistration_max_shift": And(Or(int, float), lambda x: x > 0),
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
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
        input_geoid,
        output_geoid,
        dem_roi_to_use=None,
        initial_elevation=None,
        cars_orchestrator=None,
    ):
        """
        Run dichotomic dem generation using matches

        :param triangulated_matches_list: list of triangulated matches
            positions must be in a metric system
        :type triangulated_matches_list: list(pandas.Dataframe)
        :param output_dir: directory to save dem
        :type output_dir: str
        :param input_geoid: input geoid path
        :param output_geoid: output geoid path
        :param cars_orchrestrator: the main cars orchestrator
        :param dem_roi_to_use: dem roi polygon to use as roi
        :param initial_elevation: the path to the initial elevation file

        :return: dem data computed with mean, min and max, and new initial
                 elevation file. dem is also saved in disk, and paths are
                 available in attributes.
                 (DEM_MEDIAN_PATH, DEM_MIN_PATH, DEM_MAX_PATH)
        :rtype: Tuple(CarsDataset, str | None)
        """

        # Create sequential orchestrator for savings
        self.orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )

        # Generate point cloud
        epsg = 4326

        # Optimize the case when input and output geoid are the same
        if output_geoid is True:
            input_geoid = False
            output_geoid = False

        for pair_pc in triangulated_matches_list:
            # convert to degrees for geoid offset
            if pair_pc.attrs["epsg"] != epsg:
                projection.point_cloud_conversion_dataset(pair_pc, epsg)

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
        projection.point_cloud_conversion_dataset(alti_zeros_dataset, 4326)

        if input_geoid:
            input_geoid = triangulation_wrappers.geoid_offset(
                alti_zeros_dataset, input_geoid
            )
        if output_geoid:
            output_geoid = triangulation_wrappers.geoid_offset(
                alti_zeros_dataset, output_geoid
            )

        # fillnodata
        valid = np.isfinite(list_z_grid[0])
        msd = self.fillnodata_max_search_distance
        for idx in range(3):
            list_z_grid[idx] = rasterio.fill.fillnodata(
                list_z_grid[idx], mask=valid, max_search_distance=msd
            )
            if input_geoid:
                list_z_grid[idx] += input_geoid[cst.Z].values
            if output_geoid:
                list_z_grid[idx] -= output_geoid[cst.Z].values

        dem_median = list_z_grid[0]
        dem_min = list_z_grid[1]
        dem_max = list_z_grid[2]

        if np.any((dem_max - dem_min) < 0):
            logging.error("dem min > dem max")
            raise RuntimeError("dem min > dem max")

        # apply height margin
        dem_min -= self.min_height_margin
        dem_max += self.max_height_margin

        dem_min = np.where(
            dem_median - dem_min < self.min_dem,
            dem_median + self.min_dem,
            dem_min,
        )
        dem_max = np.where(
            dem_max - dem_median > self.max_dem,
            dem_median + self.max_dem,
            dem_max,
        )

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
            dtype=np.float32,
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
            dtype=np.float32,
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
            dtype=np.float32,
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
        # cleanup
        self.orchestrator.cleanup()

        # after saving, fit initial elevation if required
        if initial_elevation is not None and self.coregistration:
            initial_elevation_out_path = os.path.join(
                output_dir, "initial_elevation_fit.tif"
            )

            coreg_offsets = fit_initial_elevation_on_dem_median(
                initial_elevation, dem_median_path, initial_elevation_out_path
            )

            coreg_info = {
                application_constants.APPLICATION_TAG: {
                    "dem_generation": {"coregistration": coreg_offsets}
                }
            }

            # save the coreg shift info in cars's main orchestrator
            cars_orchestrator.update_out_info(coreg_info)

            if (
                coreg_offsets is None
                or abs(coreg_offsets["shift_x"]) > self.coregistration_max_shift
                or abs(coreg_offsets["shift_y"]) > self.coregistration_max_shift
            ):
                logging.warning(
                    "The initial elevation will be used as-is because "
                    "coregistration failed or gave inconsistent results"
                )
                return dem, None

            return dem, initial_elevation_out_path

        return dem, None
