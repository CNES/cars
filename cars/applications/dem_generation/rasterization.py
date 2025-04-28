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


import contextlib
import logging

# Standard imports
import os
import shutil

# Third party imports
import numpy as np
import pandas
import rasterio as rio
import skimage
import xarray as xr
import yaml
from affine import Affine
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from json_checker import And, Checker, Or
from rasterio.enums import Resampling
from rasterio.warp import reproject

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation.dem_generation import DemGeneration
from cars.applications.dem_generation.dem_generation_tools import (
    fit_initial_elevation_on_dem_median,
)

# CARS imports
from cars.core import preprocessing, projection, tiling
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile


class Rasterization(DemGeneration, short_name="bulldozer_on_raster"):
    """
    Rasterization
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of Rasterization

        :param conf: configuration for Rasterization
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.resolution = self.used_config["resolution"]
        self.margin = self.used_config["margin"]
        height_margin = self.used_config["height_margin"]
        if isinstance(height_margin, list):
            self.min_height_margin = height_margin[0]
            self.max_height_margin = height_margin[1]
        else:
            self.min_height_margin = height_margin
            self.max_height_margin = height_margin
        self.morphological_filters_size = self.used_config[
            "morphological_filters_size"
        ]
        self.median_filter_size = self.used_config["median_filter_size"]
        self.fillnodata_max_search_distance = self.used_config[
            "fillnodata_max_search_distance"
        ]
        self.min_dem = self.used_config["min_dem"]
        self.max_dem = self.used_config["max_dem"]
        self.bulldozer_max_object_size = self.used_config[
            "bulldozer_max_object_size"
        ]
        self.compute_stats = self.used_config["compute_stats"]
        self.coregistration = self.used_config["coregistration"]
        self.coregistration_max_shift = self.used_config[
            "coregistration_max_shift"
        ]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

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
        overloaded_conf["method"] = conf.get("method", "bulldozer_on_raster")
        overloaded_conf["resolution"] = conf.get("resolution", 2)
        overloaded_conf["margin"] = conf.get("margin", 300)
        overloaded_conf["morphological_filters_size"] = conf.get(
            "morphological_filters_size", 30
        )
        overloaded_conf["median_filter_size"] = conf.get(
            "median_filter_size", 5
        )
        overloaded_conf["fillnodata_max_search_distance"] = conf.get(
            "fillnodata_max_search_distance", 100
        )
        overloaded_conf["min_dem"] = conf.get("min_dem", -500)
        overloaded_conf["max_dem"] = conf.get("max_dem", 1000)
        overloaded_conf["height_margin"] = conf.get("height_margin", 20)
        overloaded_conf["bulldozer_max_object_size"] = conf.get(
            "bulldozer_max_object_size", 8
        )
        overloaded_conf["compute_stats"] = conf.get("compute_stats", True)
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
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
            "margin": And(Or(float, int), lambda x: x > 0),
            "morphological_filters_size": And(int, lambda x: x > 0),
            "median_filter_size": And(int, lambda x: x > 0),
            "fillnodata_max_search_distance": And(int, lambda x: x > 0),
            "min_dem": And(Or(int, float), lambda x: x < 0),
            "max_dem": And(Or(int, float), lambda x: x > 0),
            "height_margin": And(Or(float, int), lambda x: x > 0),
            "bulldozer_max_object_size": And(int, lambda x: x > 0),
            "compute_stats": bool,
            "coregistration": bool,
            "coregistration_max_shift": And(Or(int, float), lambda x: x > 0),
            "save_intermediate_data": bool,
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
        margin_in_degrees = self.margin * conversion_factor
        resolution_in_meters = self.resolution
        resolution_in_degrees = self.resolution * conversion_factor

        # Get borders, adding margin
        xmin = xmin - margin_in_degrees
        ymin = ymin - margin_in_degrees
        xmax = xmax + margin_in_degrees
        ymax = ymax + margin_in_degrees

        # rasterize point cloud
        dem_min_path = os.path.join(output_dir, "dem_min.tif")
        dem_median_path = os.path.join(output_dir, "dem_median.tif")
        dem_max_path = os.path.join(output_dir, "dem_max.tif")

        point_clouds = cars_dataset.CarsDataset("points")
        terrain_tiling_grid = tiling.generate_tiling_grid(
            xmin,
            ymin,
            xmax,
            ymax,
            xmax - xmin,
            ymax - ymin,
        )
        point_clouds.tiling_grid = terrain_tiling_grid
        point_clouds.tiles[0][0] = merged_point_cloud
        point_clouds.attributes["bounds"] = [xmin, ymin, xmax, ymax]

        # Init rasterization app
        rasterization_application = Application("point_cloud_rasterization")

        dem = rasterization_application.run(
            point_clouds,
            epsg,
            resolution=resolution_in_degrees,
            orchestrator=self.orchestrator,
            dsm_file_name=dem_median_path,
        )

        self.orchestrator.breakpoint()

        if self.save_intermediate_data:
            raw_dsm_path = os.path.join(output_dir, "raw_dsm.tif")
            shutil.copy2(dem_median_path, raw_dsm_path)

        dem_data = dem[0, 0]["hgt"].data
        nodata = rasterization_application.dsm_no_data
        mask = dem_data == nodata

        dem_data = rio.fill.fillnodata(
            dem_data,
            mask=~mask,
            max_search_distance=self.fillnodata_max_search_distance,
        )

        not_filled_pixels = dem_data == nodata

        # Add geoid
        with rio.open(geoid_path) as in_geoid:
            # Reproject the geoid data to match the DSM
            input_geoid_data = np.empty(
                dem_data.shape, dtype=in_geoid.dtypes[0]
            )

            logging.info("Reprojection of geoid data")

            reproject(
                source=rio.band(in_geoid, 1),
                destination=input_geoid_data,
                src_transform=in_geoid.transform,
                src_crs=in_geoid.crs,
                dst_transform=dem[0, 0].profile["transform"],
                dst_crs=dem[0, 0].profile["crs"],
                resampling=Resampling.bilinear,
            )
        dem_data -= input_geoid_data

        # apply height margin
        footprint = skimage.morphology.disk(
            self.morphological_filters_size // 2, decomposition="sequence"
        )
        logging.info("Generation of DEM min")
        dem_data[not_filled_pixels] = -nodata
        dem_min = (
            skimage.morphology.erosion(dem_data, footprint=footprint)
            - self.min_height_margin
        )
        dem_data[not_filled_pixels] = nodata
        logging.info("Generation of DEM max")
        dem_max = (
            skimage.morphology.dilation(dem_data, footprint=footprint)
            + self.max_height_margin
        )
        logging.info("Generation of DEM median")
        dem_median = skimage.filters.median(
            dem_data,
            footprint=np.ones(
                (self.median_filter_size, self.median_filter_size)
            ),
        )

        mask_erosion_filter_size = self.fillnodata_max_search_distance // 2
        footprint = [
            (np.ones((mask_erosion_filter_size, 1)), 1),
            (np.ones((1, mask_erosion_filter_size)), 1),
        ]
        eroded_mask = skimage.morphology.binary_erosion(
            mask, footprint=footprint
        )

        dem_min[eroded_mask] = nodata
        dem_median[eroded_mask] = nodata
        dem_max[eroded_mask] = nodata

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

        # Generate dataset
        dem_min_xr = dem[0, 0]["hgt"].copy()
        dem_min_xr.data = dem_min
        dem_median_xr = dem[0, 0]["hgt"].copy()
        dem_median_xr.data = dem_median
        dem_max_xr = dem[0, 0]["hgt"].copy()
        dem_max_xr.data = dem_max
        dem_tile = xr.Dataset(
            data_vars={
                "dem_min": dem_min_xr,
                "dem_median": dem_median_xr,
                "dem_max": dem_max_xr,
            }
        )

        cars_dataset.fill_dataset(
            dem_tile,
            saving_info=dem[0, 0].saving_info,
            window=dem[0, 0].window,
            profile=dem[0, 0].profile,
            attributes=None,
            overlaps=None,
        )

        dem[0, 0] = dem_tile

        # Save
        self.orchestrator.breakpoint()

        if self.save_intermediate_data:
            intermediate_dem_min_path = os.path.join(
                output_dir, "dem_min_before_bulldozer.tif"
            )
            shutil.copy2(dem_min_path, intermediate_dem_min_path)
            intermediate_dem_max_path = os.path.join(
                output_dir, "dem_max_before_bulldozer.tif"
            )
            shutil.copy2(dem_max_path, intermediate_dem_max_path)

        # Launch Bulldozer on dem min
        saved_transform = edit_transform(
            dem_min_path, resolution=resolution_in_meters
        )
        temp_output_path = launch_bulldozer(
            dem_min_path,
            os.path.join(output_dir, "dem_min_bulldozer"),
            cars_orchestrator,
            self.bulldozer_max_object_size,
        )
        if temp_output_path is not None:
            shutil.copy2(temp_output_path, dem_min_path)
        edit_transform(dem_min_path, transform=saved_transform)

        # Inverse dem max and launch bulldozer
        saved_transform = edit_transform(
            dem_max_path, resolution=resolution_in_meters
        )
        reverse_dem(dem_max_path)
        temp_output_path = launch_bulldozer(
            dem_max_path,
            os.path.join(output_dir, "dem_max_bulldozer"),
            cars_orchestrator,
            self.bulldozer_max_object_size,
        )
        if temp_output_path is not None:
            shutil.copy2(temp_output_path, dem_max_path)
        reverse_dem(dem_max_path)
        edit_transform(dem_max_path, transform=saved_transform)

        # Check DEM min and max
        if self.compute_stats:
            with rio.open(dem_min_path, "r") as in_dem:
                dem_min = in_dem.read()
                nodata = in_dem.nodata
            dem_data[dem_data == nodata] = np.nan
            dem_min[dem_min == nodata] = np.nan
            diff = dem_data - dem_min
            logging.info(
                "Statistics of difference between subsampled "
                "DSM and DEM min (in meters)"
            )
            compute_stats(diff)

            with rio.open(dem_max_path, "r") as in_dem:
                dem_max = in_dem.read()
            dem_max[dem_max == nodata] = np.nan
            diff = dem_max - dem_data
            logging.info(
                "Statistics of difference between DEM max "
                "and subsampled DSM (in meters)"
            )
            compute_stats(diff)

            diff = dem_max - dem_min
            logging.info(
                "Statistics of difference between DEM max "
                "and DEM min (in meters)"
            )
            compute_stats(diff)

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


def edit_transform(input_dem, resolution=None, transform=None):
    """
    Change transform of an image
    :param input_res: path of image
    :type input_dem: str
    :param resolution: image resolution for new transform to apply
    :type resolution: float
    :param transform: new transform to apply if resolution is not given
    :type transform: affine.Affine
    """
    if resolution is not None:
        if transform is None:
            transform = Affine.from_gdal(0, resolution, 0, 0, 0, resolution)
        else:
            raise ValueError(
                "Function edit_transform take resolution or "
                "transform as parameter but not both"
            )
    with rio.open(input_dem, "r+") as in_dem:
        previous_transform = in_dem.transform
        in_dem.transform = transform
    return previous_transform


def reverse_dem(input_dem):
    """
    Compute the opposite of a DEM :
    Altitudes sign is changed

    :param input_dem: path of DEM to reverse
    :type input_dem: str
    """
    with rio.open(input_dem, "r") as in_dem:
        data = in_dem.read()
        metadata = in_dem.meta
        nodata = in_dem.nodata
    with rio.open(input_dem, "w", **metadata) as out_dem:
        out_dem.write(-data)
        out_dem.nodata = -nodata


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


def compute_stats(diff):
    """
    Compute and display statistics of difference between two DEM :
    Minimum, median, percentiles and maximum

    :param diff: altimetric difference between two DEM
    :type diff: numpy.array

    """
    mini = ("Min", np.nanmin(diff))
    median = ("Median", np.nanmedian(diff))
    p90 = ("p90", np.nanpercentile(diff, 90))
    p95 = ("p95", np.nanpercentile(diff, 95))
    p99 = ("p99", np.nanpercentile(diff, 99))
    maxi = ("Max", np.nanmax(diff))
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"| {mini[0]:6} | {median[0]:6} | {p90[0]:6} | "
        f"{p95[0]:6} | {p99[0]:6} | {maxi[0]:6} |"
    )
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"| {mini[1]:6.2f} | {median[1]:6.2f} | {p90[1]:6.2f} | "
        f"{p95[1]:6.2f} | {p99[1]:6.2f} | {maxi[1]:6.2f} |"
    )
