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
# Standard library
import logging
import math
import os
import shutil

# Third-party imports
import numpy as np
import rasterio as rio
import skimage
from json_checker import And, Checker, Or
from rasterio.enums import Resampling
from rasterio.warp import reproject

# CARS imports - Applications
from cars.applications import application_constants
from cars.applications.dem_generation.abstract_dem_generation_app import (
    DemGeneration,
)
from cars.applications.dem_generation.dem_generation_algo import (
    launch_bulldozer,
)
from cars.applications.dem_generation.dem_generation_wrappers import (
    compute_stats,
    downsample_dem,
    edit_transform,
    fit_initial_elevation_on_dem_median,
    reproject_dem,
    reverse_dem,
)

# CARS imports - Core
from cars.core import inputs
from cars.orchestrator.cluster.log_wrapper import cars_profile


class Rasterization(DemGeneration, short_name="bulldozer_on_raster"):
    """
    Rasterization
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, scaling_coeff, conf=None):
        """
        Init function of Rasterization

        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float
        :param conf: configuration for Rasterization
        :return: an application_to_use object
        """
        super().__init__(scaling_coeff, conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
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
        self.preprocessing_median_filter_size = self.used_config[
            "preprocessing_median_filter_size"
        ]
        self.postprocessing_median_filter_size = self.used_config[
            "postprocessing_median_filter_size"
        ]
        self.dem_median_downsample_factor = self.used_config[
            "dem_median_downsample_factor"
        ]
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
        overloaded_conf["margin"] = conf.get(
            "margin", float(self.scaling_coeff * 500)
        )
        overloaded_conf["morphological_filters_size"] = conf.get(
            "morphological_filters_size", 30
        )
        overloaded_conf["preprocessing_median_filter_size"] = conf.get(
            "preprocessing_median_filter_size", 5
        )
        overloaded_conf["postprocessing_median_filter_size"] = conf.get(
            "postprocessing_median_filter_size", 7
        )
        overloaded_conf["dem_median_downsample_factor"] = conf.get(
            "dem_median_downsample_factor", 15
        )
        overloaded_conf["fillnodata_max_search_distance"] = conf.get(
            "fillnodata_max_search_distance", 50
        )
        overloaded_conf["min_dem"] = conf.get("min_dem", -500)
        overloaded_conf["max_dem"] = conf.get("max_dem", 1000)
        overloaded_conf["height_margin"] = conf.get(
            "height_margin", float(math.sqrt(self.scaling_coeff * 100))
        )
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
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
            "margin": And(Or(float, int), lambda x: x > 0),
            "morphological_filters_size": And(int, lambda x: x > 0),
            "preprocessing_median_filter_size": And(int, lambda x: x > 0),
            "postprocessing_median_filter_size": And(int, lambda x: x > 0),
            "dem_median_downsample_factor": And(int, lambda x: x > 0),
            "fillnodata_max_search_distance": And(int, lambda x: x > 0),
            "min_dem": And(Or(int, float), lambda x: x < 0),
            "max_dem": And(Or(int, float), lambda x: x > 0),
            "height_margin": Or(list, float, int),
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
    def run(  # noqa: C901
        self,
        dsm_file_name,
        output_dir,
        dem_min_file_name,
        dem_max_file_name,
        dem_median_file_name,
        input_geoid,
        output_geoid,
        initial_elevation=None,
        default_alt=0,
        cars_orchestrator=None,
    ):
        """
        Run dichotomic dem generation using matches

        :param dsm_file_name: The dsm path
        :type dsm_file_name: CarsDataset
        :param output_dir: directory to save dem
        :type output_dir: str
        :param dem_min_file_name: path of dem_min
        :type dem_min_file_name: str
        :param dem_max_file_name: path of dem_max
        :type dem_max_file_name: str
        :param dem_median_file_name: path of dem_median
        :type dem_median_file_name: str
        :param input_geoid: input geoid path
        :param output_geoid: output geoid path
        :param dem_roi_to_use: dem roi polygon to use as roi

        :return: dem data computed with mean, min and max.
            dem is also saved in disk, and paths are available in attributes.
            (DEM_MEDIAN_PATH, DEM_MIN_PATH, DEM_MAX_PATH)
        :rtype: CarsDataset
        """

        # Generate point cloud
        epsg = 4326

        # Optimize the case when input and output geoid are the same
        if output_geoid is True:
            input_geoid = False
            output_geoid = False

        # rasterize point cloud
        dem_min_path = dem_min_file_name
        if dem_min_path is None:
            # File is not part of the official product, write it in dump_dir
            dem_min_path = os.path.join(output_dir, "dem_min.tif")

        dem_max_path = dem_max_file_name
        if dem_max_path is None:
            # File is not part of the official product, write it in dump_dir
            dem_max_path = os.path.join(output_dir, "dem_max.tif")

        dem_median_path_out = dem_median_file_name
        if dem_median_path_out is None:
            # File is not part of the official product, write it in dump_dir
            dem_median_path_out = os.path.join(output_dir, "dem_median.tif")

        dem_median_path_in = dsm_file_name

        dem_epsg = inputs.rasterio_get_epsg(dsm_file_name)

        if dem_epsg != epsg:
            dem_median_path_in = os.path.join(output_dir, "dem_median.tif")
            reproject_dem(dsm_file_name, "EPSG:4326", dem_median_path_in)

        with rio.open(dem_median_path_in) as src:
            dem = src.read(1)
            profile = src.profile
            nodata = src.nodata

        if self.save_intermediate_data:
            raw_dsm_path = os.path.join(output_dir, "raw_dsm.tif")
            shutil.copy2(dem_median_path_in, raw_dsm_path)

        dem_data = dem
        mask = dem_data == nodata

        # fill nodata
        max_search_distance = (
            self.fillnodata_max_search_distance
            + self.morphological_filters_size
        )
        # a margin is added for following morphological operations
        # pixels further than self.fillnodata_max_search_distance
        # will later be turned into nodata by eroded_mask
        dem_data = rio.fill.fillnodata(
            dem_data,
            mask=~mask,
            max_search_distance=max_search_distance,
        )

        not_filled_pixels = dem_data == nodata

        # Add geoid
        if input_geoid:
            with rio.open(input_geoid) as in_geoid:
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
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
            dem_data -= input_geoid_data

        if output_geoid:
            with rio.open(input_geoid) as in_geoid:
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
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
            dem_data += input_geoid_data

        # apply morphological filters and height margin
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
                (
                    self.preprocessing_median_filter_size,
                    self.preprocessing_median_filter_size,
                )
            ),
        )

        footprint = skimage.morphology.disk(
            self.fillnodata_max_search_distance // 2, decomposition="sequence"
        )
        eroded_mask = skimage.morphology.binary_erosion(
            mask, footprint=footprint
        )

        dem_min[eroded_mask] = nodata
        dem_median[eroded_mask] = nodata
        dem_max[eroded_mask] = nodata

        # Rectify pixels where DEM min < DEM median + min_depth
        dem_min = np.where(
            dem_min - dem_median < self.min_dem,
            dem_median + self.min_dem,
            dem_min,
        )
        # Rectify pixels where DEM max > DEM median + max_height
        dem_max = np.where(
            dem_max - dem_median > self.max_dem,
            dem_median + self.max_dem,
            dem_max,
        )

        # save
        with rio.open(dem_min_path, "w", **profile) as out_dem:
            out_dem.write(dem_min, 1)
        with rio.open(dem_median_path_out, "w", **profile) as out_dem:
            out_dem.write(dem_median, 1)
        with rio.open(dem_max_path, "w", **profile) as out_dem:
            out_dem.write(dem_max, 1)

        if self.save_intermediate_data:
            intermediate_dem_min_path = os.path.join(
                output_dir, "dem_min_before_bulldozer.tif"
            )
            shutil.copy2(dem_min_path, intermediate_dem_min_path)
            intermediate_dem_max_path = os.path.join(
                output_dir, "dem_max_before_bulldozer.tif"
            )
            shutil.copy2(dem_max_path, intermediate_dem_max_path)

            intermediate_dem_median_path = os.path.join(
                output_dir, "dem_median_before_downsampling.tif"
            )
            shutil.copy2(dem_median_path_out, intermediate_dem_median_path)

        # Get dsm resolution
        dsm_resolution = (
            inputs.rasterio_get_resolution(dem_median_path_in)[0]
            + inputs.rasterio_get_resolution(dem_median_path_in)[1]
        ) / 2

        # Downsample median dem only if large enough
        if np.sum(~mask) / self.dem_median_downsample_factor**2 > 10:
            downsample_dem(
                dem_median_path_out,
                scale=self.dem_median_downsample_factor,
                median_filter_size=self.postprocessing_median_filter_size,
                default_alt=default_alt,
            )

        # Launch Bulldozer on dem min
        saved_transform = edit_transform(
            dem_min_path, resolution=dsm_resolution
        )
        logging.info("Launch Bulldozer on DEM min")
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
            dem_max_path, resolution=dsm_resolution
        )
        reverse_dem(dem_max_path)
        logging.info("Launch Bulldozer on DEM max")
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
        with rio.open(dem_min_path, "r") as in_dem:
            dem_min = in_dem.read()
            dem_min_metadata = in_dem.meta
            nodata = in_dem.nodata
        with rio.open(dem_max_path, "r") as in_dem:
            dem_max = in_dem.read()
            dem_max_metadata = in_dem.meta
        dem_data[dem_data == nodata] = np.nan
        dem_min[dem_min == nodata] = np.nan
        dem_max[dem_max == nodata] = np.nan
        if self.compute_stats:
            diff = dem_data - dem_min
            logging.info(
                "Statistics of difference between subsampled "
                "DSM and DEM min (in meters)"
            )
            compute_stats(diff)

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

        # Rectify pixels where DEM min > DEM - margin
        dem_min = np.where(
            dem_min > dem_data - self.min_height_margin,
            dem_data - self.min_height_margin,
            dem_min,
        )
        # Rectify pixels where DEM max < DEM + margin
        dem_max = np.where(
            dem_max < dem_data + self.max_height_margin,
            dem_data + self.max_height_margin,
            dem_max,
        )

        with rio.open(dem_min_path, "w", **dem_min_metadata) as out_dem:
            out_dem.write(dem_min)
        with rio.open(dem_max_path, "w", **dem_max_metadata) as out_dem:
            out_dem.write(dem_max)

        paths = {
            "dem_median": dem_median_path_out,
            "dem_min": dem_min_path,
            "dem_max": dem_max_path,
        }

        # after saving, fit initial elevation if required
        if initial_elevation is not None and self.coregistration:
            initial_elevation_out_path = os.path.join(
                output_dir, "initial_elevation_fit.tif"
            )

            coreg_offsets = fit_initial_elevation_on_dem_median(
                initial_elevation,
                dem_median_path_out,
                initial_elevation_out_path,
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
                return dem, paths, None

            return dem, paths, initial_elevation_out_path

        return dem, paths, None
