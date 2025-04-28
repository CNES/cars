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
# pylint: disable=too-many-lines
"""
this module contains the LineOfSightIntersection application class.
"""

# Standard imports
import logging
import os
from typing import Dict, Tuple

# Third party imports
import numpy as np
import pandas
import xarray as xr
from json_checker import Checker
from shareloc.geofunctions.rectification_grid import RectificationGrid

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_generation import grids
from cars.applications.point_cloud_fusion import point_cloud_tools
from cars.applications.triangulation import (
    triangulation_constants,
    triangulation_sparse_tools,
    triangulation_tools,
)
from cars.applications.triangulation.triangulation import Triangulation
from cars.conf import mask_cst
from cars.core import constants as cst
from cars.core import inputs, projection, tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


class LineOfSightIntersection(
    Triangulation, short_name="line_of_sight_intersection"
):
    """
    Triangulation
    """

    def __init__(self, conf=None):
        """
        Init function of Triangulation

        :param conf: configuration for triangulation
        :return: an application_to_use object
        """

        super().__init__(conf=conf)
        # check conf
        self.used_method = self.used_config["method"]
        self.snap_to_img1 = self.used_config["snap_to_img1"]
        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

        # global value for left image to check if snap_to_img1 can
        # be applied : Need than same application object is run
        # for all pairs
        self.ref_left_img = None

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
        overloaded_conf["method"] = conf.get(
            "method", "line_of_sight_intersection"
        )
        overloaded_conf["snap_to_img1"] = conf.get("snap_to_img1", False)

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        triangulation_schema = {
            "method": str,
            "snap_to_img1": bool,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(triangulation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def save_triangulation_output(  # noqa: C901 function is too complex
        self,
        epipolar_point_cloud,
        sensor_image_left,
        output_dir,
        dump_dir=None,
        performance_maps_to_generate=None,
        save_output_coordinates=True,
        save_output_color=True,
        save_output_classification=False,
        save_output_mask=False,
        save_output_filling=False,
        save_output_performance_map=False,
        save_output_ambiguity=False,
    ):
        """
        Save the triangulation output. The different TIFs composing the depth
        map are written to the output directory. Auxiliary products can be
        requested or not using the parameters. A dump directory can also be
        provided to write any additionnal files that have not been written
        to the output directory (because they are not part of the depth map
        definition, or because they have not been requested).

        :param epipolar_point_cloud: tiled epipolar left image
        :type epipolar_point_cloud: CarsDataset
        :param sensor_image_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_left: CarsDataset
        :param output_dir: directory to write triangulation output depth
                map.
        :type output_dir: None or str
        :param dump_dir: folder used as dump directory for current pair, None to
                deactivate intermediate data writing
        :type dump_dir: str
        :param performance_maps_to_generate: None or list containing
            "risk" or "intervals"
        :type performance_maps_to_generate: None or list containing
            "risk" or "intervals"
        :param save_output_coordinates: Save X, Y and Z coords in output_dir
        :type save_output_coordinates: bool
        :param save_output_color: Save color depth map in output_dir
        :type save_output_color: bool
        :param save_output_classification: Save classification depth map in
                output_dir
        :type save_output_classification: bool
        :param save_output_mask: Save mask depth map in output_dir
        :type save_output_mask: bool
        :param save_output_filling: Save filling depth map in output_dir
        :type save_output_filling: bool
        :param save_output_performance_map: Save performance map in output_dir
        :type save_output_performance_map: bool
        :param save_output_ambiguity: Save ambiguity in output_dir
        :type save_output_ambiguity: bool
        """

        if dump_dir:
            safe_makedirs(dump_dir)

        # Propagate color type in output file
        color_type = None
        if sens_cst.INPUT_COLOR in sensor_image_left:
            color_type = inputs.rasterio_get_image_type(
                sensor_image_left[sens_cst.INPUT_COLOR]
            )
        else:
            color_type = inputs.rasterio_get_image_type(
                sensor_image_left[sens_cst.INPUT_IMG]
            )

        if output_dir is None:
            output_dir = dump_dir

        if save_output_coordinates or dump_dir:
            coords_output_dir = (
                output_dir if save_output_coordinates else dump_dir
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(coords_output_dir, "X.tif"),
                cst.X,
                epipolar_point_cloud,
                cars_ds_name="depth_map_x",
                dtype=np.float64,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(coords_output_dir, "Y.tif"),
                cst.Y,
                epipolar_point_cloud,
                cars_ds_name="depth_map_y",
                dtype=np.float64,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(coords_output_dir, "Z.tif"),
                cst.Z,
                epipolar_point_cloud,
                cars_ds_name="depth_map_z",
                dtype=np.float64,
            )

        if save_output_color or dump_dir:
            color_output_dir = output_dir if save_output_color else dump_dir
            self.orchestrator.add_to_save_lists(
                os.path.join(color_output_dir, "color.tif"),
                cst.EPI_COLOR,
                epipolar_point_cloud,
                cars_ds_name="depth_map_color",
                dtype=color_type,
            )

        if save_output_mask or dump_dir:
            mask_output_dir = output_dir if save_output_mask else dump_dir
            self.orchestrator.add_to_save_lists(
                os.path.join(mask_output_dir, "mask.tif"),
                cst.EPI_MSK,
                epipolar_point_cloud,
                cars_ds_name="depth_map_msk",
                nodata=mask_cst.NO_DATA_IN_EPIPOLAR_RECTIFICATION,
                optional_data=True,
                dtype=np.uint8,
            )

        if save_output_performance_map or dump_dir:
            map_output_dir = (
                output_dir if save_output_performance_map else dump_dir
            )
            performance_map_tail_path = "performance_map.tif"
            if "intervals" in performance_maps_to_generate:
                tail_path = performance_map_tail_path
                if len(performance_maps_to_generate) == 2:
                    tail_path = "performance_map_from_intervals.tif"
                self.orchestrator.add_to_save_lists(
                    os.path.join(map_output_dir, tail_path),
                    cst.POINT_CLOUD_PERFORMANCE_MAP_FROM_INTERVALS,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_performance_map_from_intervals",
                    optional_data=True,
                    dtype=np.float64,
                )

            if "risk" in performance_maps_to_generate:
                tail_path = performance_map_tail_path
                if len(performance_maps_to_generate) == 2:
                    tail_path = "performance_map_from_risk.tif"
                self.orchestrator.add_to_save_lists(
                    os.path.join(map_output_dir, tail_path),
                    cst.POINT_CLOUD_PERFORMANCE_MAP_FROM_RISK,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_performance_map_from_risk",
                    optional_data=True,
                    dtype=np.float64,
                )

        if save_output_ambiguity or dump_dir:
            map_output_dir = output_dir if save_output_ambiguity else dump_dir
            self.orchestrator.add_to_save_lists(
                os.path.join(map_output_dir, "ambiguity.tif"),
                cst.EPI_AMBIGUITY,
                epipolar_point_cloud,
                cars_ds_name="depth_map_ambiguity",
                optional_data=True,
                dtype=np.float64,
            )

        if save_output_classification or dump_dir:
            classif_output_dir = (
                output_dir if save_output_classification else dump_dir
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(classif_output_dir, "classification.tif"),
                cst.EPI_CLASSIFICATION,
                epipolar_point_cloud,
                cars_ds_name="depth_map_classification",
                optional_data=True,
                dtype=np.uint8,
            )

        if save_output_filling or dump_dir:
            filling_output_dir = output_dir if save_output_filling else dump_dir
            self.orchestrator.add_to_save_lists(
                os.path.join(filling_output_dir, "filling.tif"),
                cst.EPI_FILLING,
                epipolar_point_cloud,
                cars_ds_name="depth_map_filling",
                optional_data=True,
                dtype=np.uint8,
                nodata=255,
            )

        if dump_dir and performance_maps_to_generate is not None:
            if "intervals" in performance_maps_to_generate:
                self.orchestrator.add_to_save_lists(
                    os.path.join(dump_dir, "Z_inf_from_intervals.tif"),
                    cst.POINT_CLOUD_LAYER_INF_FROM_INTERVALS,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_z_inf_from_intervals",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(dump_dir, "Z_sup_from_intervals.tif"),
                    cst.POINT_CLOUD_LAYER_SUP_FROM_INTERVALS,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_z_sup_from_intervals",
                )

            if "risk" in performance_maps_to_generate:
                self.orchestrator.add_to_save_lists(
                    os.path.join(dump_dir, "Z_inf_from_risk.tif"),
                    cst.POINT_CLOUD_LAYER_INF_FROM_RISK,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_z_inf_from_risk",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(dump_dir, "Z_sup_from_risk.tif"),
                    cst.POINT_CLOUD_LAYER_SUP_FROM_RISK,
                    epipolar_point_cloud,
                    cars_ds_name="depth_map_z_sup_from_risk",
                )

        if dump_dir:
            self.orchestrator.add_to_save_lists(
                os.path.join(dump_dir, "corr_mask.tif"),
                cst.POINT_CLOUD_CORR_MSK,
                epipolar_point_cloud,
                cars_ds_name="depth_map_corr_msk",
                optional_data=True,
            )

    def fill_index(
        self,
        save_output_coordinates=True,
        save_output_color=True,
        save_output_classification=False,
        save_output_mask=False,
        save_output_filling=False,
        save_output_performance_map=False,
        save_output_ambiguity=False,
        pair_key="PAIR_0",
    ):
        """
        Fill depth map index for current pair, according to which product
        should be saved

        :param save_output_coordinates: Save X, Y and Z coords in output_dir
        :type save_output_coordinates: bool
        :param save_output_color: Save color depth map in output_dir
        :type save_output_color: bool
        :param save_output_classification: Save classification depth map in
                output_dir
        :type save_output_classification: bool
        :param save_output_mask: Save mask depth map in output_dir
        :type save_output_mask: bool
        :param save_output_filling: Save filling depth map in output_dir
        :type save_output_filling: bool
        :param save_output_performance_map: Save performance map in output_dir
        :type save_output_performance_map: bool
        :param save_output_ambiguity: Save ambiguity in output_dir
        :type save_output_ambiguity: bool
        :param pair_key: name of the current pair
        :type pair_key: str
        """

        # index file for this depth map
        index = {}

        if save_output_coordinates:
            index[cst.INDEX_DEPTH_MAP_X] = os.path.join(pair_key, "X.tif")
            index[cst.INDEX_DEPTH_MAP_Y] = os.path.join(pair_key, "Y.tif")
            index[cst.INDEX_DEPTH_MAP_Z] = os.path.join(pair_key, "Z.tif")

        if save_output_color:
            index[cst.INDEX_DEPTH_MAP_COLOR] = os.path.join(
                pair_key, "color.tif"
            )

        if save_output_mask:
            index[cst.INDEX_DEPTH_MAP_MASK] = os.path.join(pair_key, "mask.tif")

        if save_output_performance_map:
            index[cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP] = os.path.join(
                pair_key, "performance_map.tif"
            )

        if save_output_ambiguity:
            index[cst.INDEX_DEPTH_MAP_AMBIGUITY] = os.path.join(
                pair_key, "ambiguity.tif"
            )

        if save_output_classification:
            index[cst.INDEX_DEPTH_MAP_CLASSIFICATION] = os.path.join(
                pair_key, "classification.tif"
            )

        if save_output_filling:
            index[cst.INDEX_DEPTH_MAP_FILLING] = os.path.join(
                pair_key, "filling.tif"
            )

        # update orchestrator index if it has been filled
        if index:
            # Add epsg code (always lon/lat in triangulation)
            index[cst.INDEX_DEPTH_MAP_EPSG] = 4326
            self.orchestrator.update_index({"depth_map": {pair_key: index}})

    def create_point_cloud_directories(
        self, pair_dump_dir, point_cloud_dir, point_cloud
    ):
        """
        Set and create directories for point cloud disk output (laz and csv)
        The function return None path if the point cloud should not be saved

        :param pair_dump_dir: folder used as dump directory for current pair
        :type pair_dump_dir: str
        :param point_cloud_dir: folder used for laz official product directory
        :type point_cloud_dir: str
        :param point_cloud: input point cloud (for orchestrator registration)
        :type point_cloud: Dataset
        """

        csv_pc_dir_name = None
        if self.save_intermediate_data:
            csv_pc_dir_name = os.path.join(pair_dump_dir, "csv")
            safe_makedirs(csv_pc_dir_name)
            self.orchestrator.add_to_compute_lists(
                point_cloud, cars_ds_name="point_cloud_csv"
            )
        laz_pc_dir_name = None
        if self.save_intermediate_data or point_cloud_dir is not None:
            if point_cloud_dir is not None:
                laz_pc_dir_name = point_cloud_dir
            else:
                laz_pc_dir_name = os.path.join(pair_dump_dir, "laz")
            safe_makedirs(laz_pc_dir_name)
            self.orchestrator.add_to_compute_lists(
                point_cloud, cars_ds_name="point_cloud_laz"
            )

        return csv_pc_dir_name, laz_pc_dir_name

    def run(  # noqa: C901
        self,
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        epipolar_disparity_map,
        geometry_plugin,
        epipolar_image,
        epsg=None,
        denoising_overload_fun=None,
        source_pc_names=None,
        orchestrator=None,
        pair_dump_dir=None,
        pair_key="PAIR_0",
        uncorrected_grid_right=None,
        geoid_path=None,
        cloud_id=None,
        performance_maps_param=None,
        depth_map_dir=None,
        point_cloud_dir=None,
        save_output_coordinates=False,
        save_output_color=False,
        save_output_classification=False,
        save_output_mask=False,
        save_output_filling=False,
        save_output_performance_map=False,
        save_output_ambiguity=False,
    ):
        """
        Run Triangulation application.

        Created left and right CarsDataset filled with xarray.Dataset,
        corresponding to 3D point clouds, stored on epipolar geometry grid.

        :param sensor_image_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_left: CarsDataset
        :param sensor_image_right: tiled sensor right image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_right: CarsDataset
        :param grid_left: left epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",\
                "epipolar_origin_y","epipolar_spacing_x",\
                "epipolar_spacing", "disp_to_alt_ratio",\
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",
                "epipolar_origin_y","epipolar_spacing_x",
                "epipolar_spacing", "disp_to_alt_ratio",
        :type grid_right: CarsDataset
        :param epipolar_disparity_map: tiled left disparity map or \
            sparse matches:

            - if CarsDataset is instance of "arrays", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "disp", "disp_msk"
                    - attrs with keys: profile, window, overlaps
                - attributes containing:"largest_epipolar_region"\
                  "opt_epipolar_tile_size"

            - if CarsDataset is instance of "points", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future pandas DataFrame containing:

                    - data : (L, 4) shape matches
                - attributes containing:"disp_lower_bound","disp_upper_bound",\
                    "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :type epipolar_disparity_map: CarsDataset
        :param epipolar_image: tiled epipolar left image
        :type epipolar_image: CarsDataset
        :param denoising_overload_fun: function to overload dataset
        :type denoising_overload_fun: fun
        :param source_pc_names: source pc names
        :type source_pc_names: list[str]
        :param orchestrator: orchestrator used
        :param pair_dump_dir: folder used as dump directory for current pair
        :type pair_dump_dir: str
        :param pair_key: pair key id
        :type pair_key: str
        :param uncorrected_grid_right: not corrected right epipolar grid
                used if self.snap_to_img1
        :type uncorrected_grid_right: CarsDataset
        :param geoid_path: geoid path
        :type geoid_path: str
        :param performance_maps_param: parameters used
            to generate performance map
        :type performance_maps_param: dict or None
        :param depth_map_dir: directory to write triangulation output depth
                map.
        :type depth_map_dir: None or str
        :param save_output_coordinates: Save X, Y, Z coords in depth_map_dir
        :type save_output_coordinates: bool
        :param save_output_color: Save color depth map in depth_map_dir
        :type save_output_color: bool
        :param save_output_classification: Save classification depth map in
                depth_map_dir
        :type save_output_classification: bool
        :param save_output_mask: Save mask depth map in depth_map_dir
        :type save_output_mask: bool
        :param save_output_filling: Save filling depth map in depth_map_dir
        :type save_output_filling: bool
        :param save_output_performance_map: Save performance map in
                depth_map_dir
        :type save_output_performance_map: bool
        :param save_output_ambiguity: Save ambiguity in
                depth_map_dir
        :type save_output_ambiguity: bool

        :return: point cloud \
                The CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "color", "msk", "z_inf", "z_sup"
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :rtype: Tuple(CarsDataset, CarsDataset)
        """
        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if epipolar_disparity_map.dataset_type not in ("arrays", "points"):
            raise RuntimeError(
                "Triangulation application doesn't support this input "
                "data format"
            )

        # Interpolate grid
        interpolated_grid_left = RectificationGrid(
            grid_left.attributes["path"],
            interpolator=geometry_plugin.interpolator,
        )
        interpolated_grid_right = RectificationGrid(
            grid_right.attributes["path"],
            interpolator=geometry_plugin.interpolator,
        )

        broadcasted_interpolated_grid_left = self.orchestrator.cluster.scatter(
            interpolated_grid_left
        )
        broadcasted_interpolated_grid_right = self.orchestrator.cluster.scatter(
            interpolated_grid_right
        )

        if epipolar_disparity_map.dataset_type != "points":
            if source_pc_names is None:
                source_pc_names = ["PAIR_0"]

            if pair_dump_dir is None:
                pair_dump_dir = os.path.join(self.orchestrator.out_dir, "tmp")

            # Get local conf left image for this in_json iteration
            conf_left_img = sensor_image_left[sens_cst.INPUT_IMG]
            # Check left image and raise a warning
            # if different left images are used along with snap_to_img1 mode
            if self.ref_left_img is None:
                self.ref_left_img = conf_left_img
            else:
                if self.snap_to_img1 and self.ref_left_img != conf_left_img:
                    logging.warning(
                        "snap_to_left_image mode is used but inputs "
                        "have different images as their "
                        "left image in pair. This may result in "
                        "increasing registration discrepancies between pairs"
                    )

            # Add log about geoid
            if geoid_path is not None:
                alt_reference = "geoid"
            else:
                alt_reference = "ellipsoid"

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    triangulation_constants.TRIANGULATION_RUN_TAG: {
                        pair_key: {
                            triangulation_constants.ALT_REFERENCE_TAG: (
                                alt_reference
                            )
                        },
                    }
                }
            }
            self.orchestrator.update_out_info(updating_dict)

            sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
            sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
            geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
            geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

            if self.snap_to_img1:
                grid_right = uncorrected_grid_right
                if grid_right is None:
                    logging.error(
                        "Uncorrected grid was not given in order "
                        "to snap it to img1"
                    )

            # Compute disp_min and disp_max location for epipolar grid

            # Transform
            disp_min_tiling = epipolar_disparity_map.attributes[
                "disp_min_tiling"
            ]
            disp_max_tiling = epipolar_disparity_map.attributes[
                "disp_max_tiling"
            ]

            # change to N+1 M+1 dimension, fitting to tiling
            (
                disp_min_tiling,
                disp_max_tiling,
            ) = tiling.transform_disp_range_grid_to_two_layers(
                disp_min_tiling, disp_max_tiling
            )
            (
                epipolar_grid_min,
                epipolar_grid_max,
            ) = grids.compute_epipolar_grid_min_max(
                geometry_plugin,
                tiling.transform_four_layers_to_two_layers_grid(
                    epipolar_image.tiling_grid
                ),
                sensor1,
                sensor2,
                geomodel1,
                geomodel2,
                grid_left,
                grid_right,
                epsg,
                disp_min_tiling,
                disp_max_tiling,
            )
            # update attributes for corresponding tiles in cloud fusion
            # TODO remove with refactoring
            pc_attributes = {
                "used_epsg_for_terrain_grid": epsg,
                "epipolar_grid_min": epipolar_grid_min,
                "epipolar_grid_max": epipolar_grid_max,
                "largest_epipolar_region": epipolar_image.attributes[
                    "largest_epipolar_region"
                ],
                "source_pc_names": source_pc_names,
                "source_pc_name": pair_key,
                "color_type": epipolar_image.attributes["color_type"],
                "opt_epipolar_tile_size": epipolar_image.attributes[
                    "tile_width"
                ],
            }

            if geoid_path:
                pc_attributes["geoid"] = (geoid_path,)

            if epipolar_disparity_map.dataset_type not in ("arrays", "points"):
                raise RuntimeError(
                    "Triangulation application doesn't support this input "
                    "data format"
                )

            # Check performance_maps_parameters
            performance_maps_to_generate = None
            if performance_maps_param is not None:
                if "performance_map_method" not in performance_maps_param:
                    raise RuntimeError("No performance_map_method specified")
                performance_maps_to_generate = performance_maps_param[
                    "performance_map_method"
                ]
                if "perf_ambiguity_threshold" not in performance_maps_param:
                    raise RuntimeError("No perf_ambiguity_threshold specified")

            # Create CarsDataset
            # Epipolar_point_cloud
            epipolar_point_cloud = cars_dataset.CarsDataset(
                epipolar_disparity_map.dataset_type,
                name="triangulation_" + pair_key,
            )
            epipolar_point_cloud.create_empty_copy(epipolar_image)
            epipolar_point_cloud.overlaps *= 0  # Margins removed

            # Update attributes to get epipolar info
            epipolar_point_cloud.attributes.update(pc_attributes)

            # Save objects
            # Save as depth map
            self.save_triangulation_output(
                epipolar_point_cloud,
                sensor_image_left,
                depth_map_dir,
                pair_dump_dir if self.save_intermediate_data else None,
                performance_maps_to_generate,
                save_output_coordinates,
                save_output_color,
                save_output_classification,
                save_output_mask,
                save_output_filling,
                save_output_performance_map,
                save_output_ambiguity,
            )
            self.fill_index(
                save_output_coordinates,
                save_output_color,
                save_output_classification,
                save_output_mask,
                save_output_filling,
                save_output_performance_map,
                save_output_ambiguity,
                pair_key,
            )
            # Save as point cloud
            point_cloud = cars_dataset.CarsDataset(
                "points",
                name="triangulation_flatten_" + pair_key,
            )
            point_cloud.create_empty_copy(epipolar_point_cloud)
            point_cloud.attributes = epipolar_point_cloud.attributes

            csv_pc_dir_name, laz_pc_dir_name = (
                self.create_point_cloud_directories(
                    pair_dump_dir, point_cloud_dir, point_cloud
                )
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_info_epipolar] = self.orchestrator.get_saving_infos(
                [epipolar_point_cloud]
            )
            saving_info_flatten = None
            if self.save_intermediate_data or point_cloud_dir is not None:
                [saving_info_flatten] = self.orchestrator.get_saving_infos(
                    [point_cloud]
                )

            # Generate Point clouds

            # initialize empty index file for point cloud product if official
            # product is requested
            pc_index = None
            if point_cloud_dir:
                pc_index = {}

            for col in range(epipolar_disparity_map.shape[1]):
                for row in range(epipolar_disparity_map.shape[0]):
                    if epipolar_disparity_map[row, col] is not None:
                        # update saving infos  for potential replacement
                        full_saving_info_epipolar = ocht.update_saving_infos(
                            saving_info_epipolar, row=row, col=col
                        )
                        full_saving_info_flatten = None
                        if saving_info_flatten is not None:
                            full_saving_info_flatten = ocht.update_saving_infos(
                                saving_info_flatten, row=row, col=col
                            )

                        csv_pc_file_name, laz_pc_file_name = (
                            triangulation_tools.generate_point_cloud_file_names(
                                csv_pc_dir_name,
                                laz_pc_dir_name,
                                row,
                                col,
                                pc_index,
                                pair_key,
                            )
                        )

                        # Compute points
                        (
                            epipolar_point_cloud[row][col],
                            point_cloud[row][col],
                        ) = self.orchestrator.cluster.create_task(
                            triangulation_wrapper, nout=2
                        )(
                            epipolar_disparity_map[row, col],
                            sensor1,
                            sensor2,
                            geomodel1,
                            geomodel2,
                            broadcasted_interpolated_grid_left,
                            broadcasted_interpolated_grid_right,
                            geometry_plugin,
                            epsg,
                            geoid_path=geoid_path,
                            denoising_overload_fun=denoising_overload_fun,
                            cloud_id=cloud_id,
                            performance_maps_to_generate=(
                                performance_maps_to_generate
                            ),
                            performance_maps_parameters=performance_maps_param,
                            point_cloud_csv_file_name=csv_pc_file_name,
                            point_cloud_laz_file_name=laz_pc_file_name,
                            saving_info_epipolar=full_saving_info_epipolar,
                            saving_info_flatten=full_saving_info_flatten,
                        )

            # update point cloud index
            if point_cloud_dir:
                self.orchestrator.update_index(pc_index)

        else:
            epipolar_point_cloud = cars_dataset.CarsDataset(
                "points", name="pandora_sparse_matching_" + pair_key
            )
            epipolar_point_cloud.create_empty_copy(epipolar_image)

            # Update attributes to get epipolar info
            epipolar_point_cloud.attributes.update(epipolar_image.attributes)

            # Add to replace list so tiles will be readable at the same time
            self.orchestrator.add_to_replace_lists(
                epipolar_point_cloud,
                cars_ds_name="triangulated_matches",
            )

            broadcasted_grid_left = self.orchestrator.cluster.scatter(grid_left)
            broadcasted_grid_right = self.orchestrator.cluster.scatter(
                grid_right
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_info_matches] = self.orchestrator.get_saving_infos(
                [epipolar_point_cloud]
            )

            for col in range(epipolar_disparity_map.shape[1]):
                for row in range(epipolar_disparity_map.shape[0]):
                    if epipolar_disparity_map[row, col] is not None:
                        # initialize list of matches
                        full_saving_info_matches = ocht.update_saving_infos(
                            saving_info_matches, row=row, col=col
                        )

                        # Compute points
                        (
                            epipolar_point_cloud[row][col]
                        ) = self.orchestrator.cluster.create_task(
                            triangulation_wrapper_matches, nout=1
                        )(
                            epipolar_disparity_map[row, col],
                            sensor_image_left,
                            sensor_image_right,
                            broadcasted_grid_left,
                            broadcasted_grid_right,
                            broadcasted_interpolated_grid_left,
                            broadcasted_interpolated_grid_right,
                            geometry_plugin,
                            full_saving_info_matches,
                        )

        return epipolar_point_cloud


def triangulation_wrapper(
    disparity_object: xr.Dataset,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    geometry_plugin,
    epsg,
    geoid_path=None,
    denoising_overload_fun=None,
    cloud_id=None,
    performance_maps_to_generate=None,
    performance_maps_parameters=None,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
    saving_info_epipolar=None,
    saving_info_flatten=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute point clouds from image objects and disparity objects.

    :param disparity_object: Left disparity map dataset with :
            - cst_disp.MAP
            - cst_disp.VALID
            - cst.EPI_COLOR
    :type disparity_object: xr.Dataset
    :param sensor1: path to left sensor image
    :type sensor1: str
    :param sensor2: path to right sensor image
    :type sensor2: str
    :param geomodel1: path and attributes for left geomodel
    :type geomodel1: dict
    :param geomodel2: path and attributes for right geomodel
    :type geomodel2: dict
    :param grid1: dataset of the reference image grid file
    :type grid1: CarsDataset
    :param grid2: dataset of the secondary image grid file
    :type grid2: CarsDataset
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param geoid_path: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_path: str
    :param performance_maps_to_generate: None or list containing
        "risk" or "intervals"
    :param performance_maps_parameters: parameters used to
        generate performance map
    :type performance_maps_parameters: dict or None
    :param denoising_overload_fun: function to overload dataset
    :type denoising_overload_fun: fun

    :return: Left disparity object

    Returned object is composed of :
        - dataset with :
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
            - cst.Z_INF (optional)
            - cst.Z_SUP (optional)
    """

    # Get disparity maps
    disp_ref = disparity_object

    # Triangulate
    if isinstance(disp_ref, xr.Dataset):
        # Triangulate epipolar dense disparities
        points = triangulation_tools.triangulate(
            geometry_plugin,
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            grid1,
            grid2,
            disp_ref,
        )

        if performance_maps_to_generate is not None:
            for perf_method in performance_maps_to_generate:
                # Generate keys to use
                if perf_method == "risk":
                    # From pandora loader overload
                    disp_keys = [
                        "confidence_from_disp_inf_from_risk.cars_2",
                        "confidence_from_disp_sup_from_risk.cars_2",
                    ]
                    cars_inf_key = cst.POINT_CLOUD_LAYER_INF_FROM_RISK
                    cars_sup_key = cst.POINT_CLOUD_LAYER_SUP_FROM_RISK
                    cars_perf_key = cst.POINT_CLOUD_PERFORMANCE_MAP_FROM_RISK
                    use_ambiguity = True
                else:
                    # From pandora loader overload
                    disp_keys = [
                        "confidence_from_interval_bounds_inf.cars_3",
                        "confidence_from_interval_bounds_sup.cars_3",
                    ]
                    cars_inf_key = cst.POINT_CLOUD_LAYER_INF_FROM_INTERVALS
                    cars_sup_key = cst.POINT_CLOUD_LAYER_SUP_FROM_INTERVALS
                    cars_perf_key = (
                        cst.POINT_CLOUD_PERFORMANCE_MAP_FROM_INTERVALS
                    )
                    use_ambiguity = False

                ambiguity_map = None
                perf_ambiguity_threshold = None
                if use_ambiguity:
                    ambiguity_map = disp_ref["confidence_from_ambiguity.cars_1"]
                    perf_ambiguity_threshold = performance_maps_parameters[
                        "perf_ambiguity_threshold"
                    ]
                # triangulate disp inf and supp
                points_inf = triangulation_tools.triangulate(
                    geometry_plugin,
                    sensor1,
                    sensor2,
                    geomodel1,
                    geomodel2,
                    grid1,
                    grid2,
                    disp_ref,
                    disp_key=disp_keys[0],
                )

                points_sup = triangulation_tools.triangulate(
                    geometry_plugin,
                    sensor1,
                    sensor2,
                    geomodel1,
                    geomodel2,
                    grid1,
                    grid2,
                    disp_ref,
                    disp_key=disp_keys[1],
                )

                points[cst.STEREO_REF][cars_inf_key] = points_inf[
                    cst.STEREO_REF
                ][cst.Z]
                points[cst.STEREO_REF][cars_sup_key] = points_sup[
                    cst.STEREO_REF
                ][cst.Z]
                # Generate performance map
                points[cst.STEREO_REF][cars_perf_key] = (
                    triangulation_tools.compute_performance_map(
                        points[cst.STEREO_REF][cst.Z],
                        points[cst.STEREO_REF][cars_inf_key],
                        points[cst.STEREO_REF][cars_sup_key],
                        ambiguity_map=ambiguity_map,
                        perf_ambiguity_threshold=perf_ambiguity_threshold,
                    )
                )

    elif isinstance(disp_ref, pandas.DataFrame):
        # Triangulate epipolar sparse matches
        points = {}
        points[cst.STEREO_REF] = triangulation_tools.triangulate_matches(
            geometry_plugin,
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            grid1,
            grid2,
            disp_ref.to_numpy(),
        )
    else:
        logging.error(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )
        raise TypeError(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )

    if geoid_path is not None:  # if user pass a geoid, use it as alt reference
        for key, point in points.items():
            points[key] = triangulation_tools.geoid_offset(point, geoid_path)

    # Fill datasets
    pc_dataset = points[cst.STEREO_REF]
    pc_dataset.attrs["cloud_id"] = cloud_id

    # Overload dataset with denoising fun
    if denoising_overload_fun is not None:
        if isinstance(pc_dataset, xr.Dataset):
            denoising_overload_fun(
                pc_dataset,
                sensor1,
                sensor2,
                geomodel1,
                geomodel2,
                grid1,
                grid2,
                geometry_plugin,
                disp_ref,
            )
        else:
            raise RuntimeError("wrong pc type for denoising func")

    attributes = {
        cst.CROPPED_DISPARITY_RANGE: (
            ocht.get_disparity_range_cropped(disparity_object)
        )
    }
    cars_dataset.fill_dataset(
        pc_dataset,
        saving_info=saving_info_epipolar,
        window=cars_dataset.get_window_dataset(disparity_object),
        profile=cars_dataset.get_profile_rasterio(disparity_object),
        attributes=attributes,
        overlaps=cars_dataset.get_overlaps_dataset(disparity_object),
    )

    # Flatten point cloud to save it as LAZ
    flatten_pc_dataset = None
    if point_cloud_csv_file_name or point_cloud_laz_file_name:
        # Convert epipolar array into point cloud
        flatten_pc_dataset, cloud_epsg = (
            point_cloud_tools.create_combined_cloud([pc_dataset], [0], epsg)
        )
        # Convert to UTM
        if epsg is not None and cloud_epsg != epsg:
            projection.point_cloud_conversion_dataframe(
                flatten_pc_dataset, cloud_epsg, epsg
            )
            cloud_epsg = epsg

        # Fill attributes for LAZ saving
        color_type = point_cloud_tools.get_color_type([pc_dataset])
        attributes = {
            "epsg": cloud_epsg,
            "color_type": color_type,
        }
        cars_dataset.fill_dataframe(
            flatten_pc_dataset,
            saving_info=saving_info_flatten,
            attributes=attributes,
        )
    # Save point cloud in worker
    if point_cloud_csv_file_name:
        cars_dataset.run_save_points(
            flatten_pc_dataset,
            point_cloud_csv_file_name,
            overwrite=True,
            point_cloud_format="csv",
            overwrite_file_name=False,
        )
    if point_cloud_laz_file_name:
        cars_dataset.run_save_points(
            flatten_pc_dataset,
            point_cloud_laz_file_name,
            overwrite=True,
            point_cloud_format="laz",
            overwrite_file_name=False,
        )

    return pc_dataset, flatten_pc_dataset


def triangulation_wrapper_matches(
    matches,
    sensor1,
    sensor2,
    grid1,
    grid2,
    interpolated_grid1,
    interpolated_grid2,
    geometry_plugin,
    full_saving_info_matches,
):
    """
    :param matches: matches
    :type matches: cars_dataset
    :param sensor1: path to left sensor image
    :type sensor1: str
    :param sensor2: path to right sensor image
    :type sensor2: str
    :param grid1: dataset of the reference image grid file
    :type grid1: CarsDataset
    :param grid2: dataset of the secondary image grid file
    :type grid2: CarsDataset
    :param interpolated_grid1: rectification grid left
    :type interpolated_grid1: shareloc.rectificationGrid
    :param interpolated_grid2: rectification grid right
    :type interpolated_grid2: shareloc.rectificationGrid
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    """

    # Ignore this tile if there are not match
    if matches is None or matches.empty:
        return None

    epipolar_matches = matches.to_numpy()

    sensor_matches = geometry_plugin.matches_to_sensor_coords(
        interpolated_grid1,
        interpolated_grid2,
        epipolar_matches,
        cst.MATCHES_MODE,
    )
    sensor_matches = np.concatenate(sensor_matches, axis=1)
    matches = np.concatenate(
        [
            epipolar_matches,
            sensor_matches,
        ],
        axis=1,
    )

    # Triangulate matches
    triangulated_matches = (
        triangulation_sparse_tools.triangulate_sparse_matches(
            sensor1,
            sensor2,
            grid1,
            grid2,
            interpolated_grid1,
            interpolated_grid2,
            matches,
            geometry_plugin,
        )
    )

    cars_dataset.fill_dataframe(
        triangulated_matches,
        saving_info=full_saving_info_matches,
        attributes={"epsg": triangulated_matches.attrs["epsg"]},
    )

    return triangulated_matches
