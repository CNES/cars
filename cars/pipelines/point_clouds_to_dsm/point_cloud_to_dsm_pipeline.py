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
CARS point cloud to full resolution dsm pipeline class file
"""

# Standard imports
from __future__ import print_function

import json
import logging
import os

from json_checker import Checker, Or

import cars.pipelines.point_clouds_to_dsm.pc_constants as pc_cst

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.applications.point_cloud_fusion import pc_tif_tools
from cars.conf import log_conf
from cars.core import constants as cst
from cars.core import inputs, preprocessing
from cars.core.utils import make_relative_path_absolute
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    APPLICATIONS,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.sensor_to_full_resolution_dsm import dsm_output
from cars.pipelines.sensor_to_full_resolution_dsm import (
    sensor_full_res_dsm_constants as sens_cst,
)
from cars.pipelines.sensor_to_full_resolution_dsm import sensors_inputs


@Pipeline.register("dense_point_clouds_to_dense_dsm")
class PointCloudsToDsmPipeline(PipelineTemplate):
    """
    PointCloudsToDsmPipeline
    """

    def __init__(self, conf, config_json_dir=None):
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        # Merge parameters from associated json
        # priority : cars_pipeline.json << user_inputs.json
        # Get root package directory
        package_path = os.path.dirname(__file__)
        json_file = os.path.join(
            package_path,
            "..",
            "conf_pipeline",
            "point_clouds_to_dsm.json",
        )

        with open(json_file, "r", encoding="utf8") as fstream:
            pipeline_config = json.load(fstream)

        self.conf = self.merge_pipeline_conf(pipeline_config, conf)

        # check global conf
        self.check_global_schema(self.conf)

        # Used conf
        self.used_conf = {}

        # Pipeline
        self.used_conf[PIPELINE] = "dense_point_clouds_to_dense_dsm"

        # Check conf orchestrator
        self.orchestrator_conf = self.check_orchestrator(
            self.conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[ORCHESTRATOR] = self.orchestrator_conf

        # Check conf inputs
        self.inputs = self.check_inputs(
            self.conf[INPUTS], config_json_dir=config_json_dir
        )
        self.used_conf[INPUTS] = self.inputs

        # Check conf output
        self.output = self.check_output(self.conf[OUTPUT])
        self.used_conf[OUTPUT] = self.output

        # Check conf application
        application_conf = self.check_applications(
            self.conf.get(APPLICATIONS, {})
        )
        self.used_conf[APPLICATIONS] = application_conf

        # Save used conf
        out_dir = self.output["out_dir"]
        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(out_dir, "used_conf.json"),
            safe_save=True,
        )

    def check_inputs(self, conf, config_json_dir=None):
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
        overloaded_conf[sens_cst.EPSG] = conf.get(sens_cst.EPSG, None)
        overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)
        overloaded_conf[pc_cst.POINT_CLOUDS] = {}

        # Validate inputs
        inputs_schema = {
            pc_cst.POINT_CLOUDS: dict,
            sens_cst.EPSG: Or(int, None),
            sens_cst.ROI: Or(str, list, tuple, None),
        }

        checker_inputs = Checker(inputs_schema)
        checker_inputs.validate(overloaded_conf)

        # Validate point clouds

        pc_schema = {
            cst.X: str,
            cst.Y: str,
            cst.Z: str,
            cst.POINTS_CLOUD_CLR_KEY_ROOT: Or(str, None),
            cst.POINTS_CLOUD_VALID_DATA: Or(str, None),
            cst.PC_EPSG: Or(str, int, None),
        }
        checker_pc = Checker(pc_schema)

        for point_cloud_key in conf[pc_cst.POINT_CLOUDS]:
            # Get point clouds with default
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key] = {}
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.X] = conf[
                pc_cst.POINT_CLOUDS
            ][point_cloud_key].get("x", None)
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Y] = conf[
                pc_cst.POINT_CLOUDS
            ][point_cloud_key].get("y", None)
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Z] = conf[
                pc_cst.POINT_CLOUDS
            ][point_cloud_key].get("z", None)
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_CLR_KEY_ROOT
            ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("color", None)
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.POINTS_CLOUD_VALID_DATA
            ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("mask", None)
            overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                cst.PC_EPSG
            ] = conf[pc_cst.POINT_CLOUDS][point_cloud_key].get("epsg", 4326)

            # validate
            checker_pc.validate(
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key]
            )

            # check sizes
            check_input_size(
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.X],
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Y],
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][cst.Z],
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                    cst.POINTS_CLOUD_VALID_DATA
                ],
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key][
                    cst.POINTS_CLOUD_CLR_KEY_ROOT
                ],
            )

        # Modify to absolute path
        if config_json_dir is not None:
            modify_to_absolute_path(config_json_dir, overloaded_conf)

        else:
            logging.debug(
                "path of config file was not given,"
                "relative path are not transformed to absolute paths"
            )

        # Check roi
        sensors_inputs.check_roi(overloaded_conf[sens_cst.ROI])

        return overloaded_conf

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        return dsm_output.full_res_dsm_check_output(conf)

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

        # Check if all specified applications are used
        needed_applications = [
            "point_cloud_fusion",
            "point_cloud_rasterization",
            "point_cloud_outliers_removing.1",
            "point_cloud_outliers_removing.2",
        ]

        # Initialize used config
        used_conf = {}

        for app_key in conf.keys():
            if app_key not in needed_applications:
                logging.error(
                    "No {} application used in pipeline".format(app_key)
                )
                raise NameError(
                    "No {} application used in pipeline".format(app_key)
                )

        # Points cloud fusion
        self.pc_fusion_application = Application(
            "point_cloud_fusion", cfg=conf.get("point_cloud_fusion", {})
        )
        used_conf["point_cloud_fusion"] = self.pc_fusion_application.get_conf()

        # Points cloud outlier removing small components
        self.pc_outliers_removing_1_app = Application(
            "point_cloud_outliers_removing",
            cfg=conf.get(
                "point_cloud_outliers_removing.1",
                {"method": "small_components"},
            ),
        )
        used_conf[
            "point_cloud_outliers_removing.1"
        ] = self.pc_outliers_removing_1_app.get_conf()

        # Points cloud outlier removing statistical
        self.pc_outliers_removing_2_app = Application(
            "point_cloud_outliers_removing",
            cfg=conf.get(
                "point_cloud_outliers_removing.2",
                {"method": "statistical"},
            ),
        )
        used_conf[
            "point_cloud_outliers_removing.2"
        ] = self.pc_outliers_removing_2_app.get_conf()

        # Rasterization
        self.rasterization_application = Application(
            "point_cloud_rasterization",
            cfg=conf.get("point_cloud_rasterization", {}),
        )
        used_conf[
            "point_cloud_rasterization"
        ] = self.rasterization_application.get_conf()

        return used_conf

    def run(self):
        """
        Run pipeline

        """

        out_dir = self.output["out_dir"]

        log_conf.add_log_file(out_dir, "merging")

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.orchestrator_conf,
            out_dir=out_dir,
            out_json_path=os.path.join(
                out_dir, self.output[sens_cst.INFO_BASENAME]
            ),
        ) as cars_orchestrator:
            # initialize out_json
            cars_orchestrator.update_out_info(
                {
                    "version": __version__,
                    "pipeline": "point_clouds_to_dsm",
                    "inputs": self.inputs,
                }
            )

            # Run applications

            # get epsg
            epsg = self.inputs[sens_cst.EPSG]
            if epsg is None:
                # compute epsg
                epsg = pc_tif_tools.compute_epsg_from_point_cloud(
                    self.inputs["point_clouds"]
                )
            # Compute roi polygon, in input EPSG
            roi_poly = preprocessing.compute_roi_poly(
                self.inputs[sens_cst.ROI], epsg
            )

            # compute terrain bounds and transform point clouds
            (
                terrain_bounds,
                list_epipolar_points_cloud_by_tiles,
            ) = pc_tif_tools.transform_input_pc(
                self.inputs["point_clouds"],
                epsg,
                roi_poly=roi_poly,
                epipolar_tile_size=1000,  # TODO change it
                orchestrator=cars_orchestrator,
            )

            # Compute number of superposing point cloud for density
            max_number_superposing_point_clouds = (
                pc_tif_tools.compute_max_nb_point_clouds(
                    list_epipolar_points_cloud_by_tiles
                )
            )

            # Compute average distance between two points
            average_distance_point_cloud = (
                pc_tif_tools.compute_average_distance(
                    list_epipolar_points_cloud_by_tiles
                )
            )

            # Merge point clouds
            merged_points_clouds = self.pc_fusion_application.run(
                list_epipolar_points_cloud_by_tiles,
                None,
                terrain_bounds,
                epsg,
                orchestrator=cars_orchestrator,
                margins=self.pc_outliers_removing_1_app.get_on_ground_margin(
                    resolution=(self.rasterization_application.get_resolution())
                )
                + self.pc_outliers_removing_2_app.get_on_ground_margin(
                    resolution=(self.rasterization_application.get_resolution())
                )
                + self.rasterization_application.get_margins(),
                optimal_terrain_tile_width=min(
                    self.pc_outliers_removing_1_app.get_optimal_tile_size(
                        cars_orchestrator.cluster.checked_conf_cluster[
                            "max_ram_per_worker"
                        ],
                        superposing_point_clouds=(
                            max_number_superposing_point_clouds
                        ),
                        point_cloud_resolution=average_distance_point_cloud,
                    ),
                    self.pc_outliers_removing_2_app.get_optimal_tile_size(
                        cars_orchestrator.cluster.checked_conf_cluster[
                            "max_ram_per_worker"
                        ],
                        superposing_point_clouds=(
                            max_number_superposing_point_clouds
                        ),
                        point_cloud_resolution=average_distance_point_cloud,
                    ),
                    self.rasterization_application.get_optimal_tile_size(
                        cars_orchestrator.cluster.checked_conf_cluster[
                            "max_ram_per_worker"
                        ],
                        superposing_point_clouds=(
                            max_number_superposing_point_clouds
                        ),
                        point_cloud_resolution=average_distance_point_cloud,
                    ),
                ),
            )

            # Remove outliers with small components method
            filtered_1_merged_points_clouds = (
                self.pc_outliers_removing_1_app.run(
                    merged_points_clouds,
                    orchestrator=cars_orchestrator,
                )
            )

            # Remove outliers with statistical components method
            filtered_2_merged_points_clouds = (
                self.pc_outliers_removing_2_app.run(
                    filtered_1_merged_points_clouds,
                    orchestrator=cars_orchestrator,
                )
            )

            # rasterize point cloud
            _ = self.rasterization_application.run(
                filtered_2_merged_points_clouds,
                epsg,
                orchestrator=cars_orchestrator,
                dsm_file_name=os.path.join(
                    out_dir, self.output[sens_cst.DSM_BASENAME]
                ),
                color_file_name=os.path.join(
                    out_dir, self.output[sens_cst.CLR_BASENAME]
                ),
            )


def check_input_size(x_path, y_path, z_path, mask, color):
    """
    Check x, y, z, mask and color given

    Images must have same size

    :param x_path: x path
    :type x_path: str
    :param y_path: y path
    :type y_path: str
    :param z_path: z path
    :type z_path: str
    :param mask: mask path
    :type mask: str
    :param color: color path
    :type color: str
    """

    for path in [x_path, y_path, z_path]:
        if inputs.rasterio_get_nb_bands(path) != 1:
            raise RuntimeError("{} is not mono-band image".format(path))

    for path in [mask, color]:
        if path is not None:
            if inputs.rasterio_get_size(x_path) != inputs.rasterio_get_size(
                path
            ):
                raise RuntimeError(
                    "The image {} and {} "
                    "do not have the same size".format(x_path, path)
                )


def modify_to_absolute_path(config_json_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_json_dir: directory of the json configuration
    :type config_json_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for point_cloud_key in overloaded_conf[pc_cst.POINT_CLOUDS]:
        point_cloud = overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key]
        for tag in [
            cst.X,
            cst.Y,
            cst.Z,
            cst.POINTS_CLOUD_CLR_KEY_ROOT,
            cst.POINTS_CLOUD_VALID_DATA,
        ]:
            if point_cloud[tag] is not None:
                point_cloud[tag] = make_relative_path_absolute(
                    point_cloud[tag], config_json_dir
                )

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_json_dir
            )
