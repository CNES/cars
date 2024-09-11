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

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.applications.point_cloud_fusion import pc_tif_tools
from cars.core import preprocessing, roi_tools
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import output_constants, output_parameters
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.point_clouds_to_dsm import pc_inputs


@Pipeline.register(
    "dense_point_clouds_to_dense_dsm_no_merging",
    "dense_point_clouds_to_dense_dsm",
)
class PointCloudsToDsmPipeline(PipelineTemplate):
    """
    PointCloudsToDsmPipeline
    """

    # pylint: disable=too-many-instance-attributes

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
        self.used_conf[PIPELINE] = conf.get(
            PIPELINE, "dense_point_clouds_to_dense_dsm"
        )

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

        # Check advanced parameters
        # TODO static method in the base class
        self.advanced = advanced_parameters.check_advanced_parameters(
            self.conf.get(ADVANCED, {}), check_epipolar_a_priori=True
        )
        self.used_conf[ADVANCED] = self.advanced

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUTS][sens_cst.ROI]
        )

        # Check conf output
        self.output = self.check_output(self.conf[OUTPUT])
        self.used_conf[OUTPUT] = self.output

        self.save_output_dsm = (
            "dsm" in self.output[output_constants.PRODUCT_LEVEL]
        )

        self.save_output_point_cloud = (
            "point_cloud" in self.output[output_constants.PRODUCT_LEVEL]
        )

        # Check conf application
        application_conf = self.check_applications(
            self.conf.get(APPLICATIONS, {}),
            no_merging="no_merging" in self.used_conf[PIPELINE],
            save_all_intermediate_data=self.used_conf[ADVANCED][
                adv_cst.SAVE_INTERMEDIATE_DATA
            ],
            save_all_point_clouds_by_pair=self.used_conf[OUTPUT].get(
                output_constants.SAVE_BY_PAIR, False
            ),
        )
        self.used_conf[APPLICATIONS] = application_conf

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

        overloaded_conf = pc_inputs.check_point_clouds_inputs(
            conf, config_json_dir=config_json_dir
        )

        return overloaded_conf

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        return output_parameters.check_output_parameters(conf)

    def check_applications(
        self,
        conf,
        no_merging=False,
        save_all_intermediate_data=False,
        save_all_point_clouds_by_pair=False,
    ):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        :param no_merging: True if skip PC fusion and PC removing
        :type no_merging: bool
        :param save_all_intermediate_data: True to save intermediate data in all
            applications
        :type save_all_intermediate_data: bool
        :param save_all_point_clouds_by_pair: save point clouds by pair in all
            relevant applications
        :type save_all_point_clouds_by_pair: bool
        """

        # Check if all specified applications are used
        needed_applications = [
            "point_cloud_rasterization",
        ]

        if not no_merging:
            needed_applications.append("point_cloud_fusion")
            needed_applications.append("point_cloud_outliers_removing.1")
            needed_applications.append("point_cloud_outliers_removing.2")

        for app_key in conf.keys():
            if app_key not in needed_applications:
                logging.error(
                    "No {} application used in pipeline".format(app_key)
                )
                raise NameError(
                    "No {} application used in pipeline".format(app_key)
                )

        # Initialize used config
        used_conf = {}
        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            used_conf[app_key]["save_intermediate_data"] = used_conf[
                app_key
            ].get("save_intermediate_data", save_all_intermediate_data)

        for app_key in [
            "point_cloud_fusion",
            "point_cloud_outliers_removing.1",
            "point_cloud_outliers_removing.2",
        ]:
            if app_key in needed_applications:
                used_conf[app_key]["save_by_pair"] = used_conf[app_key].get(
                    "save_by_pair", save_all_point_clouds_by_pair
                )

        # Points cloud fusion
        self.pc_fusion_application = Application(
            "point_cloud_fusion", cfg=used_conf.get("point_cloud_fusion", {})
        )
        used_conf["point_cloud_fusion"] = self.pc_fusion_application.get_conf()

        # Points cloud outlier removing small components
        self.pc_outliers_removing_1_app = Application(
            "point_cloud_outliers_removing",
            cfg=used_conf.get(
                "point_cloud_outliers_removing.1",
                {"method": "small_components"},
            ),
        )
        used_conf["point_cloud_outliers_removing.1"] = (
            self.pc_outliers_removing_1_app.get_conf()
        )

        # Points cloud outlier removing statistical
        self.pc_outliers_removing_2_app = Application(
            "point_cloud_outliers_removing",
            cfg=used_conf.get(
                "point_cloud_outliers_removing.2",
                {"method": "statistical"},
            ),
        )
        used_conf["point_cloud_outliers_removing.2"] = (
            self.pc_outliers_removing_2_app.get_conf()
        )

        # Rasterization
        self.rasterization_application = Application(
            "point_cloud_rasterization",
            cfg=used_conf.get("point_cloud_rasterization", {}),
        )
        used_conf["point_cloud_rasterization"] = (
            self.rasterization_application.get_conf()
        )

        return used_conf

    @cars_profile(name="run_pc_pipeline", interval=0.5)
    def run(self):
        """
        Run pipeline

        """

        out_dir = self.output[output_constants.OUT_DIRECTORY]

        # Save used conf
        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(out_dir, "used_conf.json"),
            safe_save=True,
        )

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.orchestrator_conf,
            out_dir=out_dir,
            out_json_path=os.path.join(
                out_dir,
                self.output[output_constants.INFO_BASENAME],
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
            epsg = self.output[output_constants.EPSG]
            # compute epsg
            epsg_cloud = pc_tif_tools.compute_epsg_from_point_cloud(
                self.inputs["point_clouds"]
            )
            if epsg is None:
                epsg = epsg_cloud

            resolution = self.output[output_constants.RESOLUTION]

            # Compute roi polygon, in input EPSG
            roi_poly = preprocessing.compute_roi_poly(
                self.input_roi_poly, self.input_roi_epsg, epsg
            )

            if "no_merging" in self.used_conf[PIPELINE]:
                # compute bounds
                terrain_bounds = pc_tif_tools.get_bounds(
                    self.inputs["point_clouds"],
                    epsg,
                    roi_poly=roi_poly,
                )

                list_epipolar_points_cloud = pc_tif_tools.generate_point_clouds(
                    self.inputs["point_clouds"],
                    cars_orchestrator,
                    tile_size=1000,
                )
                # Generate cars datasets
                point_cloud_to_rasterize = (
                    list_epipolar_points_cloud,
                    terrain_bounds,
                )

                color_type = point_cloud_to_rasterize[0][0].attributes.get(
                    "color_type", None
                )

            else:
                # Compute terrain bounds and transform point clouds
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
                optimal_terrain_tile_width = min(
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
                )
                # epsg_cloud and optimal_terrain_tile_width have the same epsg
                optimal_terrain_tile_width = (
                    preprocessing.convert_optimal_tile_size_with_epsg(
                        terrain_bounds,
                        optimal_terrain_tile_width,
                        epsg,
                        epsg_cloud,
                    )
                )

                # find which application produce the final version of the
                # point cloud. The last generated point cloud will be saved
                # as official point cloud product if save_output_point_cloud
                # is True.

                last_pc_application = None
                if (
                    self.pc_outliers_removing_2_app.used_config.get(
                        "activated", False
                    )
                    is True
                ):
                    last_pc_application = "pc_outliers_removing_2"
                elif (
                    self.pc_outliers_removing_1_app.used_config.get(
                        "activated", False
                    )
                    is True
                ):
                    last_pc_application = "pc_outliers_removing_1"
                else:
                    last_pc_application = "fusion"

                # Merge point clouds
                merged_points_clouds = self.pc_fusion_application.run(
                    list_epipolar_points_cloud_by_tiles,
                    terrain_bounds,
                    epsg,
                    orchestrator=cars_orchestrator,
                    margins=(
                        self.pc_outliers_removing_1_app.get_on_ground_margin(
                            resolution=resolution
                        )
                        + self.pc_outliers_removing_2_app.get_on_ground_margin(
                            resolution=resolution
                        )
                        + self.rasterization_application.get_margins(resolution)
                    ),
                    optimal_terrain_tile_width=optimal_terrain_tile_width,
                    save_laz_output=self.save_output_point_cloud
                    and last_pc_application == "fusion",
                )

                # Add file names to retrieve source file of each point
                pc_file_names = list(self.inputs["point_clouds"])
                merged_points_clouds.attributes["source_pc_names"] = (
                    pc_file_names
                )

                # Remove outliers with small components method
                filtered_1_merged_points_clouds = (
                    self.pc_outliers_removing_1_app.run(
                        merged_points_clouds,
                        orchestrator=cars_orchestrator,
                        save_laz_output=self.save_output_point_cloud
                        and last_pc_application == "pc_outliers_removing_1",
                    )
                )

                # Remove outliers with statistical components method
                filtered_2_merged_points_clouds = (
                    self.pc_outliers_removing_2_app.run(
                        filtered_1_merged_points_clouds,
                        orchestrator=cars_orchestrator,
                        save_laz_output=self.save_output_point_cloud
                        and last_pc_application == "pc_outliers_removing_2",
                    )
                )

                point_cloud_to_rasterize = filtered_2_merged_points_clouds

                color_type = point_cloud_to_rasterize.attributes.get(
                    "color_type", None
                )

            rasterization_dump_dir = os.path.join(
                cars_orchestrator.out_dir, "dump_dir", "rasterization"
            )

            dsm_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.DSM_BASENAME],
                )
                if self.save_output_dsm
                else None
            )

            color_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.COLOR_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_COLOR
                ]
                else None
            )

            performance_map_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.PERFORMANCE_MAP_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_PERFORMANCE_MAP
                ]
                else None
            )

            classif_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.CLASSIFICATION_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_CLASSIFICATION
                ]
                else None
            )

            mask_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.MASK_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_MASK
                ]
                else None
            )

            contributing_pair_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.CONTRIBUTING_PAIR_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_CONTRIBUTING_PAIR
                ]
                else None
            )

            filling_file_name = (
                os.path.join(
                    out_dir,
                    output_constants.DSM_DIRECTORY,
                    self.output[output_constants.FILLING_BASENAME],
                )
                if self.save_output_dsm
                and self.output[output_constants.AUXILIARY][
                    output_constants.AUX_FILLING
                ]
                else None
            )

            # rasterize point cloud
            _ = self.rasterization_application.run(
                point_cloud_to_rasterize,
                epsg,
                resolution=resolution,
                orchestrator=cars_orchestrator,
                dsm_file_name=dsm_file_name,
                color_file_name=color_file_name,
                classif_file_name=classif_file_name,
                performance_map_file_name=performance_map_file_name,
                mask_file_name=mask_file_name,
                contributing_pair_file_name=contributing_pair_file_name,
                filling_file_name=filling_file_name,
                color_dtype=color_type,
                dump_dir=rasterization_dump_dir,
            )
