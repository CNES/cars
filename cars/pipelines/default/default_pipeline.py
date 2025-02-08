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
# attribute-defined-outside-init is disabled so that we can create and use
# attributes however we need, to stick to the "everything is attribute" logic
# introduced in issue#895
# pylint: disable=attribute-defined-outside-init
"""
CARS default pipeline class file
"""
# Standard imports
from __future__ import print_function

import copy
import logging
import os

import numpy as np

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation import dem_generation_tools
from cars.applications.grid_generation import grid_correction
from cars.applications.point_cloud_fusion import pc_tif_tools
from cars.applications.sparse_matching import (
    sparse_matching_tools as sparse_mtch_tools,
)
from cars.core import constants_disparity as cst_disp
from cars.core import preprocessing, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import get_descriptions_bands, rasterio_get_epsg
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import depth_map_inputs
from cars.pipelines.parameters import depth_map_inputs_constants as depth_cst
from cars.pipelines.parameters import dsm_inputs
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    GEOMETRY_PLUGIN,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate


@Pipeline.register(
    "default",
)
class DefaultPipeline(PipelineTemplate):
    """
    DefaultPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf, config_json_dir=None):
        """
        Creates pipeline

        Directly creates class attributes:
            used_conf
            generate_terrain_products
            debug_with_roi
            save_output_dsm
            save_output_depth_map
            save_output_point_clouds
            geom_plugin_without_dem_and_geoid
            geom_plugin_with_dem_and_geoid
            dem_generation_roi

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        # Used conf
        self.used_conf = {PIPELINE: "default"}

        # check global conf
        self.check_global_schema(conf)

        # Check conf orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )

        # Check conf inputs
        inputs = self.check_inputs(
            conf[INPUTS], config_json_dir=config_json_dir
        )
        self.used_conf[INPUTS] = inputs

        # Check advanced parameters
        # TODO static method in the base class
        advanced = advanced_parameters.check_advanced_parameters(
            conf.get(ADVANCED, {}), check_epipolar_a_priori=True
        )
        self.used_conf[ADVANCED] = advanced

        if self.used_conf[INPUTS][sens_cst.SENSORS] is not None:
            # Check geometry plugin and overwrite geomodel in conf inputs
            (
                inputs,
                self.used_conf[GEOMETRY_PLUGIN],
                self.geom_plugin_without_dem_and_geoid,
                self.geom_plugin_with_dem_and_geoid,
                self.dem_generation_roi,
            ) = sensor_inputs.check_geometry_plugin(
                inputs, advanced, conf.get(GEOMETRY_PLUGIN, None)
            )
            self.used_conf[INPUTS] = inputs
        elif (
            depth_cst.DEPTH_MAPS in self.used_conf[INPUTS]
            or dsm_cst.DSMS in self.used_conf[INPUTS]
        ):

            # if there's an initial elevation with
            # point clouds as inputs, generate a plugin (used in dsm_filling)

            self.geom_plugin_with_dem_and_geoid = (
                depth_map_inputs.check_geometry_plugin(
                    inputs, conf.get(GEOMETRY_PLUGIN, None)
                )
            )

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUTS][sens_cst.ROI]
        )

        self.debug_with_roi = self.used_conf[ADVANCED][adv_cst.DEBUG_WITH_ROI]

        # Check conf output
        output = self.check_output(conf[OUTPUT])
        self.used_conf[OUTPUT] = output

        prod_level = output[out_cst.PRODUCT_LEVEL]

        self.save_output_dsm = "dsm" in prod_level
        self.save_output_depth_map = "depth_map" in prod_level
        self.save_output_point_cloud = "point_cloud" in prod_level

        self.output_level_none = not (
            self.save_output_dsm
            or self.save_output_depth_map
            or self.save_output_point_cloud
        )
        self.sensors_in_inputs = sens_cst.SENSORS in self.used_conf[INPUTS]
        self.depth_maps_in_inputs = (
            depth_cst.DEPTH_MAPS in self.used_conf[INPUTS]
        )
        self.dsms_in_inputs = dsm_cst.DSMS in self.used_conf[INPUTS]
        self.merging = self.used_conf[ADVANCED][adv_cst.MERGING]

        self.phasing = self.used_conf[ADVANCED][adv_cst.PHASING]

        self.compute_depth_map = (
            self.sensors_in_inputs
            and (not self.output_level_none)
            and not self.dsms_in_inputs
            and not self.depth_maps_in_inputs
        )

        if self.output_level_none:
            self.infer_conditions_from_applications(conf)

        self.save_all_intermediate_data = self.used_conf[ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]
        self.save_all_point_clouds_by_pair = self.used_conf[OUTPUT].get(
            out_cst.SAVE_BY_PAIR, False
        )

        # Check conf application
        application_conf = self.check_applications(conf.get(APPLICATIONS, {}))

        if (
            self.sensors_in_inputs
            and not self.depth_maps_in_inputs
            and not self.dsms_in_inputs
        ):
            # Check conf application vs inputs application
            application_conf = self.check_applications_with_inputs(
                self.used_conf[INPUTS], application_conf
            )

        self.used_conf[APPLICATIONS] = application_conf

        self.config_full_res = copy.deepcopy(self.used_conf)
        self.config_full_res.__delitem__("applications")
        self.config_full_res[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI] = {}
        self.config_full_res[ADVANCED][adv_cst.TERRAIN_A_PRIORI] = {}
        self.config_full_res[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI] = True

    def quit_on_app(self, app_name):
        """
        Returns whether the pipeline should end after
        the application was called.

        Only works if the output_level is empty, so that
        the control is instead given to
        """

        if not self.output_level_none:
            # custom quit step was not set, never quit early
            return False

        return self.app_values[app_name] >= self.last_application_to_run

    def infer_conditions_from_applications(self, conf):
        """
        Fills the condition booleans used later in the pipeline by going
        through the applications and infering which application we should
        end the pipeline on.
        """

        self.last_application_to_run = 0

        sensor_to_depth_apps = {
            "grid_generation": 1,  # and 5
            "resampling": 2,  # and 8
            "hole_detection": 3,
            "sparse_matching.sift": 4,
            "sparse_matching.pandora": 6,
            "dem_generation": 7,
            "ground_truth_reprojection": 9,
            "dense_matching": 10,
            "dense_match_filling.1": 11,
            "dense_match_filling.2": 12,
            "triangulation": 13,
            "point_cloud_outlier_removal.1": 14,
            "point_cloud_outlier_removal.2": 15,
        }

        depth_merge_apps = {
            "point_cloud_fusion": 16,
        }

        depth_to_dsm_apps = {
            "pc_denoising": 17,
            "point_cloud_rasterization": 18,
            "dsm_filling": 19,
            "auxiliary_filling": 20,
        }

        self.app_values = {}
        self.app_values.update(sensor_to_depth_apps)
        self.app_values.update(depth_merge_apps)
        self.app_values.update(depth_to_dsm_apps)

        app_conf = conf.get(APPLICATIONS, {})
        for key in app_conf:

            if adv_cst.SAVE_INTERMEDIATE_DATA not in app_conf[key]:
                continue

            if not app_conf[key][adv_cst.SAVE_INTERMEDIATE_DATA]:
                continue

            if key in sensor_to_depth_apps:

                if not self.sensors_in_inputs:
                    warn_msg = (
                        "The application {} can only be used when sensor "
                        "images are given as an input. "
                        "Its configuration will be ignored."
                    ).format(key)
                    logging.warning(warn_msg)

                elif (
                    self.sensors_in_inputs
                    and not self.depth_maps_in_inputs
                    and not self.dsms_in_inputs
                ):
                    self.compute_depth_map = True
                    self.last_application_to_run = max(
                        self.last_application_to_run, self.app_values[key]
                    )

            elif key in depth_to_dsm_apps:

                if not (
                    self.sensors_in_inputs
                    or self.depth_maps_in_inputs
                    or self.dsms_in_inputs
                ):
                    warn_msg = (
                        "The application {} can only be used when sensor "
                        "images or depth maps are given as an input. "
                        "Its configuration will be ignored."
                    ).format(key)
                    logging.warning(warn_msg)

                else:
                    if (
                        self.sensors_in_inputs
                        and not self.depth_maps_in_inputs
                        and not self.dsms_in_inputs
                    ):
                        self.compute_depth_map = True

                    # enabled to start the depth map to dsm process
                    self.save_output_dsm = True

                    self.last_application_to_run = max(
                        self.last_application_to_run, self.app_values[key]
                    )

            elif key in depth_merge_apps:

                if not self.merging:
                    warn_msg = (
                        "The application {} can only be used when merging "
                        "is activated (this parameter is located in the "
                        "'advanced' config key). "
                        "The application's configuration will be ignored."
                    ).format(key)
                    logging.warning(warn_msg)

                elif not (
                    self.sensors_in_inputs
                    or self.depth_maps_in_inputs
                    or self.dsms_in_inputs
                ):
                    warn_msg = (
                        "The application {} can only be used when sensor "
                        "images or depth maps are given as an input. "
                        "Its configuration will be ignored."
                    ).format(key)
                    logging.warning(warn_msg)

                else:
                    if (
                        self.sensors_in_inputs
                        and not self.depth_maps_in_inputs
                        and not self.dsms_in_inputs
                    ):
                        self.compute_depth_map = True

                    # enabled to start the depth map to dsm process
                    self.save_output_point_cloud = True

                    self.last_application_to_run = max(
                        self.last_application_to_run, self.app_values[key]
                    )
            else:
                warn_msg = (
                    "The application {} was not recognized. Its configuration"
                    "will be ignored."
                ).format(key)
                logging.warning(warn_msg)

        if not (self.compute_depth_map or self.save_output_dsm):
            log_msg = (
                "No product level was given. CARS has not detected any "
                "data you wish to save. No computation will be done."
            )
            logging.info(log_msg)
        else:
            log_msg = (
                "No product level was given. CARS has detected that you "
                + "wish to run up to the {} application.".format(
                    next(
                        k
                        for k, v in self.app_values.items()
                        if v == self.last_application_to_run
                    )
                )
            )
            logging.warning(log_msg)

    @staticmethod
    def check_inputs(conf, config_json_dir=None):
        """
        Check the inputs given

        :param conf: configuration of inputs
        :type conf: dict
        :param config_json_dir: directory of used json, if
            user filled paths with relative paths
        :type config_json_dir: str

        :return: overloaded inputs
        :rtype: dict
        """

        output_config = {}
        if (
            sens_cst.SENSORS in conf
            and depth_cst.DEPTH_MAPS not in conf
            and dsm_cst.DSMS not in conf
        ):
            output_config = sensor_inputs.sensors_check_inputs(
                conf, config_json_dir=config_json_dir
            )
        elif depth_cst.DEPTH_MAPS in conf:
            output_config = {
                **output_config,
                **depth_map_inputs.check_depth_maps_inputs(
                    conf, config_json_dir=config_json_dir
                ),
            }
        else:
            output_config = {
                **output_config,
                **dsm_inputs.check_dsm_inputs(
                    conf, config_json_dir=config_json_dir
                ),
            }
        return output_config

    @staticmethod
    def check_output(conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        return output_parameters.check_output_parameters(conf)

    def check_applications(  # noqa: C901 : too complex
        self,
        conf,
    ):
        """
        Check the given configuration for applications,
        and generates needed applications for pipeline.

        :param conf: configuration of applications
        :type conf: dict
        """

        # Check if all specified applications are used
        # Application in terrain_application are note used in
        # the sensors_to_dense_depth_maps pipeline
        needed_applications = []

        if self.sensors_in_inputs:
            needed_applications += [
                "grid_generation",
                "resampling",
                "ground_truth_reprojection",
                "hole_detection",
                "dense_match_filling.1",
                "dense_match_filling.2",
                "sparse_matching.sift",
                "sparse_matching.pandora",
                "dense_matching",
                "triangulation",
                "dem_generation",
                "point_cloud_outlier_removal.1",
                "point_cloud_outlier_removal.2",
            ]

        if self.save_output_dsm or self.save_output_point_cloud:

            needed_applications += ["pc_denoising"]

            if self.save_output_dsm:
                needed_applications += [
                    "point_cloud_rasterization",
                    "dsm_filling",
                    "auxiliary_filling",
                ]

            if self.merging:  # we have to merge point clouds, add merging apps
                needed_applications += ["point_cloud_fusion"]

        for app_key in conf.keys():
            if app_key not in needed_applications:
                msg = (
                    f"No {app_key} application used in the "
                    + "default Cars pipeline"
                )
                logging.error(msg)
                raise NameError(msg)

        # Initialize used config
        used_conf = {}

        for app_key in [
            "sparse_matching.pandora",
            "point_cloud_outlier_removal.1",
            "point_cloud_outlier_removal.2",
            "auxiliary_filling",
        ]:
            if conf.get(app_key) is not None:
                config_app = conf.get(app_key)
                if "activated" not in config_app:
                    conf[app_key]["activated"] = True

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            used_conf[app_key]["save_intermediate_data"] = (
                self.save_all_intermediate_data
                or used_conf[app_key].get("save_intermediate_data", False)
            )

        for app_key in [
            "point_cloud_fusion",
            "pc_denoising",
        ]:
            if app_key in needed_applications:
                used_conf[app_key]["save_by_pair"] = used_conf[app_key].get(
                    "save_by_pair", self.save_all_point_clouds_by_pair
                )

        self.epipolar_grid_generation_application = None
        self.resampling_application = None
        self.ground_truth_reprojection = None
        self.hole_detection_app = None
        self.dense_match_filling_1 = None
        self.dense_match_filling_2 = None
        self.sparse_mtch_sift_app = None
        self.sparse_mtch_pandora_app = None
        self.dense_matching_app = None
        self.triangulation_application = None
        self.dem_generation_application = None
        self.pc_denoising_application = None
        self.pc_outlier_removal_1_app = None
        self.pc_outlier_removal_2_app = None
        self.rasterization_application = None
        self.pc_fusion_application = None
        self.dsm_filling_application = None

        if self.sensors_in_inputs:
            # Epipolar grid generation
            self.epipolar_grid_generation_application = Application(
                "grid_generation", cfg=used_conf.get("grid_generation", {})
            )
            used_conf["grid_generation"] = (
                self.epipolar_grid_generation_application.get_conf()
            )

            # image resampling
            self.resampling_application = Application(
                "resampling", cfg=used_conf.get("resampling", {})
            )
            used_conf["resampling"] = self.resampling_application.get_conf()

            # ground truth disparity map computation
            if self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
                used_conf["ground_truth_reprojection"][
                    "save_intermediate_data"
                ] = True

                if isinstance(
                    self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM], str
                ):
                    self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM] = {
                        "dsm": self.used_conf[ADVANCED][
                            adv_cst.GROUND_TRUTH_DSM
                        ]
                    }

                self.ground_truth_reprojection = Application(
                    "ground_truth_reprojection",
                    cfg=used_conf.get("ground_truth_reprojection", {}),
                )
            # holes detection
            self.hole_detection_app = Application(
                "hole_detection", cfg=used_conf.get("hole_detection", {})
            )
            used_conf["hole_detection"] = self.hole_detection_app.get_conf()

            # disparity filling 1 plane
            self.dense_match_filling_1 = Application(
                "dense_match_filling",
                cfg=used_conf.get(
                    "dense_match_filling.1",
                    {"method": "plane"},
                ),
            )
            used_conf["dense_match_filling.1"] = (
                self.dense_match_filling_1.get_conf()
            )

            # disparity filling 2
            self.dense_match_filling_2 = Application(
                "dense_match_filling",
                cfg=used_conf.get(
                    "dense_match_filling.2",
                    {"method": "zero_padding"},
                ),
            )
            used_conf["dense_match_filling.2"] = (
                self.dense_match_filling_2.get_conf()
            )

            # Sparse Matching
            self.sparse_mtch_sift_app = Application(
                "sparse_matching",
                cfg=used_conf.get("sparse_matching.sift", {"method": "sift"}),
            )
            used_conf["sparse_matching.sift"] = (
                self.sparse_mtch_sift_app.get_conf()
            )

            # Pandora Sparse Matching
            used_conf["sparse_matching.pandora"]["method"] = "pandora"
            self.sparse_mtch_pandora_app = Application(
                "sparse_matching",
                cfg=used_conf.get("sparse_matching.pandora"),
            )
            used_conf["sparse_matching.pandora"] = (
                self.sparse_mtch_pandora_app.get_conf()
            )

            # Matching
            generate_performance_map = (
                self.used_conf[OUTPUT]
                .get(out_cst.AUXILIARY, {})
                .get(out_cst.AUX_PERFORMANCE_MAP, False)
            )
            dense_matching_config = used_conf.get("dense_matching", {})
            if generate_performance_map is True:
                dense_matching_config["generate_performance_map"] = True
            self.dense_matching_app = Application(
                "dense_matching", cfg=dense_matching_config
            )
            used_conf["dense_matching"] = self.dense_matching_app.get_conf()

            # Triangulation
            self.triangulation_application = Application(
                "triangulation", cfg=used_conf.get("triangulation", {})
            )
            used_conf["triangulation"] = (
                self.triangulation_application.get_conf()
            )

            # MNT generation
            self.dem_generation_application = Application(
                "dem_generation", cfg=used_conf.get("dem_generation", {})
            )
            used_conf["dem_generation"] = (
                self.dem_generation_application.get_conf()
            )

            # Points cloud small component outlier removal
            self.pc_outlier_removal_1_app = Application(
                "point_cloud_outlier_removal",
                cfg=used_conf.get(
                    "point_cloud_outlier_removal.1",
                    {"method": "small_components"},
                ),
            )
            used_conf["point_cloud_outlier_removal.1"] = (
                self.pc_outlier_removal_1_app.get_conf()
            )

            # Points cloud statistical outlier removal
            self.pc_outlier_removal_2_app = Application(
                "point_cloud_outlier_removal",
                cfg=used_conf.get(
                    "point_cloud_outlier_removal.2",
                    {"method": "statistical"},
                ),
            )
            used_conf["point_cloud_outlier_removal.2"] = (
                self.pc_outlier_removal_2_app.get_conf()
            )

        if self.save_output_dsm or self.save_output_point_cloud:

            # Point cloud denoising
            self.pc_denoising_application = Application(
                "pc_denoising",
                cfg=used_conf.get("pc_denoising", {"method": "none"}),
            )
            used_conf["pc_denoising"] = self.pc_denoising_application.get_conf()

            if self.save_output_dsm:

                # Rasterization
                self.rasterization_application = Application(
                    "point_cloud_rasterization",
                    cfg=used_conf.get("point_cloud_rasterization", {}),
                )
                used_conf["point_cloud_rasterization"] = (
                    self.rasterization_application.get_conf()
                )
                # DSM filling
                self.dsm_filling_application = Application(
                    "dsm_filling", cfg=conf.get("dsm_filling", {})
                )
                used_conf["dsm_filling"] = (
                    self.dsm_filling_application.get_conf()
                )
                # Auxiliary filling
                self.auxiliary_filling_application = Application(
                    "auxiliary_filling", cfg=conf.get("auxiliary_filling", {})
                )
                used_conf["auxiliary_filling"] = (
                    self.auxiliary_filling_application.get_conf()
                )

            if self.merging:

                # Point cloud fusion
                self.pc_fusion_application = Application(
                    "point_cloud_fusion",
                    cfg=used_conf.get("point_cloud_fusion", {}),
                )
                used_conf["point_cloud_fusion"] = (
                    self.pc_fusion_application.get_conf()
                )

        return used_conf

    def check_applications_with_inputs(self, inputs_conf, application_conf):
        """
        Check for each application the input and output configuration
        consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        """

        initial_elevation = (
            inputs_conf[sens_cst.INITIAL_ELEVATION]["dem"] is not None
        )
        if self.sparse_mtch_sift_app.elevation_delta_lower_bound is None:
            self.sparse_mtch_sift_app.used_config[
                "elevation_delta_lower_bound"
            ] = (-500 if initial_elevation else -1000)
            self.sparse_mtch_sift_app.elevation_delta_lower_bound = (
                self.sparse_mtch_sift_app.used_config[
                    "elevation_delta_lower_bound"
                ]
            )
        if self.sparse_mtch_sift_app.elevation_delta_upper_bound is None:
            self.sparse_mtch_sift_app.used_config[
                "elevation_delta_upper_bound"
            ] = (1000 if initial_elevation else 9000)
            self.sparse_mtch_sift_app.elevation_delta_upper_bound = (
                self.sparse_mtch_sift_app.used_config[
                    "elevation_delta_upper_bound"
                ]
            )
        application_conf["sparse_matching.sift"] = (
            self.sparse_mtch_sift_app.get_conf()
        )

        if self.sparse_mtch_pandora_app.elevation_delta_lower_bound is None:
            self.sparse_mtch_pandora_app.used_config[
                "elevation_delta_lower_bound"
            ] = (-500 if initial_elevation else -1000)
            self.sparse_mtch_pandora_app.elevation_delta_lower_bound = (
                self.sparse_mtch_pandora_app.used_config[
                    "elevation_delta_lower_bound"
                ]
            )
        if self.sparse_mtch_pandora_app.elevation_delta_upper_bound is None:
            self.sparse_mtch_pandora_app.used_config[
                "elevation_delta_upper_bound"
            ] = (1000 if initial_elevation else 9000)
            self.sparse_mtch_pandora_app.elevation_delta_upper_bound = (
                self.sparse_mtch_pandora_app.used_config[
                    "elevation_delta_upper_bound"
                ]
            )
        application_conf["sparse_matching.pandora"] = (
            self.sparse_mtch_pandora_app.get_conf()
        )

        # check classification application parameter compare
        # to each sensors inputs classification list
        for application_key in application_conf:
            if "classification" in application_conf[application_key]:
                for item in inputs_conf["sensors"]:
                    if "classification" in inputs_conf["sensors"][item].keys():
                        if inputs_conf["sensors"][item]["classification"]:
                            descriptions = get_descriptions_bands(
                                inputs_conf["sensors"][item]["classification"]
                            )
                            if application_conf[application_key][
                                "classification"
                            ] and not set(
                                application_conf[application_key][
                                    "classification"
                                ]
                            ).issubset(
                                set(descriptions)
                            ):
                                raise RuntimeError(
                                    "The {} bands description {} ".format(
                                        inputs_conf["sensors"][item][
                                            "classification"
                                        ],
                                        list(descriptions),
                                    )
                                    + "and the {} config are not ".format(
                                        application_key
                                    )
                                    + "consistent: {}".format(
                                        application_conf[application_key][
                                            "classification"
                                        ]
                                    )
                                )
        for key1, key2 in inputs_conf["pairing"]:
            corr_cfg = self.dense_matching_app.loader.get_conf()
            corr_cfg_sparse = self.sparse_mtch_pandora_app.loader.get_conf()
            img_left = inputs_conf["sensors"][key1]["image"]
            img_right = inputs_conf["sensors"][key2]["image"]
            classif_left = None
            classif_right = None
            if "classification" in inputs_conf["sensors"][key1]:
                classif_left = inputs_conf["sensors"][key1]["classification"]
            if "classification" in inputs_conf["sensors"][key2]:
                classif_right = inputs_conf["sensors"][key2]["classification"]
            self.dense_matching_app.corr_config = (
                self.dense_matching_app.loader.check_conf(
                    corr_cfg,
                    img_left,
                    img_right,
                    classif_left,
                    classif_right,
                )
            )
            self.sparse_mtch_pandora_app.corr_config = (
                self.sparse_mtch_pandora_app.loader.check_conf(
                    corr_cfg_sparse,
                    img_left,
                    img_right,
                    classif_left,
                    classif_right,
                )
            )

        if (
            self.sparse_mtch_pandora_app.used_config.get("activated", False)
            is True
            and self.sparse_mtch_sift_app.get_decimation_factor() == 100
        ):
            self.sparse_mtch_sift_app.set_decimation_factor(30)

        return application_conf

    def sensor_to_depth_maps(self):  # noqa: C901
        """
        Creates the depth map from the sensor images given in the input,
        by following the CARS pipeline's steps.
        """
        # pylint:disable=too-many-return-statements

        inputs = self.used_conf[INPUTS]
        output = self.used_conf[OUTPUT]

        # Initialize epsg for terrain tiles
        self.phasing = self.used_conf[ADVANCED][adv_cst.PHASING]

        if self.phasing is not None:
            self.epsg = self.phasing["epsg"]
        else:
            self.epsg = output[out_cst.EPSG]

        if self.epsg is not None:
            # Compute roi polygon, in output EPSG
            self.roi_poly = preprocessing.compute_roi_poly(
                self.input_roi_poly, self.input_roi_epsg, self.epsg
            )

        self.resolution = output[out_cst.RESOLUTION]

        # List of terrain roi corresponding to each epipolar pair
        # Used to generate final terrain roi
        self.list_terrain_roi = []

        # Polygons representing the intersection of each pair of images
        # Used to fill the final DSM only inside of those Polygons
        self.list_intersection_poly = []

        # initialize lists of points
        self.list_epipolar_point_clouds = []
        self.list_sensor_pairs = sensor_inputs.generate_inputs(
            inputs, self.geom_plugin_without_dem_and_geoid
        )
        logging.info(
            "Received {} stereo pairs configurations".format(
                len(self.list_sensor_pairs)
            )
        )

        output_parameters.intialize_product_index(
            self.cars_orchestrator,
            output["product_level"],
            [sensor_pair[0] for sensor_pair in self.list_sensor_pairs],
        )

        # pairs is a dict used to store the CarsDataset of
        # all pairs, easily retrievable with pair keys
        self.pairs = {}

        # triangulated_matches_list is used to store triangulated matches
        # used in dem generation
        self.triangulated_matches_list = []

        save_matches = self.sparse_mtch_sift_app.get_save_matches()

        save_corrected_grid = (
            self.epipolar_grid_generation_application.get_save_grids()
        )

        for (
            pair_key,
            sensor_image_left,
            sensor_image_right,
        ) in self.list_sensor_pairs:

            # initialize pairs for current pair
            self.pairs[pair_key] = {}
            self.pairs[pair_key]["sensor_image_left"] = sensor_image_left
            self.pairs[pair_key]["sensor_image_right"] = sensor_image_right

            # Run applications

            # Run grid generation
            # We generate grids with dem if it is provided.
            # If not provided, grid are generated without dem and a dem
            # will be generated, to use later for a new grid generation**
            altitude_delta_min = inputs.get(sens_cst.INITIAL_ELEVATION, {}).get(
                sens_cst.ALTITUDE_DELTA_MIN, None
            )
            altitude_delta_max = inputs.get(sens_cst.INITIAL_ELEVATION, {}).get(
                sens_cst.ALTITUDE_DELTA_MAX, None
            )

            if inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] is None:
                geom_plugin = self.geom_plugin_without_dem_and_geoid

                if None not in (altitude_delta_min, altitude_delta_max):
                    raise RuntimeError(
                        "Dem path is mandatory for "
                        "the use of altitude deltas"
                    )
            else:
                geom_plugin = self.geom_plugin_with_dem_and_geoid

            # Generate rectification grids
            (
                self.pairs[pair_key]["grid_left"],
                self.pairs[pair_key]["grid_right"],
            ) = self.epipolar_grid_generation_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                geom_plugin,
                orchestrator=self.cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir,
                    "epipolar_grid_generation",
                    "initial",
                    pair_key,
                ),
                pair_key=pair_key,
            )

            if self.quit_on_app("grid_generation"):
                continue  # keep iterating over pairs, but don't go further

            # Run holes detection
            # Get classif depending on which filling is used
            # For now, 2 filling application can be used, and be configured
            # with any order. the .1 will be performed before the .2
            self.pairs[pair_key]["holes_classif"] = []
            self.pairs[pair_key]["holes_poly_margin"] = 0
            add_classif = False
            if self.dense_match_filling_1.used_method == "plane":
                self.pairs[pair_key][
                    "holes_classif"
                ] += self.dense_match_filling_1.get_classif()
                self.pairs[pair_key]["holes_poly_margin"] = max(
                    self.pairs[pair_key]["holes_poly_margin"],
                    self.dense_match_filling_1.get_poly_margin(),
                )
                add_classif = True
            if self.dense_match_filling_2.used_method == "plane":
                self.pairs[pair_key][
                    "holes_classif"
                ] += self.dense_match_filling_2.get_classif()
                self.pairs[pair_key]["holes_poly_margin"] = max(
                    self.pairs[pair_key]["holes_poly_margin"],
                    self.dense_match_filling_2.get_poly_margin(),
                )
                add_classif = True

            self.pairs[pair_key]["holes_bbox_left"] = []
            self.pairs[pair_key]["holes_bbox_right"] = []

            if self.used_conf[ADVANCED][
                adv_cst.USE_EPIPOLAR_A_PRIORI
            ] is False or (len(self.pairs[pair_key]["holes_classif"]) > 0):
                # Run resampling only if needed:
                # no a priori or needs to detect holes

                # Run epipolar resampling
                (
                    self.pairs[pair_key]["epipolar_image_left"],
                    self.pairs[pair_key]["epipolar_image_right"],
                ) = self.resampling_application.run(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["grid_left"],
                    self.pairs[pair_key]["grid_right"],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "resampling", "initial", pair_key
                    ),
                    pair_key=pair_key,
                    margins_fun=self.sparse_mtch_sift_app.get_margins_fun(),
                    tile_width=None,
                    tile_height=None,
                    add_color=False,
                    add_classif=add_classif,
                )

                if self.quit_on_app("resampling"):
                    continue  # keep iterating over pairs, but don't go further

                # Generate the holes polygons in epipolar images
                # They are only generated if dense_match_filling
                # applications are used later
                (
                    self.pairs[pair_key]["holes_bbox_left"],
                    self.pairs[pair_key]["holes_bbox_right"],
                ) = self.hole_detection_app.run(
                    self.pairs[pair_key]["epipolar_image_left"],
                    self.pairs[pair_key]["epipolar_image_right"],
                    classification=self.pairs[pair_key]["holes_classif"],
                    margin=self.pairs[pair_key]["holes_poly_margin"],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "hole_detection", pair_key
                    ),
                    pair_key=pair_key,
                )

                if self.quit_on_app("hole_detection"):
                    continue  # keep iterating over pairs, but don't go further

            if self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI] is False:
                # Run epipolar sparse_matching application
                (
                    self.pairs[pair_key]["epipolar_matches_left"],
                    _,
                ) = self.sparse_mtch_sift_app.run(
                    self.pairs[pair_key]["epipolar_image_left"],
                    self.pairs[pair_key]["epipolar_image_right"],
                    self.pairs[pair_key]["grid_left"].attributes[
                        "disp_to_alt_ratio"
                    ],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "sparse_matching.sift", pair_key
                    ),
                    pair_key=pair_key,
                )

            # Run cluster breakpoint to compute sifts: force computation
            self.cars_orchestrator.breakpoint()

            # Run grid correction application
            if self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI] is False:
                # Estimate grid correction if no epipolar a priori
                # Filter and save matches
                self.pairs[pair_key]["matches_array"] = (
                    self.sparse_mtch_sift_app.filter_matches(
                        self.pairs[pair_key]["epipolar_matches_left"],
                        self.pairs[pair_key]["grid_left"],
                        self.pairs[pair_key]["grid_right"],
                        orchestrator=self.cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=os.path.join(
                            self.dump_dir, "sparse_matching.sift", pair_key
                        ),
                        save_matches=(
                            self.sparse_mtch_sift_app.get_save_matches()
                        ),
                    )
                )

                minimum_nb_matches = (
                    self.sparse_mtch_sift_app.get_minimum_nb_matches()
                )

                # Compute grid correction
                (
                    self.pairs[pair_key]["grid_correction_coef"],
                    self.pairs[pair_key]["corrected_matches_array"],
                    self.pairs[pair_key]["corrected_matches_cars_ds"],
                    _,
                    _,
                ) = grid_correction.estimate_right_grid_correction(
                    self.pairs[pair_key]["matches_array"],
                    self.pairs[pair_key]["grid_right"],
                    initial_cars_ds=self.pairs[pair_key][
                        "epipolar_matches_left"
                    ],
                    save_matches=save_matches,
                    minimum_nb_matches=minimum_nb_matches,
                    pair_folder=os.path.join(
                        self.dump_dir, "grid_correction", "initial", pair_key
                    ),
                    pair_key=pair_key,
                    orchestrator=self.cars_orchestrator,
                )
                # Correct grid right
                self.pairs[pair_key]["corrected_grid_right"] = (
                    grid_correction.correct_grid(
                        self.pairs[pair_key]["grid_right"],
                        self.pairs[pair_key]["grid_correction_coef"],
                        save_corrected_grid,
                        os.path.join(
                            self.dump_dir,
                            "grid_correction",
                            "initial",
                            pair_key,
                        ),
                    )
                )

                self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                    pair_key
                ]["grid_left"]

                if (
                    self.sparse_mtch_pandora_app.used_config.get(
                        "activated", False
                    )
                    is True
                ):
                    # Run epipolar resampling
                    (
                        self.pairs[pair_key]["new_epipolar_image_left"],
                        self.pairs[pair_key]["new_epipolar_image_right"],
                    ) = self.resampling_application.run(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_left"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir,
                            "resampling",
                            "corrected_for_pandora",
                            pair_key,
                        ),
                        pair_key="pair_key",
                        margins_fun=(
                            self.sparse_mtch_pandora_app.get_margins_fun(
                                method="pandora"
                            )
                        ),
                        tile_width=None,
                        tile_height=None,
                        add_color=False,
                        add_classif=add_classif,
                    )

                    if self.quit_on_app("resampling"):
                        continue

                    pandora_sparse_matching_pair_folder = os.path.join(
                        self.dump_dir, "sparse_matching.pandora", pair_key
                    )

                    (
                        self.pairs[pair_key]["pandora_epipolar_matches_left"],
                        _,
                    ) = self.sparse_mtch_pandora_app.run(
                        self.pairs[pair_key]["new_epipolar_image_left"],
                        self.pairs[pair_key]["new_epipolar_image_right"],
                        orchestrator=self.cars_orchestrator,
                        pair_folder=pandora_sparse_matching_pair_folder,
                        pair_key=pair_key,
                        disp_to_alt_ratio=self.pairs[pair_key][
                            "grid_left"
                        ].attributes["disp_to_alt_ratio"],
                    )

                    self.cars_orchestrator.breakpoint()

                    self.pairs[pair_key]["pandora_matches_array"] = (
                        self.sparse_mtch_pandora_app.filter_matches(
                            self.pairs[pair_key][
                                "pandora_epipolar_matches_left"
                            ],
                            self.pairs[pair_key]["corrected_grid_left"],
                            self.pairs[pair_key]["corrected_grid_right"],
                            orchestrator=self.cars_orchestrator,
                            pair_key=pair_key,
                            pair_folder=os.path.join(
                                self.dump_dir,
                                "sparse_matching.pandora",
                                pair_key,
                            ),
                            save_matches=(
                                self.sparse_mtch_pandora_app.get_save_matches()
                            ),
                        )
                    )

                    matches = np.row_stack(
                        (
                            self.pairs[pair_key]["pandora_matches_array"],
                            self.pairs[pair_key]["corrected_matches_array"],
                        )
                    )
                else:
                    matches = self.pairs[pair_key]["corrected_matches_array"]

                # Triangulate matches
                self.pairs[pair_key]["triangulated_matches"] = (
                    dem_generation_tools.triangulate_sparse_matches(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["grid_left"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        matches,
                        geom_plugin,
                    )
                )

                if (
                    self.sparse_mtch_pandora_app.used_config.get(
                        "activated", False
                    )
                    is True
                ):
                    # filter triangulated_matches
                    connection_val = (
                        self.sparse_mtch_pandora_app.get_connection_val()
                    )
                    nb_pts_threshold = (
                        self.sparse_mtch_pandora_app.get_nb_pts_threshold()
                    )
                    clusters_dist_thresh = (
                        self.sparse_mtch_pandora_app.get_clusters_dist_thresh()
                    )
                    filtered_elt_pos = (
                        self.sparse_mtch_pandora_app.get_filtered_elt_pos()
                    )
                    filtered_matches = sparse_mtch_tools.clustering_matches(
                        self.pairs[pair_key]["triangulated_matches"],
                        connection_val=connection_val,
                        nb_pts_threshold=nb_pts_threshold,
                        clusters_distance_threshold=clusters_dist_thresh,
                        filtered_elt_pos=filtered_elt_pos,
                    )

                    app_sparse_matching = self.sparse_mtch_pandora_app
                else:
                    app_sparse_matching = self.sparse_mtch_sift_app
                    filtered_matches = copy.copy(
                        self.pairs[pair_key]["triangulated_matches"]
                    )

                matches_filter_knn = (
                    app_sparse_matching.get_matches_filter_knn()
                )
                matches_filter_dev_factor = (
                    app_sparse_matching.get_matches_filter_dev_factor()
                )
                self.pairs[pair_key]["filtered_triangulated_matches"] = (
                    sparse_mtch_tools.filter_point_cloud_matches(
                        filtered_matches,
                        matches_filter_knn=matches_filter_knn,
                        matches_filter_dev_factor=matches_filter_dev_factor,
                    )
                )

                self.triangulated_matches_list.append(
                    self.pairs[pair_key]["filtered_triangulated_matches"]
                )

                if self.quit_on_app("sparse_matching.sift") or self.quit_on_app(
                    "sparse_matching.pandora"
                ):
                    continue  # keep iterating over pairs, but don't go further

        # Clean grids at the end of processing if required. Note that this will
        # also clean refined grids
        if not (save_corrected_grid or save_matches):
            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "grid_correction")
            )
        # grids file are already cleaned in the application, but the tree
        # structure should also be cleaned
        if not save_corrected_grid:

            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "epipolar_grid_generation")
            )

        # quit if any app in the loop over the pairs was the last one
        if (
            self.quit_on_app("grid_generation")
            or self.quit_on_app("resampling")
            or self.quit_on_app("hole_detection")
            or self.quit_on_app("sparse_matching.sift")
            or self.quit_on_app("sparse_matching.pandora")
        ):
            return True

        if self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI]:
            # Use a priori
            dem_median = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                adv_cst.DEM_MEDIAN
            ]
            dem_min = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                adv_cst.DEM_MIN
            ]
            dem_max = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                adv_cst.DEM_MAX
            ]
            altitude_delta_min = self.used_conf[ADVANCED][
                adv_cst.TERRAIN_A_PRIORI
            ][adv_cst.ALTITUDE_DELTA_MIN]
            altitude_delta_max = self.used_conf[ADVANCED][
                adv_cst.TERRAIN_A_PRIORI
            ][adv_cst.ALTITUDE_DELTA_MAX]
        else:
            dem_generation_output_dir = os.path.join(
                self.dump_dir, "dem_generation"
            )
            safe_makedirs(dem_generation_output_dir)
            if not self.dem_generation_application.used_config[
                "save_intermediate_data"
            ]:
                self.cars_orchestrator.add_to_clean(dem_generation_output_dir)

            # Use initial elevation if provided, and generate dems
            # Generate MNT from matches
            dem = self.dem_generation_application.run(
                self.triangulated_matches_list,
                dem_generation_output_dir,
                inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
                dem_roi_to_use=self.dem_generation_roi,
            )
            # Same geometry plugin if we use exogenous dem
            # as initial elevation always used before if provided
            dem_median = dem.attributes[dem_gen_cst.DEM_MEDIAN_PATH]
            if (
                inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                is not None
            ):
                dem_median = inputs[sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ]

            if (
                dem_median
                != inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
            ):
                self.geom_plugin_with_dem_and_geoid = (
                    sensor_inputs.generate_geometry_plugin_with_dem(
                        self.used_conf[GEOMETRY_PLUGIN],
                        inputs,
                        dem=dem_median,
                    )
                )
            dem_min = dem.attributes[dem_gen_cst.DEM_MIN_PATH]
            dem_max = dem.attributes[dem_gen_cst.DEM_MAX_PATH]

        # update used configuration with terrain a priori
        if None not in (altitude_delta_min, altitude_delta_max):
            advanced_parameters.update_conf(
                self.used_conf,
                dem_median=dem_median,
                altitude_delta_min=altitude_delta_min,
                altitude_delta_max=altitude_delta_max,
            )
        else:
            advanced_parameters.update_conf(
                self.used_conf,
                dem_median=dem_median,
                dem_min=dem_min,
                dem_max=dem_max,
            )

        advanced_parameters.update_conf(
            self.config_full_res,
            dem_median=dem_median,
            dem_min=dem_min,
            dem_max=dem_max,
        )

        # quit only after the configuration was updated
        if self.quit_on_app("dem_generation"):
            return True

        # Define param
        use_global_disp_range = self.dense_matching_app.use_global_disp_range

        if self.pc_denoising_application is not None:
            denoising_overload_fun = (
                self.pc_denoising_application.get_triangulation_overload()
            )
        else:
            denoising_overload_fun = None

        self.pairs_names = [
            pair_name for pair_name, _, _ in self.list_sensor_pairs
        ]

        for cloud_id, (pair_key, _, _) in enumerate(self.list_sensor_pairs):
            # Geometry plugin with dem will be used for the grid generation
            geom_plugin = self.geom_plugin_with_dem_and_geoid
            if self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI] is False:

                if (
                    inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                    is None
                ):
                    # Generate grids with new MNT
                    (
                        self.pairs[pair_key]["new_grid_left"],
                        self.pairs[pair_key]["new_grid_right"],
                    ) = self.epipolar_grid_generation_application.run(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        geom_plugin,
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir,
                            "epipolar_grid_generation",
                            "new_mnt",
                            pair_key,
                        ),
                        pair_key=pair_key,
                    )

                    # Correct grids with former matches
                    # Transform matches to new grids
                    new_grid_matches_array = (
                        geom_plugin.transform_matches_from_grids(
                            matches,
                            self.pairs[pair_key]["corrected_grid_left"],
                            self.pairs[pair_key]["corrected_grid_right"],
                            self.pairs[pair_key]["new_grid_left"],
                            self.pairs[pair_key]["new_grid_right"],
                        )
                    )
                    save_matches = self.sparse_mtch_sift_app.get_save_matches()
                    # Estimate grid_correction
                    (
                        self.pairs[pair_key]["grid_correction_coef"],
                        matches,
                        self.pairs[pair_key]["corrected_matches_cars_ds"],
                        _,
                        _,
                    ) = grid_correction.estimate_right_grid_correction(
                        new_grid_matches_array,
                        self.pairs[pair_key]["new_grid_right"],
                        save_matches=save_matches,
                        initial_cars_ds=self.pairs[pair_key][
                            "epipolar_matches_left"
                        ],
                        pair_folder=os.path.join(
                            self.dump_dir, "grid_correction", "new", pair_key
                        ),
                        pair_key=pair_key,
                        orchestrator=self.cars_orchestrator,
                    )

                    # Correct grid right

                    self.pairs[pair_key]["corrected_grid_right"] = (
                        grid_correction.correct_grid(
                            self.pairs[pair_key]["new_grid_right"],
                            self.pairs[pair_key]["grid_correction_coef"],
                            save_corrected_grid,
                            os.path.join(
                                self.dump_dir,
                                "grid_correction",
                                "new",
                                pair_key,
                            ),
                        )
                    )

                    # Use the new grid as uncorrected grid
                    self.pairs[pair_key]["grid_right"] = self.pairs[pair_key][
                        "new_grid_right"
                    ]

                    self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                        pair_key
                    ]["new_grid_left"]

                # matches filter params
                matches_filter_knn = (
                    app_sparse_matching.get_matches_filter_knn()
                )
                matches_filter_dev_factor = (
                    app_sparse_matching.get_matches_filter_dev_factor()
                )
                if use_global_disp_range:
                    # Triangulate new matches
                    self.pairs[pair_key]["triangulated_matches"] = (
                        dem_generation_tools.triangulate_sparse_matches(
                            self.pairs[pair_key]["sensor_image_left"],
                            self.pairs[pair_key]["sensor_image_right"],
                            self.pairs[pair_key]["corrected_grid_left"],
                            self.pairs[pair_key]["corrected_grid_right"],
                            matches,
                            geometry_plugin=geom_plugin,
                        )
                    )

                    if (
                        self.sparse_mtch_pandora_app.used_config.get(
                            "activated", False
                        )
                        is True
                    ):
                        # filter triangulated_matches
                        # Filter outliers
                        filtered_matches = sparse_mtch_tools.clustering_matches(
                            self.pairs[pair_key]["triangulated_matches"],
                            connection_val=connection_val,
                            nb_pts_threshold=nb_pts_threshold,
                            clusters_distance_threshold=clusters_dist_thresh,
                            filtered_elt_pos=filtered_elt_pos,
                        )
                    else:
                        filtered_matches = self.pairs[pair_key][
                            "triangulated_matches"
                        ]

                    self.pairs[pair_key]["filtered_triangulated_matches"] = (
                        sparse_mtch_tools.filter_point_cloud_matches(
                            filtered_matches,
                            matches_filter_knn=matches_filter_knn,
                            matches_filter_dev_factor=matches_filter_dev_factor,
                        )
                    )

                    # Compute disp_min and disp_max
                    (
                        dmin,
                        dmax,
                    ) = sparse_mtch_tools.compute_disp_min_disp_max(
                        self.pairs[pair_key]["filtered_triangulated_matches"],
                        self.cars_orchestrator,
                        disp_margin=(
                            self.sparse_mtch_sift_app.get_disparity_margin()
                        ),
                        pair_key=pair_key,
                        disp_to_alt_ratio=self.pairs[pair_key][
                            "corrected_grid_left"
                        ].attributes["disp_to_alt_ratio"],
                    )

                    advanced_parameters.update_conf(
                        self.config_full_res,
                        dmin=dmin,
                        dmax=dmax,
                        pair_key=pair_key,
                    )
            else:
                # Use epipolar a priori
                # load the disparity range
                [dmin, dmax] = self.used_conf[ADVANCED][
                    adv_cst.EPIPOLAR_A_PRIORI
                ][pair_key][adv_cst.DISPARITY_RANGE]
                # load the grid correction coefficient
                self.pairs[pair_key]["grid_correction_coef"] = self.used_conf[
                    ADVANCED
                ][adv_cst.EPIPOLAR_A_PRIORI][pair_key][adv_cst.GRID_CORRECTION]
                self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                    pair_key
                ]["grid_left"]
                # no correction if the grid correction coefs are None
                if self.pairs[pair_key]["grid_correction_coef"] is None:
                    self.pairs[pair_key]["corrected_grid_right"] = self.pairs[
                        pair_key
                    ]["grid_right"]
                else:
                    # Correct grid right with provided epipolar a priori
                    self.pairs[pair_key]["corrected_grid_right"] = (
                        grid_correction.correct_grid_from_1d(
                            self.pairs[pair_key]["grid_right"],
                            self.pairs[pair_key]["grid_correction_coef"],
                            save_corrected_grid,
                            os.path.join(
                                self.dump_dir, "grid_correction", pair_key
                            ),
                        )
                    )
                advanced_parameters.update_conf(
                    self.config_full_res,
                    dmin=dmin,
                    dmax=dmax,
                    pair_key=pair_key,
                )
            # Run epipolar resampling

            # Update used_conf configuration with epipolar a priori
            # Add global min and max computed with grids
            advanced_parameters.update_conf(
                self.used_conf,
                grid_correction_coef=self.pairs[pair_key][
                    "grid_correction_coef"
                ],
                pair_key=pair_key,
            )
            advanced_parameters.update_conf(
                self.config_full_res,
                grid_correction_coef=self.pairs[pair_key][
                    "grid_correction_coef"
                ],
                pair_key=pair_key,
            )
            # saved used configuration
            cars_dataset.save_dict(
                self.used_conf,
                os.path.join(self.out_dir, "used_conf.json"),
                safe_save=True,
            )

            # Generate min and max disp grids
            # Global disparity min and max will be computed from
            # these grids
            dense_matching_pair_folder = os.path.join(
                self.dump_dir, "dense_matching", pair_key
            )
            if use_global_disp_range:
                # Generate min and max disp grids from constants
                # sensor image is not used here
                # TODO remove when only local diparity range will be used
                disp_range_grid = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        dmin=dmin,
                        dmax=dmax,
                        pair_folder=dense_matching_pair_folder,
                    )
                )
            elif None in (altitude_delta_min, altitude_delta_max):
                # Generate min and max disp grids from dems
                disp_range_grid = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        dem_min=dem_min,
                        dem_max=dem_max,
                        dem_median=dem_median,
                        pair_folder=dense_matching_pair_folder,
                    )
                )
            else:
                # Generate min and max disp grids from deltas
                disp_range_grid = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        altitude_delta_min=altitude_delta_min,
                        altitude_delta_max=altitude_delta_max,
                        dem_median=dem_median,
                        pair_folder=dense_matching_pair_folder,
                    )
                )

            # Get margins used in dense matching,
            dense_matching_margins_fun = (
                self.dense_matching_app.get_margins_fun(
                    self.pairs[pair_key]["corrected_grid_left"],
                    disp_range_grid,
                )
            )

            # TODO add in metadata.json max diff max - min
            # Update used_conf configuration with epipolar a priori
            # Add global min and max computed with grids
            advanced_parameters.update_conf(
                self.used_conf,
                dmin=np.min(
                    disp_range_grid[0, 0]["disp_min_grid"].values
                ),  # TODO compute dmin dans dmax
                dmax=np.max(disp_range_grid[0, 0]["disp_max_grid"].values),
                pair_key=pair_key,
            )
            advanced_parameters.update_conf(
                self.config_full_res,
                dmin=np.min(
                    disp_range_grid[0, 0]["disp_min_grid"].values
                ),  # TODO compute dmin dans dmax
                dmax=np.max(disp_range_grid[0, 0]["disp_max_grid"].values),
                pair_key=pair_key,
            )

            # saved used configuration
            cars_dataset.save_dict(
                self.used_conf,
                os.path.join(self.out_dir, "used_conf.json"),
                safe_save=True,
            )

            # Generate roi
            epipolar_roi = preprocessing.compute_epipolar_roi(
                self.input_roi_poly,
                self.input_roi_epsg,
                self.geom_plugin_with_dem_and_geoid,
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                self.pairs[pair_key]["corrected_grid_left"],
                self.pairs[pair_key]["corrected_grid_right"],
                os.path.join(self.dump_dir, "compute_epipolar_roi", pair_key),
                disp_min=np.min(
                    disp_range_grid[0, 0]["disp_min_grid"].values
                ),  # TODO compute dmin dans dmax
                disp_max=np.max(disp_range_grid[0, 0]["disp_max_grid"].values),
            )

            # Generate new epipolar images
            # Generated with corrected grids
            # Optimal size is computed for the worst case scenario
            # found with epipolar disparity range grids

            (
                optimum_tile_size,
                local_tile_optimal_size_fun,
            ) = self.dense_matching_app.get_optimal_tile_size(
                disp_range_grid,
                self.cars_orchestrator.cluster.checked_conf_cluster[
                    "max_ram_per_worker"
                ],
            )
            (
                new_epipolar_image_left,
                new_epipolar_image_right,
            ) = self.resampling_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                self.pairs[pair_key]["corrected_grid_left"],
                self.pairs[pair_key]["corrected_grid_right"],
                orchestrator=self.cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir, "resampling", "corrected_grid", pair_key
                ),
                pair_key=pair_key,
                margins_fun=dense_matching_margins_fun,
                tile_width=optimum_tile_size,
                tile_height=optimum_tile_size,
                add_color=True,
                add_classif=True,
                epipolar_roi=epipolar_roi,
            )

            # Run ground truth dsm computation
            if self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
                self.used_conf["applications"]["ground_truth_reprojection"][
                    "save_intermediate_data"
                ] = True
                new_geomplugin_dsm = AbstractGeometry(  # pylint: disable=E0110
                    self.used_conf[GEOMETRY_PLUGIN],
                    dem=self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_GROUND_TRUTH_DSM
                    ],
                    geoid=self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_GEOID
                    ],
                )
                self.ground_truth_reprojection.run(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["corrected_grid_right"],
                    new_geomplugin_dsm,
                    self.geom_plugin_with_dem_and_geoid,
                    self.pairs[pair_key]["corrected_grid_left"].attributes[
                        "disp_to_alt_ratio"
                    ],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "ground_truth_reprojection", pair_key
                    ),
                )

            # Run epipolar matching application
            epipolar_disparity_map = self.dense_matching_app.run(
                new_epipolar_image_left,
                new_epipolar_image_right,
                local_tile_optimal_size_fun,
                orchestrator=self.cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir, "dense_matching", pair_key
                ),
                pair_key=pair_key,
                disp_range_grid=disp_range_grid,
                compute_disparity_masks=False,
                disp_to_alt_ratio=self.pairs[pair_key][
                    "corrected_grid_left"
                ].attributes["disp_to_alt_ratio"],
            )

            if self.quit_on_app("dense_matching"):
                continue  # keep iterating over pairs, but don't go further

            # Dense matches filling
            if self.dense_match_filling_1.used_method == "plane":
                # Fill holes in disparity map
                (filled_with_1_epipolar_disparity_map) = (
                    self.dense_match_filling_1.run(
                        epipolar_disparity_map,
                        self.pairs[pair_key]["holes_bbox_left"],
                        self.pairs[pair_key]["holes_bbox_right"],
                        disp_min=np.min(
                            disp_range_grid[0, 0]["disp_min_grid"].values
                        ),
                        disp_max=np.max(
                            disp_range_grid[0, 0]["disp_max_grid"].values
                        ),
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir, "dense_match_filling_1", pair_key
                        ),
                        pair_key=pair_key,
                    )
                )
            else:
                # Fill with zeros
                (filled_with_1_epipolar_disparity_map) = (
                    self.dense_match_filling_1.run(
                        epipolar_disparity_map,
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir, "dense_match_filling_1", pair_key
                        ),
                        pair_key=pair_key,
                    )
                )

            if self.quit_on_app("dense_match_filling.1"):
                continue  # keep iterating over pairs, but don't go further

            if self.dense_match_filling_2.used_method == "plane":
                # Fill holes in disparity map
                (filled_with_2_epipolar_disparity_map) = (
                    self.dense_match_filling_2.run(
                        filled_with_1_epipolar_disparity_map,
                        self.pairs[pair_key]["holes_bbox_left"],
                        self.pairs[pair_key]["holes_bbox_right"],
                        disp_min=np.min(
                            disp_range_grid[0, 0]["disp_min_grid"].values
                        ),
                        disp_max=np.max(
                            disp_range_grid[0, 0]["disp_max_grid"].values
                        ),
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir, "dense_match_filling_2", pair_key
                        ),
                        pair_key=pair_key,
                    )
                )
            else:
                # Fill with zeros
                (filled_with_2_epipolar_disparity_map) = (
                    self.dense_match_filling_2.run(
                        filled_with_1_epipolar_disparity_map,
                        orchestrator=self.cars_orchestrator,
                        pair_folder=os.path.join(
                            self.dump_dir, "dense_match_filling_2", pair_key
                        ),
                        pair_key=pair_key,
                    )
                )

            if self.quit_on_app("dense_match_filling.2"):
                continue  # keep iterating over pairs, but don't go further

            if self.epsg is None:
                # compute epsg
                # Epsg uses global disparity min and max
                self.epsg = preprocessing.compute_epsg(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["corrected_grid_right"],
                    self.geom_plugin_with_dem_and_geoid,
                    disp_min=np.min(
                        disp_range_grid[0, 0]["disp_min_grid"].values
                    ),
                    disp_max=np.max(
                        disp_range_grid[0, 0]["disp_max_grid"].values
                    ),
                )
                # Compute roi polygon, in input EPSG
                self.roi_poly = preprocessing.compute_roi_poly(
                    self.input_roi_poly, self.input_roi_epsg, self.epsg
                )

            # Checking disparity intervals indicators
            if self.used_conf[APPLICATIONS]["dense_matching"][
                "generate_confidence_intervals"
            ]:
                intervals = [cst_disp.INTERVAL_INF, cst_disp.INTERVAL_SUP]
                intervals_pair_flag = False
                for key, item in self.dense_matching_app.corr_config[
                    "pipeline"
                ].items():
                    if (
                        cst_disp.CONFIDENCE_KEY in key
                        and item["confidence_method"] == cst_disp.INTERVAL
                    ):
                        indicator = key.split(".")
                        if not intervals_pair_flag:
                            if len(indicator) > 1:
                                intervals[0] += "." + indicator[-1]
                                intervals[1] += "." + indicator[-1]
                            # Only processing the first encountered interval
                            intervals_pair_flag = True
                        else:
                            warn_msg = (
                                "Multiple confidence intervals "
                                "is not supported. {} will be "
                                "ignored. Only {} will be processed"
                            ).format(key, intervals)
                            logging.warning(warn_msg)
            else:
                intervals = None

            if isinstance(output[sens_cst.GEOID], str):
                output_geoid_path = output[sens_cst.GEOID]
            elif (
                isinstance(output[sens_cst.GEOID], bool)
                and output[sens_cst.GEOID]
            ):
                package_path = os.path.dirname(__file__)
                output_geoid_path = os.path.join(
                    package_path,
                    "..",
                    "..",
                    "conf",
                    sensor_inputs.CARS_GEOID_PATH,
                )
            else:
                # default case : stay on the ellipsoid
                output_geoid_path = None

            depth_map_dir = None
            last_depth_map_application = None
            if self.save_output_depth_map:
                depth_map_dir = os.path.join(
                    self.out_dir, "depth_map", pair_key
                )
                safe_makedirs(depth_map_dir)

            point_cloud_dir = None
            if self.save_output_point_cloud:
                point_cloud_dir = os.path.join(
                    self.out_dir, "point_cloud", pair_key
                )
                safe_makedirs(point_cloud_dir)

            if self.save_output_depth_map or self.save_output_point_cloud:
                if (
                    self.pc_outlier_removal_2_app.used_config.get(
                        "activated", False
                    )
                    is True
                    and self.merging is False
                ):
                    last_depth_map_application = "pc_outlier_removal_2"
                elif (
                    self.pc_outlier_removal_1_app.used_config.get(
                        "activated", False
                    )
                    is True
                    and self.merging is False
                ):
                    last_depth_map_application = "pc_outlier_removal_1"
                else:
                    last_depth_map_application = "triangulation"

            triangulation_point_cloud_dir = (
                point_cloud_dir
                if (
                    point_cloud_dir
                    and last_depth_map_application == "triangulation"
                    and self.merging is False
                )
                else None
            )

            # Run epipolar triangulation application
            epipolar_point_cloud = self.triangulation_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                new_epipolar_image_left,
                self.pairs[pair_key]["corrected_grid_left"],
                self.pairs[pair_key]["corrected_grid_right"],
                filled_with_2_epipolar_disparity_map,
                self.epsg,
                self.geom_plugin_without_dem_and_geoid,
                denoising_overload_fun=denoising_overload_fun,
                source_pc_names=self.pairs_names,
                orchestrator=self.cars_orchestrator,
                pair_dump_dir=os.path.join(
                    self.dump_dir, "triangulation", pair_key
                ),
                pair_key=pair_key,
                uncorrected_grid_right=self.pairs[pair_key]["grid_right"],
                geoid_path=output_geoid_path,
                cloud_id=cloud_id,
                intervals=intervals,
                depth_map_dir=depth_map_dir,
                point_cloud_dir=triangulation_point_cloud_dir,
                save_output_coordinates=last_depth_map_application
                == "triangulation",
                save_output_color=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_COLOR],
                save_output_classification=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_CLASSIFICATION],
                save_output_filling=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_FILLING],
                save_output_mask=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_MASK],
                save_output_performance_map=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_PERFORMANCE_MAP],
            )

            if self.quit_on_app("triangulation"):
                continue  # keep iterating over pairs, but don't go further

            if self.merging:
                self.list_epipolar_point_clouds.append(epipolar_point_cloud)
            else:
                filtering_depth_map_dir = (
                    depth_map_dir
                    if (
                        depth_map_dir
                        and last_depth_map_application == "pc_outlier_removal_1"
                    )
                    else None
                )
                filtering_point_cloud_dir = (
                    point_cloud_dir
                    if (
                        point_cloud_dir
                        and last_depth_map_application == "pc_outlier_removal_1"
                        and self.merging is False
                    )
                    else None
                )
                filtered_epipolar_point_cloud_1 = (
                    self.pc_outlier_removal_1_app.run(
                        epipolar_point_cloud,
                        depth_map_dir=filtering_depth_map_dir,
                        point_cloud_dir=filtering_point_cloud_dir,
                        dump_dir=os.path.join(
                            self.dump_dir, "pc_outlier_removal_1", pair_key
                        ),
                        epsg=self.epsg,
                        orchestrator=self.cars_orchestrator,
                    )
                )
                if self.quit_on_app("point_cloud_outlier_removal.1"):
                    continue  # keep iterating over pairs, but don't go further
                filtering_depth_map_dir = (
                    depth_map_dir
                    if (
                        depth_map_dir
                        and last_depth_map_application == "pc_outlier_removal_2"
                    )
                    else None
                )
                filtering_point_cloud_dir = (
                    point_cloud_dir
                    if (
                        point_cloud_dir
                        and last_depth_map_application == "pc_outlier_removal_2"
                        and self.merging is False
                    )
                    else None
                )
                filtered_epipolar_point_cloud_2 = (
                    self.pc_outlier_removal_2_app.run(
                        filtered_epipolar_point_cloud_1,
                        depth_map_dir=filtering_depth_map_dir,
                        point_cloud_dir=filtering_point_cloud_dir,
                        dump_dir=os.path.join(
                            self.dump_dir, "pc_outlier_removal_2", pair_key
                        ),
                        epsg=self.epsg,
                        orchestrator=self.cars_orchestrator,
                    )
                )
                if self.quit_on_app("point_cloud_outlier_removal.2"):
                    continue  # keep iterating over pairs, but don't go further

                # denoising available only if we'll go further in the pipeline
                if self.save_output_dsm or self.save_output_point_cloud:
                    denoised_epipolar_point_clouds = (
                        self.pc_denoising_application.run(
                            filtered_epipolar_point_cloud_2,
                            orchestrator=self.cars_orchestrator,
                            pair_folder=os.path.join(
                                self.dump_dir, "denoising", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )

                    self.list_epipolar_point_clouds.append(
                        denoised_epipolar_point_clouds
                    )

                    if self.quit_on_app("pc_denoising"):
                        # keep iterating over pairs, but don't go further
                        continue

            if self.save_output_dsm or self.save_output_point_cloud:
                # Compute terrain bounding box /roi related to
                # current images
                (current_terrain_roi_bbox, intersection_poly) = (
                    preprocessing.compute_terrain_bbox(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        new_epipolar_image_left,
                        self.pairs[pair_key]["corrected_grid_left"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.epsg,
                        self.geom_plugin_with_dem_and_geoid,
                        resolution=self.resolution,
                        disp_min=np.min(
                            disp_range_grid[0, 0]["disp_min_grid"].values
                        ),
                        disp_max=np.max(
                            disp_range_grid[0, 0]["disp_max_grid"].values
                        ),
                        roi_poly=(
                            None if self.debug_with_roi else self.roi_poly
                        ),
                        orchestrator=self.cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=os.path.join(
                            self.dump_dir, "terrain_bbox", pair_key
                        ),
                        check_inputs=True,
                    )
                )
                self.list_terrain_roi.append(current_terrain_roi_bbox)
                self.list_intersection_poly.append(intersection_poly)

                # compute terrain bounds for later use
                (
                    self.terrain_bounds,
                    self.optimal_terrain_tile_width,
                ) = preprocessing.compute_terrain_bounds(
                    self.list_terrain_roi,
                    roi_poly=(None if self.debug_with_roi else self.roi_poly),
                    resolution=self.resolution,
                )

        # quit if any app in the loop over the pairs was the last one
        # pylint:disable=too-many-boolean-expressions
        if (
            self.quit_on_app("dense_matching")
            or self.quit_on_app("dense_match_filling.1")
            or self.quit_on_app("dense_match_filling.2")
            or self.quit_on_app("triangulation")
            or self.quit_on_app("point_cloud_outlier_removal.1")
            or self.quit_on_app("point_cloud_outlier_removal.2")
            or self.quit_on_app("pc_denoising")
        ):
            return True

        return False

    def rasterize_point_cloud(self):
        """
        Final step of the pipeline: rasterize the point
        cloud created in the prior steps.
        """

        rasterization_dump_dir = os.path.join(self.dump_dir, "rasterization")

        dsm_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "dsm.tif",
            )
            if self.save_output_dsm
            else None
        )

        color_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "color.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_COLOR]
            else None
        )

        performance_map_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "performance_map.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_PERFORMANCE_MAP
            ]
            else None
        )

        classif_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "classification.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_CLASSIFICATION
            ]
            else None
        )

        mask_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "mask.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_MASK]
            else None
        )

        contributing_pair_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "contributing_pair.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_CONTRIBUTING_PAIR
            ]
            else None
        )

        filling_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "filling.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING]
            else None
        )

        # rasterize point cloud
        _ = self.rasterization_application.run(
            self.point_cloud_to_rasterize,
            self.epsg,
            resolution=self.resolution,
            orchestrator=self.cars_orchestrator,
            dsm_file_name=dsm_file_name,
            color_file_name=color_file_name,
            classif_file_name=classif_file_name,
            performance_map_file_name=performance_map_file_name,
            mask_file_name=mask_file_name,
            contributing_pair_file_name=contributing_pair_file_name,
            filling_file_name=filling_file_name,
            color_dtype=self.color_type,
            dump_dir=rasterization_dump_dir,
            performance_map_classes=self.used_conf[ADVANCED][
                adv_cst.PERFORMANCE_MAP_CLASSES
            ],
            phasing=self.phasing,
        )

        # Cleaning: don't keep terrain bbox if save_intermediate_data
        # is not activated
        if not self.used_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA]:
            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "terrain_bbox")
            )

        if self.quit_on_app("point_cloud_rasterization"):
            return True

        # dsm needs to be saved before filling
        self.cars_orchestrator.breakpoint()

        return False

    def filling(self):
        """
        Fill the dsm
        """

        dsm_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "dsm.tif",
            )
            if self.save_output_dsm
            else None
        )

        if self.dsms_in_inputs:
            dsms_merging_dump_dir = os.path.join(self.dump_dir, "dsms_merging")

            dsm_dict = self.used_conf[INPUTS][dsm_cst.DSMS]
            dict_path = {}
            for key in dsm_dict.keys():
                for path_name in dsm_dict[key].keys():
                    if dsm_dict[key][path_name] is not None:
                        if path_name not in dict_path:
                            dict_path[path_name] = [dsm_dict[key][path_name]]
                        else:
                            dict_path[path_name].append(
                                dsm_dict[key][path_name]
                            )

            color_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "color.tif",
                )
                if "color" in dict_path
                else None
            )

            mask_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "mask.tif",
                )
                if "mask" in dict_path
                else None
            )

            performance_map_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "performance_map.tif",
                )
                if "performance_map" in dict_path
                else None
            )

            classif_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "classification.tif",
                )
                if "classification" in dict_path
                else None
            )

            contributing_all_pair_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "contributing_pair.tif",
                )
                if "source_pc" in dict_path
                else None
            )

            filling_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "filling.tif",
                )
                if "filling" in dict_path
                else None
            )

            self.epsg = rasterio_get_epsg(dict_path["dsm"][0])

            # Compute roi polygon, in input EPSG
            self.roi_poly = preprocessing.compute_roi_poly(
                self.input_roi_poly, self.input_roi_epsg, self.epsg
            )

            _ = dsm_inputs.merge_dsm_infos(
                dict_path,
                self.cars_orchestrator,
                self.roi_poly,
                dsms_merging_dump_dir,
                dsm_file_name,
                color_file_name,
                classif_file_name,
                filling_file_name,
                performance_map_file_name,
                mask_file_name,
                contributing_all_pair_file_name,
            )

            # dsm needs to be saved before filling
            self.cars_orchestrator.breakpoint()
        else:
            filling_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "filling.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_FILLING
                ]
                else None
            )

            color_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "color.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_COLOR]
                else None
            )

            classif_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "classification.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_CLASSIFICATION
                ]
                else None
            )

        _ = self.dsm_filling_application.run(
            orchestrator=self.cars_orchestrator,
            # path to initial elevation file via geom plugin
            initial_elevation=self.geom_plugin_with_dem_and_geoid,
            dsm_path=dsm_file_name,
            roi_polys=(
                self.list_intersection_poly if self.compute_depth_map else None
            ),
            roi_epsg=self.epsg,
            output_geoid=self.used_conf[OUTPUT][sens_cst.GEOID],
            filling_file_name=filling_file_name,
            dump_dir=self.dump_dir,
        )

        if self.quit_on_app("dsm_filling"):
            return True

        _ = self.auxiliary_filling_application.run(
            dsm_file=dsm_file_name,
            color_file=color_file_name,
            classif_file=classif_file_name,
            dump_dir=self.dump_dir,
            sensor_inputs=self.used_conf[INPUTS].get("sensors"),
            pairing=self.used_conf[INPUTS].get("pairing"),
            geom_plugin=self.geom_plugin_with_dem_and_geoid,
            orchestrator=self.cars_orchestrator,
        )

        return self.quit_on_app("auxiliary_filling")

    def preprocess_depth_maps(self):
        """
        Adds multiple processing steps to the depth maps :
        Merging, denoising.
        Creates the point cloud that will be rasterized in
        the last step of the pipeline.
        """

        if not self.merging:
            self.point_cloud_to_rasterize = (
                self.list_epipolar_point_clouds,
                self.terrain_bounds,
            )
            self.color_type = self.point_cloud_to_rasterize[0][
                0
            ].attributes.get("color_type", None)
        else:
            # find which application produce the final version of the
            # point cloud. The last generated point cloud will be saved
            # as official point cloud product if save_output_point_cloud
            # is True.

            last_pc_application = None
            # denoising application will produce a point cloud, unless
            # it uses the 'none' method.
            if self.pc_denoising_application.used_method != "none":
                last_pc_application = "denoising"
            else:
                last_pc_application = "fusion"

            raster_app_margin = 0
            if self.rasterization_application is not None:
                raster_app_margin = self.rasterization_application.get_margins(
                    self.resolution
                )

            merged_point_clouds = self.pc_fusion_application.run(
                self.list_epipolar_point_clouds,
                self.terrain_bounds,
                self.epsg,
                source_pc_names=(
                    self.pairs_names if self.compute_depth_map else None
                ),
                orchestrator=self.cars_orchestrator,
                margins=raster_app_margin,
                optimal_terrain_tile_width=self.optimal_terrain_tile_width,
                roi=(self.roi_poly if self.debug_with_roi else None),
                save_laz_output=self.save_output_point_cloud
                and last_pc_application == "fusion",
            )

            if self.quit_on_app("point_cloud_fusion"):
                return True

            # denoise point cloud
            denoised_merged_point_clouds = self.pc_denoising_application.run(
                merged_point_clouds,
                orchestrator=self.cars_orchestrator,
                save_laz_output=self.save_output_point_cloud
                and last_pc_application == "denoising",
            )

            if self.quit_on_app("pc_denoising"):
                return True

            # Rasterize merged and filtered point cloud
            self.point_cloud_to_rasterize = denoised_merged_point_clouds

            # try getting the color type from multiple sources
            self.color_type = self.list_epipolar_point_clouds[0].attributes.get(
                "color_type",
                self.point_cloud_to_rasterize.attributes.get(
                    "color_type", None
                ),
            )

        return False

    def load_input_depth_maps(self):
        """
        Loads all the data and creates all the variables used
        later when processing a depth map, as if it was just computed.
        """

        output_parameters.intialize_product_index(
            self.cars_orchestrator,
            self.used_conf[OUTPUT]["product_level"],
            self.used_conf[INPUTS][depth_cst.DEPTH_MAPS].keys(),
        )

        # get epsg
        self.epsg = self.used_conf[OUTPUT][out_cst.EPSG]

        # compute epsg
        epsg_cloud = pc_tif_tools.compute_epsg_from_point_cloud(
            self.used_conf[INPUTS][depth_cst.DEPTH_MAPS]
        )
        if self.epsg is None:
            self.epsg = epsg_cloud

        self.resolution = self.used_conf[OUTPUT][out_cst.RESOLUTION]

        # Compute roi polygon, in input EPSG
        self.roi_poly = preprocessing.compute_roi_poly(
            self.input_roi_poly, self.input_roi_epsg, self.epsg
        )

        if not self.merging:
            # compute bounds
            self.terrain_bounds = pc_tif_tools.get_bounds(
                self.used_conf[INPUTS][depth_cst.DEPTH_MAPS],
                self.epsg,
                roi_poly=self.roi_poly,
            )

            self.list_epipolar_point_clouds = (
                pc_tif_tools.generate_point_clouds(
                    self.used_conf[INPUTS][depth_cst.DEPTH_MAPS],
                    self.cars_orchestrator,
                    tile_size=1000,
                )
            )
        else:
            # Compute terrain bounds and transform point clouds
            (
                self.terrain_bounds,
                self.list_epipolar_point_clouds,
            ) = pc_tif_tools.transform_input_pc(
                self.used_conf[INPUTS][depth_cst.DEPTH_MAPS],
                self.epsg,
                roi_poly=self.roi_poly,
                epipolar_tile_size=1000,  # TODO change it
                orchestrator=self.cars_orchestrator,
            )

            # Compute number of superposing point cloud for density
            max_number_superposing_point_clouds = (
                pc_tif_tools.compute_max_nb_point_clouds(
                    self.list_epipolar_point_clouds
                )
            )

            # Compute average distance between two points
            average_distance_point_cloud = (
                pc_tif_tools.compute_average_distance(
                    self.list_epipolar_point_clouds
                )
            )
            self.optimal_terrain_tile_width = (
                self.rasterization_application.get_optimal_tile_size(
                    self.cars_orchestrator.cluster.checked_conf_cluster[
                        "max_ram_per_worker"
                    ],
                    superposing_point_clouds=(
                        max_number_superposing_point_clouds
                    ),
                    point_cloud_resolution=average_distance_point_cloud,
                )
            )
            # epsg_cloud and optimal_terrain_tile_width have the same epsg
            self.optimal_terrain_tile_width = (
                preprocessing.convert_optimal_tile_size_with_epsg(
                    self.terrain_bounds,
                    self.optimal_terrain_tile_width,
                    self.epsg,
                    epsg_cloud,
                )
            )

    def final_cleanup(self):
        """
        Clean temporary files and directory at the end of cars processing
        """

        if not self.used_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA]:
            # delete everything in tile_processing if save_intermediate_data is
            # not activated
            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "tile_processing")
            )

            # Remove dump_dir if no intermediate data should be written
            if (
                not any(
                    app.get("save_intermediate_data", False) is True
                    for app in self.used_conf[APPLICATIONS].values()
                )
                and not self.dsms_in_inputs
            ):
                self.cars_orchestrator.add_to_clean(self.dump_dir)

    @cars_profile(name="run_dense_pipeline", interval=0.5)
    def run(self):  # noqa C901
        """
        Run pipeline

        """

        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")
        self.auxiliary = self.used_conf[OUTPUT][out_cst.AUXILIARY]

        # Save used conf
        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(self.out_dir, "used_conf.json"),
            safe_save=True,
        )

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            out_json_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as self.cars_orchestrator:

            # initialize out_json
            self.cars_orchestrator.update_out_info({"version": __version__})

            if not self.dsms_in_inputs:
                if self.compute_depth_map:
                    self.sensor_to_depth_maps()
                else:
                    self.load_input_depth_maps()

                if self.save_output_dsm or self.save_output_point_cloud:
                    end_pipeline = self.preprocess_depth_maps()

                    if self.save_output_dsm and not end_pipeline:
                        self.rasterize_point_cloud()
                        self.filling()
            else:
                self.filling()

            self.final_cleanup()
