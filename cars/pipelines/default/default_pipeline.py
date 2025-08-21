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
# pylint: disable=too-many-nested-blocks
"""
CARS default pipeline class file
"""
# Standard imports
from __future__ import print_function

import copy
import json
import logging
import math
import os
import shutil

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.core import cars_logging, roi_tools
from cars.core.inputs import get_descriptions_bands, rasterio_get_size
from cars.core.utils import safe_makedirs
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster import log_wrapper
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
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
)
from cars.pipelines.pipeline_template import (
    PipelineTemplate,
    _merge_resolution_conf_rec,
)
from cars.pipelines.unit.unit_pipeline import UnitPipeline


@Pipeline.register(
    "default",
)
class DefaultPipeline(PipelineTemplate):
    """
    DefaultPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf, config_dir=None):  # noqa: C901
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
        :param config_dir: path to dir containing json or yaml file
        :type config_dir: str
        """

        # Used conf
        self.used_conf = {}

        # Transform relative path to absolute path
        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)

        # Check global conf
        self.check_global_schema(conf)

        # The inputs, outputs and advanced
        # parameters are all the same for each resolutions
        # So we do the check once

        # Check conf inputs
        inputs = self.check_inputs(conf[INPUTS], config_dir=config_dir)

        # Check advanced parameters
        # TODO static method in the base class
        (
            _,
            advanced,
            self.geometry_plugin,
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
            _,
            self.scaling_coeff,
            _,
            _,
        ) = advanced_parameters.check_advanced_parameters(
            inputs, conf.get(ADVANCED, {}), check_epipolar_a_priori=True
        )

        # Check conf output
        (
            output,
            self.scaling_coeff,
        ) = self.check_output(conf[OUTPUT], self.scaling_coeff)

        resolutions = advanced["epipolar_resolutions"]
        if isinstance(resolutions, int):
            resolutions = [resolutions]

        if (
            (depth_cst.DEPTH_MAPS in inputs) or (dsm_cst.DSMS in inputs)
        ) and len(resolutions) != 1:
            raise RuntimeError(
                "For the use of those pipelines, "
                "you have to give only one resolution"
            )

        i = 0
        last_res = False
        for res in resolutions:

            if not isinstance(res, int) or res < 0:
                raise RuntimeError("The resolution has to be an int > 0")

            # Get the current key
            key = "resolution_" + str(res)

            # Choose the right default configuration regarding the resolution
            package_path = os.path.dirname(__file__)

            if i == 0:
                json_file = os.path.join(
                    package_path,
                    "..",
                    "conf_resolution",
                    "conf_first_resolution.json",
                )
            elif i == len(resolutions) - 1 or len(resolutions) == 1:
                json_file = os.path.join(
                    package_path,
                    "..",
                    "conf_resolution",
                    "conf_final_resolution.json",
                )
            else:
                json_file = os.path.join(
                    package_path,
                    "..",
                    "conf_resolution",
                    "conf_intermediate_resolution.json",
                )

            with open(json_file, "r", encoding="utf8") as fstream:
                resolution_config = json.load(fstream)

            self.used_conf[key] = {}

            # Check conf orchestrator
            self.used_conf[key][ORCHESTRATOR] = self.check_orchestrator(
                conf.get(ORCHESTRATOR, None)
            )

            # copy inputs
            self.used_conf[key][INPUTS] = copy.deepcopy(inputs)

            # Get ROI
            (
                self.input_roi_poly,
                self.input_roi_epsg,
            ) = roi_tools.generate_roi_poly_from_inputs(
                self.used_conf[key][INPUTS][sens_cst.ROI]
            )

            # Override the resolution
            self.used_conf[key][ADVANCED] = copy.deepcopy(advanced)
            self.used_conf[key][ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS] = res

            # Copy output
            self.used_conf[key][OUTPUT] = copy.deepcopy(output)

            # Get save intermediate data per res
            # If true we don't delete the resolution directory
            if isinstance(
                self.used_conf[key][ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA],
                dict,
            ):

                self.used_conf[key][ADVANCED][
                    adv_cst.SAVE_INTERMEDIATE_DATA
                ] = self.used_conf[key][ADVANCED][
                    adv_cst.SAVE_INTERMEDIATE_DATA
                ].get(
                    key, False
                )

            self.save_intermediate_data = self.used_conf[key][ADVANCED][
                adv_cst.SAVE_INTERMEDIATE_DATA
            ]

            self.keep_low_res_dir = self.used_conf[key][ADVANCED][
                adv_cst.KEEP_LOW_RES_DIR
            ]

            if i != len(resolutions) - 1:
                # Change output dir for lower resolution
                new_dir = (
                    self.used_conf[key][OUTPUT][out_cst.OUT_DIRECTORY]
                    + "/intermediate_res/out_res"
                    + str(res)
                )
                self.used_conf[key][OUTPUT][out_cst.OUT_DIRECTORY] = new_dir
                safe_makedirs(
                    self.used_conf[key][OUTPUT][out_cst.OUT_DIRECTORY]
                )

                # Put dems in auxiliary output
                self.used_conf[key][OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MAX
                ] = True
                self.used_conf[key][OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MIN
                ] = True
                self.used_conf[key][OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MEDIAN
                ] = True

                if not self.save_intermediate_data:
                    # For each resolutions we need to calculate the dsm
                    # (except the last one)
                    self.used_conf[key][OUTPUT][out_cst.PRODUCT_LEVEL] = "dsm"

                    # The idea is to calculate the less possible things
                    # So we override those parameters
                    self.used_conf[key][ADVANCED][adv_cst.MERGING] = False
                    self.used_conf[key][ADVANCED][adv_cst.PHASING] = None
                    self.used_conf[key][OUTPUT][out_cst.SAVE_BY_PAIR] = False

                    aux_items = self.used_conf[key][OUTPUT][
                        out_cst.AUXILIARY
                    ].items()
                    for aux_key, _ in aux_items:
                        if aux_key not in ("dem_min", "dem_max", "dem_median"):
                            self.used_conf[key][OUTPUT][out_cst.AUXILIARY][
                                aux_key
                            ] = False
                else:
                    # If save_intermediate_data is true,
                    # we save the depth_maps also to debug
                    self.used_conf[key][OUTPUT][out_cst.PRODUCT_LEVEL] = [
                        "dsm",
                        "depth_map",
                    ]
            else:
                # For the final res we don't change anything
                last_res = True

            prod_level = self.used_conf[key][OUTPUT][out_cst.PRODUCT_LEVEL]

            self.save_output_dsm = "dsm" in prod_level
            self.save_output_depth_map = "depth_map" in prod_level
            self.save_output_point_cloud = "point_cloud" in prod_level

            self.output_level_none = not (
                self.save_output_dsm
                or self.save_output_depth_map
                or self.save_output_point_cloud
            )
            self.sensors_in_inputs = (
                sens_cst.SENSORS in self.used_conf[key][INPUTS]
            )
            self.depth_maps_in_inputs = (
                depth_cst.DEPTH_MAPS in self.used_conf[key][INPUTS]
            )
            self.dsms_in_inputs = dsm_cst.DSMS in self.used_conf[key][INPUTS]

            self.merging = self.used_conf[key][ADVANCED][adv_cst.MERGING]

            self.phasing = self.used_conf[key][ADVANCED][adv_cst.PHASING]

            self.save_all_point_clouds_by_pair = self.used_conf[key][
                OUTPUT
            ].get(out_cst.SAVE_BY_PAIR, False)

            # Check conf application
            if APPLICATIONS in conf:
                if all(
                    "resolution_" + str(val) not in conf[APPLICATIONS]
                    for val in resolutions
                ):
                    application_all_conf = conf.get(APPLICATIONS, {})
                else:
                    self.check_application_keys_name(
                        resolutions, conf[APPLICATIONS]
                    )
                    application_all_conf = conf[APPLICATIONS].get(key, {})
            else:
                application_all_conf = {}

            application_all_conf = self.merge_resolution_conf(
                application_all_conf, resolution_config
            )

            application_conf = self.check_applications(
                application_all_conf, key, res, last_res
            )

            if (
                self.sensors_in_inputs
                and not self.depth_maps_in_inputs
                and not self.dsms_in_inputs
            ):
                # Check conf application vs inputs application
                application_conf = self.check_applications_with_inputs(
                    self.used_conf[key][INPUTS],
                    application_conf,
                    application_all_conf,
                    res,
                )

            self.used_conf[key][APPLICATIONS] = application_conf

            i += 1

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
            "ground_truth_reprojection": 6,
            "dense_matching": 8,
            "dense_match_filling.1": 9,
            "dense_match_filling.2": 10,
            "triangulation": 11,
            "point_cloud_outlier_removal.1": 12,
            "point_cloud_outlier_removal.2": 13,
        }

        depth_merge_apps = {
            "point_cloud_fusion": 14,
        }

        depth_to_dsm_apps = {
            "pc_denoising": 15,
            "point_cloud_rasterization": 16,
            "dem_generation": 17,
            "dsm_filling.1": 18,
            "dsm_filling.2": 19,
            "dsm_filling.3": 20,
            "auxiliary_filling": 21,
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
    def check_inputs(conf, config_dir=None):
        """
        Check the inputs given

        :param conf: configuration of inputs
        :type conf: dict
        :param config_dir: directory of used json, if
            user filled paths with relative paths
        :type config_dir: str

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
                conf, config_dir=config_dir
            )
        elif depth_cst.DEPTH_MAPS in conf:
            output_config = {
                **output_config,
                **depth_map_inputs.check_depth_maps_inputs(
                    conf, config_dir=config_dir
                ),
            }
        else:
            output_config = {
                **output_config,
                **dsm_inputs.check_dsm_inputs(conf, config_dir=config_dir),
            }
        return output_config

    @staticmethod
    def check_output(conf, scaling_coeff):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float

        :return overloader output
        :rtype : dict
        """
        return output_parameters.check_output_parameters(conf, scaling_coeff)

    def merge_resolution_conf(self, config1, config2):
        """
        Merge two configuration dict, generating a new configuration

        :param conf1: configuration
        :type conf1: dict
        :param conf2: configuration
        :type conf2: dict

        :return: merged conf
        :rtype: dict

        """

        merged_dict = config1.copy()

        _merge_resolution_conf_rec(merged_dict, config2)

        return merged_dict

    def check_application_keys_name(self, resolutions, applications_conf):
        """
        Check if the application name for each res match 'resolution_res'
        """

        for key_app, _ in applications_conf.items():
            if not key_app.startswith("resolution"):
                raise RuntimeError(
                    "If you decided to define an "
                    "application per resolution, "
                    "all the keys have to be : "
                    "resolution_res"
                )
            if int(key_app.split("_")[1]) not in resolutions:
                raise RuntimeError(
                    "This resolution "
                    + key_app.split("_")[1]
                    + " is not in the resolution list"
                )

    def check_applications(  # noqa: C901 : too complex
        self,
        conf,
        key=None,
        res=None,
        last_res=False,
    ):
        """
        Check the given configuration for applications,
        and generates needed applications for pipeline.

        :param conf: configuration of applications
        :type conf: dict
        """
        scaling_coeff = self.scaling_coeff

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
                    "dsm_filling.1",
                    "dsm_filling.2",
                    "dsm_filling.3",
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
                self.save_intermediate_data
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
        self.dense_matching_app = None
        self.triangulation_application = None
        self.dem_generation_application = None
        self.pc_denoising_application = None
        self.pc_outlier_removal_1_app = None
        self.pc_outlier_removal_2_app = None
        self.rasterization_application = None
        self.pc_fusion_application = None
        self.dsm_filling_1_application = None
        self.dsm_filling_2_application = None
        self.dsm_filling_3_application = None

        if self.sensors_in_inputs:
            # Epipolar grid generation
            self.epipolar_grid_generation_application = Application(
                "grid_generation",
                cfg=used_conf.get("grid_generation", {}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["grid_generation"] = (
                self.epipolar_grid_generation_application.get_conf()
            )

            # image resampling

            self.resampling_application = Application(
                "resampling",
                cfg=used_conf.get("resampling", {}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["resampling"] = self.resampling_application.get_conf()

            # ground truth disparity map computation
            if self.used_conf[key][ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
                used_conf["ground_truth_reprojection"][
                    "save_intermediate_data"
                ] = True

                if isinstance(
                    self.used_conf[key][ADVANCED][adv_cst.GROUND_TRUTH_DSM], str
                ):
                    self.used_conf[key][ADVANCED][adv_cst.GROUND_TRUTH_DSM] = {
                        "dsm": self.used_conf[key][ADVANCED][
                            adv_cst.GROUND_TRUTH_DSM
                        ]
                    }

                self.ground_truth_reprojection = Application(
                    "ground_truth_reprojection",
                    cfg=used_conf.get("ground_truth_reprojection", {}),
                    scaling_coeff=scaling_coeff,
                )
            # holes detection
            self.hole_detection_app = Application(
                "hole_detection",
                cfg=used_conf.get("hole_detection", {}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["hole_detection"] = self.hole_detection_app.get_conf()

            # disparity filling 1 plane
            self.dense_match_filling_1 = Application(
                "dense_match_filling",
                cfg=used_conf.get(
                    "dense_match_filling.1",
                    {"method": "plane"},
                ),
                scaling_coeff=scaling_coeff,
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
                scaling_coeff=scaling_coeff,
            )
            used_conf["dense_match_filling.2"] = (
                self.dense_match_filling_2.get_conf()
            )

            # Sparse Matching
            self.sparse_mtch_sift_app = Application(
                "sparse_matching",
                cfg=used_conf.get("sparse_matching.sift", {"method": "sift"}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["sparse_matching.sift"] = (
                self.sparse_mtch_sift_app.get_conf()
            )

            # Matching
            generate_performance_map = (
                self.used_conf[key][OUTPUT]
                .get(out_cst.AUXILIARY, {})
                .get(out_cst.AUX_PERFORMANCE_MAP, False)
            )
            generate_ambiguity = (
                self.used_conf[key][OUTPUT]
                .get(out_cst.AUXILIARY, {})
                .get(out_cst.AUX_AMBIGUITY, False)
            )
            dense_matching_config = used_conf.get("dense_matching", {})
            if generate_ambiguity is True:
                dense_matching_config["generate_ambiguity"] = True

            if (
                generate_performance_map is True
                and dense_matching_config.get("performance_map_method", None)
                is None
            ):
                dense_matching_config["performance_map_method"] = "risk"
            self.dense_matching_app = Application(
                "dense_matching",
                cfg=dense_matching_config,
                scaling_coeff=scaling_coeff,
            )
            used_conf["dense_matching"] = self.dense_matching_app.get_conf()

            if not last_res:
                used_conf["dense_matching"]["performance_map_method"] = [
                    "risk",
                    "intervals",
                ]

            # Triangulation
            self.triangulation_application = Application(
                "triangulation",
                cfg=used_conf.get("triangulation", {}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["triangulation"] = (
                self.triangulation_application.get_conf()
            )

            # MNT generation
            self.dem_generation_application = Application(
                "dem_generation",
                cfg=used_conf.get("dem_generation", {}),
                scaling_coeff=scaling_coeff,
            )

            height_margin = None
            if res >= 8 and "height_margin" not in used_conf["dem_generation"]:
                height_margin = [50, 250]

            used_conf["dem_generation"] = (
                self.dem_generation_application.get_conf()
            )

            if height_margin is not None:
                used_conf["dem_generation"]["height_margin"] = height_margin

            # Points cloud small component outlier removal
            if "point_cloud_outlier_removal.1" in used_conf:
                if "method" not in used_conf["point_cloud_outlier_removal.1"]:
                    used_conf["point_cloud_outlier_removal.1"][
                        "method"
                    ] = "small_components"

            self.pc_outlier_removal_1_app = Application(
                "point_cloud_outlier_removal",
                cfg=used_conf.get(
                    "point_cloud_outlier_removal.1",
                    {"method": "small_components"},
                ),
                scaling_coeff=scaling_coeff,
            )

            connection_val = None
            if (
                "connection_distance"
                not in used_conf["point_cloud_outlier_removal.1"]
            ):
                connection_val = (
                    self.pc_outlier_removal_1_app.connection_distance * res
                )

            used_conf["point_cloud_outlier_removal.1"] = (
                self.pc_outlier_removal_1_app.get_conf()
            )

            if connection_val is not None:
                used_conf["point_cloud_outlier_removal.1"][
                    "connection_distance"
                ] = connection_val

            # Points cloud statistical outlier removal
            self.pc_outlier_removal_2_app = Application(
                "point_cloud_outlier_removal",
                cfg=used_conf.get(
                    "point_cloud_outlier_removal.2",
                    {"method": "statistical"},
                ),
                scaling_coeff=scaling_coeff,
            )
            used_conf["point_cloud_outlier_removal.2"] = (
                self.pc_outlier_removal_2_app.get_conf()
            )

        if self.save_output_dsm or self.save_output_point_cloud:

            # Point cloud denoising
            self.pc_denoising_application = Application(
                "pc_denoising",
                cfg=used_conf.get("pc_denoising", {"method": "none"}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["pc_denoising"] = self.pc_denoising_application.get_conf()

            if self.save_output_dsm:

                # Rasterization
                self.rasterization_application = Application(
                    "point_cloud_rasterization",
                    cfg=used_conf.get("point_cloud_rasterization", {}),
                    scaling_coeff=scaling_coeff,
                )
                used_conf["point_cloud_rasterization"] = (
                    self.rasterization_application.get_conf()
                )
                # DSM filling 1 : Exogenous filling
                self.dsm_filling_1_application = Application(
                    "dsm_filling",
                    cfg=conf.get(
                        "dsm_filling.1",
                        {"method": "exogenous_filling"},
                    ),
                    scaling_coeff=scaling_coeff,
                )
                used_conf["dsm_filling.1"] = (
                    self.dsm_filling_1_application.get_conf()
                )
                # DSM filling 2 : Bulldozer
                self.dsm_filling_2_application = Application(
                    "dsm_filling",
                    cfg=conf.get(
                        "dsm_filling.2",
                        {"method": "bulldozer"},
                    ),
                )
                used_conf["dsm_filling.2"] = (
                    self.dsm_filling_2_application.get_conf()
                )
                # DSM filling 3 : Border interpolation
                self.dsm_filling_3_application = Application(
                    "dsm_filling",
                    cfg=conf.get(
                        "dsm_filling.3",
                        {"method": "border_interpolation"},
                    ),
                    scaling_coeff=scaling_coeff,
                )
                used_conf["dsm_filling.3"] = (
                    self.dsm_filling_3_application.get_conf()
                )
                # Auxiliary filling
                self.auxiliary_filling_application = Application(
                    "auxiliary_filling",
                    cfg=conf.get("auxiliary_filling", {}),
                    scaling_coeff=scaling_coeff,
                )
                used_conf["auxiliary_filling"] = (
                    self.auxiliary_filling_application.get_conf()
                )

            if self.merging:

                # Point cloud fusion
                self.pc_fusion_application = Application(
                    "point_cloud_fusion",
                    cfg=used_conf.get("point_cloud_fusion", {}),
                    scaling_coeff=scaling_coeff,
                )
                used_conf["point_cloud_fusion"] = (
                    self.pc_fusion_application.get_conf()
                )

        return used_conf

    def check_applications_with_inputs(  # noqa: C901 : too complex
        self, inputs_conf, application_conf, initial_conf_app, res
    ):
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

        if (
            application_conf["dem_generation"]["method"]
            == "bulldozer_on_raster"
        ):
            first_image_path = next(iter(inputs_conf["sensors"].values()))[
                "image"
            ]["main_file"]
            first_image_size = rasterio_get_size(first_image_path)
            first_image_nb_pixels = math.prod(first_image_size)
            dem_gen_used_mem = first_image_nb_pixels / 1e8
            if dem_gen_used_mem > 8:
                logging.warning(
                    "DEM generation method is 'bulldozer_on_raster'. "
                    f"This method can use up to {dem_gen_used_mem} Gb "
                    "of memory. If you think that it is too much for "
                    "your computer, you can re-lauch the run using "
                    "'dichotomic' method for DEM generation"
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
                                set(descriptions) | {"nodata"}
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
            img_left = inputs_conf["sensors"][key1]["image"]["main_file"]
            img_right = inputs_conf["sensors"][key2]["image"]["main_file"]
            bands_left = list(
                inputs_conf["sensors"][key1]["image"]["bands"].keys()
            )
            bands_right = list(
                inputs_conf["sensors"][key2]["image"]["bands"].keys()
            )
            classif_left = None
            classif_right = None
            if (
                "classification" in inputs_conf["sensors"][key1]
                and inputs_conf["sensors"][key1]["classification"] is not None
            ):
                classif_left = inputs_conf["sensors"][key1]["classification"][
                    "main_file"
                ]
            if (
                "classification" in inputs_conf["sensors"][key2]
                and inputs_conf["sensors"][key1]["classification"] is not None
            ):
                classif_right = inputs_conf["sensors"][key2]["classification"][
                    "main_file"
                ]
            self.dense_matching_app.corr_config = (
                self.dense_matching_app.loader.check_conf(
                    corr_cfg,
                    img_left,
                    img_right,
                    bands_left,
                    bands_right,
                    classif_left,
                    classif_right,
                )
            )

        # Change the step regarding the resolution
        # For the small resolution, the resampling perform better
        # with a small step
        # For the higher ones, a step at 30 should be better
        first_image_path = next(iter(inputs_conf["sensors"].values()))["image"][
            "main_file"
        ]
        first_image_size = rasterio_get_size(first_image_path)
        size_low_res_img_row = first_image_size[0] // res
        size_low_res_img_col = first_image_size[1] // res
        if (
            "grid_generation" not in initial_conf_app
            or "epi_step" not in initial_conf_app["grid_generation"]
        ):
            if size_low_res_img_row <= 900 and size_low_res_img_col <= 900:
                application_conf["grid_generation"]["epi_step"] = res * 5
            else:
                application_conf["grid_generation"]["epi_step"] = res * 30

        return application_conf

    def cleanup_low_res_dir(self):
        """
        Clean low res dir
        """

        items = list(self.used_conf.items())
        for _, conf_res in items[:-1]:
            out_dir = conf_res[OUTPUT][out_cst.OUT_DIRECTORY]
            if os.path.exists(out_dir) and os.path.isdir(out_dir):
                try:
                    shutil.rmtree(out_dir)
                    print(f"th directory {out_dir} has been cleaned.")
                except Exception as exception:
                    print(f"Error while deleting {out_dir}: {exception}")
            else:
                print(f"The directory {out_dir} has not been deleted")

    @cars_profile(name="run_dense_pipeline", interval=0.5)
    def run(self, args=None):  # noqa C901
        """
        Run pipeline

        """
        first_res_out_dir = None
        previous_out_dir = None
        generate_dems = True
        last_key = list(self.used_conf)[-1]
        final_out_dir = self.used_conf[last_key][OUTPUT][out_cst.OUT_DIRECTORY]
        use_sift_a_priori = False

        i = 0
        nb_res = len(list(self.used_conf.items()))
        for key, conf_res in self.used_conf.items():
            out_dir = conf_res[OUTPUT][out_cst.OUT_DIRECTORY]

            if nb_res != 1 and args is not None:
                # Logging configuration with args Loglevel
                loglevel = getattr(args, "loglevel", "PROGRESS").upper()

                cars_logging.setup_logging(
                    loglevel,
                    out_dir=os.path.join(out_dir, "logs"),
                    pipeline="",
                )

            if int(key.split("_")[-1]) != 1:
                cars_logging.add_progress_message(
                    "Starting pipeline for resolution 1/" + key.split("_")[-1]
                )
            else:
                cars_logging.add_progress_message(
                    "Starting pipeline for resolution 1"
                )

            # Get the resolution step
            if previous_out_dir is not None:
                if i == len(self.used_conf) - 1:
                    which_resolution = "final"
                    generate_dems = False
                else:
                    which_resolution = "intermediate"
            else:
                if len(self.used_conf) == 1:
                    which_resolution = "single"
                else:
                    which_resolution = "first"
                    first_res_out_dir = out_dir

            # Build a priori
            if which_resolution in ("final", "intermediate"):
                dem_min = os.path.join(previous_out_dir, "dsm/dem_min.tif")
                dem_max = os.path.join(previous_out_dir, "dsm/dem_max.tif")
                dem_median = os.path.join(
                    previous_out_dir, "dsm/dem_median.tif"
                )

                conf_res[ADVANCED][adv_cst.TERRAIN_A_PRIORI] = {
                    "dem_min": dem_min,
                    "dem_max": dem_max,
                    "dem_median": dem_median,
                }

                if conf_res[INPUTS][sens_cst.INITIAL_ELEVATION]["dem"] is None:
                    conf_res[INPUTS][sens_cst.INITIAL_ELEVATION] = dem_median
                else:
                    conf_res[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                        "dem_median"
                    ] = conf_res[INPUTS][sens_cst.INITIAL_ELEVATION]["dem"]

                conf_res[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI] = True
                use_sift_a_priori = True

            # start cars orchestrator
            with orchestrator.Orchestrator(
                orchestrator_conf=conf_res[ORCHESTRATOR],
                out_dir=out_dir,
                out_json_path=os.path.join(
                    out_dir,
                    out_cst.INFO_FILENAME,
                ),
            ) as self.cars_orchestrator:

                # initialize out_json
                self.cars_orchestrator.update_out_info({"version": __version__})

                used_pipeline = UnitPipeline(conf_res)

                used_pipeline.run(
                    self.cars_orchestrator,
                    generate_dems,
                    which_resolution,
                    use_sift_a_priori,
                    first_res_out_dir,
                    final_out_dir,
                )

            if nb_res != 1 and args is not None:
                # Generate summary of tasks
                log_wrapper.generate_summary(
                    out_dir, used_pipeline.used_conf, clean_worker_logs=True
                )

            previous_out_dir = out_dir
            i += 1

        if not self.keep_low_res_dir:
            self.cleanup_low_res_dir()
