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
import logging
import os

import numpy as np
from pyproj import CRS

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars import __version__

# CARS imports
from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_wrappers as dem_wrappers,
)
from cars.applications.grid_generation import grid_correction_app
from cars.applications.grid_generation.transform_grid import transform_grid_func
from cars.applications.point_cloud_fusion import (
    pc_fusion_algo,
    pc_fusion_wrappers,
)
from cars.core import preprocessing, projection, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import (
    get_descriptions_bands,
    rasterio_get_crs,
    rasterio_get_epsg,
    rasterio_get_size,
    read_vector,
)
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import application_parameters, depth_map_inputs
from cars.pipelines.parameters import depth_map_inputs_constants as depth_cst
from cars.pipelines.parameters import dsm_inputs
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.advanced_parameters_constants import (
    USE_ENDOGENOUS_DEM,
)
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate


@Pipeline.register(
    "unit",
)
class UnitPipeline(PipelineTemplate):
    """
    UnitPipeline
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

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_dir: path to dir containing json/yaml
        :type config_dir: str
        """

        # Used conf
        self.used_conf = {}
        # refined conf
        self.refined_conf = {}

        # Transform relative path to absolute path
        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)

        # Check global conf
        self.check_global_schema(conf)

        # Check conf orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )

        # Check conf inputs
        inputs = self.check_inputs(conf[INPUTS], config_dir=config_dir)
        self.used_conf[INPUTS] = inputs
        self.refined_conf[INPUTS] = copy.deepcopy(inputs)

        # Check advanced parameters
        # TODO static method in the base class
        output_dem_dir = os.path.join(
            conf[OUTPUT][out_cst.OUT_DIRECTORY], "dump_dir", "initial_elevation"
        )
        safe_makedirs(output_dem_dir)
        (
            inputs,
            advanced,
            self.geometry_plugin,
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
            self.scaling_coeff,
            self.land_cover_map,
            self.classification_to_config_mapping,
        ) = advanced_parameters.check_advanced_parameters(
            inputs,
            conf.get(ADVANCED, {}),
            check_epipolar_a_priori=True,
            output_dem_dir=output_dem_dir,
        )
        self.used_conf[ADVANCED] = advanced

        self.refined_conf[ADVANCED] = copy.deepcopy(advanced)
        # Refined conf: resolutions 1
        self.refined_conf[ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS] = [1]

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUTS][sens_cst.ROI]
        )

        self.debug_with_roi = self.used_conf[ADVANCED][adv_cst.DEBUG_WITH_ROI]

        # Check conf output
        (
            output,
            self.scaling_coeff,
        ) = self.check_output(conf[OUTPUT], self.scaling_coeff)

        self.used_conf[OUTPUT] = output
        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

        self.refined_conf[OUTPUT] = copy.deepcopy(output)

        prod_level = output[out_cst.PRODUCT_LEVEL]

        self.save_output_dsm = "dsm" in prod_level
        self.save_output_depth_map = "depth_map" in prod_level
        self.save_output_point_cloud = "point_cloud" in prod_level
        self.save_output_classif_for_filling = False

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

        if isinstance(
            self.used_conf[ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS], list
        ):
            if len(self.used_conf[ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS]) > 1:
                raise RuntimeError(
                    "For the unit pipeline, "
                    "the epipolar resolution has to "
                    "be a single value"
                )

            self.res_resamp = self.used_conf[ADVANCED][
                adv_cst.EPIPOLAR_RESOLUTIONS
            ][0]
        else:
            self.res_resamp = self.used_conf[ADVANCED][
                adv_cst.EPIPOLAR_RESOLUTIONS
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
                self.used_conf[INPUTS], application_conf, self.res_resamp
            )

        self.used_conf[APPLICATIONS] = application_conf

        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]

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
            "sparse_matching": 4,
            "ground_truth_reprojection": 6,
            "dense_matching": 8,
            "dense_match_filling": 9,
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
        :param config_dir: directory of used json/yaml, if
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

    def save_configurations(self):
        """
        Save used_conf and refined_conf configurations
        """

        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(self.out_dir, "current_res_used_conf.json"),
            safe_save=True,
        )
        cars_dataset.save_dict(
            self.refined_conf,
            os.path.join(self.out_dir, "refined_conf.json"),
            safe_save=True,
        )

    def check_output(self, conf, scaling_coeff):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float
        :return: overloader output
        :rtype: dict
        """
        return output_parameters.check_output_parameters(conf, scaling_coeff)

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
        scaling_coeff = self.scaling_coeff

        needed_applications = application_parameters.get_needed_apps(
            self.sensors_in_inputs,
            self.save_output_dsm,
            self.save_output_point_cloud,
            self.merging,
            conf,
        )

        # Check if all specified applications are used
        # Application in terrain_application are note used in
        # the sensors_to_dense_depth_maps pipeline
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

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            if used_conf[app_key] is None:
                continue
            used_conf[app_key]["save_intermediate_data"] = (
                self.save_all_intermediate_data
                or used_conf[app_key].get("save_intermediate_data", False)
            )

            if app_key == "auxiliary_filling":
                if used_conf[app_key] is not None:
                    used_conf[app_key]["activated"] = used_conf[app_key].get(
                        "activated", True
                    )

            if app_key in [
                "point_cloud_fusion",
                "pc_denoising",
            ]:
                used_conf[app_key]["save_by_pair"] = used_conf[app_key].get(
                    "save_by_pair", self.save_all_point_clouds_by_pair
                )

        self.epipolar_grid_generation_application = None
        self.resampling_application = None
        self.ground_truth_reprojection = None
        self.dense_match_filling = None
        self.sparse_mtch_app = None
        self.dense_matching_app = None
        self.triangulation_application = None
        self.dem_generation_application = None
        self.pc_denoising_application = None
        self.pc_outlier_removal_apps = {}
        self.rasterization_application = None
        self.pc_fusion_application = None
        self.dsm_filling_1_application = None
        self.dsm_filling_2_application = None
        self.dsm_filling_3_application = None
        self.dsm_filling_apps = {}

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
                    scaling_coeff=scaling_coeff,
                )

            # disparity filling
            self.dense_match_filling = Application(
                "dense_match_filling",
                cfg=used_conf.get(
                    "dense_match_filling",
                    {"method": "zero_padding"},
                ),
                scaling_coeff=scaling_coeff,
            )
            used_conf["dense_match_filling"] = (
                self.dense_match_filling.get_conf()
            )

            # Sparse Matching
            self.sparse_mtch_app = Application(
                "sparse_matching",
                cfg=used_conf.get("sparse_matching", {"method": "sift"}),
                scaling_coeff=scaling_coeff,
            )
            used_conf["sparse_matching"] = self.sparse_mtch_app.get_conf()

            # Matching
            generate_performance_map = (
                self.used_conf[OUTPUT]
                .get(out_cst.AUXILIARY, {})
                .get(out_cst.AUX_PERFORMANCE_MAP, False)
            )
            generate_ambiguity = (
                self.used_conf[OUTPUT]
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

            # particular case for some epipolar resolutions
            if not dense_matching_config:
                used_conf["dense_matching"]["performance_map_method"] = [
                    "risk",
                    "intervals",
                ]

            self.dense_matching_app = Application(
                "dense_matching",
                cfg=dense_matching_config,
                scaling_coeff=scaling_coeff,
            )
            used_conf["dense_matching"] = self.dense_matching_app.get_conf()

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
            used_conf["dem_generation"] = (
                self.dem_generation_application.get_conf()
            )

            for app_key, app_conf in used_conf.items():
                if not app_key.startswith("point_cloud_outlier_removal"):
                    continue

                if app_conf is None:
                    self.pc_outlier_removal_apps = {}
                    # keep over multiple runs
                    used_conf["point_cloud_outlier_removal"] = None
                    break

                if app_key in self.pc_outlier_removal_apps:
                    msg = (
                        f"The key {app_key} is defined twice in the input "
                        "configuration."
                    )
                    logging.error(msg)
                    raise NameError(msg)

                if app_key[27:] == ".1":
                    app_conf.setdefault("method", "small_components")
                if app_key[27:] == ".2":
                    app_conf.setdefault("method", "statistical")

                self.pc_outlier_removal_apps[app_key] = Application(
                    "point_cloud_outlier_removal",
                    cfg=app_conf,
                    scaling_coeff=scaling_coeff,
                )
                used_conf[app_key] = self.pc_outlier_removal_apps[
                    app_key
                ].get_conf()

            methods_str = "\n".join(
                f" - {k}={a.used_method}"
                for k, a in self.pc_outlier_removal_apps.items()
            )
            logging.info(
                f"{len(self.pc_outlier_removal_apps)} point cloud outlier "
                + f"removal apps registered:\n{methods_str}"
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

                for app_key, app_conf in used_conf.items():
                    if not app_key.startswith("dsm_filling"):
                        continue

                    if app_conf is None:
                        self.dsm_filling_apps = {}
                        # keep over multiple runs
                        used_conf["dsm_filling"] = None
                        break

                    if app_key in self.dsm_filling_apps:
                        msg = (
                            f"The key {app_key} is defined twice in the input "
                            "configuration."
                        )
                        logging.error(msg)
                        raise NameError(msg)

                    if app_key[11:] == ".1":
                        app_conf.setdefault("method", "exogenous_filling")
                    if app_key[11:] == ".2":
                        app_conf.setdefault("method", "bulldozer")
                    if app_key[11:] == ".3":
                        app_conf.setdefault("method", "border_interpolation")

                    self.dsm_filling_apps[app_key] = Application(
                        "dsm_filling",
                        cfg=app_conf,
                        scaling_coeff=scaling_coeff,
                    )
                    used_conf[app_key] = self.dsm_filling_apps[
                        app_key
                    ].get_conf()

                methods_str = "\n".join(
                    f" - {k}={a.used_method}"
                    for k, a in self.dsm_filling_apps.items()
                )
                logging.info(
                    f"{len(self.dsm_filling_apps)} dsm filling apps "
                    + f"registered:\n{methods_str}"
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

            if any(
                app_obj.classification != ["nodata"]
                for app_key, app_obj in self.dsm_filling_apps.items()
            ):
                self.save_output_classif_for_filling = True

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
        self, inputs_conf, application_conf, epipolar_resolution
    ):
        """
        Check for each application the input and output configuration
        consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        :param epipolar_resolution: epipolar resolution
        :type epipolar_resolution: int
        """

        initial_elevation = (
            inputs_conf[sens_cst.INITIAL_ELEVATION]["dem"] is not None
        )
        if self.sparse_mtch_app.elevation_delta_lower_bound is None:
            self.sparse_mtch_app.used_config["elevation_delta_lower_bound"] = (
                -500 if initial_elevation else -1000
            )
            self.sparse_mtch_app.elevation_delta_lower_bound = (
                self.sparse_mtch_app.used_config["elevation_delta_lower_bound"]
            )
        if self.sparse_mtch_app.elevation_delta_upper_bound is None:
            self.sparse_mtch_app.used_config["elevation_delta_upper_bound"] = (
                1000 if initial_elevation else 9000
            )
            self.sparse_mtch_app.elevation_delta_upper_bound = (
                self.sparse_mtch_app.used_config["elevation_delta_upper_bound"]
            )
        application_conf["sparse_matching"] = self.sparse_mtch_app.get_conf()

        # check classification application parameter compare
        # to each sensors inputs classification list
        for application_key in application_conf:
            if application_conf[application_key] is None:
                continue
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
                and inputs_conf["sensors"][key2]["classification"] is not None
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
        size_low_res_img_row = first_image_size[0] // epipolar_resolution
        size_low_res_img_col = first_image_size[1] // epipolar_resolution
        if epipolar_resolution > 1:
            if size_low_res_img_row <= 900 and size_low_res_img_col <= 900:
                application_conf["grid_generation"]["epi_step"] = (
                    epipolar_resolution * 5
                )
            else:
                application_conf["grid_generation"]["epi_step"] = (
                    epipolar_resolution * 30
                )

        return application_conf

    def generate_grid_correction_on_dem(self, pair_key, geo_plugin_on_dem):
        """
        Generate the epipolar grid correction for a given pair, using given dem
        """

        # Generate new grids with dem
        # Generate rectification grids
        (
            grid_left_new_dem,
            grid_right_new_dem,
        ) = self.epipolar_grid_generation_application.run(
            self.pairs[pair_key]["sensor_image_left"],
            self.pairs[pair_key]["sensor_image_right"],
            geo_plugin_on_dem,
            orchestrator=self.cars_orchestrator,
            pair_folder=os.path.join(
                self.dump_dir,
                "epipolar_grid_generation",
                "new_dem",
                pair_key,
            ),
            pair_key=pair_key,
        )

        if self.pairs[pair_key].get("sensor_matches_left", None) is None:
            logging.error(
                "No sensor matches available to compute grid correction"
            )
            return None

        # Generate new matches with new grids
        new_grid_matches_array = geo_plugin_on_dem.transform_matches_from_grids(
            self.pairs[pair_key]["sensor_matches_left"],
            self.pairs[pair_key]["sensor_matches_right"],
            grid_left_new_dem,
            grid_right_new_dem,
        )

        # Generate grid_correction
        # Compute grid correction
        (
            new_grid_correction_coef,
            _,
            _,
            _,
        ) = grid_correction_app.estimate_right_grid_correction(
            new_grid_matches_array,
            grid_right_new_dem,
            save_matches=False,
            minimum_nb_matches=0,
            pair_folder=os.path.join(
                self.dump_dir, "grid_correction", " new_dem", pair_key
            ),
            pair_key=pair_key,
            orchestrator=self.cars_orchestrator,
        )

        return new_grid_correction_coef

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

        save_matches = self.sparse_mtch_app.get_save_matches()

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

            if inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] is None:
                geom_plugin = self.geom_plugin_without_dem_and_geoid

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

            if self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] in (
                None,
                {},
            ):
                # Run resampling only if needed:
                # no a priori

                # Get required bands of first resampling
                required_bands = self.sparse_mtch_app.get_required_bands()

                # Run first epipolar resampling
                (
                    self.pairs[pair_key]["epipolar_image_left"],
                    self.pairs[pair_key]["epipolar_image_right"],
                ) = self.resampling_application.run(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["grid_left"],
                    self.pairs[pair_key]["grid_right"],
                    geom_plugin,
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "resampling", "initial", pair_key
                    ),
                    pair_key=pair_key,
                    margins_fun=self.sparse_mtch_app.get_margins_fun(),
                    tile_width=None,
                    tile_height=None,
                    required_bands=required_bands,
                )

                if self.quit_on_app("resampling"):
                    continue  # keep iterating over pairs, but don't go further

            if self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] in (
                None,
                {},
            ):
                # Run epipolar sparse_matching application
                (
                    self.pairs[pair_key]["epipolar_matches_left"],
                    _,
                ) = self.sparse_mtch_app.run(
                    self.pairs[pair_key]["epipolar_image_left"],
                    self.pairs[pair_key]["epipolar_image_right"],
                    self.pairs[pair_key]["grid_left"]["disp_to_alt_ratio"],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "sparse_matching", pair_key
                    ),
                    pair_key=pair_key,
                )

            # Run cluster breakpoint to compute sifts: force computation
            self.cars_orchestrator.breakpoint()

            minimum_nb_matches = self.sparse_mtch_app.get_minimum_nb_matches()

            # Run grid correction application
            if self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] in (None, {}):
                # Estimate grid correction if no epipolar a priori
                # Filter and save matches
                self.pairs[pair_key]["matches_array"] = (
                    self.sparse_mtch_app.filter_matches(
                        self.pairs[pair_key]["epipolar_matches_left"],
                        self.pairs[pair_key]["grid_left"],
                        self.pairs[pair_key]["grid_right"],
                        geom_plugin,
                        orchestrator=self.cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=os.path.join(
                            self.dump_dir, "sparse_matching", pair_key
                        ),
                        save_matches=(self.sparse_mtch_app.get_save_matches()),
                    )
                )

                # Compute grid correction
                (
                    self.pairs[pair_key]["grid_correction_coef"],
                    self.pairs[pair_key]["corrected_matches_array"],
                    _,
                    _,
                ) = grid_correction_app.estimate_right_grid_correction(
                    self.pairs[pair_key]["matches_array"],
                    self.pairs[pair_key]["grid_right"],
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
                    grid_correction_app.correct_grid(
                        self.pairs[pair_key]["grid_right"],
                        self.pairs[pair_key]["grid_correction_coef"],
                        os.path.join(
                            self.dump_dir,
                            "grid_correction",
                            "initial",
                            pair_key,
                        ),
                        save_corrected_grid,
                    )
                )

                self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                    pair_key
                ]["grid_left"]

                if self.quit_on_app("sparse_matching"):
                    continue

                # Shrink disparity intervals according to SIFT disparities
                disp_to_alt_ratio = self.pairs[pair_key]["grid_left"][
                    "disp_to_alt_ratio"
                ]
                disp_bounds_params = (
                    self.sparse_mtch_app.disparity_bounds_estimation
                )

                if disp_bounds_params["activated"]:
                    matches = self.pairs[pair_key]["matches_array"]
                    sift_disp = matches[:, 2] - matches[:, 0]
                    disp_min = np.percentile(
                        sift_disp, disp_bounds_params["percentile"]
                    )
                    disp_max = np.percentile(
                        sift_disp, 100 - disp_bounds_params["percentile"]
                    )
                    logging.info(
                        "Global disparity interval without margin : "
                        f"[{disp_min:.2f} pix, {disp_max:.2f} pix]"
                    )
                    disp_min -= (
                        disp_bounds_params["upper_margin"] / disp_to_alt_ratio
                    )
                    disp_max += (
                        disp_bounds_params["lower_margin"] / disp_to_alt_ratio
                    )
                    logging.info(
                        "Global disparity interval with margin : "
                        f"[{disp_min:.2f} pix, {disp_max:.2f} pix]"
                    )
                else:
                    disp_min = (
                        -self.sparse_mtch_app.elevation_delta_upper_bound
                        / disp_to_alt_ratio
                    )
                    disp_max = (
                        -self.sparse_mtch_app.elevation_delta_lower_bound
                        / disp_to_alt_ratio
                    )
                    logging.info(
                        "Global disparity interval : "
                        f"[{disp_min:.2f} pix, {disp_max:.2f} pix]"
                    )

                if self.epsg is None:
                    # compute epsg
                    # Epsg uses global disparity min and max
                    self.epsg = preprocessing.compute_epsg(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_left"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        disp_min=0,
                        disp_max=0,
                    )
                    # Compute roi polygon, in input EPSG
                    self.roi_poly = preprocessing.compute_roi_poly(
                        self.input_roi_poly, self.input_roi_epsg, self.epsg
                    )

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
            or self.quit_on_app("sparse_matching")
        ):
            return True

        if not self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] in (None, {}):
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

        for _, (pair_key, _, _) in enumerate(self.list_sensor_pairs):
            # Geometry plugin with dem will be used for the grid generation
            geom_plugin = self.geom_plugin_with_dem_and_geoid

            if self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] in (None, {}):
                save_matches = True

                (
                    self.pairs[pair_key]["sensor_matches_left"],
                    self.pairs[pair_key]["sensor_matches_right"],
                ) = geom_plugin.get_sensor_matches(
                    self.pairs[pair_key]["corrected_matches_array"],
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["corrected_grid_right"],
                    pair_folder=os.path.join(
                        self.out_dir, "dsm/sensor_matches", pair_key
                    ),
                    save_matches=save_matches,
                )
            elif (
                not self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI]
                in (None, {})
                and not self.use_sift_a_priori
            ):
                # Use epipolar a priori
                # load the disparity range
                if use_global_disp_range:
                    [dmin, dmax] = self.used_conf[ADVANCED][
                        adv_cst.EPIPOLAR_A_PRIORI
                    ][pair_key][adv_cst.DISPARITY_RANGE]

                    advanced_parameters.update_conf(
                        self.refined_conf,
                        dmin=dmin,
                        dmax=dmax,
                        pair_key=pair_key,
                    )
                else:

                    # load the grid correction coefficient
                    self.pairs[pair_key][
                        "grid_correction_coef"
                    ] = self.used_conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI][
                        pair_key
                    ][
                        adv_cst.GRID_CORRECTION
                    ]
                    self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                        pair_key
                    ]["grid_left"]
                    # no correction if the grid correction coefs are None
                    if self.pairs[pair_key]["grid_correction_coef"] is None:
                        self.pairs[pair_key]["corrected_grid_right"] = (
                            self.pairs[pair_key]["grid_right"]
                        )
                    else:
                        # Correct grid right with provided epipolar a priori
                        self.pairs[pair_key]["corrected_grid_right"] = (
                            grid_correction_app.correct_grid_from_1d(
                                self.pairs[pair_key]["grid_right"],
                                self.pairs[pair_key]["grid_correction_coef"],
                                save_corrected_grid,
                                os.path.join(
                                    self.dump_dir, "grid_correction", pair_key
                                ),
                            )
                        )
            else:
                # Correct grids with former matches
                # Transform matches to new grids

                save_matches = self.sparse_mtch_app.get_save_matches()

                self.sensor_matches_left = os.path.join(
                    self.first_res_out_dir,
                    "dsm/sensor_matches",
                    pair_key,
                    "sensor_matches_left.npy",
                )
                self.sensor_matches_right = os.path.join(
                    self.first_res_out_dir,
                    "dsm/sensor_matches",
                    pair_key,
                    "sensor_matches_right.npy",
                )

                self.pairs[pair_key]["sensor_matches_left"] = np.load(
                    self.sensor_matches_left
                )
                self.pairs[pair_key]["sensor_matches_right"] = np.load(
                    self.sensor_matches_right
                )

                new_grid_matches_array = (
                    geom_plugin.transform_matches_from_grids(
                        self.pairs[pair_key]["sensor_matches_left"],
                        self.pairs[pair_key]["sensor_matches_right"],
                        self.pairs[pair_key]["grid_left"],
                        self.pairs[pair_key]["grid_right"],
                    )
                )

                # Estimate grid_correction
                (
                    self.pairs[pair_key]["grid_correction_coef"],
                    self.pairs[pair_key]["corrected_matches_array"],
                    _,
                    _,
                ) = grid_correction_app.estimate_right_grid_correction(
                    new_grid_matches_array,
                    self.pairs[pair_key]["grid_right"],
                    save_matches=save_matches,
                    minimum_nb_matches=minimum_nb_matches,
                    pair_folder=os.path.join(
                        self.dump_dir, "grid_correction", "new", pair_key
                    ),
                    pair_key=pair_key,
                    orchestrator=self.cars_orchestrator,
                )

                # Correct grid right

                self.pairs[pair_key]["corrected_grid_right"] = (
                    grid_correction_app.correct_grid(
                        self.pairs[pair_key]["grid_right"],
                        self.pairs[pair_key]["grid_correction_coef"],
                        os.path.join(
                            self.dump_dir,
                            "grid_correction",
                            "new",
                            pair_key,
                        ),
                        save_corrected_grid,
                    )
                )

                # Use the new grid as uncorrected grid
                self.pairs[pair_key]["corrected_grid_left"] = self.pairs[
                    pair_key
                ]["grid_left"]

            # Run epipolar resampling
            self.pairs[pair_key]["corrected_grid_left"] = transform_grid_func(
                self.pairs[pair_key]["corrected_grid_left"],
                self.res_resamp,
            )

            self.pairs[pair_key]["corrected_grid_right"] = transform_grid_func(
                self.pairs[pair_key]["corrected_grid_right"],
                self.res_resamp,
                right=True,
            )

            # Update refined_conf configuration with epipolar a priori
            advanced_parameters.update_conf(
                self.refined_conf,
                grid_correction_coef=self.pairs[pair_key][
                    "grid_correction_coef"
                ],
                pair_key=pair_key,
                reference_dem=self.used_conf[INPUTS][
                    sens_cst.INITIAL_ELEVATION
                ][sens_cst.DEM_PATH],
            )
            # saved used configuration
            self.save_configurations()

            # Generate min and max disp grids
            # Global disparity min and max will be computed from
            # these grids
            dense_matching_pair_folder = os.path.join(
                self.dump_dir, "dense_matching", pair_key
            )

            if self.which_resolution in ("first", "single") and self.used_conf[
                ADVANCED
            ][adv_cst.TERRAIN_A_PRIORI] in (None, {}):
                dmin = disp_min / self.res_resamp
                dmax = disp_max / self.res_resamp
                # generate_disparity_grids runs orchestrator.breakpoint()
                self.pairs[pair_key]["disp_range_grid"] = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        dmin=dmin,
                        dmax=dmax,
                        pair_folder=dense_matching_pair_folder,
                        orchestrator=self.cars_orchestrator,
                    )
                )

                dsp_marg = self.sparse_mtch_app.get_disparity_margin()
                updating_infos = {
                    application_constants.APPLICATION_TAG: {
                        sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                            pair_key: {
                                sm_cst.DISPARITY_MARGIN_PARAM_TAG: dsp_marg,
                                sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                                sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                            }
                        }
                    }
                }
                self.cars_orchestrator.update_out_info(updating_infos)

                advanced_parameters.update_conf(
                    self.refined_conf,
                    dmin=dmin,
                    dmax=dmax,
                    pair_key=pair_key,
                )
            else:
                # Generate min and max disp grids from dems
                # generate_disparity_grids runs orchestrator.breakpoint()
                self.pairs[pair_key]["disp_range_grid"] = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        dem_min=dem_min,
                        dem_max=dem_max,
                        dem_median=dem_median,
                        pair_folder=dense_matching_pair_folder,
                        orchestrator=self.cars_orchestrator,
                    )
                )

                if use_global_disp_range:
                    # Generate min and max disp grids from constants
                    # sensor image is not used here
                    # TODO remove when only local diparity range will be used

                    if self.use_sift_a_priori:
                        dmin = self.pairs[pair_key]["disp_range_grid"][
                            "global_min"
                        ]
                        dmax = self.pairs[pair_key]["disp_range_grid"][
                            "global_max"
                        ]

                        # update orchestrator_out_json
                        marg = self.sparse_mtch_app.get_disparity_margin()
                        updating_infos = {
                            application_constants.APPLICATION_TAG: {
                                sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                                    pair_key: {
                                        sm_cst.DISPARITY_MARGIN_PARAM_TAG: marg,
                                        sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                                        sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                                    }
                                }
                            }
                        }
                        self.cars_orchestrator.update_out_info(updating_infos)

                        advanced_parameters.update_conf(
                            self.refined_conf,
                            dmin=dmin,
                            dmax=dmax,
                            pair_key=pair_key,
                        )

                    # generate_disparity_grids runs orchestrator.breakpoint()
                    self.pairs[pair_key]["disp_range_grid"] = (
                        self.dense_matching_app.generate_disparity_grids(
                            self.pairs[pair_key]["sensor_image_right"],
                            self.pairs[pair_key]["corrected_grid_right"],
                            self.geom_plugin_with_dem_and_geoid,
                            dmin=dmin,
                            dmax=dmax,
                            pair_folder=dense_matching_pair_folder,
                            orchestrator=self.cars_orchestrator,
                        )
                    )

            # TODO add in metadata.json max diff max - min
            # Update used_conf configuration with epipolar a priori
            # Add global min and max computed with grids
            advanced_parameters.update_conf(
                self.refined_conf,
                dmin=self.pairs[pair_key]["disp_range_grid"]["global_min"],
                dmax=self.pairs[pair_key]["disp_range_grid"]["global_max"],
                pair_key=pair_key,
            )
            advanced_parameters.update_conf(
                self.refined_conf,
                dmin=self.pairs[pair_key]["disp_range_grid"]["global_min"],
                dmax=self.pairs[pair_key]["disp_range_grid"]["global_max"],
                pair_key=pair_key,
            )

            # saved used configuration
            # saved used configuration
            self.save_configurations()

            # end of for loop, to finish computing disparity range grids

        for cloud_id, (pair_key, _, _) in enumerate(self.list_sensor_pairs):

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
                disp_min=self.pairs[pair_key]["disp_range_grid"]["global_min"],
                disp_max=self.pairs[pair_key]["disp_range_grid"]["global_max"],
            )

            # Generate new epipolar images
            # Generated with corrected grids
            # Optimal size is computed for the worst case scenario
            # found with epipolar disparity range grids

            (
                optimum_tile_size,
                local_tile_optimal_size_fun,
            ) = self.dense_matching_app.get_optimal_tile_size(
                self.pairs[pair_key]["disp_range_grid"],
                self.cars_orchestrator.cluster.checked_conf_cluster[
                    "max_ram_per_worker"
                ],
            )

            # Get required bands of third resampling
            required_bands = self.dense_matching_app.get_required_bands()

            # Add left required bands for texture
            required_bands["left"] = sorted(
                set(required_bands["left"]).union(set(self.texture_bands))
            )

            # Find index of texture band in left_dataset
            texture_bands_indices = [
                required_bands["left"].index(band)
                for band in self.texture_bands
            ]

            # Get margins used in dense matching,
            dense_matching_margins_fun = (
                self.dense_matching_app.get_margins_fun(
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["disp_range_grid"],
                )
            )

            # Quick fix to reduce memory usage
            if self.res_resamp >= 16:
                optimum_tile_size = 200

            # Run third epipolar resampling
            (
                new_epipolar_image_left,
                new_epipolar_image_right,
            ) = self.resampling_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                self.pairs[pair_key]["corrected_grid_left"],
                self.pairs[pair_key]["corrected_grid_right"],
                geom_plugin,
                orchestrator=self.cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir, "resampling", "corrected_grid", pair_key
                ),
                pair_key=pair_key,
                margins_fun=dense_matching_margins_fun,
                tile_width=optimum_tile_size,
                tile_height=optimum_tile_size,
                add_classif=True,
                epipolar_roi=epipolar_roi,
                resolution=self.res_resamp,
                required_bands=required_bands,
                texture_bands=self.texture_bands,
            )

            # Run ground truth dsm computation
            if self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
                self.used_conf["applications"]["ground_truth_reprojection"][
                    "save_intermediate_data"
                ] = True
                new_geomplugin_dsm = AbstractGeometry(  # pylint: disable=E0110
                    self.geometry_plugin,
                    dem=self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_GROUND_TRUTH_DSM
                    ],
                    geoid=self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_GEOID
                    ],
                    scaling_coeff=self.scaling_coeff,
                )
                self.ground_truth_reprojection.run(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["corrected_grid_right"],
                    new_geomplugin_dsm,
                    self.geom_plugin_with_dem_and_geoid,
                    self.pairs[pair_key]["corrected_grid_left"][
                        "disp_to_alt_ratio"
                    ],
                    self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_AUX_PATH
                    ],
                    self.used_conf[ADVANCED][adv_cst.GROUND_TRUTH_DSM][
                        adv_cst.INPUT_AUX_INTERP
                    ],
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "ground_truth_reprojection", pair_key
                    ),
                )

            if self.epsg is None:
                # compute epsg
                # Epsg uses global disparity min and max
                self.epsg = preprocessing.compute_epsg(
                    self.pairs[pair_key]["sensor_image_left"],
                    self.pairs[pair_key]["sensor_image_right"],
                    self.pairs[pair_key]["corrected_grid_left"],
                    self.pairs[pair_key]["corrected_grid_right"],
                    self.geom_plugin_with_dem_and_geoid,
                    disp_min=self.pairs[pair_key]["disp_range_grid"][
                        "global_min"
                    ],
                    disp_max=self.pairs[pair_key]["disp_range_grid"][
                        "global_max"
                    ],
                )
                # Compute roi polygon, in input EPSG
                self.roi_poly = preprocessing.compute_roi_poly(
                    self.input_roi_poly, self.input_roi_epsg, self.epsg
                )

            self.vertical_crs = projection.get_output_crs(self.epsg, output)

            if (
                self.save_output_dsm
                or self.save_output_point_cloud
                or self.dense_matching_app.get_method() == "auto"
            ):
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
                        disp_min=self.pairs[pair_key]["disp_range_grid"][
                            "global_min"
                        ],
                        disp_max=self.pairs[pair_key]["disp_range_grid"][
                            "global_max"
                        ],
                        roi_poly=(
                            None if self.debug_with_roi else self.roi_poly
                        ),
                        orchestrator=self.cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=os.path.join(
                            self.dump_dir, "terrain_bbox", pair_key
                        ),
                        check_inputs=False,
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

                if self.which_resolution not in ("final", "single"):
                    self.terrain_bounds = dem_wrappers.modify_terrain_bounds(
                        self.terrain_bounds,
                        self.dem_generation_application.margin[0],
                        self.dem_generation_application.margin[1],
                    )

            if self.dense_matching_app.get_method() == "auto":
                # Copy the initial corr_config in order to keep
                # the inputs that have already been checked
                corr_cfg = self.dense_matching_app.corr_config.copy()

                # Find the conf that correspond to the land cover map
                conf = self.dense_matching_app.loader.find_auto_conf(
                    intersection_poly,
                    self.land_cover_map,
                    self.classification_to_config_mapping,
                    self.epsg,
                )

                # Update the used_conf if order to reinitialize
                # the dense matching app
                # Because we kept the information regarding the ambiguity,
                # performance_map calculus..
                self.used_conf["applications"]["dense_matching"]["loader_conf"][
                    "pipeline"
                ] = conf["pipeline"]

                # Re initialization of the dense matching application
                self.dense_matching_app = Application(
                    "dense_matching",
                    cfg=self.used_conf["applications"]["dense_matching"],
                )

                # Update the corr_config with the inputs that have
                # already been checked
                self.dense_matching_app.corr_config["input"] = corr_cfg["input"]

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
                disp_range_grid=self.pairs[pair_key]["disp_range_grid"],
                compute_disparity_masks=False,
                margins_to_keep=sum(
                    app.get_epipolar_margin()
                    for _, app in self.pc_outlier_removal_apps.items()
                ),
                texture_bands=texture_bands_indices,
            )

            if self.quit_on_app("dense_matching"):
                continue  # keep iterating over pairs, but don't go further

            # Fill with zeros
            (filled_epipolar_disparity_map) = self.dense_match_filling.run(
                epipolar_disparity_map,
                orchestrator=self.cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir, "dense_match_filling", pair_key
                ),
                pair_key=pair_key,
            )

            if self.quit_on_app("dense_match_filling"):
                continue  # keep iterating over pairs, but don't go further

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

            triangulation_point_cloud_dir = (
                point_cloud_dir
                if (
                    point_cloud_dir
                    and len(self.pc_outlier_removal_apps) == 0
                    and self.merging is False
                )
                else None
            )

            # Run epipolar triangulation application
            epipolar_point_cloud = self.triangulation_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                self.pairs[pair_key]["corrected_grid_left"],
                self.pairs[pair_key]["corrected_grid_right"],
                filled_epipolar_disparity_map,
                self.geom_plugin_without_dem_and_geoid,
                new_epipolar_image_left,
                epsg=self.epsg,
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
                performance_maps_param=(
                    self.dense_matching_app.get_performance_map_parameters()
                ),
                depth_map_dir=depth_map_dir,
                point_cloud_dir=triangulation_point_cloud_dir,
                save_output_coordinates=(len(self.pc_outlier_removal_apps) == 0)
                and (
                    self.save_output_depth_map or self.save_output_point_cloud
                ),
                save_output_color=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_TEXTURE],
                save_output_classification=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_CLASSIFICATION],
                save_output_filling=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_FILLING],
                save_output_mask=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_MASK],
                save_output_performance_map=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_PERFORMANCE_MAP],
                save_output_ambiguity=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_AMBIGUITY],
            )

            if self.quit_on_app("triangulation"):
                continue  # keep iterating over pairs, but don't go further

            if self.merging:
                self.list_epipolar_point_clouds.append(epipolar_point_cloud)
            else:

                filtered_epipolar_point_cloud = epipolar_point_cloud
                for app_key, app in self.pc_outlier_removal_apps.items():

                    app_key_is_last = (
                        app_key == list(self.pc_outlier_removal_apps)[-1]
                    )
                    filtering_depth_map_dir = (
                        depth_map_dir if app_key_is_last else None
                    )
                    filtering_point_cloud_dir = (
                        point_cloud_dir if app_key_is_last else None
                    )

                    filtered_epipolar_point_cloud = app.run(
                        filtered_epipolar_point_cloud,
                        depth_map_dir=filtering_depth_map_dir,
                        point_cloud_dir=filtering_point_cloud_dir,
                        dump_dir=os.path.join(
                            self.dump_dir,
                            (
                                "pc_outlier_removal"
                                f"{str(app_key[27:]).replace('.', '_')}"
                            ),
                            pair_key,
                        ),
                        epsg=self.epsg,
                        orchestrator=self.cars_orchestrator,
                    )
                if self.quit_on_app("point_cloud_outlier_removal"):
                    continue  # keep iterating over pairs, but don't go further

                # denoising available only if we'll go further in the pipeline
                if self.save_output_dsm or self.save_output_point_cloud:
                    denoised_epipolar_point_clouds = (
                        self.pc_denoising_application.run(
                            filtered_epipolar_point_cloud,
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

        # quit if any app in the loop over the pairs was the last one
        # pylint:disable=too-many-boolean-expressions
        if (
            self.quit_on_app("dense_matching")
            or self.quit_on_app("dense_match_filling")
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

        self.rasterization_dump_dir = os.path.join(
            self.dump_dir, "rasterization"
        )

        dsm_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "dsm.tif",
            )
            if self.save_output_dsm
            else None
        )

        weights_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "weights.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_WEIGHTS]
            else None
        )

        color_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "texture.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_TEXTURE]
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

        ambiguity_file_name = (
            os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "ambiguity.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_AMBIGUITY]
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

        if classif_file_name is None and self.save_output_classif_for_filling:
            classif_file_name = os.path.join(
                self.rasterization_dump_dir,
                "classification_for_filling.tif",
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
            self.vertical_crs,
            resolution=self.resolution,
            orchestrator=self.cars_orchestrator,
            dsm_file_name=dsm_file_name,
            weights_file_name=weights_file_name,
            color_file_name=color_file_name,
            classif_file_name=classif_file_name,
            performance_map_file_name=performance_map_file_name,
            ambiguity_file_name=ambiguity_file_name,
            mask_file_name=mask_file_name,
            contributing_pair_file_name=contributing_pair_file_name,
            filling_file_name=filling_file_name,
            color_dtype=self.color_type,
            dump_dir=self.rasterization_dump_dir,
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

        if self.generate_dems:
            dsm_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "dsm.tif",
                )
                if self.save_output_dsm
                else None
            )

            dem_min_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "dem_min.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MIN
                ]
                else None
            )

            dem_max_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "dem_max.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MAX
                ]
                else None
            )

            dem_median_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "dem_median.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_DEM_MEDIAN
                ]
                else None
            )

            dem_generation_output_dir = os.path.join(
                self.dump_dir, "dem_generation"
            )
            safe_makedirs(dem_generation_output_dir)
            if not self.dem_generation_application.used_config[
                "save_intermediate_data"
            ]:
                self.cars_orchestrator.add_to_clean(dem_generation_output_dir)

            # Use initial elevation if provided, and generate dems
            _, paths, _ = self.dem_generation_application.run(
                dsm_file_name,
                dem_generation_output_dir,
                dem_min_file_name,
                dem_max_file_name,
                dem_median_file_name,
                input_geoid=self.used_conf[INPUTS][sens_cst.INITIAL_ELEVATION][
                    sens_cst.GEOID
                ],
                output_geoid=self.used_conf[OUTPUT][out_cst.OUT_GEOID],
                initial_elevation=(
                    self.used_conf[INPUTS][sens_cst.INITIAL_ELEVATION][
                        sens_cst.DEM_PATH
                    ]
                ),
                default_alt=self.geom_plugin_with_dem_and_geoid.default_alt,
                cars_orchestrator=self.cars_orchestrator,
            )

            # Update refined conf configuration with dem paths
            dem_median = paths["dem_median"]
            dem_min = paths["dem_min"]
            dem_max = paths["dem_max"]

            advanced_parameters.update_conf(
                self.refined_conf,
                dem_median=dem_median,
                dem_min=dem_min,
                dem_max=dem_max,
            )

            if self.used_conf[ADVANCED][USE_ENDOGENOUS_DEM]:
                # Generate new geom plugin with dem
                output_dem_dir = os.path.join(
                    self.dump_dir, "initial_elevation"
                )
                new_geom_plugin = (
                    sensor_inputs.generate_geometry_plugin_with_dem(
                        self.geometry_plugin,
                        self.used_conf[INPUTS],
                        dem=dem_median,
                        output_dem_dir=output_dem_dir,
                    )
                )

                for (
                    pair_key,
                    _,
                    _,
                ) in self.list_sensor_pairs:
                    new_grid_correction_coef = (
                        self.generate_grid_correction_on_dem(
                            pair_key,
                            new_geom_plugin,
                        )
                    )
                    if new_grid_correction_coef is not None:
                        # Update refined_conf configuration with epipolar
                        # a priori
                        advanced_parameters.update_conf(
                            self.refined_conf,
                            grid_correction_coef=new_grid_correction_coef,
                            pair_key=pair_key,
                            reference_dem=dem_median,
                        )

            # saved used configuration
            self.save_configurations()

        return False

    def filling(self):  # noqa: C901 : too complex
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
                        if isinstance(dsm_dict[key][path_name], str):
                            if path_name not in dict_path:
                                dict_path[path_name] = [
                                    dsm_dict[key][path_name]
                                ]
                            else:
                                dict_path[path_name].append(
                                    dsm_dict[key][path_name]
                                )

            color_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "texture.tif",
                )
                if "texture" in dict_path
                or self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_TEXTURE
                ]
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

            ambiguity_bool = any("ambiguity" in key for key in dict_path)
            ambiguity_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                )
                if ambiguity_bool
                else None
            )

            classif_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "classification.tif",
                )
                if "classification" in dict_path
                or self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_CLASSIFICATION
                ]
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
            self.vertical_crs = rasterio_get_crs(dict_path["dsm"][0])

            # Compute roi polygon, in input EPSG
            self.roi_poly = preprocessing.compute_roi_poly(
                self.input_roi_poly, self.input_roi_epsg, self.epsg
            )

            _ = dsm_inputs.merge_dsm_infos(
                dict_path,
                self.cars_orchestrator,
                self.roi_poly,
                self.used_conf[ADVANCED][adv_cst.DSM_MERGING_TILE_SIZE],
                dsms_merging_dump_dir,
                dsm_file_name,
                color_file_name,
                classif_file_name,
                filling_file_name,
                performance_map_file_name,
                ambiguity_file_name,
                mask_file_name,
                contributing_all_pair_file_name,
            )

            # dsm needs to be saved before filling
            self.cars_orchestrator.breakpoint()
        else:
            color_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "texture.tif",
                )
                if self.save_output_dsm
                and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_TEXTURE
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

            if (
                classif_file_name is None
                and self.save_output_classif_for_filling
            ):
                classif_file_name = os.path.join(
                    self.rasterization_dump_dir,
                    "classification_for_filling.tif",
                )

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

        if not hasattr(self, "list_intersection_poly"):
            if (
                self.used_conf[INPUTS][sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ]
                is not None
                and self.sensors_in_inputs
            ):
                self.list_sensor_pairs = sensor_inputs.generate_inputs(
                    self.used_conf[INPUTS],
                    self.geom_plugin_without_dem_and_geoid,
                )

                self.list_intersection_poly = []
                for _, (
                    pair_key,
                    sensor_image_left,
                    sensor_image_right,
                ) in enumerate(self.list_sensor_pairs):
                    pair_folder = os.path.join(
                        self.dump_dir, "terrain_bbox", pair_key
                    )
                    safe_makedirs(pair_folder)
                    geojson1 = os.path.join(
                        pair_folder, "left_envelope.geojson"
                    )
                    geojson2 = os.path.join(
                        pair_folder, "right_envelope.geojson"
                    )
                    out_envelopes_intersection = os.path.join(
                        pair_folder, "envelopes_intersection.geojson"
                    )

                    inter_poly, _ = projection.ground_intersection_envelopes(
                        sensor_image_left[sens_cst.INPUT_IMG][
                            sens_cst.MAIN_FILE
                        ],
                        sensor_image_right[sens_cst.INPUT_IMG][
                            sens_cst.MAIN_FILE
                        ],
                        sensor_image_left[sens_cst.INPUT_GEO_MODEL],
                        sensor_image_right[sens_cst.INPUT_GEO_MODEL],
                        self.geom_plugin_with_dem_and_geoid,
                        geojson1,
                        geojson2,
                        out_envelopes_intersection,
                        envelope_file_driver="GeoJSON",
                        intersect_file_driver="GeoJSON",
                    )

                    # Retrieve bounding box of the grd inters of the envelopes
                    inter_poly, inter_epsg = read_vector(
                        out_envelopes_intersection
                    )

                    # Project polygon if epsg is different
                    if self.vertical_crs != CRS(inter_epsg):
                        inter_poly = projection.polygon_projection_crs(
                            inter_poly, CRS(inter_epsg), self.vertical_crs
                        )

                self.list_intersection_poly.append(inter_poly)
            else:
                self.list_intersection_poly = None

        dtm_file_name = None
        for app_key, app in self.dsm_filling_apps.items():

            app_dump_dir = os.path.join(
                self.dump_dir, app_key.replace(".", "_")
            )

            if app.get_conf()["method"] == "exogenous_filling":
                _ = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                    output_geoid=self.used_conf[OUTPUT][sens_cst.GEOID],
                    geom_plugin=self.geom_plugin_with_dem_and_geoid,
                )
            elif app.get_conf()["method"] == "bulldozer":
                dtm_file_name = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                    orchestrator=self.cars_orchestrator,
                )
            elif app.get_conf()["method"] == "border_interpolation":
                _ = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dtm_file=dtm_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                )

            if not app.save_intermediate_data:
                self.cars_orchestrator.add_to_clean(app_dump_dir)

        _ = self.auxiliary_filling_application.run(
            dsm_file=dsm_file_name,
            color_file=color_file_name,
            classif_file=classif_file_name,
            dump_dir=self.dump_dir,
            roi_epsg=self.epsg,
            sensor_inputs=self.used_conf[INPUTS].get("sensors"),
            pairing=self.used_conf[INPUTS].get("pairing"),
            geom_plugin=self.geom_plugin_with_dem_and_geoid,
            texture_bands=self.texture_bands,
            orchestrator=self.cars_orchestrator,
        )
        self.cars_orchestrator.breakpoint()

        return self.quit_on_app("auxiliary_filling")

    @cars_profile(name="Preprocess depth maps", interval=0.5)
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
        # get epsg
        self.epsg = self.used_conf[OUTPUT][out_cst.EPSG]

        output_parameters.intialize_product_index(
            self.cars_orchestrator,
            self.used_conf[OUTPUT]["product_level"],
            self.used_conf[INPUTS][depth_cst.DEPTH_MAPS].keys(),
        )

        # compute epsg
        epsg_cloud = pc_fusion_wrappers.compute_epsg_from_point_cloud(
            self.used_conf[INPUTS][depth_cst.DEPTH_MAPS]
        )
        if self.epsg is None:
            self.epsg = epsg_cloud

        self.vertical_crs = projection.get_output_crs(
            self.epsg, self.used_conf[OUTPUT]
        )

        self.resolution = self.used_conf[OUTPUT][out_cst.RESOLUTION]

        # Compute roi polygon, in input EPSG
        self.roi_poly = preprocessing.compute_roi_poly(
            self.input_roi_poly, self.input_roi_epsg, self.epsg
        )

        if not self.merging:
            # compute bounds
            self.terrain_bounds = pc_fusion_wrappers.get_bounds(
                self.used_conf[INPUTS][depth_cst.DEPTH_MAPS],
                self.epsg,
                roi_poly=self.roi_poly,
            )

            self.list_epipolar_point_clouds = (
                pc_fusion_algo.generate_point_clouds(
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
            ) = pc_fusion_algo.transform_input_pc(
                self.used_conf[INPUTS][depth_cst.DEPTH_MAPS],
                self.epsg,
                roi_poly=self.roi_poly,
                epipolar_tile_size=1000,  # TODO change it
                orchestrator=self.cars_orchestrator,
            )

            # Compute number of superposing point cloud for density
            max_number_superposing_point_clouds = (
                pc_fusion_wrappers.compute_max_nb_point_clouds(
                    self.list_epipolar_point_clouds
                )
            )

            # Compute average distance between two points
            average_distance_point_cloud = (
                pc_fusion_wrappers.compute_average_distance(
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

    @cars_profile(name="Final cleanup", interval=0.5)
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
                    if app is not None
                )
                and not self.dsms_in_inputs
            ):
                self.cars_orchestrator.add_to_clean(self.dump_dir)

    @cars_profile(name="run_unit_pipeline", interval=0.5)
    def run(
        self,
        generate_dems=False,
        which_resolution="single",
        use_sift_a_priori=False,
        first_res_out_dir=None,
        log_dir=None,
    ):  # noqa C901
        """
        Run pipeline

        """
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(self.out_dir, "logs")

        self.first_res_out_dir = first_res_out_dir
        self.texture_bands = self.used_conf[ADVANCED][adv_cst.TEXTURE_BANDS]

        self.auxiliary = self.used_conf[OUTPUT][out_cst.AUXILIARY]

        self.use_sift_a_priori = use_sift_a_priori

        self.generate_dems = generate_dems

        self.which_resolution = which_resolution

        # saved used configuration
        self.save_configurations()
        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=self.log_dir,
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
