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
# pylint: disable=C0302
"""
CARS surface modeling pipeline class file
"""
# Standard imports
from __future__ import print_function

import copy
import logging
import os
import warnings
from collections import OrderedDict

import numpy as np
import rasterio
from json_checker import Checker, OptionalKey
from rasterio.errors import NodataShadowWarning

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars import __version__

# CARS imports
from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_wrappers as dem_wrappers,
)
from cars.core import preprocessing, projection, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import get_descriptions_bands
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import (
    advanced_parameters,
)
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import (
    application_parameters,
)
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import (
    output_parameters,
    sensor_inputs,
)
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.advanced_parameters_constants import (
    USE_ENDOGENOUS_DEM,
)
from cars.pipelines.parameters.output_constants import AUXILIARY
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
    TIE_POINTS,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.tie_points.tie_points import TiePointsPipeline

PIPELINE = "surface_modeling"


@Pipeline.register(PIPELINE)
class SurfaceModelingPipeline(PipelineTemplate):
    """
    SurfaceModelingPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        conf,
        config_dir=None,
    ):  # noqa: C901
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
        self.config_dir = config_dir

        # Check global conf
        self.check_global_schema(conf)

        if PIPELINE in conf:
            self.check_pipeline_conf(conf)

        self.out_dir = conf[OUTPUT][out_cst.OUT_DIRECTORY]

        # Check conf orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )

        # Check conf inputs
        inputs = self.check_inputs(conf[INPUT], config_dir=config_dir)
        self.used_conf[INPUT] = inputs
        self.refined_conf[INPUT] = copy.deepcopy(inputs)

        self.dem_scaling_coeff = None
        if inputs[sens_cst.LOW_RES_DSM] is not None:
            low_res_dsm = rasterio.open(inputs[sens_cst.LOW_RES_DSM])
            self.dem_scaling_coeff = np.mean(low_res_dsm.res) * 2

        # Init tie points pipelines
        self.tie_points_pipelines = {}
        for key1, key2 in inputs[sens_cst.PAIRING]:
            pair_key = key1 + "_" + key2
            tie_points_config = {}
            tie_points_input = {}
            tie_points_input[sens_cst.SENSORS] = {
                key1: inputs[sens_cst.SENSORS][key1],
                key2: inputs[sens_cst.SENSORS][key2],
            }
            tie_points_input[sens_cst.PAIRING] = [[key1, key2]]
            tie_points_input[sens_cst.LOADERS] = inputs[sens_cst.LOADERS]
            tie_points_input[sens_cst.INITIAL_ELEVATION] = inputs[
                sens_cst.INITIAL_ELEVATION
            ]
            tie_points_config[INPUT] = tie_points_input
            tie_points_output = os.path.join(self.out_dir, TIE_POINTS, pair_key)
            tie_points_config["output"] = {"directory": tie_points_output}
            if TIE_POINTS in conf:
                tie_points_config[TIE_POINTS] = conf[TIE_POINTS]
            self.tie_points_pipelines[pair_key] = TiePointsPipeline(
                tie_points_config,
                config_dir=self.config_dir,
            )
            self.used_conf[TIE_POINTS] = {}
            self.used_conf[TIE_POINTS][APPLICATIONS] = (
                self.tie_points_pipelines[pair_key].used_conf[TIE_POINTS][
                    APPLICATIONS
                ]
            )
            self.used_conf[TIE_POINTS][ADVANCED] = self.tie_points_pipelines[
                pair_key
            ].used_conf[TIE_POINTS][ADVANCED]

        # Check advanced parameters
        # TODO static method in the base class
        output_dem_dir = os.path.join(
            self.out_dir, "dump_dir", "initial_elevation"
        )
        pipeline_conf = conf.get(PIPELINE, {})
        self.used_conf[PIPELINE] = {}
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
            pipeline_conf.get(ADVANCED, {}),
            output_dem_dir=output_dem_dir,
        )

        self.used_conf[PIPELINE][ADVANCED] = advanced

        self.refined_conf[ADVANCED] = copy.deepcopy(advanced)
        # Refined conf: resolutions 1
        self.refined_conf[ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS] = [1]

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUT][sens_cst.ROI]
        )

        self.debug_with_roi = self.used_conf[PIPELINE][ADVANCED][
            adv_cst.DEBUG_WITH_ROI
        ]

        # Check conf output
        (
            output,
            self.scaling_coeff,
        ) = self.check_output(inputs, conf[OUTPUT], self.scaling_coeff)

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

        # Used classification values, for filling -> will be masked
        self.used_classif_values_for_filling = self.get_classif_values_filling(
            self.used_conf[INPUT]
        )

        self.phasing = self.used_conf[PIPELINE][ADVANCED][adv_cst.PHASING]

        self.compute_depth_map = not self.output_level_none

        if self.output_level_none:
            self.infer_conditions_from_applications(conf[PIPELINE])

        self.save_all_intermediate_data = self.used_conf[PIPELINE][ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]

        self.save_all_point_clouds_by_pair = self.used_conf[OUTPUT].get(
            out_cst.SAVE_BY_PAIR, False
        )

        # Check conf application
        application_conf = self.check_applications(
            pipeline_conf.get(APPLICATIONS, {})
        )

        # Check conf application vs inputs application
        application_conf = self.check_applications_with_inputs(
            self.used_conf[INPUT], application_conf
        )

        self.used_conf[PIPELINE][APPLICATIONS] = application_conf

    def check_pipeline_conf(self, conf):
        """
        Check pipeline configuration
        """
        # Validate inputs
        pipeline_schema = {
            OptionalKey(ADVANCED): dict,
            OptionalKey(APPLICATIONS): dict,
        }

        checker_inputs = Checker(pipeline_schema)
        checker_inputs.validate(conf[PIPELINE])

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
            "dem_generation": 1,
            "grid_generation": 2,
            "grid_correction": 3,
            "resampling": 4,
            "ground_truth_reprojection": 7,
            "dense_matching": 9,
            "dense_match_filling": 10,
            "triangulation": 12,
            "point_cloud_outlier_removal.1": 13,
            "point_cloud_outlier_removal.2": 14,
        }

        depth_to_dsm_apps = {
            "point_cloud_rasterization": 16,
            "dsm_filling.1": 18,
            "dsm_filling.2": 19,
            "dsm_filling.3": 20,
            "auxiliary_filling": 21,
        }

        self.app_values = {}
        self.app_values.update(sensor_to_depth_apps)
        self.app_values.update(depth_to_dsm_apps)

        app_conf = conf.get(APPLICATIONS, {})
        for key in app_conf:

            if adv_cst.SAVE_INTERMEDIATE_DATA not in app_conf[key]:
                continue

            if not app_conf[key][adv_cst.SAVE_INTERMEDIATE_DATA]:
                continue

            if key in sensor_to_depth_apps:
                self.compute_depth_map = True
                self.last_application_to_run = max(
                    self.last_application_to_run, self.app_values[key]
                )

            elif key in depth_to_dsm_apps:
                self.compute_depth_map = True

                # enabled to start the depth map to dsm process
                self.save_output_dsm = True

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

        return sensor_inputs.sensors_check_inputs(conf, config_dir=config_dir)

    def save_configurations(self):
        """
        Save used_conf and refined_conf configurations
        """

        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(self.out_dir, "current_res_used_conf.yaml"),
        )
        cars_dataset.save_dict(
            self.refined_conf,
            os.path.join(self.out_dir, "refined_conf.yaml"),
        )

    def check_output(self, inputs, conf, scaling_coeff):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float
        :return: overloader output
        :rtype: dict
        """
        return output_parameters.check_output_parameters(
            inputs, conf, scaling_coeff
        )

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
            True,
            self.save_output_dsm,
            self.save_output_point_cloud,
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

        self.epipolar_grid_generation_application = None
        self.resampling_application = None
        self.ground_truth_reprojection = None
        self.dense_match_filling = None
        self.sparse_mtch_app = None
        self.dense_matching_app = None
        self.triangulation_application = None
        self.dem_generation_application = None
        self.pc_outlier_removal_apps = {}
        self.rasterization_application = None
        self.pc_fusion_application = None
        self.grid_correction_app = None

        # Epipolar grid generation
        self.epipolar_grid_generation_application = Application(
            "grid_generation",
            cfg=used_conf.get("grid_generation", {}),
            scaling_coeff=scaling_coeff,
        )
        used_conf["grid_generation"] = (
            self.epipolar_grid_generation_application.get_conf()
        )

        # Epipolar grid correction
        self.grid_correction_app = Application(
            "grid_correction",
            cfg=used_conf.get("grid_correction", {}),
        )
        used_conf["grid_correction"] = self.grid_correction_app.get_conf()

        # image resampling
        self.resampling_application = Application(
            "resampling",
            cfg=used_conf.get("resampling", {}),
            scaling_coeff=scaling_coeff,
        )
        used_conf["resampling"] = self.resampling_application.get_conf()

        # ground truth disparity map computation
        if self.used_conf[PIPELINE][ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
            used_conf["ground_truth_reprojection"][
                "save_intermediate_data"
            ] = True

            if isinstance(
                self.used_conf[PIPELINE][ADVANCED][adv_cst.GROUND_TRUTH_DSM],
                str,
            ):
                self.used_conf[PIPELINE][ADVANCED][adv_cst.GROUND_TRUTH_DSM] = {
                    "dsm": self.used_conf[PIPELINE][ADVANCED][
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
        used_conf["dense_match_filling"] = self.dense_match_filling.get_conf()

        # Matching
        generate_performance_map = bool(
            self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_PERFORMANCE_MAP
            ]
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
        used_conf["triangulation"] = self.triangulation_application.get_conf()

        # MNT generation
        self.dem_generation_application = Application(
            "dem_generation",
            cfg=used_conf.get("dem_generation", {}),
            scaling_coeff=(
                self.dem_scaling_coeff
                if self.dem_scaling_coeff is not None
                else scaling_coeff
            ),
        )
        used_conf["dem_generation"] = self.dem_generation_application.get_conf()

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
            "{} point cloud outlier removal apps registered:\n{}".format(
                len(self.pc_outlier_removal_apps), methods_str
            )
        )

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

        return used_conf

    def get_classif_values_filling(self, inputs):
        """
        Get values in classif, used for filling

        :param inputs: inputs
        :type inputs: dict

        :return: list of values
        :rtype: list
        """

        filling_classif_values = []

        if sens_cst.FILLING not in inputs or inputs[sens_cst.FILLING] is None:
            logging.info("No filling in input configuration")
            return None

        filling_classif_values = []
        for _, classif_values in inputs[sens_cst.FILLING].items():
            # Add new value to filling bands
            if classif_values is not None:
                if isinstance(classif_values, str):
                    classif_values = [classif_values]
                filling_classif_values += classif_values

        simplified_list = list(OrderedDict.fromkeys(filling_classif_values))
        res_as_string_list = [str(value) for value in simplified_list]
        return res_as_string_list

    def check_applications_with_inputs(  # noqa: C901 : too complex
        self, inputs_conf, application_conf
    ):
        """
        Check for each application the input and output configuration
        consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        """

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
            nodata_left = inputs_conf["sensors"][key2]["image"]["no_data"]
            nodata_right = inputs_conf["sensors"][key2]["image"]["no_data"]
            bands_left = list(
                inputs_conf["sensors"][key1]["image"]["bands"].keys()
            )
            bands_right = list(
                inputs_conf["sensors"][key2]["image"]["bands"].keys()
            )
            values_classif_left = None
            values_classif_right = None
            if (
                "classification" in inputs_conf["sensors"][key1]
                and inputs_conf["sensors"][key1]["classification"] is not None
            ):
                values_classif_left = inputs_conf["sensors"][key1][
                    "classification"
                ]["values"]
                values_classif_left = list(map(str, values_classif_left))
            if (
                "classification" in inputs_conf["sensors"][key2]
                and inputs_conf["sensors"][key2]["classification"] is not None
            ):
                values_classif_right = inputs_conf["sensors"][key2][
                    "classification"
                ]["values"]
                values_classif_right = list(map(str, values_classif_right))
            self.dense_matching_app.corr_config = (
                self.dense_matching_app.loader.check_conf(
                    corr_cfg,
                    nodata_left,
                    nodata_right,
                    bands_left,
                    bands_right,
                    values_classif_left,
                    values_classif_right,
                )
            )

        return application_conf

    def sensor_to_depth_maps(self):  # noqa: C901
        """
        Creates the depth map from the sensor images given in the input,
        by following the CARS pipeline's steps.
        """
        # pylint:disable=too-many-return-statements
        inputs = self.used_conf[INPUT]
        output = self.used_conf[OUTPUT]

        # Initialize epsg for terrain tiles
        self.phasing = self.used_conf[PIPELINE][ADVANCED][adv_cst.PHASING]

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
        sensor_inputs.load_geomodels(
            inputs, self.geom_plugin_without_dem_and_geoid
        )
        self.list_sensor_pairs = sensor_inputs.generate_pairs(
            self.used_conf[INPUT]
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

        save_corrected_grid = (
            self.epipolar_grid_generation_application.get_save_grids()
        )

        # Compute dems
        dems = {}
        if self.used_conf[INPUT][sens_cst.LOW_RES_DSM] is not None:
            # Create dsm directory
            dsm_dir = os.path.join(
                self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY],
                "dsm",
            )
            safe_makedirs(dsm_dir)

            # DSM file name
            dsm_file_name = self.used_conf[INPUT][sens_cst.LOW_RES_DSM]

            dem_generation_output_dir = os.path.join(
                self.dump_dir, "dem_generation"
            )
            safe_makedirs(dem_generation_output_dir)

            # Use initial elevation if provided, and generate dems
            _, paths, _ = self.dem_generation_application.run(
                dsm_file_name,
                dem_generation_output_dir,
                input_geoid=self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                    sens_cst.GEOID
                ],
                output_geoid=self.used_conf[OUTPUT][out_cst.OUT_GEOID],
                initial_elevation=(
                    self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                        sens_cst.DEM_PATH
                    ]
                ),
                default_alt=self.geom_plugin_with_dem_and_geoid.default_alt,
                cars_orchestrator=self.cars_orchestrator,
            )

            if self.quit_on_app("dem_generation"):
                return True

            dem_median = paths["dem_median"]
            dem_min = paths["dem_min"]
            dem_max = paths["dem_max"]

            dems["dem_median"] = dem_median
            dems["dem_min"] = dem_min
            dems["dem_max"] = dem_max

            # Override initial elevation
            if (
                inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] is None
                or "dem_median"
                in inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
            ):
                inputs[sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ] = dem_median
            elif (
                inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                is not None
                and self.used_conf[PIPELINE][ADVANCED][USE_ENDOGENOUS_DEM]
            ):
                inputs[sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ] = dem_median

            # Check advanced parameters with new initial elevation
            output_dem_dir = os.path.join(
                self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY],
                "dump_dir",
                "initial_elevation",
            )
            safe_makedirs(output_dem_dir)
            (
                inputs,
                _,
                self.geometry_plugin,
                self.geom_plugin_without_dem_and_geoid,
                self.geom_plugin_with_dem_and_geoid,
                _,
                _,
                _,
            ) = advanced_parameters.check_advanced_parameters(
                inputs,
                self.used_conf.get(ADVANCED, {}),
                output_dem_dir=output_dem_dir,
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

            # Prepare tie point pipeline

            # Update tie points pipeline with rectification grids
            tie_points_config = self.tie_points_pipelines[pair_key].used_conf
            image_keys = list(tie_points_config[INPUT][sens_cst.SENSORS])
            tie_points_config[INPUT][sens_cst.RECTIFICATION_GRIDS] = {
                image_keys[0]: self.pairs[pair_key]["grid_left"],
                image_keys[1]: self.pairs[pair_key]["grid_right"],
            }
            if self.tie_points_out_dir is not None:
                tie_points_config[OUTPUT][
                    out_cst.OUT_DIRECTORY
                ] = self.tie_points_out_dir
            tie_points_pipeline = TiePointsPipeline(
                tie_points_config,
                config_dir=self.config_dir,
            )
            sparse_mtch_app = tie_points_pipeline.sparse_matching_app

            tie_points_output = tie_points_config[OUTPUT][out_cst.OUT_DIRECTORY]

            # If dem are provided, launch disparity grids generation
            disp_range_grid_dir = os.path.join(
                tie_points_output, "disparity_grids"
            )
            disp_range_grid = None
            if dems:
                disp_range_grid = (
                    self.dense_matching_app.generate_disparity_grids(
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        dem_min=dem_min,
                        dem_max=dem_max,
                        pair_folder=disp_range_grid_dir,
                        orchestrator=self.cars_orchestrator,
                    )
                )

            # Launch tie points pipeline
            tie_points_pipeline.run(
                disp_range_grid=disp_range_grid, log_dir=self.log_dir
            )
            self.pairs[pair_key]["matches_array"] = np.load(
                os.path.join(
                    tie_points_output, pair_key, "filtered_matches.npy"
                )
            )

            minimum_nb_matches = sparse_mtch_app.minimum_nb_matches
            nb_matches = self.pairs[pair_key]["matches_array"].shape[0]
            save_matches = sparse_mtch_app.get_save_matches()

            if nb_matches > minimum_nb_matches:
                # Compute grid correction
                (self.pairs[pair_key]["corrected_grid_right"], _, _, _) = (
                    self.grid_correction_app.run(
                        self.pairs[pair_key]["matches_array"],
                        self.pairs[pair_key]["grid_right"],
                        save_matches=save_matches,
                        minimum_nb_matches=minimum_nb_matches,
                        pair_folder=os.path.join(
                            self.dump_dir,
                            "grid_correction",
                            "initial",
                            pair_key,
                        ),
                        pair_key=pair_key,
                        orchestrator=self.cars_orchestrator,
                        save_corrected_grid=save_corrected_grid,
                    )
                )
            else:
                logging.warning(
                    "Grid correction is not applied because numer of matches "
                    "found ({}) is less than minimum numer of matches "
                    "required for grid correction ({})".format(
                        nb_matches,
                        minimum_nb_matches,
                    )
                )
                self.pairs[pair_key]["corrected_grid_right"] = self.pairs[
                    pair_key
                ]["grid_right"]

            self.pairs[pair_key]["corrected_grid_left"] = self.pairs[pair_key][
                "grid_left"
            ]

            # Shrink disparity intervals according to SIFT disparities
            disp_to_alt_ratio = self.pairs[pair_key]["grid_left"][
                "disp_to_alt_ratio"
            ]
            disp_bounds_params = sparse_mtch_app.disparity_bounds_estimation

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
        if not save_corrected_grid:
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
        if self.quit_on_app("grid_generation") or self.quit_on_app(
            "resampling"
        ):
            return True

        # Define param
        use_global_disp_range = self.dense_matching_app.use_global_disp_range

        self.pairs_names = [
            pair_name for pair_name, _, _ in self.list_sensor_pairs
        ]

        for _, (pair_key, _, _) in enumerate(self.list_sensor_pairs):
            # Geometry plugin with dem will be used for the grid generation
            geom_plugin = self.geom_plugin_with_dem_and_geoid

            # saved used configuration
            self.save_configurations()

            # Generate min and max disp grids
            # Global disparity min and max will be computed from
            # these grids
            dense_matching_pair_folder = os.path.join(
                self.dump_dir, "dense_matching", pair_key
            )

            if self.which_resolution in ("first", "single") and dems in (
                None,
                {},
            ):
                dmin = disp_min
                dmax = disp_max
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

                dsp_marg = self.tie_points_pipelines[
                    pair_key
                ].sparse_matching_app.get_disparity_margin()
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
                        pair_folder=dense_matching_pair_folder,
                        orchestrator=self.cars_orchestrator,
                    )
                )

                if use_global_disp_range:
                    # Generate min and max disp grids from constants
                    # sensor image is not used here
                    # TODO remove when only local diparity range will be used
                    dmin = self.pairs[pair_key]["disp_range_grid"]["global_min"]
                    dmax = self.pairs[pair_key]["disp_range_grid"]["global_max"]

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
                required_bands=required_bands,
                texture_bands=self.texture_bands,
            )
            # Run ground truth dsm computation
            if self.used_conf[PIPELINE][ADVANCED][adv_cst.GROUND_TRUTH_DSM]:
                self.used_conf["applications"]["ground_truth_reprojection"][
                    "save_intermediate_data"
                ] = True
                new_geomplugin_dsm = AbstractGeometry(  # pylint: disable=E0110
                    self.geometry_plugin,
                    dem=self.used_conf[PIPELINE][ADVANCED][
                        adv_cst.GROUND_TRUTH_DSM
                    ][adv_cst.INPUT_GROUND_TRUTH_DSM],
                    geoid=self.used_conf[PIPELINE][ADVANCED][
                        adv_cst.GROUND_TRUTH_DSM
                    ][adv_cst.INPUT_GEOID],
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
                    self.used_conf[PIPELINE][ADVANCED][
                        adv_cst.GROUND_TRUTH_DSM
                    ][adv_cst.INPUT_AUX_PATH],
                    self.used_conf[PIPELINE][ADVANCED][
                        adv_cst.GROUND_TRUTH_DSM
                    ][adv_cst.INPUT_AUX_INTERP],
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
                self.used_conf[PIPELINE]["applications"]["dense_matching"][
                    "loader_conf"
                ] = conf
                self.used_conf[PIPELINE]["applications"]["dense_matching"][
                    "method"
                ] = "custom"

                # Re initialization of the dense matching application
                self.dense_matching_app = Application(
                    "dense_matching",
                    cfg=self.used_conf[PIPELINE]["applications"][
                        "dense_matching"
                    ],
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
                classif_bands_to_mask=self.used_classif_values_for_filling,
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
                if (point_cloud_dir and len(self.pc_outlier_removal_apps) == 0)
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
                denoising_overload_fun=None,
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
                and self.auxiliary[out_cst.AUX_IMAGE],
                save_output_classification=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_CLASSIFICATION],
                save_output_filling=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_FILLING],
                save_output_performance_map=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_PERFORMANCE_MAP],
                save_output_ambiguity=bool(depth_map_dir)
                and self.auxiliary[out_cst.AUX_AMBIGUITY],
            )

            if self.quit_on_app("triangulation"):
                continue  # keep iterating over pairs, but don't go further

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
                        (  # pylint: disable=inconsistent-quotes
                            f"pc_outlier_removal"
                            f"{str(app_key[27:]).replace('.', '_')}"
                        ),
                        pair_key,
                    ),
                    epsg=self.epsg,
                    orchestrator=self.cars_orchestrator,
                )
            if self.quit_on_app("point_cloud_outlier_removal"):
                continue  # keep iterating over pairs, but don't go further

            self.list_epipolar_point_clouds.append(
                filtered_epipolar_point_cloud
            )

        # quit if any app in the loop over the pairs was the last one
        # pylint:disable=too-many-boolean-expressions
        if (
            self.quit_on_app("dense_matching")
            or self.quit_on_app("dense_match_filling")
            or self.quit_on_app("triangulation")
            or self.quit_on_app("point_cloud_outlier_removal.1")
            or self.quit_on_app("point_cloud_outlier_removal.2")
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
                "image.tif",
            )
            if self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_IMAGE]
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
            contributing_pair_file_name=contributing_pair_file_name,
            filling_file_name=filling_file_name,
            color_dtype=self.color_type,
            dump_dir=self.rasterization_dump_dir,
            performance_map_classes=self.used_conf[OUTPUT][AUXILIARY][
                out_cst.AUX_PERFORMANCE_MAP
            ],
            phasing=self.phasing,
        )

        # Cleaning: don't keep terrain bbox if save_intermediate_data
        # is not activated
        if not self.used_conf[PIPELINE][ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]:
            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "terrain_bbox")
            )

        if self.quit_on_app("point_cloud_rasterization"):
            return True

        # dsm needs to be saved before filling
        self.cars_orchestrator.breakpoint()

        # saved used configuration
        self.save_configurations()

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

        if (
            classif_file_name is not None
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_CLASSIFICATION
            ]
        ):
            self.merge_classif_bands(
                classif_file_name,
                self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_CLASSIFICATION
                ],
                dsm_file_name,
            )

        if (
            filling_file_name is not None
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING]
        ):
            self.merge_filling_bands(
                filling_file_name,
                self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING],
                dsm_file_name,
            )

        return False

    @cars_profile(name="merge filling bands", interval=0.5)
    def merge_filling_bands(self, filling_path, aux_filling, dsm_file):
        """
        Merge filling bands to get mono band in output
        """

        with rasterio.open(dsm_file) as in_dsm:
            dsm_msk = in_dsm.read_masks(1)

        with rasterio.open(filling_path) as src:
            nb_bands = src.count

            if nb_bands == 1:
                return False

            filling_multi_bands = src.read()
            filling_mono_bands = np.zeros(filling_multi_bands.shape[1:3])
            descriptions = src.descriptions
            dict_temp = {name: i for i, name in enumerate(descriptions)}
            profile = src.profile

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NodataShadowWarning)
                filling_mask = src.read_masks(1)

            filling_mono_bands[filling_mask == 0] = 0

            filling_bands_list = {
                "fill_with_geoid": ["filling_exogenous"],
                "interpolate_from_borders": [
                    "bulldozer",
                    "border_interpolation",
                ],
                "fill_with_endogenous_dem": [
                    "filling_exogenous",
                    "bulldozer",
                ],
                "fill_with_exogenous_dem": ["bulldozer"],
            }

            # To get the right footprint
            filling_mono_bands = np.logical_or(dsm_msk, filling_mask).astype(
                np.uint8
            )

            # to keep the previous classif convention
            filling_mono_bands[filling_mono_bands == 0] = src.nodata
            filling_mono_bands[filling_mono_bands == 1] = 0

            no_match = False
            for key, value in aux_filling.items():
                if isinstance(value, str):
                    value = [value]

                if isinstance(value, list):
                    for elem in value:
                        if elem != "other":
                            filling_method = filling_bands_list[elem]

                            if all(
                                method in descriptions
                                for method in filling_method
                            ):
                                indices_true = [
                                    dict_temp[m] for m in filling_method
                                ]

                                mask_true = np.all(
                                    filling_multi_bands[indices_true, :, :]
                                    == 1,
                                    axis=0,
                                )

                                indices_false = [
                                    i
                                    for i in range(filling_multi_bands.shape[0])
                                    if i not in indices_true
                                ]

                                mask_false = np.all(
                                    filling_multi_bands[indices_false, :, :]
                                    == 0,
                                    axis=0,
                                )

                                mask = mask_true & mask_false

                                filling_mono_bands[mask] = key
                            else:
                                no_match = True

            if no_match:
                mask_1 = np.all(
                    filling_multi_bands[1:, :, :] == 1,
                    axis=0,
                )

                mask_2 = np.all(
                    filling_mono_bands == 0,
                    axis=0,
                )

                filling_mono_bands[mask_1 & mask_2] = (
                    aux_filling["other"] if "other" in aux_filling else 50
                )

            profile.update(count=1, dtype=filling_mono_bands.dtype)
            with rasterio.open(filling_path, "w", **profile) as src:
                src.write(filling_mono_bands, 1)

        return True

    @cars_profile(name="merge classif bands", interval=0.5)
    def merge_classif_bands(self, classif_path, aux_classif, dsm_file):
        """
        Merge classif bands to get mono band in output
        """
        with rasterio.open(dsm_file) as in_dsm:
            dsm_msk = in_dsm.read_masks(1)

        with rasterio.open(classif_path) as src:
            nb_bands = src.count

            if nb_bands == 1:
                return False

            classif_multi_bands = src.read()
            classif_mono_band = np.zeros(classif_multi_bands.shape[1:3])
            descriptions = src.descriptions
            profile = src.profile
            classif_mask = src.read_masks(1)
            classif_mono_band[classif_mask == 0] = 0

            # To get the right footprint
            classif_mono_band = np.logical_or(dsm_msk, classif_mask).astype(
                np.uint8
            )

            # to keep the previous classif convention
            classif_mono_band[classif_mono_band == 0] = src.nodata
            classif_mono_band[classif_mono_band == 1] = 0

            for key, value in aux_classif.items():
                if isinstance(value, int):
                    num_band = descriptions.index(str(value))
                    mask_1 = classif_mono_band == 0
                    mask_2 = classif_multi_bands[num_band, :, :] == 1
                    classif_mono_band[mask_1 & mask_2] = key
                elif isinstance(value, list):
                    for elem in value:
                        num_band = descriptions.index(str(elem))
                        mask_1 = classif_mono_band == 0
                        mask_2 = classif_multi_bands[num_band, :, :] == 1
                        classif_mono_band[mask_1 & mask_2] = key

        profile.update(count=1, dtype=classif_mono_band.dtype)
        with rasterio.open(classif_path, "w", **profile) as src:
            src.write(classif_mono_band, 1)

        return True

    @cars_profile(name="Preprocess depth maps", interval=0.5)
    def preprocess_depth_maps(self):
        """
        Adds multiple processing steps to the depth maps :
        Merging.
        Creates the point cloud that will be rasterized in
        the last step of the pipeline.
        """

        self.point_cloud_to_rasterize = (
            self.list_epipolar_point_clouds,
            self.terrain_bounds,
        )
        self.color_type = self.point_cloud_to_rasterize[0][0].attributes.get(
            "color_type", None
        )

    @cars_profile(name="Final cleanup", interval=0.5)
    def final_cleanup(self):
        """
        Clean temporary files and directory at the end of cars processing
        """

        if not self.used_conf[PIPELINE][ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]:
            # delete everything in tile_processing if save_intermediate_data is
            # not activated
            self.cars_orchestrator.add_to_clean(
                os.path.join(self.dump_dir, "tile_processing")
            )

            self.cars_orchestrator.add_to_clean(
                os.path.join(self.out_dir, "tie_points")
            )

            # Remove dump_dir if no intermediate data should be written
            if not any(
                app.get("save_intermediate_data", False) is True
                for app in self.used_conf[PIPELINE][APPLICATIONS].values()
                if app is not None
            ):
                self.cars_orchestrator.add_to_clean(self.dump_dir)

    @cars_profile(name="run_surface_modeling_pipeline", interval=0.5)
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        which_resolution="single",
        log_dir=None,
        tie_points_out_dir=None,
    ):  # noqa C901
        """
        Run pipeline

        """
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(self.out_dir, "logs")

        self.texture_bands = self.used_conf[OUTPUT][AUXILIARY][
            out_cst.AUX_IMAGE
        ]

        self.auxiliary = self.used_conf[OUTPUT][out_cst.AUXILIARY]

        self.which_resolution = which_resolution

        self.tie_points_out_dir = tie_points_out_dir

        # saved used configuration
        self.save_configurations()
        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=self.log_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as self.cars_orchestrator:
            # initialize out_json
            self.cars_orchestrator.update_out_info({"version": __version__})

            if self.compute_depth_map:
                self.sensor_to_depth_maps()

            if self.save_output_dsm or self.save_output_point_cloud:
                self.preprocess_depth_maps()

                if self.save_output_dsm:
                    self.rasterize_point_cloud()

            self.final_cleanup()
