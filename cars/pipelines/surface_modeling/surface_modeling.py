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
# pylint: disable=design
"""
CARS surface modeling pipeline class file
"""
# Standard imports
from __future__ import print_function

import copy
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rasterio
from json_checker import Checker, OptionalKey

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars import __version__

# CARS imports
from cars.applications import application_constants
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_wrappers as dem_wrappers,
)
from cars.core import preprocessing, projection, roi_tools, tiling
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import get_descriptions_bands
from cars.core.progress.progress import ProgressTree
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import application_parameters
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
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
from cars.pipelines.surface_modeling.surface_modeling_pipeline_wrappers import (
    merge_filling_bands_wrapper,
)
from cars.pipelines.tie_points.tie_points import TiePointsPipeline

PIPELINE = "surface_modeling"


@Pipeline.register(PIPELINE)
class SurfaceModelingPipeline(PipelineTemplate):
    """
    SurfaceModelingPipeline
    """

    @staticmethod
    def _should_run_disparity_to_depth_maps(
        compute_depth_map,
        quit_after_dem_generation,
        quit_after_grid_or_resampling,
        quit_after_dense_matching,
    ):
        """Return whether disparity_to_depth_maps phase is expected to run."""
        return not (
            (not compute_depth_map)
            or quit_after_dem_generation
            or quit_after_grid_or_resampling
            or quit_after_dense_matching
        )

    @staticmethod
    def _compute_generate_disparity_grids_runs(
        num_sensor_pairs,
        has_tie_points_pipeline,
        has_low_res_dsm,
        is_first_or_single,
        use_global_disp_range,
        quit_after_grid_or_resampling,
        save_output_dsm,
        should_run_disparity_to_depth_maps,
    ):
        """Calculate expected runs for generate_disparity_grids."""
        expected_runs = 0

        if not quit_after_grid_or_resampling:
            # One preparation run per pair only when tie points are enabled
            # and a low resolution DSM is provided.
            expected_runs += (
                num_sensor_pairs
                if has_tie_points_pipeline and has_low_res_dsm
                else 0
            )

            # Main disparity grids generation:
            # one run per pair, plus an optional second run per pair when
            # global disparity range is enabled in the DEM-based branch.
            main_runs_per_pair = 1
            if (
                (not is_first_or_single) or has_low_res_dsm
            ) and use_global_disp_range:
                main_runs_per_pair += 1
            expected_runs += num_sensor_pairs * main_runs_per_pair

        # When DSM rasterization is disabled and disparity_to_depth_maps
        # won't run, orchestrator.__exit__ is the final compute pass
        if not save_output_dsm and (not should_run_disparity_to_depth_maps):
            expected_runs += 1

        return expected_runs

    @staticmethod
    def _compute_tie_points_runs(
        has_tie_points_pipeline,
        quit_after_grid_or_resampling,
        num_sensor_pairs,
    ):
        """Calculate expected runs for tie_points."""
        if not has_tie_points_pipeline or quit_after_grid_or_resampling:
            return 0
        return num_sensor_pairs

    @staticmethod
    def _compute_disparity_to_depth_maps_runs(
        should_run_disparity_to_depth_maps,
        quit_after_triangulation,
        use_sensor_disp,
        num_sensor_pairs,
        save_output_dsm,
    ):
        """Calculate expected runs for disparity_to_depth_maps."""
        if not should_run_disparity_to_depth_maps:
            return 0

        expected_runs = 0

        # There is one explicit breakpoint pass per iter item
        # in disparity_to_depth_maps(), unless triangulation phase is
        # configured as a stopping point
        if not quit_after_triangulation:
            expected_runs += 1 if use_sensor_disp else num_sensor_pairs

        # If DSM rasterization is disabled, this task remains the active
        # target until orchestrator __exit__, which adds a final pass
        if not save_output_dsm:
            expected_runs += 1

        return expected_runs

    @staticmethod
    def _compute_rasterize_point_cloud_runs(
        save_output_dsm,
        quit_after_rasterization,
        has_aux_filling,
    ):
        """Calculate expected runs for rasterize_point_cloud."""
        if not save_output_dsm:
            return 0

        expected_runs = 0
        if not quit_after_rasterization:
            # One compute pass to flush rasterization outputs.
            expected_runs += 1
            # Optional second pass when auxiliary filling merge is enabled.
            if has_aux_filling:
                expected_runs += 1
        return expected_runs

    def setup_progress_tracking(
        self,
        parent_pipeline_id=None,
        tie_points_pipeline_id=None,
    ):
        """
        Setup progress tracking for surface modeling.

        :param parent_pipeline_id: Optional parent pipeline ID
        :type parent_pipeline_id: int or None
        :param tie_points_pipeline_id: Optional tie_points pipeline ID
            pre-created by default pipeline
        :type tie_points_pipeline_id: int or None
        :return: Task ID to pass to orchestrator via set_target_task()
        :rtype: int
        """
        # Create pipeline if standalone, otherwise use parent
        if parent_pipeline_id is None:
            self.pipeline_progress_id = ProgressTree().begin_pipeline(
                "surface_modeling"
            )
        else:
            self.pipeline_progress_id = parent_pipeline_id

        # Calculate number of sensor image pairs
        num_sensor_pairs = len(self.used_conf[INPUT][sens_cst.PAIRING])

        # Register tasks for surface modeling workflow
        use_global_disp_range = self.dense_matching_app.use_global_disp_range
        has_tie_points_pipeline = self.tie_points_pipelines is not None
        has_low_res_dsm = (
            self.used_conf[INPUT][sens_cst.LOW_RES_DSM] is not None
        )
        is_first_or_single = self.which_resolution in ("first", "single")
        has_aux_filling = (
            self.save_output_dsm
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING]
        )

        # Early stops configured through output_level_none/application setup
        quit_after_dem_generation = self.quit_on_app("dem_generation")
        quit_after_grid_or_resampling = self.quit_on_app(
            "grid_generation"
        ) or self.quit_on_app("resampling")
        quit_after_dense_matching = self.quit_on_app(
            "dense_matching"
        ) or self.quit_on_app("dense_match_filling")
        quit_after_triangulation = (
            self.quit_on_app("triangulation")
            or self.quit_on_app("point_cloud_outlier_removal.1")
            or self.quit_on_app("point_cloud_outlier_removal.2")
        )
        quit_after_rasterization = self.quit_on_app("point_cloud_rasterization")

        should_run_disparity_to_depth_maps = (
            self._should_run_disparity_to_depth_maps(
                self.compute_depth_map,
                quit_after_dem_generation,
                quit_after_grid_or_resampling,
                quit_after_dense_matching,
            )
        )

        generate_disparity_grids_runs = (
            self._compute_generate_disparity_grids_runs(
                num_sensor_pairs,
                has_tie_points_pipeline,
                has_low_res_dsm,
                is_first_or_single,
                use_global_disp_range,
                quit_after_grid_or_resampling,
                self.save_output_dsm,
                should_run_disparity_to_depth_maps,
            )
        )
        tie_points_runs = self._compute_tie_points_runs(
            has_tie_points_pipeline,
            quit_after_grid_or_resampling,
            num_sensor_pairs,
        )
        disparity_to_depth_maps_runs = (
            self._compute_disparity_to_depth_maps_runs(
                should_run_disparity_to_depth_maps,
                quit_after_triangulation,
                self.use_sensor_disp,
                num_sensor_pairs,
                self.save_output_dsm,
            )
        )
        rasterize_point_cloud_runs = self._compute_rasterize_point_cloud_runs(
            self.save_output_dsm,
            quit_after_rasterization,
            has_aux_filling,
        )

        # clamp expected_runs to >= 1
        generate_disparity_grids_runs = max(1, generate_disparity_grids_runs)
        tie_points_runs = max(1, tie_points_runs)
        disparity_to_depth_maps_runs = max(1, disparity_to_depth_maps_runs)
        rasterize_point_cloud_runs = max(1, rasterize_point_cloud_runs)

        self.task_ids = {}
        self.task_ids[
            "generate_disparity_grids"
        ] = ProgressTree().register_task(
            self.pipeline_progress_id,
            "generate_disparity_grids",
            weight=2,
            expected_runs=generate_disparity_grids_runs,
        )

        tie_points_task_pipeline_id = (
            tie_points_pipeline_id
            if tie_points_pipeline_id is not None
            else self.pipeline_progress_id
        )
        self.task_ids["tie_points"] = ProgressTree().register_task(
            tie_points_task_pipeline_id,
            "tie_points",
            weight=20,
            expected_runs=tie_points_runs,
        )

        self.task_ids["disparity_to_depth_maps"] = ProgressTree().register_task(
            self.pipeline_progress_id,
            "disparity_to_depth_maps",
            weight=20,
            expected_runs=disparity_to_depth_maps_runs,
        )

        # rasterize_point_cloud - final phase
        # Additional compute passes can run under the same target task:
        # - mono-band filling merge when auxiliary filling output is enabled
        # - a final orchestrator compute pass (standalone auxiliary filling)
        # - dtm generation when DTM output is requested (on single/final run)
        self.task_ids["rasterize_point_cloud"] = ProgressTree().register_task(
            self.pipeline_progress_id,
            "rasterize_point_cloud",
            weight=58,
            expected_runs=rasterize_point_cloud_runs,
        )

        # Return first task ID for orchestrator to track
        return self.task_ids["generate_disparity_grids"]

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

        # metadata
        self.metadata = None

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

        initial_elevation = (
            inputs[sens_cst.INITIAL_ELEVATION]["dem"] is not None
        )
        self.elevation_delta_lower_bound = -500 if initial_elevation else -1000
        self.elevation_delta_upper_bound = 1000 if initial_elevation else 9000

        self.dem_scaling_coeff = None
        if inputs[sens_cst.LOW_RES_DSM] is not None:
            low_res_dsm = rasterio.open(inputs[sens_cst.LOW_RES_DSM])
            self.dem_scaling_coeff = np.mean(low_res_dsm.res) * 2
            crs = low_res_dsm.crs
            if crs.is_geographic:
                xmin = low_res_dsm.bounds.left
                ymin = low_res_dsm.bounds.bottom
                utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
                conversion_factor = preprocessing.get_conversion_factor(
                    low_res_dsm.bounds, utm_epsg, crs.to_epsg()
                )
                self.dem_scaling_coeff = (
                    self.dem_scaling_coeff * conversion_factor
                )

        # Init tie points pipelines
        self.tie_point_save = False
        if TIE_POINTS in conf and conf[TIE_POINTS] is None:
            self.tie_points_pipelines = None
        else:
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
                tie_points_output = os.path.join(
                    self.out_dir, TIE_POINTS, pair_key
                )
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

                self.used_conf[TIE_POINTS][ADVANCED] = (
                    self.tie_points_pipelines[pair_key].used_conf[TIE_POINTS][
                        ADVANCED
                    ]
                )

                if self.used_conf[TIE_POINTS][ADVANCED][
                    adv_cst.SAVE_INTERMEDIATE_DATA
                ]:
                    self.tie_point_save = True

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
            bounds,
            self.use_sensor_disp,
        ) = advanced_parameters.check_advanced_parameters(
            inputs,
            pipeline_conf.get(ADVANCED, {}),
            output_dem_dir=output_dem_dir,
        )

        self.used_conf[PIPELINE][ADVANCED] = advanced

        self.refined_conf[PIPELINE] = copy.deepcopy(self.used_conf[PIPELINE])
        self.refined_conf[PIPELINE][ADVANCED] = copy.deepcopy(advanced)
        # Refined conf: resolutions 1
        self.refined_conf[PIPELINE][ADVANCED][adv_cst.RESOLUTIONS] = [1]

        self.refined_conf["pipeline"] = "surface_modeling"

        if sens_cst.SCALING_COEFF in conf[INPUT]:
            if conf[INPUT][sens_cst.SCALING_COEFF] is not None:
                self.scaling_coeff = conf[INPUT][sens_cst.SCALING_COEFF]

        # Check conf output
        (
            output,
            self.scaling_coeff,
        ) = self.check_output(inputs, conf[OUTPUT], self.scaling_coeff, bounds)

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUT][sens_cst.ROI]
        )

        if self.input_roi_poly is not None:
            xmin = bounds[0]
            ymin = bounds[1]
            utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
            conversion_factor = preprocessing.get_conversion_factor(
                bounds, utm_epsg, self.input_roi_epsg
            )
            res_roi_epsg = output["resolution"] / conversion_factor

            terrain_margin = 10 * self.scaling_coeff * res_roi_epsg

            self.input_roi_poly = self.input_roi_poly.buffer(
                terrain_margin, join_style=2
            )

        self.debug_with_roi = self.used_conf[PIPELINE][ADVANCED][
            adv_cst.DEBUG_WITH_ROI
        ]

        self.used_conf[OUTPUT] = output
        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

        self.refined_conf[OUTPUT] = copy.deepcopy(output)

        prod_level = output[out_cst.PRODUCT_LEVEL]

        self.product_format = output[out_cst.PRODUCT_FORMAT]

        self.save_output_dsm = "dsm" in prod_level
        self.save_output_point_cloud = "point_cloud" in prod_level
        self.save_output_dtm = "dtm" in prod_level

        if self.save_output_dtm and not self.save_output_dsm:
            self.output_dsm = True

        self.output_level_none = not (
            self.save_output_dsm
            or self.save_output_point_cloud
            or self.save_output_dtm
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

    def check_output(self, inputs, conf, scaling_coeff, bounds):
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
            inputs, conf, scaling_coeff, bounds
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
            self.save_output_dtm,
            conf,
            self.use_sensor_disp,
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

        self.epipolar_to_sensor_matching_app = None
        self.triangulation_n_los_app = None

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

        # epipolar to sensor matching
        if self.use_sensor_disp:
            self.epipolar_to_sensor_matching_app = Application(
                "epipolar_to_sensor_matching",
                cfg=used_conf.get("epipolar_to_sensor_matching", {}),
            )
            used_conf["epipolar_to_sensor_matching"] = (
                self.epipolar_to_sensor_matching_app.get_conf()
            )

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

        if self.save_output_dtm:
            # dtm generation
            self.dtm_generation_application = Application(
                "dtm_generation",
                cfg=used_conf.get("dtm_generation", {}),
            )
            used_conf["dtm_generation"] = (
                self.dtm_generation_application.get_conf()
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
            values_to_add = []
            # Add new value to filling bands
            if classif_values is not None:
                if isinstance(classif_values, str):
                    values_to_add = [classif_values]
                else:
                    for elem in classif_values:
                        if elem not in ("mismatch", "occlusion"):
                            values_to_add.append(elem)
                filling_classif_values += values_to_add

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
            if "fill_classification" in application_conf[application_key]:
                for item in inputs_conf["sensors"]:
                    if (
                        "fill_classification"
                        in inputs_conf["sensors"][item].keys()
                    ):
                        if inputs_conf["sensors"][item]["fill_classification"]:
                            descriptions = get_descriptions_bands(
                                inputs_conf["sensors"][item][
                                    "fill_classification"
                                ]
                            )
                            if application_conf[application_key][
                                "fill_classification"
                            ] and not set(
                                application_conf[application_key][
                                    "fill_classification"
                                ]
                            ).issubset(
                                set(descriptions) | {"nodata"}
                            ):
                                raise RuntimeError(
                                    "The {} bands description {} ".format(
                                        inputs_conf["sensors"][item][
                                            "fill_classification"
                                        ],
                                        list(descriptions),
                                    )
                                    + "and the {} config are not ".format(
                                        application_key
                                    )
                                    + "consistent: {}".format(
                                        application_conf[application_key][
                                            "fill_classification"
                                        ]
                                    )
                                )
        for key1, key2 in inputs_conf["pairing"]:
            corr_cfg = self.dense_matching_app.loader.get_conf()
            nodata_left = inputs_conf["sensors"][key1]["image"]["no_data"]
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
            self.dense_matching_app.dense_matching_method.corr_config = (
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

    def sensor_to_disparity(self):  # noqa: C901
        """
        Creates the disparity map from the sensor images given in the input,
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

        _ = output_parameters.intialize_product_index(
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
                _,
                _,
            ) = advanced_parameters.check_advanced_parameters(
                inputs,
                self.used_conf.get(PIPELINE, {}).get(ADVANCED, {}),
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
                resolution=self.working_res,
            )

            if self.quit_on_app("grid_generation"):
                continue  # keep iterating over pairs, but don't go further

            # Prepare tie point pipeline

            # Update tie points pipeline with rectification grids
            if self.tie_points_pipelines is not None:
                tie_points_config = self.tie_points_pipelines[
                    pair_key
                ].used_conf
                image_keys = list(tie_points_config[INPUT][sens_cst.SENSORS])
                tie_points_config[INPUT][sens_cst.RECTIFICATION_GRIDS] = {
                    image_keys[0]: self.pairs[pair_key]["grid_left"],
                    image_keys[1]: self.pairs[pair_key]["grid_right"],
                }
                tie_points_pipeline = TiePointsPipeline(
                    tie_points_config,
                    config_dir=self.config_dir,
                )
                sparse_mtch_app = tie_points_pipeline.sparse_matching_app

                tie_points_output = tie_points_config[OUTPUT][
                    out_cst.OUT_DIRECTORY
                ]

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

                    disp_min_sparse = disp_range_grid["global_min"]
                    disp_max_sparse = disp_range_grid["global_max"]
                else:
                    disp_min_sparse = self.elevation_delta_lower_bound
                    disp_max_sparse = self.elevation_delta_upper_bound

                epipolar_roi = None
                ignore_roi_during_a_priori = inputs[
                    "ignore_roi_during_a_priori"
                ]
                if not ignore_roi_during_a_priori:
                    epipolar_roi = preprocessing.compute_epipolar_roi(
                        self.input_roi_poly,
                        self.input_roi_epsg,
                        self.geom_plugin_with_dem_and_geoid,
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["grid_left"],
                        self.pairs[pair_key]["grid_right"],
                        os.path.join(
                            self.dump_dir,
                            "compute_epipolar_roi_apriori",
                            pair_key,
                        ),
                        disp_min=disp_min_sparse,
                        disp_max=disp_max_sparse,
                    )

                # Launch tie points pipeline
                tie_points_pipeline.run(
                    disp_range_grid=disp_range_grid,
                    epipolar_roi=epipolar_roi,
                    log_dir=self.log_dir,
                    cars_orchestrator=self.cars_orchestrator,
                    previous_dir=self.previous_out_dir,
                    res_factor=self.res_factor,
                    parent_pipeline_id=self.pipeline_progress_id,
                    task_progress_id=self.task_ids.get("tie_points"),
                )
                self.pairs[pair_key]["matches_array"] = np.load(
                    os.path.join(
                        tie_points_output, pair_key, "filtered_matches.npy"
                    )
                )

                self.cars_orchestrator.set_target_task(
                    self.task_ids["generate_disparity_grids"]
                )
                minimum_nb_matches = (
                    self.grid_correction_app.get_minimum_nb_matches()
                )
                nb_matches = self.pairs[pair_key]["matches_array"].shape[0]
                save_matches = sparse_mtch_app.save_intermediate_data

                if nb_matches > minimum_nb_matches:
                    # Compute grid correction
                    (self.pairs[pair_key]["corrected_grid_right"], _, _, _) = (
                        self.grid_correction_app.run(
                            self.pairs[pair_key]["matches_array"],
                            self.pairs[pair_key]["grid_right"],
                            save_matches=save_matches,
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
                        "Grid correction is not applied because number of "
                        "matches found ({}) is less than minimum number of "
                        "matches required for grid correction ({})".format(
                            nb_matches,
                            minimum_nb_matches,
                        )
                    )
                    self.pairs[pair_key]["corrected_grid_right"] = self.pairs[
                        pair_key
                    ]["grid_right"]
            else:
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
            if self.tie_points_pipelines is not None:
                disp_bounds_params = sparse_mtch_app.disparity_bounds_estimation
                matches = self.pairs[pair_key]["matches_array"]
                if disp_bounds_params["activated"] and matches.shape[0] > 0:
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
                        -self.elevation_delta_upper_bound / disp_to_alt_ratio
                    )
                    disp_max = (
                        -self.elevation_delta_lower_bound / disp_to_alt_ratio
                    )
                    logging.info(
                        "Global disparity interval : "
                        f"[{disp_min:.2f} pix, {disp_max:.2f} pix]"
                    )
            else:
                disp_min = -self.elevation_delta_upper_bound / disp_to_alt_ratio
                disp_max = -self.elevation_delta_lower_bound / disp_to_alt_ratio
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

                updating_infos = {
                    application_constants.APPLICATION_TAG: {
                        sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                            pair_key: {
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
                    updating_infos = {
                        application_constants.APPLICATION_TAG: {
                            sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                                pair_key: {
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

        for _, (pair_key, _, _) in enumerate(self.list_sensor_pairs):

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
                self.pairs[pair_key]["new_epipolar_image_left"],
                self.pairs[pair_key]["new_epipolar_image_right"],
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
                self.used_conf[PIPELINE][APPLICATIONS][
                    "ground_truth_reprojection"
                ]["save_intermediate_data"] = True
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
                or self.dense_matching_app.get_method() == "pandora_auto"
            ):
                # Compute terrain bounding box /roi related to
                # current images
                (current_terrain_roi_bbox, intersection_poly) = (
                    preprocessing.compute_terrain_bbox(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["sensor_image_right"],
                        self.pairs[pair_key]["new_epipolar_image_left"],
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
                        self.epsg,
                    )

            if self.dense_matching_app.get_method() == "pandora_auto":
                # Copy the initial corr_config in order to keep
                # the inputs that have already been checked
                method = self.dense_matching_app.dense_matching_method
                corr_cfg = method.corr_config.copy()

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
                self.used_conf[PIPELINE][APPLICATIONS]["dense_matching"][
                    "loader_conf"
                ] = conf
                self.used_conf[PIPELINE][APPLICATIONS]["dense_matching"][
                    "method"
                ] = "pandora_custom"

                # Re initialization of the dense matching application
                self.dense_matching_app = Application(
                    "dense_matching",
                    cfg=self.used_conf[PIPELINE][APPLICATIONS][
                        "dense_matching"
                    ],
                )

                # Update the corr_config with the inputs that have
                # already been checked
                self.dense_matching_app.dense_matching_method.corr_config[
                    "input"
                ] = corr_cfg["input"]

            # Run epipolar matching application
            epipolar_disparity_map = self.dense_matching_app.run(
                self.pairs[pair_key]["new_epipolar_image_left"],
                self.pairs[pair_key]["new_epipolar_image_right"],
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
            (self.pairs[pair_key]["filled_epipolar_disparity_map"]) = (
                self.dense_match_filling.run(
                    epipolar_disparity_map,
                    orchestrator=self.cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir, "dense_match_filling", pair_key
                    ),
                    pair_key=pair_key,
                )
            )

            if self.quit_on_app("dense_match_filling"):
                continue  # keep iterating over pairs, but don't go further

        # quit if any app in the loop over the pairs was the last one
        # pylint:disable=too-many-boolean-expressions
        if self.quit_on_app("dense_matching") or self.quit_on_app(
            "dense_match_filling"
        ):
            return True

        return False

    def disparity_to_depth_maps(self):  # noqa: C901
        """
        Creates the depth map from the disparity maps,
        by following the CARS pipeline's steps.
        """
        self.cars_orchestrator.set_target_task(
            self.task_ids["disparity_to_depth_maps"]
        )

        output = self.used_conf[OUTPUT]

        if isinstance(output[sens_cst.GEOID], str):
            output_geoid_path = output[sens_cst.GEOID]
        elif (
            isinstance(output[sens_cst.GEOID], bool) and output[sens_cst.GEOID]
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

        if self.use_sensor_disp:

            sensor_depth_maps_dir = os.path.join(
                self.dump_dir, "sensor_dense_matching"
            )
            safe_makedirs(sensor_depth_maps_dir)

            # TODO: add warning in checker if not always common image

            sensor_matches = []
            sensor_common_image = None
            sensor_inputs_secondary = []
            # Generate sensor matches
            for _, (pair_key, _, _) in enumerate(self.list_sensor_pairs):
                depth_map_pair_dir = os.path.join(
                    sensor_depth_maps_dir, pair_key
                )
                if sensor_common_image is None:
                    sensor_common_image = self.pairs[pair_key][
                        "sensor_image_left"
                    ]
                else:
                    if (
                        self.pairs[pair_key]["sensor_image_left"]
                        != sensor_common_image
                    ):
                        raise RuntimeError("No common reference image.")
                safe_makedirs(depth_map_pair_dir)
                self.pairs[pair_key]["sensor_dense_matches"] = (
                    self.epipolar_to_sensor_matching_app.run(
                        self.pairs[pair_key]["sensor_image_left"],
                        self.pairs[pair_key]["corrected_grid_left"],
                        self.pairs[pair_key]["corrected_grid_right"],
                        self.pairs[pair_key]["filled_epipolar_disparity_map"],
                        orchestrator=self.cars_orchestrator,
                        pair_folder=depth_map_pair_dir,
                        pair_key=pair_key,
                    )
                )
                # Generate triangulation input
                sensor_matches.append(
                    self.pairs[pair_key]["sensor_dense_matches"]
                )
                sensor_inputs_secondary.append(
                    self.pairs[pair_key]["sensor_image_right"]
                )

                if self.quit_on_app("sensor_matching"):
                    continue

            triangulation_kwargs = {"sensor_matches": sensor_matches}

            cloud_id_mono = 0
            pair_key = "common_sensor"
            self.pairs[pair_key] = {
                "sensor_image_left": sensor_common_image,
                "sensor_image_right": sensor_inputs_secondary,
            }

            iter_list = [(cloud_id_mono, pair_key, triangulation_kwargs)]

        else:
            iter_list = []
            for cloud_id, (pair_key, _, _) in enumerate(self.list_sensor_pairs):
                triangulation_kwargs = {
                    "grid_left": self.pairs[pair_key]["corrected_grid_left"],
                    "grid_right": self.pairs[pair_key]["corrected_grid_right"],
                    "epipolar_disparity_map": self.pairs[pair_key][
                        "filled_epipolar_disparity_map"
                    ],
                    "epipolar_image": self.pairs[pair_key][
                        "new_epipolar_image_left"
                    ],
                    "uncorrected_grid_right": self.pairs[pair_key][
                        "grid_right"
                    ],
                }

                iter_list.append((cloud_id, pair_key, triangulation_kwargs))

        for cloud_id, pair_key, triangulation_kwargs in iter_list:
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
                    and (
                        len(self.pc_outlier_removal_apps) == 0
                        or "tif" in self.product_format["point_cloud"]
                    )
                )
                else None
            )

            # Run triangulation application : sensor or epipolar
            point_cloud = self.triangulation_application.run(
                self.pairs[pair_key]["sensor_image_left"],
                self.pairs[pair_key]["sensor_image_right"],
                self.geom_plugin_without_dem_and_geoid,
                epsg=self.epsg,
                denoising_overload_fun=None,
                source_pc_names=self.pairs_names,
                orchestrator=self.cars_orchestrator,
                pair_dump_dir=os.path.join(
                    self.dump_dir, "triangulation", pair_key
                ),
                pair_key=pair_key,
                geoid_path=output_geoid_path,
                cloud_id=cloud_id,
                performance_maps_param=(
                    self.dense_matching_app.get_performance_map_parameters()
                ),
                point_cloud_format=self.product_format["point_cloud"],
                point_cloud_dir=triangulation_point_cloud_dir,
                save_output_coordinates=(len(self.pc_outlier_removal_apps) == 0)
                and self.save_output_point_cloud
                and "tif" in self.product_format["point_cloud"],
                save_output_color=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_IMAGE]
                and "tif" in self.product_format["point_cloud"],
                save_output_classification=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_CLASSIFICATION]
                and "tif" in self.product_format["point_cloud"],
                save_output_filling=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_FILLING]
                and "tif" in self.product_format["point_cloud"],
                save_output_performance_map=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_PERFORMANCE_MAP]
                and "tif" in self.product_format["point_cloud"],
                save_output_ambiguity=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_AMBIGUITY]
                and "tif" in self.product_format["point_cloud"],
                save_output_edges=bool(point_cloud_dir)
                and self.auxiliary[out_cst.AUX_EDGES]
                and "tif" in self.product_format["point_cloud"],
                save_residues=bool(point_cloud_dir) and self.use_sensor_disp,
                **triangulation_kwargs,
            )

            if self.quit_on_app("triangulation"):
                continue  # keep iterating over pairs, but don't go further

            filtered_epipolar_point_cloud = point_cloud
            filtering_point_cloud_dir = None
            for app_key, app in self.pc_outlier_removal_apps.items():
                app_key_is_last = (
                    app_key == list(self.pc_outlier_removal_apps)[-1]
                )
                filtering_point_cloud_dir = (
                    point_cloud_dir if app_key_is_last else None
                )

                filtered_epipolar_point_cloud = app.run(
                    filtered_epipolar_point_cloud,
                    point_cloud_dir=filtering_point_cloud_dir,
                    point_cloud_format=self.product_format["point_cloud"],
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

            disparity_to_depth_task_id = self.task_ids[
                "disparity_to_depth_maps"
            ]
            progress_tree = ProgressTree()
            runs_before = progress_tree.get_task_started_runs(
                disparity_to_depth_task_id
            )

            self.cars_orchestrator.breakpoint()

            runs_after = progress_tree.get_task_started_runs(
                disparity_to_depth_task_id
            )
            if runs_before is not None and runs_after == runs_before:
                # the breakpoint didn't compute anything
                # still notify the progress tree to avoid blocking the pipeline
                progress_tree.notify(
                    disparity_to_depth_task_id,
                    "started",
                    total=1,
                )
                progress_tree.notify(
                    disparity_to_depth_task_id,
                    "completed",
                )

            dir_to_check = [
                d
                for d in (
                    triangulation_point_cloud_dir,
                    filtering_point_cloud_dir,
                )
                if d is not None
            ]

            if len(dir_to_check) == 0:
                continue
            for folder in dir_to_check:
                if "tif" not in self.product_format["point_cloud"]:
                    continue
                true_path = os.path.join(folder, "tif")
                true_path = Path(true_path)
                for file in true_path.iterdir():
                    if not os.path.exists(file) or os.path.getsize(file) == 0:
                        raise RuntimeError(
                            "The file {} generated at resolution "
                            "{} is empty".format(file, self.working_res)
                        )

                    with rasterio.open(file) as src:
                        is_full_nan = True
                        for _, window in src.block_windows(1):
                            data = src.read(1, window=window)
                            is_full_nan = np.isnan(data).all()

                            if not is_full_nan:
                                break

                        if is_full_nan:
                            raise RuntimeError(
                                "The file {} generated at "
                                "resolution"
                                "{} is full of nan".format(
                                    file, self.working_res
                                )
                            )

        # quit if any app in the loop over the pairs was the last one
        # pylint:disable=too-many-boolean-expressions
        if (
            self.quit_on_app("triangulation")
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
        interval_was_cropped = self.cars_orchestrator.breakpoint()

        cropped_file_name = os.path.join(
            os.path.dirname(dsm_file_name), "cropped_disp_range.tif"
        )
        if not interval_was_cropped and os.path.exists(cropped_file_name):
            os.remove(cropped_file_name)

        if (
            not os.path.exists(dsm_file_name)
            or os.path.getsize(dsm_file_name) == 0
        ):
            raise RuntimeError(
                "The DSM generated at resolution {} is empty".format(
                    self.working_res
                )
            )

        # saved used configuration
        self.save_configurations()

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
            filling_file_name_inter = os.path.join(
                os.path.dirname(dsm_file_name), "filling_inter.tif"
            )
            filling_file_name_out = os.path.join(
                os.path.dirname(dsm_file_name), "filling.tif"
            )

            if not os.path.exists(filling_file_name):
                # create input filling file
                with rasterio.open(dsm_file_name) as src:
                    profile = src.profile

                profile.update(dtype="uint8", nodata=0)

                with rasterio.open(filling_file_name, "w", **profile):
                    pass

            self.merge_filling_bands(
                filling_file_name,
                filling_file_name_inter,
                self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING],
                dsm_file_name,
                os.path.join(
                    os.path.dirname(dsm_file_name), "invalidity_mask.tif"
                ),
                local_orchestrator=self.cars_orchestrator,
            )

            self.cars_orchestrator.breakpoint()

            try:
                shutil.move(filling_file_name_inter, filling_file_name_out)
            except FileNotFoundError:
                logging.warning("Filling file not found")

        invalidity_mask_file = os.path.join(
            os.path.dirname(dsm_file_name), "invalidity_mask.tif"
        )
        if os.path.exists(invalidity_mask_file):
            self.merge_invalidity_mask_bands(
                invalidity_mask_file,
                dsm_file_name,
            )

        self.dtm_generation_dump_dir = os.path.join(
            self.dump_dir, "dtm_generation"
        )

        if self.save_output_dtm and self.which_resolution in (
            "single",
            "final",
        ):
            _ = self.dtm_generation_application.run(
                dsm_file_name,
                self.dtm_generation_dump_dir,
                self.cars_orchestrator,
                os.path.join(self.out_dir, "dsm"),
            )

        return False

    @cars_profile(name="merge filling bands", interval=0.5)
    def merge_filling_bands(  # pylint: disable=R0917
        self,
        in_filling_path,
        out_filling_path,
        aux_filling,
        dsm_file,
        invalidity_mask_file,
        local_orchestrator=None,
        tile_size=10000,
    ):
        """
        Merge filling bands to get mono band in output
        """
        if local_orchestrator is None:
            local_orchestrator = orchestrator.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        with rasterio.open(in_filling_path) as src:
            profile = src.profile
            height = src.height
            width = src.width
            filling_dtype = src.dtypes[0]
            nodata_value = src.nodata

        # Update to one band
        profile.update(count=1, dtype=filling_dtype)

        filling_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Monoband Filling"
        )
        # Compute tiling grid
        filling_cars_ds.tiling_grid = tiling.generate_tiling_grid(
            0,
            0,
            height,
            width,
            tile_size,
            tile_size,
        )

        # Saving infos
        [
            saving_info,
        ] = local_orchestrator.get_saving_infos([filling_cars_ds])

        # Save list
        local_orchestrator.add_to_save_lists(
            out_filling_path,
            "mono_filling",
            filling_cars_ds,
            dtype=filling_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="MonoBand Filling",
        )

        for row in range(filling_cars_ds.shape[0]):
            for col in range(filling_cars_ds.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = orchestrator.update_saving_infos(
                    saving_info, row=row, col=col
                )
                window = filling_cars_ds.get_window_as_dict(row, col)
                # Compute images
                (
                    filling_cars_ds[row, col]
                ) = local_orchestrator.cluster.create_task(
                    merge_filling_bands_wrapper, nout=1
                )(
                    in_filling_path,
                    aux_filling,
                    dsm_file,
                    invalidity_mask_file,
                    window=window,
                    saving_info=full_saving_info,
                    profile_filling=profile,
                )

        return filling_cars_ds

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

    @cars_profile(name="merge invalidity mask bands", interval=0.5)
    def merge_invalidity_mask_bands(self, invalidity_mask_path, dsm_file):
        """
        Merge invalidity mask bands to get mono band in output
        """
        if not os.path.exists(invalidity_mask_path):
            logging.warning("no invalidity mask to merge")
            return False
        with rasterio.open(dsm_file) as in_dsm:
            dsm_msk = in_dsm.read_masks(1)

        with rasterio.open(invalidity_mask_path) as src:
            nb_bands = src.count

            if nb_bands == 1:
                return False

            mask_multi_bands = src.read()
            mask_mono_band = np.zeros(mask_multi_bands.shape[1:3])
            profile = src.profile
            mask = src.read_masks(1)
            mask_mono_band[mask == 0] = 0

            # To get the right footprint
            mask_mono_band = np.logical_or(dsm_msk, mask).astype(np.uint8)

            # to keep the previous classif convention
            mask_mono_band[mask_mono_band == 0] = src.nodata
            mask_mono_band[mask_mono_band == 1] = 0

            for num_band in range(0, nb_bands):
                mask_1 = mask_mono_band == 0
                mask_2 = mask_multi_bands[num_band, :, :] == 1
                mask_mono_band[mask_1 & mask_2] = num_band + 1

        profile.update(count=1, dtype=mask_mono_band.dtype)
        with rasterio.open(invalidity_mask_path, "w", **profile) as src:
            src.write(mask_mono_band, 1)

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

        if (
            not self.used_conf[PIPELINE][ADVANCED][
                adv_cst.SAVE_INTERMEDIATE_DATA
            ]
            and not self.tie_point_save
        ):
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
        args=None,  # pylint: disable=W0613
        which_resolution="single",
        working_res=1,
        res_factor=None,
        log_dir=None,
        previous_out_dir=None,
        parent_pipeline_id=None,
        tie_points_pipeline_id=None,
    ):  # noqa C901
        """
        Run pipeline

        :param parent_pipeline_id: Optional parent pipeline ID if nested
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

        self.working_res = working_res

        self.res_factor = res_factor

        self.previous_out_dir = previous_out_dir

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
            # link metadata
            self.metadata = self.cars_orchestrator.out_yaml
            # initialize out_json
            self.cars_orchestrator.update_out_info({"version": __version__})

            # Register surface modeling tasks with progress tracking
            task_id = self.setup_progress_tracking(
                parent_pipeline_id,
                tie_points_pipeline_id=tie_points_pipeline_id,
            )
            self.cars_orchestrator.set_target_task(task_id)

            if self.compute_depth_map:
                stop = self.sensor_to_disparity()
                if not stop:
                    self.disparity_to_depth_maps()

            if self.save_output_dsm or self.save_output_point_cloud:
                self.preprocess_depth_maps()

                if self.save_output_dsm:
                    if (
                        hasattr(self, "task_ids")
                        and "rasterize_point_cloud" in self.task_ids
                    ):
                        self.cars_orchestrator.set_target_task(
                            self.task_ids["rasterize_point_cloud"]
                        )
                    self.rasterize_point_cloud()

            self.final_cleanup()
