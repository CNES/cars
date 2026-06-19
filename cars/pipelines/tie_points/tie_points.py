#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
CARS tie points pipeline class file
"""
import logging
import os

import yaml
from json_checker import Checker, Or

import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars.applications.application import Application
from cars.core import roi_tools
from cars.core.progress.progress import ProgressTree
from cars.core.utils import safe_makedirs
from cars.orchestrator import orchestrator
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate

PIPELINE = "tie_points"


@Pipeline.register(
    PIPELINE,
)
class TiePointsPipeline(PipelineTemplate):  # pylint: disable=R0902
    """
    Tie points pipeline
    """

    def setup_progress_tracking(self, parent_pipeline_id=None):
        """
        Setup progress tracking for tie points.

        :param parent_pipeline_id: Optional parent pipeline ID
        :type parent_pipeline_id: int or None
        :return: Task ID to pass to orchestrator via set_target_task()
        :rtype: int
        """
        progress_tree = ProgressTree()
        self.pipeline_progress_id = progress_tree.begin_pipeline(
            "Tie Points", parent_id=parent_pipeline_id
        )
        self.task_progress_id = progress_tree.register_task(
            self.pipeline_progress_id,
            "tie_points",
            weight=1,
        )
        return self.task_progress_id

    def __init__(self, conf, config_dir=None):
        """
        Creates pipeline

        Directly creates class attributes:
            used_conf
            geom_plugin_without_dem_and_geoid
            geom_plugin_with_dem_and_geoid

        :param conf: configuration
        :type conf: dictionary
        :param config_dir: path to dir containing json/yaml
        :type config_dir: str
        """

        # Used conf
        self.used_conf = {}

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
        inputs = self.check_inputs(conf[INPUT], config_dir=config_dir)
        self.used_conf[INPUT] = inputs

        # Check conf output
        output = self.check_output(conf[OUTPUT])

        self.used_conf[OUTPUT] = output
        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

        # Check advanced parameters
        output_dem_dir = os.path.join(self.dump_dir, "initial_elevation")
        safe_makedirs(output_dem_dir)
        pipeline_conf = conf.get(PIPELINE, {})
        self.used_conf[PIPELINE] = {}
        (
            inputs,
            advanced,
            self.geometry_plugin,
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
        ) = self.check_advanced_parameters(
            inputs,
            pipeline_conf.get(ADVANCED, {}),
            output_dem_dir=output_dem_dir,
        )
        self.used_conf[PIPELINE][ADVANCED] = advanced
        self.epipolar_roi_margin_factor = advanced["epipolar_roi_margin_factor"]

        # Check conf output
        output = self.check_output(conf[OUTPUT])

        self.used_conf[OUTPUT] = output
        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

        self.save_all_intermediate_data = self.used_conf[PIPELINE][ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]

        # Check conf application
        application_conf = self.check_applications(
            pipeline_conf.get(APPLICATIONS, {})
        )
        self.cars_orchestrator = None
        # Check conf application vs inputs application
        application_conf = self.check_applications_with_inputs(
            self.used_conf[INPUT], application_conf
        )

        self.used_conf[PIPELINE][APPLICATIONS] = application_conf

        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]

        # progress tracking attributes
        self.pipeline_progress_id = None
        self.task_progress_id = None

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
        overloaded_conf = conf.copy()

        overloaded_conf[sens_cst.RECTIFICATION_GRIDS] = conf.get(
            sens_cst.RECTIFICATION_GRIDS, None
        )

        overloaded_conf[sens_cst.PAIRING] = conf.get(sens_cst.PAIRING, None)

        overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)

        overloaded_conf[sens_cst.INITIAL_ELEVATION] = (
            sensor_inputs.get_initial_elevation(
                conf.get(sens_cst.INITIAL_ELEVATION, None)
            )
        )
        overloaded_conf[sens_cst.LOADERS] = sensor_inputs.check_loaders(
            conf.get(sens_cst.LOADERS, {})
        )

        classif_loader = overloaded_conf[sens_cst.LOADERS][
            sens_cst.INPUT_CLASSIFICATION
        ]

        overloaded_conf[sens_cst.FILLING] = sensor_inputs.check_filling(
            conf.get(sens_cst.FILLING, {}), classif_loader
        )

        # Validate inputs
        inputs_schema = {
            sens_cst.SENSORS: dict,
            sens_cst.PAIRING: Or([[str]], None),
            sens_cst.RECTIFICATION_GRIDS: Or(dict, None),
            sens_cst.INITIAL_ELEVATION: Or(str, dict, None),
            sens_cst.ROI: Or(str, dict, None),
            sens_cst.LOADERS: dict,
            sens_cst.FILLING: dict,
        }

        checker_inputs = Checker(inputs_schema)
        checker_inputs.validate(overloaded_conf)

        sensor_inputs.check_sensors(conf, overloaded_conf, config_dir)

        # Check srtm dir
        sensor_inputs.check_srtm(
            overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
        )

        return overloaded_conf

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
        if self.sparse_matching_app.elevation_delta_lower_bound is None:
            self.sparse_matching_app.used_config[
                "elevation_delta_lower_bound"
            ] = (-500 if initial_elevation else -1000)
            self.sparse_matching_app.elevation_delta_lower_bound = (
                self.sparse_matching_app.used_config[
                    "elevation_delta_lower_bound"
                ]
            )
        if self.sparse_matching_app.elevation_delta_upper_bound is None:
            self.sparse_matching_app.used_config[
                "elevation_delta_upper_bound"
            ] = (1000 if initial_elevation else 9000)
            self.sparse_matching_app.elevation_delta_upper_bound = (
                self.sparse_matching_app.used_config[
                    "elevation_delta_upper_bound"
                ]
            )
        application_conf["sparse_matching"] = (
            self.sparse_matching_app.get_conf()
        )

        sparse_method = self.sparse_matching_app.sparse_matching_method

        self.method_margins = None
        if type(sparse_method).__name__ == "Pandora2DSparseMethod":
            for key1, key2 in inputs_conf["pairing"]:
                corr_cfg = sparse_method.loader.get_conf()
                nodata_left = inputs_conf["sensors"][key1]["image"]["no_data"]
                nodata_right = inputs_conf["sensors"][key2]["image"]["no_data"]

                cfg, method_margins = sparse_method.loader.check_conf(
                    corr_cfg,
                    nodata_left,
                    nodata_right,
                )
                sparse_method.corr_config = cfg
                self.method_margins = method_margins

        return application_conf

    @staticmethod
    def check_advanced_parameters(inputs, conf, output_dem_dir=None):
        """
        Check the advanced parameters consistency

        :param conf: configuration of inputs
        :type conf: dict
        :param config_dir: directory of used json/yaml, if
            user filled paths with relative paths
        :type config_dir: str

        :return: overloaded inputs
        :rtype: dict
        """

        overloaded_conf = conf.copy()

        overloaded_conf["epipolar_roi_margin_factor"] = conf.get(
            "epipolar_roi_margin_factor", 0.2
        )

        overloaded_conf[adv_cst.SAVE_INTERMEDIATE_DATA] = conf.get(
            adv_cst.SAVE_INTERMEDIATE_DATA, False
        )

        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            inputs,
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_without_dem_and_geoid,
            geom_plugin_with_dem_and_geoid,
            _,
            _,
        ) = sensor_inputs.check_geometry_plugin(
            inputs,
            conf.get(adv_cst.GEOMETRY_PLUGIN, None),
            output_dem_dir,
        )

        # Validate inputs
        schema = {
            adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
            adv_cst.GEOMETRY_PLUGIN: Or(str, dict),
            "epipolar_roi_margin_factor": float,
        }

        checker_advanced_parameters = Checker(schema)
        checker_advanced_parameters.validate(overloaded_conf)

        return (
            inputs,
            overloaded_conf,
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            geom_plugin_without_dem_and_geoid,
            geom_plugin_with_dem_and_geoid,
        )

    @staticmethod
    def check_output(conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :return: overloader output
        :rtype: dict
        """
        overloaded_conf = conf.copy()
        out_dir = conf[out_cst.OUT_DIRECTORY]
        out_dir = os.path.abspath(out_dir)
        # Ensure that output directory and its subdirectories exist
        safe_makedirs(out_dir)

        # Overload some parameters
        overloaded_conf[out_cst.OUT_DIRECTORY] = out_dir

        # Check schema
        output_schema = {out_cst.OUT_DIRECTORY: str}
        checker_output = Checker(output_schema)
        checker_output.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check the given configuration for applications,
        and generates needed applications for pipeline.

        :param conf: configuration of applications
        :type conf: dict
        """

        # Initialize used config
        used_conf = {}

        needed_applications = [
            "grid_generation",
            "resampling",
            "sparse_matching",
        ]

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            if used_conf[app_key] is not None:
                used_conf[app_key]["save_intermediate_data"] = (
                    self.save_all_intermediate_data
                    or used_conf[app_key].get("save_intermediate_data", False)
                )

        # Epipolar grid generation
        self.epipolar_grid_generation_application = Application(
            "grid_generation",
            cfg=used_conf.get("grid_generation", {}),
        )
        used_conf["grid_generation"] = (
            self.epipolar_grid_generation_application.get_conf()
        )

        # Image resampling
        self.resampling_application = Application(
            "resampling",
            cfg=used_conf.get("resampling", {}),
        )
        used_conf["resampling"] = self.resampling_application.get_conf()

        # Sparse Matching
        self.sparse_matching_app = Application(
            "sparse_matching",
            cfg=used_conf.get("sparse_matching", {}),
        )
        used_conf["sparse_matching"] = self.sparse_matching_app.get_conf()

        return used_conf

    def update_via_metadata_file(self, previous_dir, pair_key, res_factor):
        """
        Update the eipolar error values via the metatdata.yaml file
        """
        path_metadata = os.path.join(previous_dir, "metadata.yaml")
        with open(path_metadata, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if self.sparse_matching_app.epipolar_error_maximum_bias == "auto":
                self.sparse_matching_app.epipolar_error_maximum_bias = (
                    data["applications"]["match_filtering"][pair_key][
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_STD
                    ]
                    * res_factor
                )

            if self.sparse_matching_app.epipolar_error_upper_bound == "auto":
                self.sparse_matching_app.epipolar_error_upper_bound = (
                    2
                    * data["applications"]["match_filtering"][pair_key][
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_STD
                    ]
                    * res_factor
                )

            if self.sparse_matching_app.epipolar_error_estimation == "auto":
                self.sparse_matching_app.epipolar_error_estimation = (
                    data["applications"]["match_filtering"][pair_key][
                        sm_cst.BEFORE_CORRECTION_EPI_ERROR_MEAN
                    ]
                    * res_factor
                )

            self.sparse_matching_app.epipolar_error_maximum_bias = min(
                self.sparse_matching_app.epipolar_error_maximum_bias, 50
            )

            self.sparse_matching_app.epipolar_error_upper_bound = min(
                self.sparse_matching_app.epipolar_error_upper_bound, 10
            )

    def run(  # pylint: disable=too-many-positional-arguments
        self,
        args=None,  # pylint: disable=W0613
        log_dir=None,
        disp_range_grid=None,
        epipolar_roi=None,
        cars_orchestrator=None,
        previous_dir=None,
        res_factor=None,
        parent_pipeline_id=None,
        task_progress_id=None,
    ):
        """
        Run pipeline

        """
        if log_dir is None:
            log_dir = os.path.join(self.out_dir, "logs")

        sparse_method = self.sparse_matching_app.sparse_matching_method

        # Load geomodels directly on conf object
        sensor_inputs.load_geomodels(
            self.used_conf[INPUT], self.geom_plugin_without_dem_and_geoid
        )
        list_sensor_pairs = sensor_inputs.generate_pairs(self.used_conf[INPUT])

        inherent_orchestrator = False
        if cars_orchestrator is None:
            cars_orchestrator = orchestrator.Orchestrator(
                orchestrator_conf=self.used_conf[ORCHESTRATOR],
                out_dir=self.out_dir,
                log_dir=log_dir,
                out_yaml_path=os.path.join(
                    self.out_dir,
                    out_cst.INFO_FILENAME,
                ),
            )
            inherent_orchestrator = True

        self.cars_orchestrator = cars_orchestrator
        if task_progress_id is not None:
            # Route orchestrator progress events to the given progress task
            cars_orchestrator.set_target_task(task_progress_id)
        elif parent_pipeline_id is not None:
            # Fallback to nested creation for standalone usage.
            task_id = self.setup_progress_tracking(parent_pipeline_id)
            cars_orchestrator.set_target_task(task_id)

        # Run applications
        if epipolar_roi is not None:
            roi_tools.expand_roi(
                epipolar_roi, margin_ratio=self.epipolar_roi_margin_factor
            )

        # Run grid generation
        # We generate grids with dem if it is provided.
        if self.geom_plugin_with_dem_and_geoid.dem is None:
            geom_plugin = self.geom_plugin_without_dem_and_geoid
        else:
            geom_plugin = self.geom_plugin_with_dem_and_geoid

        for (
            pair_key,
            sensor_image_left,
            sensor_image_right,
        ) in list_sensor_pairs:

            if previous_dir is not None:
                self.update_via_metadata_file(
                    previous_dir, pair_key, res_factor
                )

            method_sparse = self.sparse_matching_app.sparse_matching_method
            if hasattr(method_sparse, "corr_config"):
                method_sparse.corr_config["segment_mode"]["memory_per_work"] = (
                    cars_orchestrator.cluster.checked_conf_cluster[
                        "max_ram_per_worker"
                    ]
                )

            if self.used_conf[INPUT][sens_cst.RECTIFICATION_GRIDS] is None:
                # Generate rectification grids
                (
                    grid_left,
                    grid_right,
                ) = self.epipolar_grid_generation_application.run(
                    sensor_image_left,
                    sensor_image_right,
                    geom_plugin,
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(
                        self.dump_dir,
                        "epipolar_grid_generation",
                        "initial",
                        pair_key,
                    ),
                    pair_key=pair_key,
                )
            else:
                image_keys = list(self.used_conf[INPUT][sens_cst.SENSORS])
                grid_left = self.used_conf[INPUT][sens_cst.RECTIFICATION_GRIDS][
                    image_keys[0]
                ]
                grid_right = self.used_conf[INPUT][
                    sens_cst.RECTIFICATION_GRIDS
                ][image_keys[1]]

            # Get required bands of resampling
            required_bands = self.sparse_matching_app.get_required_bands()

            tile_width = sparse_method.tile_width
            tile_height = sparse_method.tile_height
            if disp_range_grid is not None:
                margins_fun_resam = (
                    self.sparse_matching_app.get_margins_tile_fun(
                        grid_left,
                        disp_range_grid,
                    )
                )

                disp_min = disp_range_grid["global_min"]
                disp_max = disp_range_grid["global_max"]
                logging.info(
                    "Global disparity range for sparse matching : "
                    "[{} pix, {} pix]".format(disp_min, disp_max)
                )

                disp_to_alt_ratio = grid_left["disp_to_alt_ratio"]
                self.sparse_matching_app.elevation_delta_lower_bound = (
                    -disp_max * disp_to_alt_ratio
                )
                self.sparse_matching_app.elevation_delta_upper_bound = (
                    -disp_min * disp_to_alt_ratio
                )

                margins_fun = sparse_method.add_margin_wrapper(
                    margins_fun_resam, self.method_margins
                )

            else:
                margins_fun_resam = (
                    self.sparse_matching_app.get_margins_strip_fun()
                )
                margins_fun = sparse_method.add_margin_wrapper(
                    margins_fun_resam, self.method_margins
                )
            (
                epipolar_image_left,
                epipolar_image_right,
            ) = self.resampling_application.run(
                sensor_image_left,
                sensor_image_right,
                grid_left,
                grid_right,
                geom_plugin,
                orchestrator=cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir, "resampling", "initial", pair_key
                ),
                pair_key=pair_key,
                margins_fun=margins_fun,
                tile_width=tile_width,
                tile_height=tile_height,
                epipolar_roi=epipolar_roi,
                required_bands=required_bands,
            )

            # Compute sparse matching
            (
                epipolar_matches_left,
                _,
            ) = self.sparse_matching_app.run(
                epipolar_image_left,
                epipolar_image_right,
                grid_left["disp_to_alt_ratio"],
                orchestrator=cars_orchestrator,
                pair_folder=os.path.join(
                    self.dump_dir,
                    "sparse_matching",
                    pair_key,
                ),
                pair_key=pair_key,
                disp_range_grid=disp_range_grid,
            )

            cars_orchestrator.breakpoint()

            # Filter and save matches
            _ = self.sparse_matching_app.filter_matches(
                epipolar_matches_left,
                grid_left,
                grid_right,
                geom_plugin,
                orchestrator=cars_orchestrator,
                pair_folder=os.path.join(
                    self.out_dir,
                    pair_key,
                ),
                pair_key=pair_key,
                save_matches=True,
            )

        if inherent_orchestrator:
            cars_orchestrator.cleanup()
