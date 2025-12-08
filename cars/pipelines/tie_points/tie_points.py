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
import os

from json_checker import Checker, Or

from cars.applications.application import Application
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


@Pipeline.register(
    "tie_points",
)
class TiePointsPipeline(PipelineTemplate):
    """
    Tie points pipeline
    """

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

        # Check advanced parameters
        output_dem_dir = os.path.join(
            conf[OUTPUT][out_cst.OUT_DIRECTORY], "dump_dir", "initial_elevation"
        )
        safe_makedirs(output_dem_dir)
        pipeline_conf = conf.get("tie_points", {})
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
        self.used_conf[ADVANCED] = advanced

        # Check conf output
        output = self.check_output(conf[OUTPUT])

        self.used_conf[OUTPUT] = output
        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        self.dump_dir = os.path.join(self.out_dir, "dump_dir")

        self.save_all_intermediate_data = self.used_conf[ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]

        # Check conf application
        application_conf = self.check_applications(
            pipeline_conf.get(APPLICATIONS, {})
        )

        self.used_conf[APPLICATIONS] = application_conf

        self.out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]

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

        input_config = sensor_inputs.sensors_check_inputs(
            conf, config_dir=config_dir
        )
        return input_config

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
        ) = sensor_inputs.check_geometry_plugin(
            inputs,
            conf.get(adv_cst.GEOMETRY_PLUGIN, None),
            output_dem_dir,
        )

        # Validate inputs
        schema = {
            adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
            adv_cst.GEOMETRY_PLUGIN: Or(str, dict),
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

    def run(self, log_dir=None):
        """
        Run pipeline

        """
        if log_dir is None:
            log_dir = os.path.join(self.out_dir, "logs")

        # Load geomodels directly on conf object
        (
            pair_key,
            sensor_image_left,
            sensor_image_right,
        ) = sensor_inputs.generate_inputs(
            self.used_conf[INPUT], self.geom_plugin_without_dem_and_geoid
        )[
            0
        ]

        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=log_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as cars_orchestrator:

            # Run applications

            # Run grid generation
            # We generate grids with dem if it is provided.
            if (
                self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ]
                is None
            ):
                geom_plugin = self.geom_plugin_without_dem_and_geoid
            else:
                geom_plugin = self.geom_plugin_with_dem_and_geoid

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
                ),
                pair_key=pair_key,
            )

            # Get required bands of resampling
            required_bands = self.sparse_matching_app.get_required_bands()

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
                    self.dump_dir, "resampling", "initial"
                ),
                pair_key=pair_key,
                margins_fun=self.sparse_matching_app.get_margins_fun(),
                tile_width=None,
                tile_height=None,
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
                ),
                pair_key=pair_key,
            )

            cars_orchestrator.breakpoint()

            # Filter and save matches
            _ = self.sparse_matching_app.filter_matches(
                epipolar_matches_left,
                grid_left,
                grid_right,
                geom_plugin,
                orchestrator=cars_orchestrator,
                pair_folder=self.out_dir,
                pair_key=pair_key,
                save_matches=True,
            )
