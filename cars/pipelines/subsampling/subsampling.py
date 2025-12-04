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
CARS subsampling pipeline class file
"""
# Standard imports
from __future__ import print_function

import os
from pathlib import Path

import yaml

from cars.applications.application import Application
from cars.core.utils import safe_makedirs
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters
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
from cars.pipelines.unit.unit_pipeline import UnitPipeline


@Pipeline.register(
    "subsampling",
)
class SubsamplingPipeline(PipelineTemplate):
    """
    SubsamplingPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf, config_dir=None):  # noqa: C901
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_dir: path to dir containing json or yaml file
        :type config_dir: str
        """

        self.config_dir = config_dir
        # Transform relative path to absolute path
        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)

        # Check global conf
        self.check_global_schema(conf)

        self.out_dir = conf[OUTPUT][out_cst.OUT_DIRECTORY]

        self.subsampling_dir = os.path.join(
            os.path.dirname(self.out_dir), "subsampling"
        )

        # Get epipolar resolutions to use
        self.epipolar_resolutions = (
            advanced_parameters.get_epipolar_resolutions(conf.get(ADVANCED, {}))
        )
        if isinstance(self.epipolar_resolutions, int):
            self.epipolar_resolutions = [self.epipolar_resolutions]

        # Check input
        conf[INPUT] = self.check_inputs(conf)
        # check advanced
        conf[ADVANCED] = self.check_advanced(conf)
        # check output
        conf[OUTPUT] = self.check_output(conf)

        self.used_conf = {}

        # Check conf orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[INPUT] = conf[INPUT]
        self.used_conf[OUTPUT] = conf[OUTPUT]
        self.used_conf[ADVANCED] = conf[ADVANCED]

        self.used_conf[APPLICATIONS] = self.check_applications(
            conf[APPLICATIONS]
        )

        self.intermediate_data_dir = os.path.join(
            self.out_dir, "intermediate_data"
        )

    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs given

        :param conf: configuration
        :type conf: dict
        :param config_dir: directory of used json, if
            user filled paths with relative paths
        :type config_dir: str

        :return: overloader inputs
        :rtype: dict
        """
        return UnitPipeline.check_inputs(
            conf[INPUT], config_dir=self.config_dir
        )

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        conf_output, _ = output_parameters.check_output_parameters(
            conf[INPUT], conf[OUTPUT], None
        )
        return conf_output

    def check_advanced(self, conf):
        """
        Check all conf for advanced configuration

        :return: overridden advanced conf
        :rtype: dict
        """
        (_, advanced, _, _, _, _, _, _) = (
            advanced_parameters.check_advanced_parameters(
                conf[INPUT],
                conf.get(ADVANCED, {}),
                check_epipolar_a_priori=True,
            )
        )
        return advanced

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

        used_conf = {}

        self.sensors_subsampling = Application(
            "sensors_subsampling",
            cfg=used_conf.get("sensors_subsampling", {}),
        )
        used_conf["sensors_subsampling"] = self.sensors_subsampling.get_conf()

        return used_conf

    def formatting(self, key, out_dir):
        """
        Format the input.yaml with new inputs
        """

        inputs = self.used_conf[INPUT]
        sensor = inputs[sens_cst.SENSORS][key]

        def replace_path(path):
            return os.path.join(out_dir, key, Path(path).name)

        sensor[sens_cst.INPUT_IMG]["bands"]["b0"]["path"] = replace_path(
            sensor[sens_cst.INPUT_IMG]["bands"]["b0"]["path"]
        )

        if (
            sens_cst.INPUT_CLASSIFICATION in sensor
            and sensor[sens_cst.INPUT_CLASSIFICATION] is not None
        ):
            sensor[sens_cst.INPUT_CLASSIFICATION]["path"] = replace_path(
                sensor[sens_cst.INPUT_CLASSIFICATION]["path"]
            )

        if (
            sens_cst.INPUT_MSK in sensor
            and sensor[sens_cst.INPUT_MSK] is not None
        ):
            sensor[sens_cst.INPUT_MSK]["path"] = replace_path(
                sensor[sens_cst.INPUT_MSK]["path"]
            )

        for band_name, band_info in sensor[sens_cst.INPUT_IMG]["bands"].items():
            if band_name == "b0":
                continue
            band_info["path"] = replace_path(band_info["path"])

    @cars_profile(name="Run_subsampling_pipeline", interval=0.5)
    def run(self):  # noqa C901
        """
        Run pipeline
        """
        inputs = self.used_conf[INPUT]

        self.log_dir = os.path.join(self.out_dir, "logs")

        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=self.log_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as self.cars_orchestrator:
            for res in self.epipolar_resolutions:
                for key, val in inputs[sens_cst.SENSORS].items():
                    # Define the output directory
                    out_directory = os.path.join(
                        self.subsampling_dir, "res_" + str(res)
                    )
                    safe_makedirs(out_directory)

                    _ = self.sensors_subsampling.run(
                        key,
                        val,
                        res,
                        out_directory,
                        self.cars_orchestrator,
                    )

                    self.formatting(key, os.path.abspath(out_directory))

                out_yaml = os.path.abspath(
                    os.path.join(out_directory, "input.yaml")
                )
                with open(out_yaml, "w", encoding="utf-8") as f:
                    yaml.dump(
                        self.used_conf[INPUT],
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
