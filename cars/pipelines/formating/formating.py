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

import os
import shutil
from pathlib import Path

from json_checker import Checker

from cars.pipelines import pipeline_constants as pipeline_cst

# CARS imports
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    INPUT,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate


@Pipeline.register(
    "formating",
)
class FormatingPipeline(PipelineTemplate):
    """
    DefaultPipeline
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

        conf[INPUT] = self.check_inputs(conf)

        conf[OUTPUT] = self.check_output(conf[OUTPUT])

        self.used_conf = {}
        self.used_conf[INPUT] = conf[INPUT]
        self.used_conf[OUTPUT] = conf[OUTPUT]

        self.out_dir = conf[OUTPUT][out_cst.OUT_DIRECTORY]

    def check_global_schema(self, conf):
        """
        Check the global schema
        """

        # Validate inputs
        global_schema = {
            pipeline_cst.INPUT: dict,
            pipeline_cst.OUTPUT: dict,
        }

        checker_inputs = Checker(global_schema)
        checker_inputs.validate(conf)

    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs
        """

        input_conf = conf.get(INPUT, {})

        overloaded_conf = input_conf.copy()

        overloaded_conf["input_path"] = input_conf.get("input_path", None)

        formating_schema_input = {
            "input_path": str,
        }

        checker_input = Checker(formating_schema_input)
        checker_input.validate(overloaded_conf)

        return overloaded_conf

    def check_output(self, conf):
        """
        Check the inputs
        """

        input_conf = conf.get(INPUT, {})

        overloaded_conf = input_conf.copy()

        overloaded_conf["directory"] = conf.get("directory", None)

        formating_schema_output = {"directory": str}

        checker_input = Checker(formating_schema_output)
        checker_input.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check applications
        """

    def run(self):
        """
        Run the formating pipeline
        """
        source_dir = Path(self.used_conf[INPUT]["input_path"])
        destination_dir = Path(self.used_conf[OUTPUT]["directory"])

        for element in source_dir.iterdir():
            dest = destination_dir / element.name

            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            shutil.move(element, dest)
