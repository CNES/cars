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

from cars.core import cars_logging

# CARS imports
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    INPUT,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate


@Pipeline.register(
    "formatting",
)
class FormattingPipeline(PipelineTemplate):
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

    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs
        """

        input_conf = conf.get(INPUT, {})

        overloaded_conf = input_conf.copy()

        overloaded_conf["input_path"] = input_conf.get("input_path", None)

        formatting_schema_input = {
            "input_path": str,
        }

        checker_input = Checker(formatting_schema_input)
        checker_input.validate(overloaded_conf)

        return overloaded_conf

    def check_output(self, conf):
        """
        Check the inputs
        """

        input_conf = conf.get(INPUT, {})

        overloaded_conf = input_conf.copy()

        overloaded_conf["directory"] = conf.get("directory", None)

        formatting_schema_output = {"directory": str}

        checker_input = Checker(formatting_schema_output)
        checker_input.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check applications
        """

    def move_replace_files_only(
        self, source_dir: Path, destination_dir: Path, check=True
    ):
        """
        Replace files in dsm directory
        """
        destination_dir.mkdir(parents=True, exist_ok=True)

        for element in source_dir.iterdir():
            if (
                element.name in ["dsm", "depth_map", "point_cloud", "logs"]
                or not check
            ):
                dest = destination_dir / element.name

                if element.is_dir():
                    self.move_replace_files_only(element, dest, check=False)
                else:
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(element), str(dest))

    def move_replace_dir(self, source_dir: Path, destination_dir: Path):
        """
        replace directory
        """
        for element in source_dir.iterdir():
            if element.name in ["dsm", "depth_map", "point_cloud"]:
                dest = destination_dir / element.name

                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()

                shutil.move(str(element), str(dest))

    def run(self, surface_modeling_dir):
        """
        Run the formatting pipeline
        """
        cars_logging.add_progress_message("Starting formatting pipeline")

        source_dir = Path(self.used_conf[INPUT]["input_path"])
        destination_dir = Path(self.used_conf[OUTPUT]["directory"])

        if surface_modeling_dir is not None:
            if (
                source_dir != Path(surface_modeling_dir)
                and surface_modeling_dir is not None
            ):
                self.move_replace_dir(
                    Path(surface_modeling_dir), destination_dir
                )

        self.move_replace_files_only(source_dir, destination_dir)
