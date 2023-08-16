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
"""
This module contains class pipeline template for
templating the pipeline concept.
"""

from abc import ABCMeta, abstractmethod

from json_checker import Checker, OptionalKey

# CARS imports
from cars.orchestrator import orchestrator
from cars.pipelines import pipeline_constants

# Disable pylint error: too few public method


class PipelineTemplate(metaclass=ABCMeta):  # pylint: disable=R0903
    """
    Class for general specification of an pipeline
    """

    def check_orchestrator(self, conf):
        """
        Check the configuration of orchestrator

        :param conf: configuration of orchestrator
        :type conf: dict
        :return: overloaded orchestrator conf
        :rtype: dict
        """

        with orchestrator.Orchestrator(
            orchestrator_conf=conf, out_dir=None, launch_worker=False
        ) as orchestrator_obj:
            conf = orchestrator_obj.get_conf()
        return conf

    def check_global_schema(self, conf):
        """
        Check the given global configuration

        :param conf: configuration
        :type conf: dict
        """

        # Validate inputs
        global_schema = {
            pipeline_constants.INPUTS: dict,
            pipeline_constants.OUTPUT: dict,
            OptionalKey(pipeline_constants.APPLICATIONS): dict,
            OptionalKey(pipeline_constants.GEOMETRY_PLUGIN): str,
            OptionalKey(pipeline_constants.ORCHESTRATOR): dict,
            OptionalKey(pipeline_constants.PIPELINE): str,
        }

        checker_inputs = Checker(global_schema)
        checker_inputs.validate(conf)

    def merge_pipeline_conf(self, config1, config2):
        """
        Merge two pipeline dict, generating a new configuration

        :param conf1: configuration
        :type conf1: dict
        :param conf2: configuration
        :type conf2: dict

        :return: merged conf
        :rtype: dict

        """

        merged_dict = config1.copy()

        _merge_pipeline_conf_rec(merged_dict, config2)

        return merged_dict

    @abstractmethod
    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs given

        :param conf: configuration of inputs
        :type conf: dict
        :param config_json_dir: directory of used json, if
            user filled paths with relative paths
        :type config_json_dir: str

        :return: overloader inputs
        :rtype: dict
        """

    @abstractmethod
    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """

    @abstractmethod
    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

    @abstractmethod
    def run(self):
        """
        Run pipeline
        """


def _merge_pipeline_conf_rec(conf1, conf2):
    """
    Merge secondary configuration on primary one

    :param conf1: configuration
    :type conf1: dict
    :param conf2: configuration
    :type conf2: dict

    :return: merged conf
    :rtype: dict

    """

    for key in conf2.keys():
        if key in conf1:
            if isinstance(conf1[key], dict) and isinstance(conf2[key], dict):
                _merge_pipeline_conf_rec(conf1[key], conf2[key])
            else:
                conf1[key] = conf2[key]

        else:
            conf1[key] = conf2[key]
