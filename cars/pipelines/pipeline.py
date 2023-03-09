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
This module contains class pipeline factory.
"""

import logging
from typing import Dict, Union


class Pipeline:
    """
    Pipeline factory:
    A class designed for registered all available Cars Pipeline and
    instantiate when needed.

    """

    # Dict (pipeline_name:str, class: object) containing registered
    # pipelines
    available_pipeline = {}

    def __new__(
        cls,
        pipeline_name: str,
        cfg: Dict[str, Union[str, int]],
        config_json_dir,
    ):
        """
        Return the instance of pipeline associated with the pipeline
        name given as parameter

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        return cls.create_pipeline(pipeline_name, cfg, config_json_dir)

    @classmethod
    def create_pipeline(
        cls, name: str, cfg: Dict[str, Union[str, int]], config_json_dir
    ):
        """Factory command to create the pipeline
        Return the instance of pipeline associated with the pipeline
        name given as parameter

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: cars input configuration
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        pipeline = None
        try:
            pipeline_class = cls.available_pipeline[name]

        except KeyError as kerr:
            logging.error("No pipeline named {0} supported".format(name))
            raise NameError(
                "No pipeline named {0} supported".format(name)
            ) from kerr

        pipeline = pipeline_class(cfg, config_json_dir)

        return pipeline

    @classmethod
    def print_available_pipelines(cls):
        """
        Print all registered pipelines
        """

        for pipeline_name in cls.available_pipeline:
            print(pipeline_name)

    @classmethod
    def register(cls, *pipeline_names: str):
        """
        Allows to register the pipeline with its name
        :param pipeline_name: the pipelines to be registered
        :type pipeline_name: string
        """

        def decorator(app):
            """
            Registers the class in the available methods
            :param app: the app class to be registered
            :type app: object
            """
            for pipeline_name in pipeline_names:
                cls.available_pipeline[pipeline_name] = app
            return app

        return decorator
