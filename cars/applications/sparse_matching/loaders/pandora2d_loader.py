# !/usr/bin/env python
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
CARS pandora2d loaders file
"""

import json
import logging
import os

import numpy as np
import pandora2d
from json_checker import Or
from pandora2d.check_configuration import (
    check_pipeline_section,
    check_segment_mode_section,
)
from pandora2d.state_machine import Pandora2DMachine
from pandora.check_configuration import (
    concat_conf,
    get_config_pipeline,
)

from cars.applications.dense_matching.loaders.pandora_loader import (
    check_input_section_custom_cars,
    get_config_input_custom_cars,
    overload_pandora_conf_with_confidence,
)


class Pandora2DLoader:
    """
    PandoraLoader

    """

    def __init__(  # pylint: disable=too-many-positional-arguments  # noqa: C901
        self,
        conf=None,
        method_name=None,
        generate_ambiguity=False,
        perf_eta_max_ambiguity=0.99,
        perf_eta_step=0.04,
        step=None,
    ):
        """
        Init function of PandoraLoader

        If conf is profided, pandora2d will use it
        If not, Pandora2d will use intern configuration :
        census or mccnn, depending on method_name

        :param conf: configuration of pandora to use
        :type conf: dict
        :param method_name: name of method to use
        """

        if method_name is None:
            method_name = "default"

        self.pandora_config = None

        if isinstance(conf, str):
            # load file
            with open(conf, "r", encoding="utf8") as fstream:
                conf = json.load(fstream)

        elif conf is None:
            package_path = os.path.dirname(__file__)

            config_map = {
                "default": "config_default.json",
                "mccnn": "config_mc_cnn.json",
                "mutual_information": "config_mutual_information.json",
                "zncc-optim-1": "config_zncc_optim-1.json",
                "zncc-optim-2": "config_zncc_optim-2.json",
                "optical_flow": "config_optical_flow.json",
            }

            try:
                filename = config_map[method_name]
            except KeyError as err:
                logging.error(
                    "No method named {} in pandora2d loaders".format(
                        method_name
                    )
                )
                raise NameError(
                    "No method named {} in pandora2d loaders".format(
                        method_name
                    )
                ) from err

            conf_file_path = os.path.join(package_path, filename)

            with open(conf_file_path, "r", encoding="utf8") as fstream:
                conf = json.load(fstream)

        perf_ambiguity_conf = {
            "cost_volume_confidence.cars_1": {
                "confidence_method": "ambiguity",
                "eta_max": perf_eta_max_ambiguity,
                "eta_step": perf_eta_step,
            }
        }

        segment_mode = {
            "enable": True,
            "memory_per_work": 500,
        }

        conf["segment_mode"] = segment_mode

        confidences = {}

        if generate_ambiguity:
            confidences.update(perf_ambiguity_conf)

        if confidences:
            conf["pipeline"] = overload_pandora_conf_with_confidence(
                conf["pipeline"], confidences
            )

        if step is not None:
            conf["pipeline"]["matching_cost"]["step"] = step

        # Check conf
        self.pandora_config = conf

    def get_conf(self):
        """
        Get pandora configuration used

        :return: pandora configuration
        :rtype: dict

        """

        return self.pandora_config

    def check_conf(  # pylint: disable=too-many-positional-arguments
        self,
        user_cfg,
        nodata_left,
        nodata_right,
    ):
        """
        Check configuration

        :param user_cfg: configuration
        :type user_cfg: dict

        :return: pandora configuration
        :rtype: dict

        """
        # Import plugins before checking configuration
        pandora2d.import_plugin()
        # Check configuration and update the configuration with default values
        # Instantiate pandora state machine
        pandora2d_machine = Pandora2DMachine()

        user_cfg_pipeline = get_config_pipeline(user_cfg)

        check_pipeline_section(user_cfg_pipeline, pandora2d_machine)

        check_segment_mode_section(user_cfg)

        # check a part of input section
        user_cfg_input = get_config_input_custom_cars(
            user_cfg, nodata_left, nodata_right
        )
        cfg_input = check_input_section_custom_cars(user_cfg_input)

        # concatenate updated config
        cfg = concat_conf([cfg_input, user_cfg_pipeline])

        cfg["segment_mode"] = user_cfg["segment_mode"]

        margins = {"left": 0, "up": 0, "right": 0, "down": 0}
        for val in pandora2d_machine.margins_img.__dict__[
            "_cumulatives"
        ].values():
            margins["left"] += val.left
            margins["right"] += val.right
            margins["up"] += val.up
            margins["down"] += val.down

        return cfg, margins


input_configuration_schema_custom_cars = {
    "nodata_left": Or(
        int, lambda x: np.isnan(x)  # pylint: disable=unnecessary-lambda
    ),
    "nodata_right": Or(
        int, lambda x: np.isnan(x)  # pylint: disable=unnecessary-lambda
    ),
}

default_short_configuration_input_custom_cars = {
    "input": {
        "nodata_left": -9999,
        "nodata_right": -9999,
    }
}
