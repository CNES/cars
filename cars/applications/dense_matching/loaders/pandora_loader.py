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
CARS pandora loader file
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import pandora
from json_checker import Checker, Or
from pandora.check_json import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
    update_conf,
)
from pandora.state_machine import PandoraMachine


class PandoraLoader:
    """
    PandoraLoader

    """

    def __init__(
        self,
        conf=None,
        method_name=None,
        generate_performance_map=False,
        perf_eta_max_ambiguity=0.99,
        perf_eta_max_risk=0.25,
        perf_eta_step=0.04,
    ):
        """
        Init function of PandoraLoader

        If conf is profided, pandora will use it
        If not, Pandora will use intern configuration :
        census or mccnn, depending on method_name

        :param conf: configuration of pandora to use
        :type conf: dict
        :param method_name: name of method to use
        :param performance_map_conf: true if generate performance maps

        """

        self.pandora_config = None

        uses_cars_pandora_conf = False

        if conf is None:
            uses_cars_pandora_conf = True
            package_path = os.path.dirname(__file__)

            if "mccnn" in method_name:
                # Use mccn_conf

                conf_file_path = os.path.join(package_path, "config_mccnn.json")
                # Read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)

            elif "census" in method_name:
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm.json"
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)

            else:
                logging.error(
                    "No method named {} in pandora loader".format(method_name)
                )
                raise NameError(
                    "No method named {} in pandora loader".format(method_name)
                )

        # add performance map in configuration
        if generate_performance_map:
            perf_conf = {
                "cost_volume_confidence.cars_1": {
                    "confidence_method": "ambiguity",
                    "eta_max": perf_eta_max_ambiguity,
                    "eta_step": perf_eta_step,
                },
                "cost_volume_confidence.cars_2": {
                    "confidence_method": "risk",
                    "eta_max": perf_eta_max_risk,
                    "eta_step": perf_eta_step,
                },
            }

            conf["pipeline"] = overload_pandora_conf_with_confidence(
                conf["pipeline"], perf_conf
            )

        else:
            # by default generate ambiguity user uses
            # cars loaded configurations
            if uses_cars_pandora_conf:
                ambi_conf = {
                    "cost_volume_confidence": {
                        "confidence_method": "ambiguity",
                        "eta_max": 0.7,
                        "eta_step": 0.01,
                    },
                }
                conf["pipeline"] = overload_pandora_conf_with_confidence(
                    conf["pipeline"], ambi_conf
                )

        # Check conf
        self.pandora_configuration = self.check_conf(conf)

    def get_conf(self):
        """
        Get pandora configuration used

        :return: pandora configuration
        :rtype: dict

        """

        return self.pandora_configuration

    def check_conf(self, user_cfg):
        """
        Check configuration

        :param user_cfg: configuration
        :type user_cfg: dict

        :return: pandora configuration
        :rtype: dict

        """

        # Import plugins before checking configuration
        pandora.import_plugin()
        # Check configuration and update the configuration with default values
        # Instantiate pandora state machine
        pandora_machine = PandoraMachine()
        # check pipeline
        user_cfg_pipeline = get_config_pipeline(user_cfg)
        cfg_pipeline = check_pipeline_section(
            user_cfg_pipeline, pandora_machine
        )
        # check a part of input section
        user_cfg_input = get_config_input_custom_cars(user_cfg)
        cfg_input = check_input_section_custom_cars(user_cfg_input)
        # concatenate updated config
        cfg = concat_conf([cfg_input, cfg_pipeline])

        return cfg


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


def get_config_input_custom_cars(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the input configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "input" in user_cfg:
        cfg["input"] = {}

        if "nodata_left" in user_cfg["input"]:
            cfg["input"]["nodata_left"] = user_cfg["input"]["nodata_left"]

        if "nodata_right" in user_cfg["input"]:
            cfg["input"]["nodata_right"] = user_cfg["input"]["nodata_right"]

    return cfg


def check_input_section_custom_cars(
    user_cfg: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input_custom_cars, user_cfg)

    # check schema
    configuration_schema = {"input": input_configuration_schema_custom_cars}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def overload_pandora_conf_with_confidence(conf, confidence_conf):
    """
    Overload pandora pipeline configuration with given confidence to add
    just before disparity computation.

    :param conf: current pandora configuration
    :type conf: OrderedDict
    :param confidence_conf: confidence applications config
    :type confidence_conf: OrderedDict

    :return: updated pandora pipeline conf
    :rtype: OrderedDict
    """

    out_dict = OrderedDict()
    out_dict.update(conf)

    conf_keys = list(conf.keys())
    confidence_conf_keys = list(confidence_conf.keys())

    for key in confidence_conf_keys:
        if key in conf_keys:
            logging.error("{} pandora key already in configuration".format(key))

    # update confidence
    out_dict.update(confidence_conf)

    # move confidence keys right before disparity computation

    # get position of key "disparity"
    if "disparity" not in conf_keys:
        raise RuntimeError("disparity key not in pandora configuration")
    disp_index = conf_keys.index("disparity")

    # move to end every key from disparity

    for ind in range(disp_index, len(conf_keys)):
        out_dict.move_to_end(conf_keys[ind])

    return out_dict
