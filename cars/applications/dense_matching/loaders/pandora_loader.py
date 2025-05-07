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

import copy
import json
import logging
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import pandora
from json_checker import Checker, Or
from pandora.check_configuration import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
    update_conf,
)
from pandora.img_tools import get_metadata
from pandora.state_machine import PandoraMachine


class PandoraLoader:
    """
    PandoraLoader

    """

    def __init__(  # noqa: C901
        self,
        conf=None,
        method_name=None,
        generate_performance_map_from_risk=False,
        generate_performance_map_from_intervals=False,
        generate_ambiguity=False,
        perf_eta_max_ambiguity=0.99,
        perf_eta_max_risk=0.25,
        perf_eta_step=0.04,
        use_cross_validation=True,
        denoise_disparity_map=False,
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
        :param use_cross_validation: true to add crossvalidation
        :param denoise_disparity_map: true to add the disparity denoiser filter

        """

        if method_name is None:
            method_name = "census_sgm_default"

        self.pandora_config = None

        uses_cars_pandora_conf = False

        if isinstance(conf, str):
            # load file
            with open(conf, "r", encoding="utf8") as fstream:
                conf = json.load(fstream)

        elif conf is None:
            uses_cars_pandora_conf = True
            package_path = os.path.dirname(__file__)

            if method_name == "mccnn_sgm":
                # Use mccn_conf

                conf_file_path = os.path.join(package_path, "config_mccnn.json")
                # Read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_urban":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm_urban.json"
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_shadow":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm_shadow.json"
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_mountain_and_vegetation":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path,
                    "config_census_sgm_mountain_and_vegetation.json",
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_homogeneous":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm_homogeneous.json"
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_default":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm_default.json"
                )
                # read conf
                with open(conf_file_path, "r", encoding="utf8") as fstream:
                    conf = json.load(fstream)
            elif method_name == "census_sgm_sparse":
                # Use census sgm conf
                conf_file_path = os.path.join(
                    package_path, "config_census_sgm_sparse.json"
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

        perf_ambiguity_conf = {
            "cost_volume_confidence.cars_1": {
                "confidence_method": "ambiguity",
                "eta_max": perf_eta_max_ambiguity,
                "eta_step": perf_eta_step,
            }
        }

        perf_risk_conf = {
            "cost_volume_confidence.cars_2": {
                "confidence_method": "risk",
                "eta_max": perf_eta_max_risk,
                "eta_step": perf_eta_step,
            }
        }
        intervals_conf = {
            "cost_volume_confidence.cars_3": {
                "confidence_method": "interval_bounds",
            }
        }
        # Cross validation
        cross_validation_acc_conf = {
            "validation": {
                "validation_method": "cross_checking_accurate",
                "cross_checking_threshold": 1.0,
            }
        }

        cross_validation_fast_conf = {
            "validation": {
                "validation_method": "cross_checking_fast",
            }
        }

        disparity_denoiser_conf = {
            "filter": {"filter_method": "disparity_denoiser"}
        }

        confidences = {}
        if generate_performance_map_from_risk:
            confidences.update(perf_ambiguity_conf)
            confidences.update(perf_risk_conf)
        if generate_performance_map_from_intervals:
            confidences.update(perf_ambiguity_conf)
            confidences.update(intervals_conf)

        if generate_ambiguity:
            confidences.update(perf_ambiguity_conf)

        if confidences:
            conf["pipeline"] = overload_pandora_conf_with_confidence(
                conf["pipeline"], confidences
            )

        # update with cross validation
        if "validation" not in conf["pipeline"]:
            if use_cross_validation in (True, "fast"):
                conf["pipeline"].update(cross_validation_fast_conf)
            elif use_cross_validation == "accurate":
                conf["pipeline"].update(cross_validation_acc_conf)

        if (
            denoise_disparity_map
            and conf["pipeline"]["filter"]["filter_method"]
            != "disparity_denoiser"
        ):
            conf["pipeline"].update(disparity_denoiser_conf)

        if generate_performance_map_from_intervals:
            # To ensure the consistency between the disparity map
            # and the intervals, the median filter for intervals
            # must be similar to the median filter. The filter is
            # added at the end of the conf as it is applied during
            # the disp_map state.
            try:
                filter_size = conf["pipeline"]["filter"]["filter_size"]
            except KeyError:
                filter_size = 3

            conf_filter_interval = {
                "filter.cars_3": {
                    "filter_method": "median_for_intervals",
                    "filter_size": filter_size,
                    "interval_indicator": "cars_3",
                    "regularization": True,
                    "ambiguity_indicator": "cars_1",
                }
            }
            pipeline_dict = OrderedDict()
            pipeline_dict.update(conf["pipeline"])
            # Filter is placed after validation in config
            # and should be placed before.
            # However it does not have any incidence on operation
            if uses_cars_pandora_conf:
                pipeline_dict.update(conf_filter_interval)

            conf["pipeline"] = pipeline_dict

        if "filter" in conf["pipeline"]:
            filter_conf = conf["pipeline"]["filter"]
            if filter_conf["filter_method"] == "disparity_denoiser":
                if "band" in filter_conf:
                    band = filter_conf["band"]
                    if band is not None:
                        logging.warning(
                            "As multiband support is not "
                            "yet implemented in CARS, "
                            "the filter will be applied to "
                            "the panchromatic band."
                        )
                        conf["pipeline"]["filter"]["band"] = None

        for key in list(conf.get("pipeline")):
            if key.startswith("filter"):
                if (
                    conf["pipeline"][key]["filter_method"]
                    == "median_for_intervals"
                    and "validation" in conf["pipeline"]
                ):
                    if (
                        conf["pipeline"]["validation"]["validation_method"]
                        == "cross_checking_fast"
                    ):
                        conf["pipeline"]["validation"][
                            "validation_method"
                        ] = "cross_checking_accurate"
                        logging.warning(
                            "You can not use median_for_intervals with "
                            "the fast cross checking validation for now. "
                            "It therefore has been overrided to accurate"
                        )

        # Check conf
        self.pandora_config = conf

    def get_conf(self):
        """
        Get pandora configuration used

        :return: pandora configuration
        :rtype: dict

        """

        return self.pandora_config

    def check_conf(
        self,
        user_cfg,
        img_left,
        img_right,
        classif_left=None,
        classif_right=None,
    ):
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
        metadata_left = get_metadata(img_left, classif=classif_left)
        metadata_right = get_metadata(img_right, classif=classif_right)

        user_cfg_pipeline = get_config_pipeline(user_cfg)
        saved_schema = copy.deepcopy(
            pandora.matching_cost.matching_cost.AbstractMatchingCost.schema
        )
        cfg_pipeline = check_pipeline_section(
            user_cfg_pipeline, metadata_left, metadata_right, pandora_machine
        )
        # quick fix to remove when the problem is solved in pandora
        pandora.matching_cost.matching_cost.AbstractMatchingCost.schema = (
            saved_schema
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
    user_cfg: Dict[str, dict],
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
