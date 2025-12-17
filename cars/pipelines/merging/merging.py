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
CARS merging pipeline class file
"""

import os

from json_checker import Checker, Or

from cars.applications.application import Application
from cars.core import preprocessing, roi_tools
from cars.core.inputs import rasterio_get_epsg
from cars.core.utils import safe_makedirs
from cars.orchestrator import orchestrator
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import dsm_inputs
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import output_constants as out_cst
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

PIPELINE = "merging"


@Pipeline.register(
    PIPELINE,
)
class MergingPipeline(PipelineTemplate):
    """
    Merging pipeline
    """

    def __init__(self, conf, config_dir=None):
        """
        Creates pipeline

        Directly creates class attributes:
            used_conf

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
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
        pipeline_conf = conf.get(PIPELINE, {})
        advanced = self.check_advanced_parameters(
            pipeline_conf.get(ADVANCED, {})
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

        input_config = dsm_inputs.check_dsm_inputs(conf, config_dir=config_dir)
        return input_config

    @staticmethod
    def check_advanced_parameters(conf):
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

        # Validate inputs
        schema = {
            adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
        }

        checker_advanced_parameters = Checker(schema)
        checker_advanced_parameters.validate(overloaded_conf)

        return overloaded_conf

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

        # Load auxiliary and subfields
        overloaded_conf[out_cst.AUXILIARY] = overloaded_conf.get(
            out_cst.AUXILIARY, {}
        )

        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_IMAGE] = overloaded_conf[
            out_cst.AUXILIARY
        ].get(out_cst.AUX_IMAGE, True)
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_DEM_MIN] = (
            overloaded_conf[out_cst.AUXILIARY].get(out_cst.AUX_DEM_MIN, False)
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_DEM_MAX] = (
            overloaded_conf[out_cst.AUXILIARY].get(out_cst.AUX_DEM_MAX, False)
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_DEM_MEDIAN] = (
            overloaded_conf[out_cst.AUXILIARY].get(
                out_cst.AUX_DEM_MEDIAN, False
            )
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_WEIGHTS] = (
            overloaded_conf[out_cst.AUXILIARY].get(out_cst.AUX_WEIGHTS, False)
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_CLASSIFICATION] = (
            overloaded_conf[out_cst.AUXILIARY].get(
                out_cst.AUX_CLASSIFICATION, False
            )
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_PERFORMANCE_MAP] = (
            overloaded_conf[out_cst.AUXILIARY].get(
                out_cst.AUX_PERFORMANCE_MAP, False
            )
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_CONTRIBUTING_PAIR] = (
            overloaded_conf[out_cst.AUXILIARY].get(
                out_cst.AUX_CONTRIBUTING_PAIR, False
            )
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_FILLING] = (
            overloaded_conf[out_cst.AUXILIARY].get(out_cst.AUX_FILLING, False)
        )
        overloaded_conf[out_cst.AUXILIARY][out_cst.AUX_AMBIGUITY] = (
            overloaded_conf[out_cst.AUXILIARY].get(out_cst.AUX_AMBIGUITY, False)
        )

        # Check schema
        output_schema = {
            out_cst.OUT_DIRECTORY: str,
            out_cst.AUXILIARY: dict,
        }
        checker_output = Checker(output_schema)
        checker_output.validate(overloaded_conf)

        # Check auxiliary keys
        auxiliary_schema = {
            out_cst.AUX_IMAGE: Or(bool, str, list),
            out_cst.AUX_WEIGHTS: bool,
            out_cst.AUX_CLASSIFICATION: Or(bool, dict, list),
            out_cst.AUX_PERFORMANCE_MAP: Or(bool, list),
            out_cst.AUX_CONTRIBUTING_PAIR: bool,
            out_cst.AUX_FILLING: Or(bool, dict),
            out_cst.AUX_AMBIGUITY: bool,
            out_cst.AUX_DEM_MIN: bool,
            out_cst.AUX_DEM_MAX: bool,
            out_cst.AUX_DEM_MEDIAN: bool,
        }

        checker_auxiliary = Checker(auxiliary_schema)
        checker_auxiliary.validate(overloaded_conf[out_cst.AUXILIARY])

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

        needed_applications = ["dsm_merging"]

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            if used_conf[app_key] is not None:
                used_conf[app_key]["save_intermediate_data"] = (
                    self.save_all_intermediate_data
                    or used_conf[app_key].get("save_intermediate_data", False)
                )

        # DSM merging
        self.dsm_merging_application = Application(
            "dsm_merging",
            cfg=used_conf.get("dsm_merging", {}),
        )
        used_conf["dsm_merging"] = self.dsm_merging_application.get_conf()

        return used_conf

    def run(self, log_dir=None):
        """
        Run pipeline

        """
        if log_dir is None:
            log_dir = os.path.join(self.out_dir, "logs")

        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
            log_dir=log_dir,
            out_yaml_path=os.path.join(
                self.out_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as cars_orchestrator:

            dsms_merging_dump_dir = os.path.join(self.dump_dir, "dsms_merging")

            dsm_dict = self.used_conf[INPUT][dsm_cst.DSMS]
            dict_path = {}
            for key in dsm_dict.keys():
                for path_name in dsm_dict[key].keys():
                    if dsm_dict[key][path_name] is not None:
                        if isinstance(dsm_dict[key][path_name], str):
                            if path_name not in dict_path:
                                dict_path[path_name] = [
                                    dsm_dict[key][path_name]
                                ]
                            else:
                                dict_path[path_name].append(
                                    dsm_dict[key][path_name]
                                )

            dsm_file_name = os.path.join(
                self.out_dir,
                out_cst.DSM_DIRECTORY,
                "dsm.tif",
            )

            color_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "image.tif",
                )
                if "texture" in dict_path
                or self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_IMAGE]
                else None
            )

            performance_map_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "performance_map.tif",
                )
                if "performance_map" in dict_path
                else None
            )

            ambiguity_bool = any("ambiguity" in key for key in dict_path)
            ambiguity_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "ambiguity.tif",
                )
                if ambiguity_bool
                else None
            )

            classif_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "classification.tif",
                )
                if "merging_classification" in dict_path
                or self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_CLASSIFICATION
                ]
                else None
            )

            contributing_all_pair_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "contributing_pair.tif",
                )
                if "contributing_pair" in dict_path
                else None
            )

            filling_file_name = (
                os.path.join(
                    self.out_dir,
                    out_cst.DSM_DIRECTORY,
                    "filling.tif",
                )
                if "merging_filling" in dict_path
                else None
            )

            # Get ROI
            epsg = rasterio_get_epsg(dict_path["dsm"][0])
            (
                input_roi_poly,
                input_roi_epsg,
            ) = roi_tools.generate_roi_poly_from_inputs(
                self.used_conf[INPUT][sens_cst.ROI]
            )
            roi_poly = preprocessing.compute_roi_poly(
                input_roi_poly, input_roi_epsg, epsg
            )

            # Launch merging
            _ = self.dsm_merging_application.run(
                dict_path,
                cars_orchestrator,
                roi_poly,
                dsms_merging_dump_dir,
                dsm_file_name,
                color_file_name,
                classif_file_name,
                filling_file_name,
                performance_map_file_name,
                ambiguity_file_name,
                contributing_all_pair_file_name,
            )
