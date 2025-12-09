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

import copy
import json
import logging
import os
import shutil
from collections import OrderedDict
from datetime import datetime

import yaml

# CARS imports
from cars.core import cars_logging
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster import log_wrapper
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines import pipeline_constants as pipeline_cst
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
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
from cars.pipelines.subsampling.subsampling import SubsamplingPipeline
from cars.pipelines.unit.unit_pipeline import UnitPipeline

package_path = os.path.dirname(__file__)
FIRST_RES = "first_resolution"
INTERMEDIATE_RES = "intermediate_resolution"
FINAL_RES = "final_resolution"

PIPELINE_CONFS = {
    FIRST_RES: os.path.join(
        package_path,
        "..",
        "conf_resolution",
        "conf_first_resolution.yaml",
    ),
    INTERMEDIATE_RES: os.path.join(
        package_path,
        "..",
        "conf_resolution",
        "conf_intermediate_resolution.yaml",
    ),
    FINAL_RES: os.path.join(
        package_path,
        "..",
        "conf_resolution",
        "conf_final_resolution.yaml",
    ),
}


@Pipeline.register(
    "default",
)
class DefaultPipeline(PipelineTemplate):
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

        self.out_dir = conf[OUTPUT][out_cst.OUT_DIRECTORY]

        # Get epipolar resolutions to use
        self.epipolar_resolutions = (
            advanced_parameters.get_epipolar_resolutions(conf.get(ADVANCED, {}))
        )
        if isinstance(self.epipolar_resolutions, int):
            self.epipolar_resolutions = [self.epipolar_resolutions]

        # Check application
        self.check_applications(conf)
        # Check input
        conf[INPUT] = self.check_inputs(conf)

        # check advanced
        conf[ADVANCED] = self.check_advanced(conf)
        # check output
        conf[OUTPUT] = self.check_output(conf)

        self.intermediate_data_dir = os.path.join(
            self.out_dir, "intermediate_data"
        )

        self.subsampling_conf = self.construct_subsampling_conf(conf)
        conf[pipeline_cst.SUBSAMPLING] = self.check_subsampling(
            self.subsampling_conf
        )

        if dsm_cst.DSMS in conf[INPUT] and len(self.epipolar_resolutions) != 1:
            logging.info(
                "For the use of those pipelines, "
                "you have to give only one resolution"
            )
            # overide epipolar resolutions
            # TODO: delete with external dsm pipeline (refactoring)
            self.epipolar_resolutions = [1]

        used_configurations = {}
        self.unit_pipelines = {}
        self.positions = {}
        self.used_conf = {}

        self.keep_low_res_dir = conf[ADVANCED][adv_cst.KEEP_LOW_RES_DIR]

        # Get first res outdir for sift matches
        self.first_res_out_dir_with_sensors = None

        for epipolar_resolution_index, epipolar_res in enumerate(
            self.epipolar_resolutions
        ):
            first_res = epipolar_resolution_index == 0
            last_res = (
                epipolar_resolution_index == len(self.epipolar_resolutions) - 1
            )
            intermediate_res = not first_res and not last_res

            # set computed bool
            self.positions[epipolar_resolution_index] = {
                "first_res": first_res,
                "intermediate_res": intermediate_res,
                "last_res": last_res,
            }

            current_conf = copy.deepcopy(conf)
            current_conf = extract_conf_with_resolution(
                current_conf,
                epipolar_res,
                first_res,
                intermediate_res,
                last_res,
                self.intermediate_data_dir,
            )

            if first_res:
                self.first_res_out_dir_with_sensors = current_conf[OUTPUT][
                    "directory"
                ]
                previous_scaling_coeff = None

            if not isinstance(epipolar_res, int) or epipolar_res < 0:
                raise RuntimeError("The resolution has to be an int > 0")

            self.used_conf[epipolar_resolution_index] = current_conf

            # Initialize unit pipeline in order to retrieve the
            # used configuration
            # This pipeline will not be run
            _ = current_conf.pop(pipeline_cst.SUBSAMPLING, None)

            current_unit_pipeline = UnitPipeline(
                current_conf,
                config_dir=self.config_dir,
                previous_scaling_coeff=previous_scaling_coeff,
            )

            previous_scaling_coeff = current_unit_pipeline.scaling_coeff

            self.unit_pipelines[epipolar_resolution_index] = (
                current_unit_pipeline
            )
            # Get used_conf
            used_configurations[epipolar_res] = current_unit_pipeline.used_conf

        # Generate full used_conf
        full_used_conf = merge_used_conf(
            used_configurations, self.epipolar_resolutions
        )
        # Save used_conf
        cars_dataset.save_dict(
            full_used_conf,
            os.path.join(self.out_dir, "global_used_conf.yaml"),
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
            conf[INPUT], conf[OUTPUT]
        )
        return conf_output

    def check_subsampling(self, conf):
        """
        Check the subsampling section

        :param conf: configuration of subsampling
        type conf: dict
        """

        pipeline = SubsamplingPipeline(conf)
        conf_subsampling = conf.get(pipeline_cst.SUBSAMPLING, {})
        advanced = pipeline.check_advanced(conf_subsampling.get(ADVANCED, {}))
        applications = pipeline.check_applications(
            conf_subsampling.get(APPLICATIONS, {})
        )

        return {"advanced": advanced, "applications": applications}

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
            )
        )

        return advanced

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """
        applications_conf = conf.get(APPLICATIONS, {})
        # check format: contains "all" of "resolutions

        int_keys = [int(epi_res) for epi_res in self.epipolar_resolutions]
        string_keys = [str(key) for key in int_keys]

        possible_keys = ["all"] + int_keys + string_keys

        # Check conf keys in possible keys
        for app_base_key in applications_conf.keys():
            if app_base_key not in possible_keys:
                raise RuntimeError(
                    "Application key {} not in possibles keys in : 'all', {} , "
                    "as int or str".format(app_base_key, string_keys)
                )

        # Key str and int key are not defined for the same resolution
        for resolution in int_keys:
            if (
                resolution in applications_conf
                and str(resolution) in applications_conf
            ):
                raise RuntimeError(
                    "Application configuration for {} resolution "
                    "is defined both "
                    "with int and str key".format(resolution)
                )

    def cleanup_low_res_dir(self):
        """
        Clean low res dir
        """

        if os.path.exists(self.intermediate_data_dir) and os.path.isdir(
            self.intermediate_data_dir
        ):
            try:
                shutil.rmtree(self.intermediate_data_dir)
                logging.info(
                    f"th directory {self.intermediate_data_dir} "
                    f" has been cleaned."
                )
            except Exception as exception:
                logging.error(
                    f"Error while deleting {self.intermediate_data_dir}: "
                    f"{exception}"
                )
        else:
            logging.info(
                f"The directory {self.intermediate_data_dir} has not "
                f"been deleted"
            )

    def construct_subsampling_conf(self, conf):
        """
        Construct the right conf for subsampling
        """
        subsampling_conf = {}
        subsampling_conf[INPUT] = copy.deepcopy(conf[INPUT])
        subsampling_conf[OUTPUT] = {}
        subsampling_conf[OUTPUT]["directory"] = self.intermediate_data_dir

        subsampling_conf[pipeline_cst.ADVANCED] = conf.get(
            pipeline_cst.SUBSAMPLING, {}
        ).get(ADVANCED, {})
        subsampling_conf[pipeline_cst.APPLICATIONS] = conf.get(
            pipeline_cst.SUBSAMPLING, {}
        ).get(APPLICATIONS, {})

        return subsampling_conf

    @cars_profile(name="Run_default_pipeline", interval=0.5)
    def run(self, args=None):  # noqa C901
        """
        Run pipeline

        """

        global_log_file = os.path.join(
            self.out_dir,
            "logs",
            "{}_{}.log".format(
                datetime.now().strftime("%y-%m-%d_%Hh%Mm"), "default_pipeline"
            ),
        )

        previous_out_dir = None
        updated_conf = {}
        step = 0

        subsampling_pipeline = SubsamplingPipeline(
            self.subsampling_conf, self.config_dir
        )
        subsampling_pipeline.run()

        for resolution_index, epipolar_res in enumerate(
            self.epipolar_resolutions
        ):

            # Get tested unit pipeline
            current_conf = self.used_conf[resolution_index]
            current_out_dir = current_conf[OUTPUT]["directory"]

            # Put right directory for subsampling
            if epipolar_res != 1:
                yaml_file = os.path.join(
                    self.intermediate_data_dir,
                    "subsampling/res_" + str(epipolar_res) + "/input.yaml",
                )
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                json_str = json.dumps(data, indent=4)
                data = json.loads(json_str)

                current_conf[INPUT] = data

            # update directory for unit pipeline
            current_conf[OUTPUT]["directory"] = current_out_dir

            used_pipeline = UnitPipeline(
                current_conf, config_dir=self.config_dir
            )

            # get position
            first_res, _, last_res = (
                self.positions[resolution_index]["first_res"],
                self.positions[resolution_index]["intermediate_res"],
                self.positions[resolution_index]["last_res"],
            )

            # setup logging
            loglevel = getattr(args, "loglevel", "PROGRESS").upper()

            current_log_dir = os.path.join(
                self.out_dir, "logs", "res_" + str(epipolar_res)
            )

            cars_logging.setup_logging(
                loglevel,
                out_dir=current_log_dir,
                pipeline="unit_pipeline",
                global_log_file=global_log_file,
            )

            cars_logging.add_progress_message(
                "Starting pipeline for resolution 1/" + str(epipolar_res)
            )

            # use sift a priori if not first
            use_sift_a_priori = False
            if not first_res:
                use_sift_a_priori = True

            # define wich resolution
            if first_res and last_res:
                which_resolution = "single"
            elif first_res:
                which_resolution = "first"
            elif last_res:
                which_resolution = "final"
            else:
                which_resolution = "intermediate"

            # Generate dem
            generate_dems = True
            if last_res:
                generate_dems = False

            # Overide with a priori
            if not first_res:
                dsm = os.path.join(previous_out_dir, "dsm/dsm.tif")
                used_pipeline.used_conf[INPUT][sens_cst.LOW_RES_DSM] = dsm
            else:
                previous_scaling_coeff = None

            updated_pipeline = UnitPipeline(
                used_pipeline.used_conf,
                config_dir=self.config_dir,
                previous_scaling_coeff=previous_scaling_coeff,
            )
            updated_pipeline.run(
                generate_dems=generate_dems,
                which_resolution=which_resolution,
                use_sift_a_priori=use_sift_a_priori,
                first_res_out_dir=self.first_res_out_dir_with_sensors,
                log_dir=current_log_dir,
            )

            # update previous out dir
            previous_out_dir = current_out_dir

            # keep previous scaling coeff for dem_generation
            previous_scaling_coeff = updated_pipeline.scaling_coeff

            # generate summary
            log_wrapper.generate_summary(
                current_log_dir, updated_pipeline.used_conf
            )

            updated_conf[epipolar_res] = updated_pipeline.used_conf

            step += 1

        # Generate full used_conf
        full_used_conf = merge_used_conf(
            updated_conf, self.epipolar_resolutions
        )
        # Save used_conf
        cars_dataset.save_dict(
            full_used_conf,
            os.path.join(self.out_dir, "global_used_conf.yaml"),
        )

        # Merge profiling in pdf
        log_wrapper.generate_pdf_profiling(os.path.join(self.out_dir, "logs"))

        # clean outdir
        if not self.keep_low_res_dir:
            self.cleanup_low_res_dir()


def extract_applications(
    current_applications_conf, res, default_conf_for_res, filling_applications
):
    """
    Extract applications for current resolution

    :param current_applications_conf: current applications configuration
    :type current_applications_conf: dict
    :param res: resolution to extract
    :type res: int
    :param default_conf_for_res: default configuration for resolution
    :type default_conf_for_res: dict
    :param filling_applications: filling applications configuration
    :type filling_applications: dict

    :return: configuration for the given resolution
    :rtype: dict
    """

    # "all" : applied to all conf
    # int  (1, 2, 4, 8, 16, ...) applied for specified resolution

    all_conf = current_applications_conf.get("all", {})
    # Overide with default_conf_for_res
    all_conf = overide_pipeline_conf(all_conf, default_conf_for_res)
    # Get configuration for current res
    if res in current_applications_conf:
        # key is int
        key = res
    else:
        key = str(res)

    res_conf = current_applications_conf.get(key, {})

    # Overide all conf with current res conf
    new_application_conf = overide_pipeline_conf(all_conf, res_conf)

    # Overide with filling applications
    new_application_conf = overide_pipeline_conf(
        new_application_conf, filling_applications, append_classification=True
    )
    return new_application_conf


# pylint: disable=too-many-positional-arguments
def extract_conf_with_resolution(
    current_conf,
    res,
    first_res,
    intermediate_res,
    last_res,
    intermediate_data_dir,
):
    """
    Extract the configuration for the given resolution

    :param current_conf: current configuration
    :type current_conf: dict
    :param res: resolution to extract
    :type res: int
    :return: configuration for the given resolution
    :rtype: dict
    :param first_res: is first resolution
    :type first_res: bool
    :param intermediate_res: is intermediate resolution
    :type intermediate_res: bool
    :param last_res: is last resolution
    :type last_res: bool
    :param previous_out_dir: path to previous outdir
    :type: previous_out_dir: str
    """

    new_dir_out_dir = current_conf[OUTPUT][out_cst.OUT_DIRECTORY]
    if not last_res:
        new_dir_out_dir = os.path.join(
            intermediate_data_dir, "out_res" + str(res)
        )
        safe_makedirs(new_dir_out_dir)

    new_conf = copy.deepcopy(current_conf)

    # Get save intermediate data
    if isinstance(new_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA], dict):
        # If save_intermediate_data is not a dict, we set it to False
        new_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA] = new_conf[ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ].get("resolution_" + str(res), False)

    # Overide epipolar resolution
    new_conf[ADVANCED][adv_cst.EPIPOLAR_RESOLUTIONS] = res

    # Overide  configuration with pipeline conf
    if first_res:
        # read the first resolution conf with json package
        with open(PIPELINE_CONFS[FIRST_RES], "r", encoding="utf-8") as file:
            overiding_conf = yaml.safe_load(file)
    elif intermediate_res:
        with open(
            PIPELINE_CONFS[INTERMEDIATE_RES], "r", encoding="utf-8"
        ) as file:
            overiding_conf = yaml.safe_load(file)
    else:
        with open(PIPELINE_CONFS[FINAL_RES], "r", encoding="utf-8") as file:
            overiding_conf = yaml.safe_load(file)

    if last_res and dsm_cst.DSMS not in current_conf[INPUT]:
        # Use filling applications only for last resolution
        filling_applications = generate_filling_applications(
            current_conf[INPUT]
        )
    else:
        filling_applications = {}

    # extract application
    new_conf[APPLICATIONS] = extract_applications(
        current_conf.get(APPLICATIONS, {}),
        res,
        overiding_conf.get(APPLICATIONS, {}),
        filling_applications,
    )

    overiding_conf = {
        OUTPUT: {
            out_cst.AUXILIARY: {
                out_cst.AUX_DEM_MAX: True,
                out_cst.AUX_DEM_MIN: True,
                out_cst.AUX_DEM_MEDIAN: True,
            },
        },
    }
    new_conf = overide_pipeline_conf(new_conf, overiding_conf)

    # Overide output to not compute data
    # Overide resolution to let unit pipeline manage it
    if not last_res:
        overiding_conf = {
            OUTPUT: {
                out_cst.OUT_DIRECTORY: new_dir_out_dir,
                out_cst.RESOLUTION: None,
                out_cst.SAVE_BY_PAIR: True,
                out_cst.AUXILIARY: {
                    out_cst.AUX_DEM_MAX: True,
                    out_cst.AUX_DEM_MIN: True,
                    out_cst.AUX_DEM_MEDIAN: True,
                },
            },
            APPLICATIONS: {
                "dense_matching": {
                    "performance_map_method": ["risk", "intervals"]
                }
            },
        }
        new_conf = overide_pipeline_conf(new_conf, overiding_conf)

        # set product level to dsm
        new_conf[OUTPUT][out_cst.PRODUCT_LEVEL] = ["dsm"]
        # remove resolution to let CARS compute it for current
        # epipolar resolution
        new_conf[OUTPUT]["resolution"] = None

        if not new_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA]:
            # Save the less possible things
            aux_items = new_conf[OUTPUT][out_cst.AUXILIARY].items()
            for aux_key, _ in aux_items:
                if aux_key not in ("dem_min", "dem_max", "dem_median", "image"):
                    new_conf[OUTPUT][out_cst.AUXILIARY][aux_key] = False

    return new_conf


def generate_filling_applications(inputs_conf):
    """
    Generate filling applications configuration according to inputs

    :param inputs_conf: inputs configuration
    :type inputs_conf: dict
    """

    filling_applications = {}

    # Generate applications configuration
    for filling_name, classif_values in inputs_conf[sens_cst.FILLING].items():
        # No filling
        if classif_values is None:
            continue

        classif_values = list(map(str, classif_values))

        # Update application configuration
        if filling_name == "fill_with_geoid":
            new_filling_conf = {
                "dense_match_filling": {
                    "method": "zero_padding",
                    "classification": classif_values,
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                    "classification": classif_values,
                    "fill_with_geoid": classif_values,
                },
            }
        elif filling_name == "interpolate_from_borders":
            new_filling_conf = {
                "dense_match_filling": {
                    "method": "zero_padding",
                    "classification": classif_values,
                },
                "dsm_filling.2": {
                    "method": "bulldozer",
                    "classification": classif_values,
                },
                "dsm_filling.3": {
                    "method": "border_interpolation",
                    "classification": classif_values,
                },
            }
        elif filling_name == "fill_with_endogenous_dem":
            new_filling_conf = {
                "dense_match_filling": {
                    "method": "zero_padding",
                    "classification": classif_values,
                },
                "dsm_filling.1": {
                    "method": "exogenous_filling",
                    "classification": classif_values,
                },
                "dsm_filling.2": {
                    "method": "bulldozer",
                    "classification": classif_values,
                },
            }
        elif filling_name == "fill_with_exogenous_dem":
            new_filling_conf = {
                "dense_match_filling": {
                    "method": "zero_padding",
                    "classification": classif_values,
                },
                "dsm_filling.2": {
                    "method": "bulldozer",
                    "classification": classif_values,
                },
            }
        else:
            new_filling_conf = {}

        # Update application configuration
        filling_applications = overide_pipeline_conf(
            filling_applications, new_filling_conf, append_classification=True
        )

    return filling_applications


def overide_pipeline_conf(conf, overiding_conf, append_classification=False):
    """
    Merge two dictionaries recursively without removing keys from the base conf.

    :param conf: base configuration dictionary
    :type conf: dict
    :param overiding_conf: overriding configuration dictionary
    :type overiding_conf: dict
    :return: merged configuration
    :rtype: dict
    """
    result = copy.deepcopy(conf)

    def merge_recursive(base_dict, override_dict):
        """
        Main recursive function
        """
        for key, value in override_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                merge_recursive(base_dict[key], value)
            elif (
                append_classification
                and key in base_dict
                and isinstance(base_dict[key], list)
                and isinstance(value, list)
                and key == "classification"
            ):
                # extend list, avoiding duplicates
                base_dict[key] = list(
                    OrderedDict.fromkeys(base_dict[key] + value)
                )
            else:
                base_dict[key] = value

    merge_recursive(result, overiding_conf)
    return result


def merge_used_conf(used_configurations, epipolar_resolutions):
    """
    Merge all used configuration
    """
    used_configurations = copy.deepcopy(used_configurations)

    merged_conf = {
        INPUT: used_configurations[epipolar_resolutions[-1]][INPUT],
        ADVANCED: used_configurations[epipolar_resolutions[0]][ADVANCED],
        OUTPUT: used_configurations[epipolar_resolutions[0]][OUTPUT],
        ORCHESTRATOR: used_configurations[epipolar_resolutions[0]][
            ORCHESTRATOR
        ],
    }

    merged_conf[APPLICATIONS] = {}
    merged_conf[APPLICATIONS]["all"] = {}

    # Merge applications
    for res in epipolar_resolutions:
        merged_conf[APPLICATIONS][res] = used_configurations[res][APPLICATIONS]

    # apply epipolar resolutions
    merged_conf[ADVANCED]["epipolar_resolutions"] = epipolar_resolutions
    return merged_conf
