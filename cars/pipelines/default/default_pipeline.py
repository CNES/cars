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

# CARS imports
from cars.core import cars_logging
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import depth_map_inputs_constants as depth_cst
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUTS,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate
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
        "conf_first_resolution.json",
    ),
    INTERMEDIATE_RES: os.path.join(
        package_path,
        "..",
        "conf_resolution",
        "conf_intermediate_resolution.json",
    ),
    FINAL_RES: os.path.join(
        package_path,
        "..",
        "conf_resolution",
        "conf_final_resolution.json",
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

        # Check application
        self.check_application_keys_name(
            self.epipolar_resolutions, conf.get(APPLICATIONS, {})
        )
        # Check input
        conf[INPUTS] = self.check_inputs(conf)
        # check advanced
        conf[ADVANCED] = self.check_advanced(conf)
        # check output
        conf[OUTPUT] = self.check_output(conf)

        if isinstance(self.epipolar_resolutions, int):
            self.epipolar_resolutions = [self.epipolar_resolutions]

        if (
            (depth_cst.DEPTH_MAPS in conf[INPUTS])
            or (dsm_cst.DSMS in conf[INPUTS])
        ) and len(self.epipolar_resolutions) != 1:
            raise RuntimeError(
                "For the use of those pipelines, "
                "you have to give only one resolution"
            )

        used_configurations = {}
        self.unit_pipelines = {}
        self.positions = {}

        self.intermediate_out_dirs = []

        self.keep_low_res_dir = False  # TODO update

        for epipolar_resolution_index, epipolar_res in enumerate(
            self.epipolar_resolutions
        ):
            first_res = epipolar_resolution_index == 0
            intermediate_res = (
                epipolar_resolution_index == len(self.epipolar_resolutions) - 1
                or len(self.epipolar_resolutions) == 1
            )
            last_res = (
                epipolar_resolution_index == len(self.epipolar_resolutions) - 1
            )

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
            )
            # get output directory
            self.intermediate_out_dirs.append(
                current_conf[OUTPUT][out_cst.OUT_DIRECTORY]
            )

            if not isinstance(epipolar_res, int) or epipolar_res < 0:
                raise RuntimeError("The resolution has to be an int > 0")

            # Initialize unit pipeline in order to retrieve the
            # used configuration
            # This pipeline will not be run

            print("INIT PIPELINE", epipolar_res)
            current_unit_pipeline = UnitPipeline(
                current_conf, config_dir=self.config_dir
            )
            self.unit_pipelines[epipolar_resolution_index] = (
                current_unit_pipeline
            )
            # Get used_conf
            used_configurations[epipolar_res] = current_unit_pipeline.used_conf

        # Generate full used_conf
        full_used_conf = merge_used_conf(used_configurations)
        # Save used_conf
        cars_dataset.save_dict(
            full_used_conf,
            os.path.join(self.out_dir, "used_conf.json"),
            safe_save=True,
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
            conf[INPUTS], config_dir=self.config_dir
        )

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """

        print("initial", conf[OUTPUT])
        print("scaling", self.scaling_coeff)
        conf_output, self.scaling_coeff = (
            output_parameters.check_output_parameters(
                conf[OUTPUT], self.scaling_coeff
            )
        )
        return conf_output

    def check_advanced(self, conf):
        """
        Check all conf for advanced configuration

        :return: overridden advanced conf
        :rtype: dict
        """
        (_, advanced, _, _, _, _, self.scaling_coeff) = (
            advanced_parameters.check_advanced_parameters(
                conf[INPUTS],
                conf.get(ADVANCED, {}),
                check_epipolar_a_priori=True,
            )
        )
        return advanced

    def check_applications(self, conf, key=None, res=None, last_res=False):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

    def check_application_keys_name(self, resolutions, applications_conf):
        """
        Check if the application name for each res match 'resolution_res'
        """

        if applications_conf != {}:
            for key_app, _ in applications_conf.items():
                if not key_app.startswith("resolution"):
                    raise RuntimeError(
                        "If you decided to define an "
                        "application per resolution, "
                        "all the keys have to be : "
                        "resolution_res"
                    )
                if int(key_app.split("_")[1]) not in resolutions:
                    raise RuntimeError(
                        "This resolution "
                        + key_app.split("_")[1]
                        + " is not in the resolution list"
                    )

    def cleanup_low_res_dir(self):
        """
        Clean low res dir
        """

        for out_dir in self.intermediate_out_dirs[:-1]:
            if os.path.exists(out_dir) and os.path.isdir(out_dir):
                try:
                    shutil.rmtree(out_dir)
                    logging.info(f"th directory {out_dir} has been cleaned.")
                except Exception as exception:
                    logging.error(
                        f"Error while deleting {out_dir}: {exception}"
                    )
            else:
                logging.info(f"The directory {out_dir} has not been deleted")

    def overide_with_apriori(self, conf, previous_out_dir, first_res):
        """
        Override configuration with terrain a priori

        :param new_conf: configuration
        :type new_conf: dict
        """

        new_conf = copy.deepcopy(conf)

        # Extract avanced parameters configuration
        # epipolar and terrain a priori can only be used on firt resolution
        if not first_res:
            dem_min = os.path.join(previous_out_dir, "dsm/dem_min.tif")
            dem_max = os.path.join(previous_out_dir, "dsm/dem_max.tif")
            dem_median = os.path.join(previous_out_dir, "dsm/dem_median.tif")

            new_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI] = {
                "dem_min": dem_min,
                "dem_max": dem_max,
                "dem_median": dem_median,
            }
            # Use initial elevation or dem median according to use wish
            if new_conf[INPUTS][sens_cst.INITIAL_ELEVATION]["dem"] is None:
                new_conf[INPUTS][sens_cst.INITIAL_ELEVATION] = dem_median
            else:
                new_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI]["dem_median"] = (
                    new_conf[INPUTS][sens_cst.INITIAL_ELEVATION]["dem"]
                )
            if new_conf[ADVANCED][adv_cst.USE_ENDOGENOUS_DEM] and not first_res:
                new_conf[INPUTS][sens_cst.INITIAL_ELEVATION] = dem_median

            new_conf[ADVANCED][adv_cst.EPIPOLAR_A_PRIORI] = None

        return new_conf

    @cars_profile(name="run_dense_pipeline", interval=0.5)
    def run(self, args=None):  # noqa C901
        """
        Run pipeline

        """

        print("RUN")

        # Get first res outdir for sift matches
        first_res_out_dir = self.intermediate_out_dirs[0]
        previous_out_dir = None
        for resolution_index, epipolar_res in enumerate(
            self.epipolar_resolutions
        ):
            current_out_dir = self.intermediate_out_dirs[resolution_index]

            # get position
            first_res, _, last_res = (
                self.positions[resolution_index]["first_res"],
                self.positions[resolution_index]["intermediate_res"],
                self.positions[resolution_index]["last_res"],
            )

            # Setup logging
            if not (first_res and last_res):
                loglevel = getattr(args, "loglevel", "PROGRESS").upper()

                cars_logging.setup_logging(
                    loglevel,
                    out_dir=os.path.join(current_out_dir, "logs"),
                    pipeline="",
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

            # Launch unit pipeline
            used_pipeline = self.unit_pipelines[resolution_index]

            overridden_conf = self.overide_with_apriori(
                used_pipeline.used_conf, previous_out_dir, first_res
            )
            updated_pipeline = UnitPipeline(
                overridden_conf, config_dir=self.config_dir
            )
            updated_pipeline.run(
                generate_dems=generate_dems,
                which_resolution=which_resolution,
                use_sift_a_priori=use_sift_a_priori,
                first_res_out_dir=first_res_out_dir,
                final_out_dir=self.out_dir,
            )

            # update previous out dir
            previous_out_dir = current_out_dir

        # Merge profiling in pdf
        # TODO store used_conf

        # clean outdir
        if not self.keep_low_res_dir:
            self.cleanup_low_res_dir()


def extract_conf_with_resolution(
    current_conf, res, first_res, intermediate_res, last_res
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

    print("current_conf", current_conf)
    print(OUTPUT, out_cst.OUT_DIRECTORY)
    new_dir_out_dir = current_conf[OUTPUT][out_cst.OUT_DIRECTORY]
    if not last_res:
        new_dir_out_dir = (
            current_conf[OUTPUT][out_cst.OUT_DIRECTORY]
            + "/intermediate_res/out_res"
            + str(res)
        )
        safe_makedirs(new_dir_out_dir)

    new_conf = copy.deepcopy(current_conf)

    # Extract application configuration
    current_application_conf = current_conf.get(APPLICATIONS, {})
    new_application_conf = {}
    if not any(
        key.startswith("resolution_") for key in current_application_conf
    ):
        # configuration given per resolution
        for key in current_application_conf:
            if key.startswith("resolution_") and int(key.split("_")[1]) == res:
                new_application_conf[key] = current_application_conf[key]
    else:
        new_application_conf = current_application_conf

    # apply new configuration
    new_conf[APPLICATIONS] = new_application_conf

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
            overiding_conf = json.load(file)
    elif intermediate_res:
        with open(
            PIPELINE_CONFS[INTERMEDIATE_RES], "r", encoding="utf-8"
        ) as file:
            overiding_conf = json.load(file)
    else:
        with open(PIPELINE_CONFS[FINAL_RES], "r", encoding="utf-8") as file:
            overiding_conf = json.load(file)

    new_conf = overide_pipeline_conf(new_conf, overiding_conf)

    # Overide output to not compute data
    if not last_res:
        overiding_conf = {
            OUTPUT: {
                out_cst.OUT_DIRECTORY: new_dir_out_dir,
                out_cst.SAVE_BY_PAIR: True,
                out_cst.AUXILIARY: {
                    out_cst.AUX_DEM_MAX: True,
                    out_cst.AUX_DEM_MIN: True,
                    out_cst.AUX_DEM_MEDIAN: True,
                },
            }
        }
        new_conf = overide_pipeline_conf(new_conf, overiding_conf)

        if not new_conf[ADVANCED][adv_cst.SAVE_INTERMEDIATE_DATA]:
            # Save the less possible things
            aux_items = new_conf[OUTPUT][out_cst.AUXILIARY].items()
            for aux_key, _ in aux_items:
                if aux_key not in ("dem_min", "dem_max", "dem_median"):
                    new_conf[OUTPUT][out_cst.AUXILIARY][aux_key] = False
        else:
            # we save the depth_maps also to debug
            new_conf[OUTPUT][out_cst.PRODUCT_LEVEL] = [
                "dsm",
                "depth_map",
            ]

    return new_conf


def overide_pipeline_conf(conf, overiding_conf):
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
        for key, value in override_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                merge_recursive(base_dict[key], value)
            else:
                base_dict[key] = value

    merge_recursive(result, overiding_conf)
    return result


def merge_used_conf(used_configurations):
    """
    Merge all used configuration
    """

    return {"WIP": used_configurations}
