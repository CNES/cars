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
from cars.pipelines.filling.filling import FillingPipeline
from cars.pipelines.formatting.formatting import FormattingPipeline
from cars.pipelines.merging.merging import MergingPipeline
from cars.pipelines.parameters import advanced_parameters, dsm_inputs
from cars.pipelines.parameters import dsm_inputs_constants as dsm_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.output_constants import AUXILIARY
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.subsampling.subsampling import SubsamplingPipeline
from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

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
            advanced_parameters.get_epipolar_resolutions(
                conf.get(pipeline_cst.SUBSAMPLING, {}).get(ADVANCED, {})
            )
        )
        if isinstance(self.epipolar_resolutions, int):
            self.epipolar_resolutions = [self.epipolar_resolutions]

        conf[PIPELINE] = self.check_pipeline(conf)

        self.pipeline_to_use = conf[PIPELINE]

        # Check input
        conf[INPUT] = self.check_inputs(conf, config_json_dir=config_dir)

        # check output
        conf[OUTPUT] = self.check_output(conf)

        self.intermediate_data_dir = os.path.join(
            self.out_dir, "intermediate_data"
        )

        conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )

        for pipeline, val in self.pipeline_to_use.items():
            if pipeline in conf and not val:
                logging.warning(
                    f"You tried to override the {pipeline} pipeline but "
                    f"didn't specify it in the pipeline section. "
                    "Therefore, this pipeline will not be used"
                )

        if pipeline_cst.SURFACE_MODELING not in conf:
            conf[pipeline_cst.SURFACE_MODELING] = {}
        if pipeline_cst.TIE_POINTS not in conf:
            conf[pipeline_cst.TIE_POINTS] = {}

        if dsm_cst.DSMS in conf[INPUT] and len(self.epipolar_resolutions) != 1:
            logging.info(
                "For the use of those pipelines, "
                "you have to give only one resolution"
            )
            # overide epipolar resolutions
            # TODO: delete with external dsm pipeline (refactoring)
            self.epipolar_resolutions = [1]
        elif (
            not self.pipeline_to_use[pipeline_cst.SUBSAMPLING]
            and len(self.epipolar_resolutions) != 1
        ):
            logging.warning(
                "As you're not using the subsampling pipeline, "
                "the working resolution will be 1"
            )

            self.epipolar_resolutions = [1]

        used_configurations = {}
        self.positions = {}
        self.used_conf = {}

        self.keep_low_res_dir = True

        if self.pipeline_to_use[pipeline_cst.SUBSAMPLING]:
            self.subsampling_conf = self.construct_subsampling_conf(conf)
            conf[pipeline_cst.SUBSAMPLING] = self.check_subsampling(
                self.subsampling_conf
            )

        if self.pipeline_to_use[pipeline_cst.FILLING]:
            self.filling_conf = self.construct_filling_conf(conf)
            conf[pipeline_cst.FILLING] = self.check_filling(self.filling_conf)

        subsampling_used_conf = conf.get(pipeline_cst.SUBSAMPLING, {})

        filling_used_conf = conf.get(pipeline_cst.FILLING, {})

        if self.pipeline_to_use[pipeline_cst.SURFACE_MODELING]:
            for epipolar_resolution_index, epipolar_res in enumerate(
                self.epipolar_resolutions
            ):
                first_res = epipolar_resolution_index == 0
                last_res = (
                    epipolar_resolution_index
                    == len(self.epipolar_resolutions) - 1
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

                if not isinstance(epipolar_res, int) or epipolar_res < 0:
                    raise RuntimeError("The resolution has to be an int > 0")

                self.used_conf[epipolar_resolution_index] = current_conf

                # Initialize unit pipeline in order to retrieve the
                # used configuration
                # This pipeline will not be run
                _ = current_conf.pop(pipeline_cst.SUBSAMPLING, None)
                _ = current_conf.pop(pipeline_cst.FILLING, None)

                current_unit_pipeline = SurfaceModelingPipeline(
                    current_conf,
                    config_dir=self.config_dir,
                )
                # Get used_conf
                used_configurations[epipolar_res] = (
                    current_unit_pipeline.used_conf
                )

            # Generate full used_conf
            full_used_conf = merge_used_conf(
                used_configurations,
                self.epipolar_resolutions,
                os.path.abspath(self.out_dir),
            )
        else:
            self.used_conf = copy.deepcopy(conf)
            full_used_conf = self.used_conf

        full_used_conf[pipeline_cst.SUBSAMPLING] = subsampling_used_conf
        full_used_conf[pipeline_cst.PIPELINE] = conf[PIPELINE]

        full_used_conf[pipeline_cst.FILLING] = filling_used_conf

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
        output_config = {}
        if sens_cst.SENSORS in conf[INPUT] and dsm_cst.DSMS not in conf[INPUT]:
            output_config = sensor_inputs.sensors_check_inputs(
                conf[INPUT], config_dir=config_json_dir
            )
        elif dsm_cst.DSMS in conf[INPUT]:
            output_config = {
                **output_config,
                **dsm_inputs.check_dsm_inputs(
                    conf[INPUT], config_dir=config_json_dir
                ),
            }
        else:
            raise RuntimeError("No sensors or dsms in inputs")

        return output_config

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

    def check_pipeline(self, conf):  # noqa: C901
        """
        Check the pipeline section
        """
        possible_pipeline = [
            pipeline_cst.SUBSAMPLING,
            pipeline_cst.SURFACE_MODELING,
            pipeline_cst.FILLING,
            pipeline_cst.MERGING,
            pipeline_cst.FORMATTING,
        ]
        dict_pipeline = {}

        if PIPELINE not in conf:
            if dsm_cst.DSMS in conf[INPUT]:
                conf[PIPELINE] = [pipeline_cst.MERGING, pipeline_cst.FORMATTING]
            elif sens_cst.SENSORS in conf[INPUT]:
                conf[PIPELINE] = [
                    pipeline_cst.SUBSAMPLING,
                    pipeline_cst.SURFACE_MODELING,
                    pipeline_cst.FORMATTING,
                ]

        if isinstance(conf[PIPELINE], str):
            if conf[PIPELINE] not in possible_pipeline:
                raise RuntimeError("This pipeline does not exist")
            dict_pipeline = {conf[PIPELINE]: True}
        elif isinstance(conf[PIPELINE], list):
            for elem in conf[PIPELINE]:
                if elem not in possible_pipeline:
                    raise RuntimeError(f"The pipeline {elem} does not exist")
                dict_pipeline.update({elem: True})
        elif isinstance(conf[PIPELINE], dict):
            for key, _ in conf[PIPELINE].items():
                if key not in possible_pipeline:
                    raise RuntimeError(f"The pipeline {key} does not exist")

        for key in possible_pipeline:
            if key not in dict_pipeline:
                dict_pipeline.update({key: False})

        if (
            dsm_cst.DSMS in conf[INPUT]
            and not dict_pipeline[pipeline_cst.MERGING]
        ):
            dict_pipeline[pipeline_cst.MERGING] = True
        elif (
            dsm_cst.DSMS in conf[INPUT]
            and dict_pipeline[pipeline_cst.SURFACE_MODELING]
        ):
            raise RuntimeError(
                "You can not use the surface modeling pipeline with dsm inputs"
            )
        elif (
            sens_cst.SENSORS in conf[INPUT]
            and dict_pipeline[pipeline_cst.MERGING]
            and dsm_cst.DSMS not in conf[INPUT]
        ):
            raise RuntimeError(
                "You can not use the merging pipeline with sensors inputs only"
            )

        if (
            pipeline_cst.FILLING in conf[INPUT] or pipeline_cst.FILLING in conf
        ) and not dict_pipeline[pipeline_cst.FILLING]:
            dict_pipeline[pipeline_cst.FILLING] = True

        if (
            pipeline_cst.SURFACE_MODELING in conf[INPUT]
            and not dict_pipeline[pipeline_cst.SURFACE_MODELING]
        ):
            dict_pipeline[pipeline_cst.SURFACE_MODELING] = True

        if (
            pipeline_cst.MERGING in conf[INPUT]
            and not dict_pipeline[pipeline_cst.MERGING]
        ):
            dict_pipeline[pipeline_cst.MERGING] = True

        if (
            pipeline_cst.SUBSAMPLING in conf[INPUT]
            and not dict_pipeline[pipeline_cst.SUBSAMPLING]
        ):
            dict_pipeline[pipeline_cst.SUBSAMPLING] = True

        return dict_pipeline

    def check_subsampling(self, conf):
        """
        Check the subsampling section

        :param conf: configuration of subsampling
        :type conf: dict
        """

        pipeline = SubsamplingPipeline(conf)
        advanced = pipeline.check_advanced(conf.get(ADVANCED, {}))
        applications = pipeline.check_applications(conf.get(APPLICATIONS, {}))

        return {"advanced": advanced, "applications": applications}

    def check_filling(self, conf):
        """
        Check the filling section

        :param conf: configuration of subsampling
        :type conf: dict
        """

        pipeline = FillingPipeline(conf, pre_check=True)
        advanced = pipeline.check_advanced(
            conf[pipeline_cst.FILLING], conf[INPUT]
        )
        applications = pipeline.check_applications(conf.get(APPLICATIONS, {}))

        return {"advanced": advanced, "applications": applications}

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

    def construct_merging_conf(self, conf):
        """
        Construct the right conf for merging
        """
        merging_conf = {}
        merging_conf[INPUT] = copy.deepcopy(conf[INPUT])
        merging_conf[OUTPUT] = {}
        merging_conf[OUTPUT]["directory"] = os.path.join(
            self.intermediate_data_dir, pipeline_cst.MERGING
        )
        merging_conf[OUTPUT][AUXILIARY] = conf[OUTPUT].get(AUXILIARY, {})

        merging_conf[pipeline_cst.MERGING] = conf.get(pipeline_cst.MERGING, {})

        return merging_conf

    def construct_subsampling_conf(self, conf):
        """
        Construct the right conf for subsampling
        """
        subsampling_conf = {}
        subsampling_conf[INPUT] = copy.deepcopy(conf[INPUT])
        subsampling_conf[OUTPUT] = {}
        subsampling_conf[OUTPUT]["directory"] = self.intermediate_data_dir

        subsampling_conf[pipeline_cst.SUBSAMPLING] = conf.get(
            pipeline_cst.SUBSAMPLING, {}
        )

        return subsampling_conf

    def construct_formatting_conf(self, input_dir):
        """
        Construct the right conf for formatting
        """

        formatting_conf = {}
        formatting_conf[INPUT] = {}
        formatting_conf[INPUT]["input_path"] = input_dir
        formatting_conf[OUTPUT] = {}
        formatting_conf[OUTPUT]["directory"] = self.out_dir

        return formatting_conf

    def construct_filling_conf(self, conf):
        """
        Construct the right conf for filling
        """
        filling_conf = {}
        filling_conf[INPUT] = copy.deepcopy(conf[INPUT])
        _ = filling_conf[INPUT].pop(dsm_cst.DSMS, None)
        filling_conf[OUTPUT] = copy.deepcopy(conf[OUTPUT])
        filling_conf[OUTPUT]["directory"] = self.intermediate_data_dir
        filling_conf[pipeline_cst.FILLING] = conf.get(pipeline_cst.FILLING, {})
        return filling_conf

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

        if self.pipeline_to_use[pipeline_cst.SUBSAMPLING]:
            subsampling_pipeline = SubsamplingPipeline(
                self.subsampling_conf, self.config_dir
            )
            subsampling_pipeline.run()

        if self.pipeline_to_use[pipeline_cst.SURFACE_MODELING]:
            for resolution_index, epipolar_res in enumerate(
                self.epipolar_resolutions
            ):

                # Get tested unit pipeline
                current_conf = self.used_conf[resolution_index]
                current_out_dir = current_conf[OUTPUT]["directory"]

                # Put right directory for subsampling
                if self.pipeline_to_use[pipeline_cst.SUBSAMPLING]:
                    if epipolar_res != 1:
                        yaml_file = os.path.join(
                            self.intermediate_data_dir,
                            "subsampling/res_"
                            + str(epipolar_res)
                            + "/input.yaml",
                        )
                        with open(yaml_file, encoding="utf-8") as f:
                            data = yaml.safe_load(f)

                        json_str = json.dumps(data, indent=4)
                        data = json.loads(json_str)

                        current_conf[INPUT] = data

                # update directory for unit pipeline
                current_conf[OUTPUT]["directory"] = current_out_dir

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
                    pipeline="surface_modeling",
                    global_log_file=global_log_file,
                )

                cars_logging.add_progress_message(
                    "Starting surface modeling pipeline for resolution 1/"
                    + str(epipolar_res)
                )

                # define wich resolution
                if first_res and last_res:
                    which_resolution = "single"
                elif first_res:
                    which_resolution = "first"
                elif last_res:
                    which_resolution = "final"
                else:
                    which_resolution = "intermediate"

                # Overide with a priori
                if not first_res:
                    dsm = os.path.join(previous_out_dir, "dsm/dsm.tif")
                    current_conf[INPUT][sens_cst.LOW_RES_DSM] = dsm

                # Define tie points output dir
                tie_points_out_dir = os.path.join(
                    self.intermediate_data_dir,
                    "tie_points",
                    "res" + str(epipolar_res),
                )
                safe_makedirs(tie_points_out_dir)

                updated_pipeline = SurfaceModelingPipeline(
                    current_conf,
                    config_dir=self.config_dir,
                )
                updated_pipeline.run(
                    which_resolution=which_resolution,
                    log_dir=current_log_dir,
                    tie_points_out_dir=tie_points_out_dir,
                )

                # update previous out dir
                previous_out_dir = current_out_dir

                # generate summary
                log_wrapper.generate_summary(
                    current_log_dir,
                    updated_pipeline.used_conf,
                    pipeline_cst.SURFACE_MODELING,
                )

                updated_conf[epipolar_res] = updated_pipeline.used_conf

            # Generate full used_conf
            full_used_conf = merge_used_conf(
                updated_conf,
                self.epipolar_resolutions,
                os.path.abspath(self.out_dir),
            )
        else:
            full_used_conf = self.used_conf

        final_conf = None
        if self.pipeline_to_use[pipeline_cst.MERGING]:
            merging_conf = self.construct_merging_conf(self.used_conf)
            merging_pipeline = MergingPipeline(merging_conf, self.config_dir)
            merging_pipeline.run()

            final_conf = merging_pipeline.used_conf

        if updated_conf and final_conf is None:
            last_key = list(updated_conf.keys())[-1]
            final_conf = updated_conf[last_key]
        elif not updated_conf and final_conf is None:
            final_conf = self.used_conf

        formatting_input_dir = final_conf[OUTPUT][out_cst.OUT_DIRECTORY]

        if self.pipeline_to_use[pipeline_cst.FILLING]:
            if self.filling_conf[INPUT]["dsm_to_fill"] is None:
                if (
                    not self.pipeline_to_use[pipeline_cst.SURFACE_MODELING]
                    and not self.pipeline_to_use[pipeline_cst.MERGING]
                ):
                    raise RuntimeError(
                        "You have to fill the dsm_to_fill part of the input if "
                        "you want to use the filling pipeline separately"
                    )

                self.filling_conf[INPUT]["dsm_to_fill"] = {}
                aux_path = os.path.join(
                    final_conf[OUTPUT][out_cst.OUT_DIRECTORY], "dsm/"
                )
                self.filling_conf[INPUT]["dsm_to_fill"]["dsm"] = os.path.join(
                    aux_path, "dsm.tif"
                )

                for aux_output, val in final_conf[OUTPUT][
                    out_cst.AUXILIARY
                ].items():
                    if val:
                        self.filling_conf[INPUT]["dsm_to_fill"][aux_output] = (
                            os.path.join(aux_path, aux_output + ".tif")
                        )
            initial_elevation = final_conf[INPUT][
                sens_cst.INITIAL_ELEVATION
            ].get("dem", None)

            if (
                initial_elevation is not None
                and "dem_median" in initial_elevation
            ):
                self.filling_conf[INPUT][sens_cst.INITIAL_ELEVATION] = None

            filling_pipeline = FillingPipeline(
                self.filling_conf, self.config_dir
            )
            filling_pipeline.run()

            formatting_input_dir = os.path.join(
                filling_pipeline.used_conf[OUTPUT][out_cst.OUT_DIRECTORY],
                pipeline_cst.FILLING,
            )

        if self.pipeline_to_use[pipeline_cst.FORMATTING]:
            formatting_conf = self.construct_formatting_conf(
                formatting_input_dir
            )
            formatting_pipeline = FormattingPipeline(
                formatting_conf, self.config_dir
            )
            formatting_pipeline.run()

        if self.pipeline_to_use[pipeline_cst.FILLING]:
            full_used_conf[pipeline_cst.FILLING] = {
                ADVANCED: filling_pipeline.used_conf[ADVANCED],
                APPLICATIONS: filling_pipeline.used_conf[APPLICATIONS],
            }

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


def extract_conf_section(
    current_conf_section,
    res,
    default_conf_for_res=None,
    filling_applications=None,
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

    all_conf = current_conf_section.get("all", {})
    # Overide with default_conf_for_res
    if default_conf_for_res is not None:
        all_conf = overide_pipeline_conf(all_conf, default_conf_for_res)
    # Get configuration for current res
    if res in current_conf_section:
        # key is int
        key = res
    else:
        key = str(res)

    res_conf = current_conf_section.get(key, {})

    # Overide all conf with current res conf
    new_application_conf = overide_pipeline_conf(all_conf, res_conf)

    # Overide with filling applications
    if filling_applications is not None:
        new_application_conf = overide_pipeline_conf(
            new_application_conf,
            filling_applications,
            append_classification=True,
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

    surface_modeling_out_dir = os.path.join(
        intermediate_data_dir, "surface_modeling", "res" + str(res)
    )
    safe_makedirs(surface_modeling_out_dir)

    new_conf = copy.deepcopy(current_conf)

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
        filling_applications_for_surface_modeling = (
            generate_filling_applications_for_surface_modeling(
                current_conf[INPUT]
            )
        )
    else:
        filling_applications_for_surface_modeling = {}

    # Extract surface modeling conf
    new_conf[pipeline_cst.SURFACE_MODELING] = {}
    new_conf[pipeline_cst.SURFACE_MODELING][APPLICATIONS] = (
        extract_conf_section(
            current_conf[pipeline_cst.SURFACE_MODELING].get(APPLICATIONS, {}),
            res,
            overiding_conf.get(APPLICATIONS, {}),
            filling_applications_for_surface_modeling,
        )
    )
    new_conf[pipeline_cst.SURFACE_MODELING][ADVANCED] = extract_conf_section(
        current_conf[pipeline_cst.SURFACE_MODELING].get(ADVANCED, {}),
        res,
    )

    # Extract tie points conf
    new_conf[pipeline_cst.TIE_POINTS] = {}
    new_conf[pipeline_cst.TIE_POINTS][APPLICATIONS] = extract_conf_section(
        current_conf[pipeline_cst.TIE_POINTS].get(APPLICATIONS, {}),
        res,
        overiding_conf.get(APPLICATIONS, {}),
    )
    new_conf[pipeline_cst.TIE_POINTS][ADVANCED] = extract_conf_section(
        current_conf[pipeline_cst.TIE_POINTS].get(ADVANCED, {}),
        res,
    )

    overiding_conf = {
        OUTPUT: {out_cst.OUT_DIRECTORY: surface_modeling_out_dir},
    }
    new_conf = overide_pipeline_conf(new_conf, overiding_conf)

    # Overide output to not compute data
    if not last_res:
        overiding_conf = {
            pipeline_cst.SURFACE_MODELING: {
                APPLICATIONS: {
                    "dense_matching": {
                        "performance_map_method": ["risk", "intervals"]
                    }
                }
            },
        }
        new_conf = overide_pipeline_conf(new_conf, overiding_conf)

        # set product level to dsm
        new_conf[OUTPUT][out_cst.PRODUCT_LEVEL] = ["dsm"]
        # remove resolution to let CARS compute it for current
        # epipolar resolution
        new_conf[OUTPUT]["resolution"] = None

        # Save the less possible things
        for aux_key in new_conf[OUTPUT][out_cst.AUXILIARY]:
            if aux_key != "image":
                new_conf[OUTPUT][out_cst.AUXILIARY][aux_key] = False

    return new_conf


def generate_filling_applications_for_surface_modeling(inputs_conf):
    """
    Generate filling applications configuration according to inputs

    :param inputs_conf: inputs configuration
    :type inputs_conf: dict
    """

    filling_applications = {}

    # Generate applications configuration
    for _, classif_values in inputs_conf[sens_cst.FILLING].items():
        # No filling
        if classif_values is None:
            continue

        classif_values = list(map(str, classif_values))

        # Update application configuration
        new_filling_conf = {
            "dense_match_filling": {
                "method": "zero_padding",
                "classification": classif_values,
            }
        }

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


def merge_used_conf(used_configurations, epipolar_resolutions, out_dir):
    """
    Merge all used configuration
    """
    used_configurations = copy.deepcopy(used_configurations)

    merged_conf = {
        INPUT: used_configurations[epipolar_resolutions[-1]][INPUT],
        OUTPUT: used_configurations[epipolar_resolutions[0]][OUTPUT],
        ORCHESTRATOR: used_configurations[epipolar_resolutions[0]][
            ORCHESTRATOR
        ],
    }

    merged_conf[OUTPUT]["directory"] = out_dir

    merged_conf[pipeline_cst.TIE_POINTS] = {APPLICATIONS: {}, ADVANCED: {}}
    merged_conf[pipeline_cst.SURFACE_MODELING] = {
        APPLICATIONS: {},
        ADVANCED: {},
    }

    for resolution in epipolar_resolutions:
        used_conf = used_configurations[resolution]
        merged_conf[pipeline_cst.TIE_POINTS][APPLICATIONS][str(resolution)] = (
            used_conf[pipeline_cst.TIE_POINTS][APPLICATIONS]
        )
        merged_conf[pipeline_cst.TIE_POINTS][ADVANCED][str(resolution)] = (
            used_conf[pipeline_cst.TIE_POINTS][ADVANCED]
        )
        merged_conf[pipeline_cst.SURFACE_MODELING][APPLICATIONS][
            str(resolution)
        ] = used_conf[pipeline_cst.SURFACE_MODELING][APPLICATIONS]
        merged_conf[pipeline_cst.SURFACE_MODELING][ADVANCED][
            str(resolution)
        ] = used_conf[pipeline_cst.SURFACE_MODELING][ADVANCED]

    return merged_conf
