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
This module refers to the prepare outputs
"""

# Standard imports
import json
import os
from typing import Dict, Union

# Third party imports
from json_checker import And, OptionalKey, Or

# CARS imports
from cars.conf import input_parameters, mask_classes, static_conf
from cars.core.inputs import rasterio_can_open
from cars.core.utils import make_relative_path_absolute


def write_preprocessing_content_file(config, filename, indent=2):
    """
    Write a preprocessing json content file.
    Relative paths in preprocessing/output section will be made absolute.

    :param config: dictionnary holding the config to write
    :type config: dict
    :param filename: Path to json file
    :type filename: str
    :param indent: indentations in output file
    :type indent: int
    """
    with open(filename, "w") as fstream:
        # Make absolute path relative
        for tag in [
            LEFT_EPIPOLAR_GRID_TAG,
            RIGHT_EPIPOLAR_GRID_TAG,
            LEFT_ENVELOPE_TAG,
            RIGHT_ENVELOPE_TAG,
            MATCHES_TAG,
            RAW_MATCHES_TAG,
            RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG,
            LEFT_ENVELOPE_TAG,
            RIGHT_ENVELOPE_TAG,
            ENVELOPES_INTERSECTION_TAG,
            LOWRES_DSM_TAG,
            LOWRES_INITIAL_DEM_TAG,
            LOWRES_ELEVATION_DIFFERENCE_TAG,
            LOWRES_DEM_SPLINES_FIT_TAG,
            CORRECTED_LOWRES_DSM_TAG,
            CORRECTED_LOWRES_ELEVATION_DIFFERENCE_TAG,
        ]:
            if (
                tag
                in config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ]
            ):

                value = config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ][tag]

                config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ][tag] = os.path.basename(value)

        json.dump(config, fstream, indent=indent)


def read_preprocessing_content_file(filename):
    """
    Read a json content file from preprocessing step.
    Relative paths in preprocessing/output section  will be made absolute.

    :param filename: Path to json file
    :type filename: str

    :returns: The dictionnary read from file
    :rtype: dict
    """
    config = {}
    with open(filename, "r") as fstream:
        config = json.load(fstream)
        json_dir = os.path.abspath(os.path.dirname(filename))
        # Make relative path absolute
        for tag in [
            input_parameters.IMG1_TAG,
            input_parameters.IMG2_TAG,
            input_parameters.MASK1_TAG,
            input_parameters.MASK2_TAG,
            input_parameters.COLOR1_TAG,
            input_parameters.SRTM_DIR_TAG,
        ]:
            if tag in config[input_parameters.INPUT_SECTION_TAG]:
                # Get config input parameters tag paths
                value = config[input_parameters.INPUT_SECTION_TAG][tag]
                # Update relative paths to absolute ones
                config[input_parameters.INPUT_SECTION_TAG][
                    tag
                ] = make_relative_path_absolute(value, json_dir)

        for tag in [
            LEFT_EPIPOLAR_GRID_TAG,
            RIGHT_EPIPOLAR_GRID_TAG,
            LEFT_ENVELOPE_TAG,
            RIGHT_ENVELOPE_TAG,
            MATCHES_TAG,
            RAW_MATCHES_TAG,
            RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG,
            LEFT_ENVELOPE_TAG,
            RIGHT_ENVELOPE_TAG,
            ENVELOPES_INTERSECTION_TAG,
            LOWRES_DSM_TAG,
            LOWRES_INITIAL_DEM_TAG,
            LOWRES_ELEVATION_DIFFERENCE_TAG,
            LOWRES_DEM_SPLINES_FIT_TAG,
            CORRECTED_LOWRES_DSM_TAG,
            CORRECTED_LOWRES_ELEVATION_DIFFERENCE_TAG,
        ]:
            if (
                tag
                in config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ]
            ):
                # Get config preprocessing section tag paths
                value = config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ][tag]
                # Update relative paths to absolute ones
                config[PREPROCESSING_SECTION_TAG][
                    PREPROCESSING_OUTPUT_SECTION_TAG
                ][tag] = make_relative_path_absolute(value, json_dir)

    # Return config with absolute paths updated
    return config


# Tags for preprocessing output section in content.json of preprocessing step
MINIMUM_DISPARITY_TAG = "minimum_disparity"
MAXIMUM_DISPARITY_TAG = "maximum_disparity"
LEFT_EPIPOLAR_GRID_TAG = "left_epipolar_grid"
RIGHT_EPIPOLAR_GRID_TAG = "right_epipolar_grid"
LEFT_ENVELOPE_TAG = "left_envelope"
RIGHT_ENVELOPE_TAG = "right_envelope"
ENVELOPES_INTERSECTION_TAG = "envelopes_intersection"
ENVELOPES_INTERSECTION_BB_TAG = "envelopes_intersection_bounding_box"
EPIPOLAR_SIZE_X_TAG = "epipolar_size_x"
EPIPOLAR_SIZE_Y_TAG = "epipolar_size_y"
EPIPOLAR_ORIGIN_X_TAG = "epipolar_origin_x"
EPIPOLAR_ORIGIN_Y_TAG = "epipolar_origin_y"
EPIPOLAR_SPACING_X_TAG = "epipolar_spacing_x"
EPIPOLAR_SPACING_Y_TAG = "epipolar_spacing_y"
MATCHES_TAG = "matches"
RAW_MATCHES_TAG = "raw_matches"
LOWRES_DSM_TAG = "lowres_dsm"
LOWRES_INITIAL_DEM_TAG = "lowres_initial_dem"
LOWRES_ELEVATION_DIFFERENCE_TAG = "lowres_elevation_difference"
RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG = "right_epipolar_uncorrected_grid"
DISP_TO_ALT_RATIO_TAG = "disp_to_alt_ratio"
LEFT_AZIMUTH_ANGLE_TAG = "left_azimuth_angle"
LEFT_ELEVATION_ANGLE_TAG = "left_elevation_angle"
RIGHT_AZIMUTH_ANGLE_TAG = "right_azimuth_angle"
RIGHT_ELEVATION_ANGLE_TAG = "right_elevation_angle"
CONVERGENCE_ANGLE_TAG = "convergence_angle"
TIME_DIRECTION_LINE_ORIGIN_X_TAG = "time_direction_line_origin_x"
TIME_DIRECTION_LINE_ORIGIN_Y_TAG = "time_direction_line_origin_y"
TIME_DIRECTION_LINE_VECTOR_X_TAG = "time_direction_line_vector_x"
TIME_DIRECTION_LINE_VECTOR_Y_TAG = "time_direction_line_vector_y"
LOWRES_DEM_SPLINES_FIT_TAG = "lowres_dem_splines_fit"
CORRECTED_LOWRES_DSM_TAG = "corrected_lowres_dsm"
CORRECTED_LOWRES_ELEVATION_DIFFERENCE_TAG = (
    "corrected_lowres_elevation_difference"
)

# Tags for preprocessing parameters
EPI_STEP_TAG = "epi_step"
DISPARITY_MARGIN_TAG = "disparity_margin"
ELEVATION_DELTA_LOWER_BOUND_TAG = "elevation_delta_lower_bound"
ELEVATION_DELTA_UPPER_BOUND_TAG = "elevation_delta_upper_bound"
EPIPOLAR_ERROR_UPPER_BOUND_TAG = "epipolar_error_upper_bound"
EPIPOLAR_ERROR_MAXIMUM_BIAS_TAG = "epipolar_error_maximum_bias"
PREPARE_MASK_CLASSES_USAGE_TAG = "mask_classes_usage_in_prepare"
MASK1_IGNORED_BY_SIFT_MATCHING_TAG = "%s_%s" % (
    input_parameters.MASK1_TAG,
    mask_classes.ignored_by_sift_matching_tag,
)
MASK2_IGNORED_BY_SIFT_MATCHING_TAG = "%s_%s" % (
    input_parameters.MASK2_TAG,
    mask_classes.ignored_by_sift_matching_tag,
)

# Tags for content.json of preprocessing step
PREPROCESSING_SECTION_TAG = "preprocessing"
PREPROCESSING_OUTPUT_SECTION_TAG = "output"
PREPROCESSING_PARAMETERS_SECTION_TAG = "parameters"
PREPROCESSING_VERSION_TAG = "version"

# tags for dask configuration file
# TODO : Move to a class parameter as not changeable by user.
# TODO: in dask orchestration ?
PREPROCESSING_DASK_CONFIG_TAG = "dask_config_prepare"

# Schema of preprocessing/output section
PREPROCESSING_OUTPUT_SCHEMA = {
    MINIMUM_DISPARITY_TAG: float,
    MAXIMUM_DISPARITY_TAG: float,
    LEFT_EPIPOLAR_GRID_TAG: And(str, rasterio_can_open),
    RIGHT_EPIPOLAR_GRID_TAG: And(str, rasterio_can_open),
    RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG: And(str, rasterio_can_open),
    LEFT_ENVELOPE_TAG: And(str, os.path.isfile),
    RIGHT_ENVELOPE_TAG: And(str, os.path.isfile),
    EPIPOLAR_SIZE_X_TAG: And(int, lambda x: x > 0),
    EPIPOLAR_SIZE_Y_TAG: And(int, lambda x: x > 0),
    EPIPOLAR_ORIGIN_X_TAG: float,
    EPIPOLAR_ORIGIN_Y_TAG: float,
    EPIPOLAR_SPACING_X_TAG: float,
    EPIPOLAR_SPACING_Y_TAG: float,
    DISP_TO_ALT_RATIO_TAG: float,
    LEFT_AZIMUTH_ANGLE_TAG: float,
    LEFT_ELEVATION_ANGLE_TAG: float,
    RIGHT_AZIMUTH_ANGLE_TAG: float,
    RIGHT_ELEVATION_ANGLE_TAG: float,
    CONVERGENCE_ANGLE_TAG: float,
    OptionalKey(TIME_DIRECTION_LINE_ORIGIN_X_TAG): float,
    OptionalKey(TIME_DIRECTION_LINE_ORIGIN_Y_TAG): float,
    OptionalKey(TIME_DIRECTION_LINE_VECTOR_X_TAG): float,
    OptionalKey(TIME_DIRECTION_LINE_VECTOR_Y_TAG): float,
    LOWRES_DSM_TAG: And(str, os.path.isfile),
    LOWRES_INITIAL_DEM_TAG: And(str, os.path.isfile),
    LOWRES_ELEVATION_DIFFERENCE_TAG: And(str, os.path.isfile),
    OptionalKey(LOWRES_DEM_SPLINES_FIT_TAG): And(str, os.path.isfile),
    OptionalKey(CORRECTED_LOWRES_DSM_TAG): And(str, os.path.isfile),
    OptionalKey(CORRECTED_LOWRES_ELEVATION_DIFFERENCE_TAG): And(
        str, os.path.isfile
    ),
    OptionalKey(MATCHES_TAG): And(str, os.path.isfile),
    OptionalKey(RAW_MATCHES_TAG): And(str, os.path.isfile),
    OptionalKey(ENVELOPES_INTERSECTION_TAG): str,
    OptionalKey(ENVELOPES_INTERSECTION_BB_TAG): list,
}

# Type of preprocessing/output section
PreprocessingOutputType = Dict[str, Union[float, str, int]]

# schema of the preprocessing/parameters section
PREPROCESSING_PARAMETERS_SCHEMA = {
    EPI_STEP_TAG: And(int, lambda x: x > 0),
    DISPARITY_MARGIN_TAG: And(float, lambda x: 0.0 <= x <= 1.0),
    EPIPOLAR_ERROR_UPPER_BOUND_TAG: And(float, lambda x: x > 0),
    EPIPOLAR_ERROR_MAXIMUM_BIAS_TAG: And(float, lambda x: x >= 0),
    ELEVATION_DELTA_LOWER_BOUND_TAG: float,
    ELEVATION_DELTA_UPPER_BOUND_TAG: float,
    OptionalKey(PREPARE_MASK_CLASSES_USAGE_TAG): {
        MASK1_IGNORED_BY_SIFT_MATCHING_TAG: Or([int], None),
        OptionalKey(MASK2_IGNORED_BY_SIFT_MATCHING_TAG): Or([int], None),
    },
}

# Type of the preprocessing/parameters section
PreprocessingParametersType = Dict[str, Union[float, int]]

# Schema of the full content.json for preprocessing output
IN_CONF_SCHEMA = input_parameters.INPUT_CONFIGURATION_SCHEMA  # local variable
PREPROCESSING_CONTENT_SCHEMA = {
    input_parameters.INPUT_SECTION_TAG: IN_CONF_SCHEMA,
    PREPROCESSING_SECTION_TAG: {
        PREPROCESSING_VERSION_TAG: str,
        PREPROCESSING_PARAMETERS_SECTION_TAG: PREPROCESSING_PARAMETERS_SCHEMA,
        input_parameters.STATIC_PARAMS_TAG: {
            static_conf.prepare_tag: static_conf.prepare_params_schema,
            static_conf.plugins_tag: static_conf.plugins_schema,
        },
        PREPROCESSING_OUTPUT_SECTION_TAG: PREPROCESSING_OUTPUT_SCHEMA,
    },
}


# Type of the full content.json for preprocessing output
PreprocessingContentType = Dict[
    str, Union[str, PreprocessingParametersType, PreprocessingOutputType]
]
