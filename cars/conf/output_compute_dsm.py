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
This module refers to the compute_dsm outputs
"""

# Standard imports
import json
import os

# Third party imports
from json_checker import And, OptionalKey, Or

# CARS imports
from cars.conf import (
    input_parameters,
    mask_classes,
    output_prepare,
    static_conf,
)
from cars.core.inputs import ncdf_can_open, rasterio_can_open

# TODO : If we are going to create multiple pipelines to stop and
# different points (for validation or else) then are we going to
# create more files like this one ot the output_prepare one that
# will be using as output tags for a pipeline but also input tags
# for a step ? Maybe we could move all the tags to the actual steps
# that managing them ? Like moving the matching tags to the matching step ?


def write_stereo_content_file(config, filename, indent=2):
    """
    Write a stereo json content file.
    Relative paths in stereo/output section will be made absolute.

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
            DSM_TAG,
            COLOR_TAG,
            DSM_MEAN_TAG,
            DSM_STD_TAG,
            DSM_N_PTS_TAG,
            DSM_POINTS_IN_CELL_TAG,
        ]:
            if (
                tag
                in config[COMPUTE_DSM_SECTION_TAG][
                    COMPUTE_DSM_OUTPUT_SECTION_TAG
                ]
            ):

                value = config[COMPUTE_DSM_SECTION_TAG][
                    COMPUTE_DSM_OUTPUT_SECTION_TAG
                ][tag]

                config[COMPUTE_DSM_SECTION_TAG][COMPUTE_DSM_OUTPUT_SECTION_TAG][
                    tag
                ] = os.path.basename(value)
        json.dump(config, fstream, indent=indent)


# Tags for content.json stereo/parameters section of stereo step
RESOLUTION_TAG = "resolution"
SIGMA_TAG = "sigma"
DSM_RADIUS_TAG = "dsm_radius"
COMPUTE_DSM_MASK_CLASSES_USAGE_TAG = "mask_classes_usage_in_compute_dsm"
MASK1_IGNORED_BY_CORR_TAG = "%s_%s" % (
    input_parameters.MASK1_TAG,
    mask_classes.ignored_by_corr_tag,
)
MASK2_IGNORED_BY_CORR_TAG = "%s_%s" % (
    input_parameters.MASK2_TAG,
    mask_classes.ignored_by_corr_tag,
)
MASK1_SET_TO_REF_ALT_TAG = "%s_%s" % (
    input_parameters.MASK1_TAG,
    mask_classes.set_to_ref_alt_tag,
)
MASK2_SET_TO_REF_ALT_TAG = "%s_%s" % (
    input_parameters.MASK2_TAG,
    mask_classes.set_to_ref_alt_tag,
)

# Tags for content.json stereo/output section of stereo step
DSM_TAG = "dsm"
COLOR_TAG = "color"
MSK_TAG = "msk"
DSM_MEAN_TAG = "dsm_mean"
DSM_STD_TAG = "dsm_std"
DSM_N_PTS_TAG = "dsm_n_pts"
DSM_POINTS_IN_CELL_TAG = "dsm_points_in_cell"
DSM_NO_DATA_TAG = "dsm_no_data"
COLOR_NO_DATA_TAG = "color_no_data"
EPSG_TAG = "epsg"
ALT_REFERENCE_TAG = "altimetric_reference"
ALIGN_OPTION = "align_option"
SNAP_TO_IMG1_OPTION = "snap_to_img1"

# tags from content.json of compute dsm pipeline
COMPUTE_DSM_INPUTS_SECTION_TAG = "input_configurations"
COMPUTE_DSM_INPUT_TAG = "input_configuration"
COMPUTE_DSM_SECTION_TAG = "stereo"
COMPUTE_DSM_OUTPUT_SECTION_TAG = "output"
COMPUTE_DSM_PARAMETERS_SECTION_TAG = "parameters"
COMPUTE_DSM_VERSION_TAG = "version"

# tags for dask configuration file
COMPUTE_DSM_DASK_CONFIG_TAG = "dask_config_compute_dsm"

# Schema of the output section of compute dsm content.json
COMPUTE_DSM_OUTPUT_SCHEMA = {
    DSM_TAG: And(str, Or(rasterio_can_open, ncdf_can_open)),
    OptionalKey(COLOR_TAG): And(str, rasterio_can_open),
    OptionalKey(MSK_TAG): And(str, rasterio_can_open),
    DSM_NO_DATA_TAG: float,
    OptionalKey(COLOR_NO_DATA_TAG): float,
    OptionalKey(DSM_MEAN_TAG): And(str, rasterio_can_open),
    OptionalKey(DSM_STD_TAG): And(str, rasterio_can_open),
    OptionalKey(DSM_N_PTS_TAG): And(str, rasterio_can_open),
    OptionalKey(DSM_POINTS_IN_CELL_TAG): And(str, rasterio_can_open),
    EPSG_TAG: int,
    ALT_REFERENCE_TAG: str,
    OptionalKey(ALIGN_OPTION): bool,
    OptionalKey(SNAP_TO_IMG1_OPTION): bool,
    OptionalKey(output_prepare.ENVELOPES_INTERSECTION_BB_TAG): list,
}

# schema of the parameters section
COMPUTE_DSM_PARAMETERS_SCHEMA = {
    RESOLUTION_TAG: And(float, lambda x: x > 0),
    OptionalKey(EPSG_TAG): And(int, lambda x: x > 0),
    SIGMA_TAG: Or(None, And(float, lambda x: x >= 0)),
    DSM_RADIUS_TAG: And(int, lambda x: x >= 0),
}

COMPUTE_DSM_CLASSES_USAGE_SCHEMA = {
    MASK1_IGNORED_BY_CORR_TAG: Or([int], None),
    OptionalKey(MASK2_IGNORED_BY_CORR_TAG): Or([int], None),
    MASK1_SET_TO_REF_ALT_TAG: Or([int], None),
    OptionalKey(MASK2_SET_TO_REF_ALT_TAG): Or([int], None),
}

# Schema of the full json for compute dsm output
COMPUTE_DSM_CONTENT_SCHEMA = {
    COMPUTE_DSM_INPUTS_SECTION_TAG: [
        {
            COMPUTE_DSM_INPUT_TAG: output_prepare.PREPROCESSING_CONTENT_SCHEMA,
            OptionalKey(
                COMPUTE_DSM_MASK_CLASSES_USAGE_TAG
            ): COMPUTE_DSM_CLASSES_USAGE_SCHEMA,
        }
    ],
    COMPUTE_DSM_SECTION_TAG: {
        COMPUTE_DSM_VERSION_TAG: str,
        COMPUTE_DSM_PARAMETERS_SECTION_TAG: COMPUTE_DSM_PARAMETERS_SCHEMA,
        # fmt: off
        input_parameters.STATIC_PARAMS_TAG:{
            static_conf.compute_dsm_tag: static_conf.compute_dsm_params_schema,
            static_conf.plugins_tag: static_conf.plugins_schema,
        },
        # fmt: on
        COMPUTE_DSM_OUTPUT_SECTION_TAG: COMPUTE_DSM_OUTPUT_SCHEMA,
    },
}
