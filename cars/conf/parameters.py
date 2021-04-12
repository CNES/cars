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
Parameters module:
contains schema for all json cars files,
string tags associated with fields as well as function to read, write
and check json parameters files.
"""


# Standard imports
import os

# json checker imports
import json
from json_checker import OptionalKey, And, Or

# cars imports
from cars.conf import mask_classes, static_conf, \
    input_parameters, output_prepare
from cars.utils import rasterio_can_open, ncdf_can_open

# TODO : with refactoring : constants in UPPER_CASE
# TODO : not use a global parameters variable ?
#pylint: disable=invalid-name

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
    with open(filename, 'w') as f:
        # Make absolute path relative
        for tag in [
            dsm_tag,
            color_tag,
            dsm_mean_tag,
            dsm_std_tag,
            dsm_n_pts_tag,
            dsm_points_in_cell_tag]:
            if tag in config[stereo_section_tag][stereo_output_section_tag]:

                v = config[stereo_section_tag][stereo_output_section_tag][tag]

                config[stereo_section_tag][stereo_output_section_tag][tag] =\
                    os.path.basename(v)

        json.dump(config, f, indent=indent)


# Tags for content.json stereo/parameters section of stereo step
resolution_tag = "resolution"
sigma_tag = "sigma"
dsm_radius_tag = "dsm_radius"
stereo_mask_classes_usage_tag = "mask_classes_usage_in_compute_dsm"
mask1_ignored_by_corr_tag =\
    '%s_%s' % (input_parameters.MASK1_TAG, mask_classes.ignored_by_corr_tag)
mask2_ignored_by_corr_tag =\
    '%s_%s' % (input_parameters.MASK2_TAG, mask_classes.ignored_by_corr_tag)
mask1_set_to_ref_alt_tag =\
    '%s_%s' % (input_parameters.MASK1_TAG, mask_classes.set_to_ref_alt_tag)
mask2_set_to_ref_alt_tag =\
    '%s_%s' % (input_parameters.MASK2_TAG, mask_classes.set_to_ref_alt_tag)

# Tags for content.json stereo/output section of stereo step
dsm_tag = "dsm"
color_tag = "color"
msk_tag = "msk"
dsm_mean_tag = "dsm_mean"
dsm_std_tag = "dsm_std"
dsm_n_pts_tag = "dsm_n_pts"
dsm_points_in_cell_tag = "dsm_points_in_cell"
dsm_no_data_tag = "dsm_no_data"
color_no_data_tag = "color_no_data"
epsg_tag = "epsg"
alt_reference_tag = 'altimetric_reference'

# tags from content.json of stereo step
stereo_inputs_section_tag = "input_configurations"
stereo_input_tag = "input_configuration"
stereo_section_tag = "stereo"
stereo_output_section_tag = "output"
stereo_parameters_section_tag = "parameters"
stereo_version_tag = "version"

# tags for dask configuration file
compute_dsm_dask_config_tag = "dask_config_compute_dsm"

# Schema of the output section of stereo content.json
stereo_output_schema = {
    dsm_tag: And(str, Or(rasterio_can_open, ncdf_can_open)),
    OptionalKey(color_tag): And(str, rasterio_can_open),
    OptionalKey(msk_tag): And(str, rasterio_can_open),
    dsm_no_data_tag: float,
    OptionalKey(color_no_data_tag): float,
    OptionalKey(dsm_mean_tag): And(str, rasterio_can_open),
    OptionalKey(dsm_std_tag): And(str, rasterio_can_open),
    OptionalKey(dsm_n_pts_tag): And(str, rasterio_can_open),
    OptionalKey(dsm_points_in_cell_tag): And(str, rasterio_can_open),
    epsg_tag: int,
    alt_reference_tag: str,
    OptionalKey(output_prepare.ENVELOPES_INTERSECTION_BB_TAG): list
}

# schema of the parameters section
stereo_parameters_schema = {
    resolution_tag: And(float, lambda x: x > 0),
    OptionalKey(epsg_tag): And(int, lambda x: x > 0),
    sigma_tag: Or(None, And(float, lambda x: x >= 0)),
    dsm_radius_tag: And(int, lambda x: x >= 0)
}

stereo_classes_usage_schema = {
    mask1_ignored_by_corr_tag: Or([int], None),
    mask2_ignored_by_corr_tag: Or([int], None),
    mask1_set_to_ref_alt_tag: Or([int], None),
    mask2_set_to_ref_alt_tag: Or([int], None)
}

# Schema of the full json for stereo output
stereo_content_schema = {
    stereo_inputs_section_tag:
    [
        {
            #TODO ça me perturbe un poil de dépendre du output_prepare
            stereo_input_tag: output_prepare.PREPROCESSING_CONTENT_SCHEMA,
            OptionalKey(stereo_mask_classes_usage_tag):\
                                        stereo_classes_usage_schema
        }
    ],
    stereo_section_tag:
    {
        stereo_version_tag: str,
        stereo_parameters_section_tag: stereo_parameters_schema,
        input_parameters.STATIC_PARAMS_TAG:
         static_conf.compute_dsm_params_schema,
        stereo_output_section_tag: stereo_output_schema
    }
}
