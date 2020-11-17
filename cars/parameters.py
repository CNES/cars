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
This module contains schema for all json files manipulated by cars,
string tags associated with fields as well as function to read, write
and check json parameters files.
"""

# Standard imports
import os
from typing import Dict, Union

# json checker imports
import json
from json_checker import OptionalKey, And, Or

# cars imports
from cars import configuration as static_cfg
from cars.utils import rasterio_can_open, ncdf_can_open, \
    make_relative_path_absolute


static_params_tag = 'static_parameters'

def read_input_parameters(filename):
    """
    Read an input parameters json file.
    Relative paths will be made absolute.

    :param filename: Path to json file
    :type filename: str

    :returns: The dictionary read from file
    :rtype: dict
    """
    config = {}
    with open(filename, 'r') as f:
        # Load json file
        config = json.load(f)
        json_dir = os.path.abspath(os.path.dirname(filename))
        # make potential relative paths absolute
        for tag in [
                img1_tag,
                img2_tag,
                mask1_tag,
                mask2_tag,
                color1_tag,
                srtm_dir_tag]:
            if tag in config:
                config[tag] = make_relative_path_absolute(
                    config[tag], json_dir)
    return config


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
    with open(filename, 'w') as f:
        # Make absolute path relative
        for tag in [
                left_epipolar_grid_tag,
                right_epipolar_grid_tag,
                left_envelope_tag,
                right_envelope_tag,
                matches_tag,
                raw_matches_tag,
                right_epipolar_uncorrected_grid_tag,
                left_envelope_tag,
                right_envelope_tag,
                envelopes_intersection_tag,
                lowres_dsm_tag,
                lowres_initial_dem_tag,
                lowres_elevation_difference_tag,
                lowres_dem_splines_fit_tag,
                corrected_lowres_dsm_tag,
                corrected_lowres_elevation_difference_tag]:
            if tag in config[preprocessing_section_tag][preprocessing_output_section_tag]:
                v = config[preprocessing_section_tag][preprocessing_output_section_tag][tag]
                config[preprocessing_section_tag][preprocessing_output_section_tag][tag] = os.path.basename(
                    v)
        json.dump(config, f, indent=indent)


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
    with open(filename, 'r') as f:
        config = json.load(f)
        json_dir = os.path.abspath(os.path.dirname(filename))
        # Make relative path absolute
        for tag in [
                img1_tag,
                img2_tag,
                mask1_tag,
                mask2_tag,
                color1_tag,
                srtm_dir_tag]:
            if tag in config[input_section_tag]:
                v = config[input_section_tag][tag]
                config[input_section_tag][tag] = make_relative_path_absolute(
                    v, json_dir)
        for tag in [
                left_epipolar_grid_tag,
                right_epipolar_grid_tag,
                left_envelope_tag,
                right_envelope_tag,
                matches_tag,
                raw_matches_tag,
                right_epipolar_uncorrected_grid_tag,
                left_envelope_tag,
                right_envelope_tag,
                envelopes_intersection_tag,
                lowres_dsm_tag,
                lowres_initial_dem_tag,
                lowres_elevation_difference_tag,
                lowres_dem_splines_fit_tag,
                corrected_lowres_dsm_tag,
                corrected_lowres_elevation_difference_tag
                ]:
            if tag in config[preprocessing_section_tag][preprocessing_output_section_tag]:
                v = config[preprocessing_section_tag][preprocessing_output_section_tag][tag]
                config[preprocessing_section_tag][preprocessing_output_section_tag][tag] = make_relative_path_absolute(
                    v, json_dir)
    return config


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
        for tag in [dsm_tag, color_tag, dsm_mean_tag, dsm_std_tag, dsm_n_pts_tag, dsm_points_in_cell_tag]:
            if tag in config[stereo_section_tag][stereo_output_section_tag]:
                v = config[stereo_section_tag][stereo_output_section_tag][tag]
                config[stereo_section_tag][stereo_output_section_tag][tag] = os.path.basename(
                    v)
        json.dump(config, f, indent=indent)


# tags for input parameters
img1_tag = "img1"
img2_tag = "img2"
srtm_dir_tag = "srtm_dir"
color1_tag = "color1"
mask1_tag = "mask1"
mask2_tag = "mask2"
nodata1_tag = "nodata1"
nodata2_tag = "nodata2"
default_alt_tag = "default_alt"

# Tags for preprocessing output section in content.json of preprocessing step
minimum_disparity_tag = "minimum_disparity"
maximum_disparity_tag = "maximum_disparity"
left_epipolar_grid_tag = "left_epipolar_grid"
right_epipolar_grid_tag = "right_epipolar_grid"
left_envelope_tag = "left_envelope"
right_envelope_tag = "right_envelope"
envelopes_intersection_tag = "envelopes_intersection"
epipolar_size_x_tag = "epipolar_size_x"
epipolar_size_y_tag = "epipolar_size_y"
epipolar_origin_x_tag = "epipolar_origin_x"
epipolar_origin_y_tag = "epipolar_origin_y"
epipolar_spacing_x_tag = "epipolar_spacing_x"
epipolar_spacing_y_tag = "epipolar_spacing_y"
matches_tag = "matches"
raw_matches_tag = "raw_matches"
lowres_dsm_tag = "lowres_dsm"
lowres_initial_dem_tag = "lowres_initial_dem"
lowres_elevation_difference_tag = "lowres_elevation_difference"
right_epipolar_uncorrected_grid_tag = "right_epipolar_uncorrected_grid"
disp_to_alt_ratio_tag = "disp_to_alt_ratio"
left_azimuth_angle_tag = "left_azimuth_angle"
left_elevation_angle_tag = "left_elevation_angle"
right_azimuth_angle_tag = "right_azimuth_angle"
right_elevation_angle_tag = "right_elevation_angle"
convergence_angle_tag = "convergence_angle"
time_direction_line_origin_x_tag = "time_direction_line_origin_x"
time_direction_line_origin_y_tag = "time_direction_line_origin_y"
time_direction_line_vector_x_tag = "time_direction_line_vector_x"
time_direction_line_vector_y_tag = "time_direction_line_vector_y"
lowres_dem_splines_fit_tag = "lowres_dem_splines_fit"
corrected_lowres_dsm_tag = "corrected_lowres_dsm"
corrected_lowres_elevation_difference_tag = "corrected_lowres_elevation_difference"

# Tags for preprocessing parameters
epi_step_tag = "epi_step"
disparity_margin_tag = "disparity_margin"
elevation_delta_lower_bound_tag = "elevation_delta_lower_bound"
elevation_delta_upper_bound_tag = "elevation_delta_upper_bound"
epipolar_error_upper_bound_tag = "epipolar_error_upper_bound"
epipolar_error_maximum_bias_tag = "epipolar_error_maximum_bias"

# Tags for content.json of preprocessing step
input_section_tag = "input"
preprocessing_section_tag = "preprocessing"
preprocessing_output_section_tag = "output"
preprocessing_parameters_section_tag = "parameters"
preprocessing_version_tag = "version"

# Tags for content.json stereo/parameters section of stereo step
resolution_tag = "resolution"
sigma_tag = "sigma"
dsm_radius_tag = "dsm_radius"

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
envelopes_intersection_bb_tag = "envelopes_intersection_bounding_box"

# tags from content.json of stereo step
stereo_inputs_section_tag = "input_configurations"
stereo_section_tag = "stereo"
stereo_output_section_tag = "output"
stereo_parameters_section_tag = "parameters"
stereo_version_tag = "version"


# Schema for input configuration json
input_configuration_schema = {
    img1_tag: And(str, rasterio_can_open),
    img2_tag: And(str, rasterio_can_open),
    OptionalKey(srtm_dir_tag): And(str, os.path.isdir),
    OptionalKey(color1_tag): And(str, rasterio_can_open),
    OptionalKey(mask1_tag): And(str, rasterio_can_open),
    OptionalKey(mask2_tag): And(str, rasterio_can_open),
    OptionalKey(default_alt_tag): float,
    nodata1_tag: int,
    nodata2_tag: int
}

# Type for input configuration json
input_configuration_type = Dict[str, Union[int, str]]

# Schema of preprocessing/output section
preprocessing_output_schema = {
    minimum_disparity_tag: float,
    maximum_disparity_tag: float,
    left_epipolar_grid_tag: And(str, rasterio_can_open),
    right_epipolar_grid_tag: And(str, rasterio_can_open),
    right_epipolar_uncorrected_grid_tag: And(str, rasterio_can_open),
    left_envelope_tag: And(str, os.path.isfile),
    right_envelope_tag: And(str, os.path.isfile),
    epipolar_size_x_tag: And(int, lambda x: x > 0),
    epipolar_size_y_tag: And(int, lambda x: x > 0),
    epipolar_origin_x_tag: float,
    epipolar_origin_y_tag: float,
    epipolar_spacing_x_tag: float,
    epipolar_spacing_y_tag: float,
    disp_to_alt_ratio_tag: float,
    left_azimuth_angle_tag: float,
    left_elevation_angle_tag: float,
    right_azimuth_angle_tag: float,
    right_elevation_angle_tag: float,
    convergence_angle_tag: float,
    OptionalKey(time_direction_line_origin_x_tag): float,
    OptionalKey(time_direction_line_origin_y_tag): float,
    OptionalKey(time_direction_line_vector_x_tag): float,
    OptionalKey(time_direction_line_vector_y_tag): float,
    lowres_dsm_tag: And(str, os.path.isfile),
    lowres_initial_dem_tag: And(str, os.path.isfile),
    lowres_elevation_difference_tag: And(str, os.path.isfile),
    OptionalKey(lowres_dem_splines_fit_tag): And(str, os.path.isfile),
    OptionalKey(corrected_lowres_dsm_tag): And(str, os.path.isfile),
    OptionalKey(corrected_lowres_elevation_difference_tag): And(str, os.path.isfile),
    OptionalKey(matches_tag): And(str, os.path.isfile),
    OptionalKey(raw_matches_tag): And(str, os.path.isfile),
    OptionalKey(envelopes_intersection_tag): str,
    OptionalKey(envelopes_intersection_bb_tag): list
}

# Type of preprocessing/output section
preprocessing_output_type = Dict[str, Union[float, str, int]]

# schema of the preprocessing/parameters section
preprocessing_parameters_schema = {
    epi_step_tag: And(int, lambda x: x > 0),
    disparity_margin_tag: And(float, lambda x: x >= 0. and x <= 1.),
    epipolar_error_upper_bound_tag: And(float, lambda x: x > 0),
    epipolar_error_maximum_bias_tag: And(float, lambda x: x >= 0),
    elevation_delta_lower_bound_tag: float,
    elevation_delta_upper_bound_tag: float
}

# Type of the preprocessing/parameters section
preprocessing_parameters_type = Dict[str, Union[float, int]]

# Schema of the full content.json for preprocessing output
preprocessing_content_schema = {
    input_section_tag: input_configuration_schema,
    preprocessing_section_tag:
    {
        preprocessing_version_tag: str,
        preprocessing_parameters_section_tag: preprocessing_parameters_schema,
        static_params_tag: static_cfg.prepare_params_schema,
        preprocessing_output_section_tag: preprocessing_output_schema
    }
}

# Type of the full content.json for preprocessing output
preprocessing_content_type = Dict[str, Union[str,
                                             preprocessing_parameters_type,
                                             preprocessing_output_type]]

# Schema of the output section of stereo content.json
stereo_output_schema = {
    dsm_tag: And(str, Or(rasterio_can_open, ncdf_can_open)),
    OptionalKey(color_tag):
        And(str, rasterio_can_open),
    dsm_no_data_tag: float,
    OptionalKey(color_no_data_tag): float,
    OptionalKey(dsm_mean_tag) : And(str, rasterio_can_open),
    OptionalKey(dsm_std_tag) : And(str, rasterio_can_open),
    OptionalKey(dsm_n_pts_tag) : And(str, rasterio_can_open),
    OptionalKey(dsm_points_in_cell_tag) : And(str, rasterio_can_open),
    epsg_tag: int,
    alt_reference_tag: str
}

# schema of the parameters section
stereo_parameters_schema = {
    resolution_tag: And(float, lambda x: x > 0),
    OptionalKey(epsg_tag): And(int, lambda x: x > 0),
    sigma_tag: Or(None, And(float, lambda x: x >= 0)),
    dsm_radius_tag: And(int, lambda x: x >= 0)
}

# Schema of the full json for stereo output
stereo_content_schema = {
    stereo_inputs_section_tag:
    [
        preprocessing_content_schema
    ],
    stereo_section_tag:
    {
        stereo_version_tag: str,
        stereo_parameters_section_tag: stereo_parameters_schema,
        static_params_tag: static_cfg.compute_dsm_params_schema,
        stereo_output_section_tag: stereo_output_schema
    }
}
