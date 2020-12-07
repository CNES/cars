#
# coding: utf8
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
=========================
  Module "configuration"
=========================
This module is the main CARS configuration python file.
It contains all the functions associated with CARS configuration.
"""
import os
import logging
from collections import namedtuple
import json
from json_checker import Or
from numpy import dtype

from cars import utils
from cars import filtering

# TODO : not use a global cfg variable ?
# TODO : with refactoring : constants in UPPER_CASE
#pylint: disable=invalid-name
cfg = None

#### Prepare ####

# sift tags and schema
sift_tag = 'sift'
sift_matching_threshold_tag = 'matching_threshold'
sift_n_octave_tag = 'n_octave'
sift_n_scale_per_octave_tag = 'n_scale_per_octave'
sift_dog_threshold_tag = 'dog_threshold'
sift_edge_threshold_tag = 'edge_threshold'
sift_magnification_tag = 'magnification'
sift_back_matching_tag = 'back_matching'
sift_parameters_schema = {
    sift_matching_threshold_tag: float,
    sift_n_octave_tag: int,
    sift_n_scale_per_octave_tag: int,
    sift_dog_threshold_tag: float,
    sift_edge_threshold_tag: float,
    sift_magnification_tag: float,
    sift_back_matching_tag: bool
}

# low res dsm tags and schema
low_res_dsm_tag = 'low_res_dsm'
low_res_dsm_resolution_in_degree_tag = 'low_res_dsm_resolution_in_degree'
low_res_dsm_min_sizex_for_align_tag = 'lowres_dsm_min_sizex'
low_res_dsm_min_sizey_for_align_tag = 'lowres_dsm_min_sizey'
low_res_dsm_ext_tag = 'low_res_dsm_ext'
low_res_dsm_order_tag = 'low_res_dsm_order'
low_res_dsm_parameters_schema = {
    low_res_dsm_resolution_in_degree_tag: float,
    low_res_dsm_min_sizex_for_align_tag: int,
    low_res_dsm_min_sizey_for_align_tag: int,
    low_res_dsm_ext_tag: int,
    low_res_dsm_order_tag: int
}

# disparity range estimation
disparity_range_tag = "disparity_range"
disparity_outliers_rejection_percent_tag =\
    "disparity_outliers_rejection_percent"

disparity_range_parameters_schema = {
    disparity_outliers_rejection_percent_tag : float
}

# prepare schema
prepare_params_schema = {
    sift_tag: sift_parameters_schema,
    low_res_dsm_tag: low_res_dsm_parameters_schema,
    disparity_range_tag : disparity_range_parameters_schema
}


#### Compute DSM ####

# tiling configuration tags and schema
tiling_conf_tag = 'tiling_configuration'
epi_tile_margin_tag = 'epipolar_tile_margin_in_percent'
tiling_conf_schema = {
    epi_tile_margin_tag: Or(None, int)
}

# rasterization tags and schema
rasterization_tag = 'rasterization'
grid_points_division_factor_tag = 'grid_points_division_factor'
rasterization_schema = {
    grid_points_division_factor_tag: Or(None, int)
}

# cloud filtering tags and schema
cloud_filtering_tag = 'cloud_filtering'
small_cpnts_filter_tag = 'small_components'
small_cpnts_on_ground_margin_tag = 'on_ground_margin'
small_cpnts_connection_dist_tag = 'connection_distance'
small_cpnts_nb_points_threshold_tag = 'nb_points_threshold'
small_cpnts_clusters_dist_threshold_tag = 'clusters_distance_threshold'
small_cpnts_removed_elt_mask_tag = 'removed_elt_mask'
small_cpnts_mask_value_tag = 'mask_value'
small_cpnts_schema = {
    small_cpnts_on_ground_margin_tag: int,
    small_cpnts_connection_dist_tag: float,
    small_cpnts_nb_points_threshold_tag: int,
    small_cpnts_clusters_dist_threshold_tag: None,
    small_cpnts_removed_elt_mask_tag: bool,
    small_cpnts_mask_value_tag: int
}
stat_outliers_filter_tag = 'statistical_outliers'
stat_outliers_k_tag = 'k'
stat_outliers_stdev_factor_tag = 'std_dev_factor'
stat_outliers_removed_elt_mask_tag = 'removed_elt_mask'
stat_outliers_mask_value_tag = 'mask_value'
stat_outliers_schema = {
    stat_outliers_k_tag: int,
    stat_outliers_stdev_factor_tag: float,
    stat_outliers_removed_elt_mask_tag: bool,
    stat_outliers_mask_value_tag: int
}
cloud_filtering_schema = {
    small_cpnts_filter_tag: Or(None, small_cpnts_schema),
    stat_outliers_filter_tag: Or(None,stat_outliers_schema)
}

# output tags and schema
output_tag = 'output'
color_image_encoding_tag = "color_image_encoding"
output_schema = {
    color_image_encoding_tag : str
}

# compute dsm params schema
compute_dsm_params_schema = {
    tiling_conf_tag: tiling_conf_schema,
    rasterization_tag: rasterization_schema,
    cloud_filtering_tag: cloud_filtering_schema,
    output_tag: output_schema
}


#### final static conf file ####
prepare_tag = 'prepare'
compute_dsm_tag = 'compute_dsm'
static_conf_schema = {
    prepare_tag: prepare_params_schema,
    compute_dsm_tag: compute_dsm_params_schema
}


#### namedTuple for parameters ####
SiftParams = namedtuple('SiftParams', sift_parameters_schema.keys())
LowResDSMParams = namedtuple(
    'LowResDSMParams', low_res_dsm_parameters_schema.keys())
RasterizationParams = namedtuple(
    'RasterizationParams', rasterization_schema.keys())


def load_cfg():
    """
    Load the static configuration file.
    It shall be automatically installed
    as $CARS_STATIC_CONFIGURATION/static_conf/static_configuration.json.
    """
    path = os.environ.get('CARS_STATIC_CONFIGURATION')

    if path is None:
        logger = logging.getLogger()
        logger.critical(
            'CARS_STATIC_CONFIGURATION environment variable is not set')

    # Read the static configuration file
    conf_file_path = os.path.join(path,
                    'static_conf', 'static_configuration.json')
    if not os.path.exists(conf_file_path):
        logger = logging.getLogger()
        logger.critical(
            'The file indicated {} does not exist'.format(conf_file_path))

    with open(conf_file_path, 'r') as conf_file:
        global cfg
        cfg = json.load(conf_file)
        utils.check_json(cfg, static_conf_schema)


def get_cfg():
    """
    Get the static configuration dictionary (global variable cfg).
    Initialize it if it has not been read yet.

    :return: the static configuration dictionary
    """
    if cfg is None:
        load_cfg()
    return cfg


def get_sift_params() -> SiftParams:
    """
    Construct the SiftParams namedtuple from the static configuration file

    :return: the sift parameters
    """
    if cfg is None:
        load_cfg()

    # get sift section
    sift_dict = cfg[prepare_tag][sift_tag]
    sift_params = SiftParams(*sift_dict.values())

    return sift_params


def get_low_res_dsm_params() -> LowResDSMParams:
    """
    Construct the LowResDSMParams namedtuple from the static configuration file

    :return: the low res dsm parameters
    """
    if cfg is None:
        load_cfg()

    # get low res dsm section
    low_res_dsm_dict = cfg[prepare_tag][low_res_dsm_tag]
    low_res_dsm_params = LowResDSMParams(*low_res_dsm_dict.values())

    return low_res_dsm_params


def get_disparity_outliers_rejection_percent() -> float:
    """
    :return: Disparity outliers rejection percent from static configuration file
    """
    if cfg is None:
        load_cfg()

    return cfg[prepare_tag][disparity_range_tag][
                disparity_outliers_rejection_percent_tag]


def get_epi_tile_margin_percent() -> int:
    """
    :return: The epipolar tile margin to use as a percent
        of the estimated tile size from static configuration file
    """
    if cfg is None:
        load_cfg()

    return cfg[compute_dsm_tag][tiling_conf_tag][epi_tile_margin_tag]


def get_rasterization_params() -> RasterizationParams:
    """
    Construct the RasterizationParams namedtuple
    from the static configuration file

    :return: the rasterization parameters
    """
    if cfg is None:
        load_cfg()

    # get rasterization section
    rasterization_dict = cfg[compute_dsm_tag][rasterization_tag]
    rasterization_params = RasterizationParams(*rasterization_dict.values())

    return rasterization_params


def get_small_components_filter_params()\
    -> filtering.SmallComponentsFilterParams:
    """
    Construct the filtering.SmallComponentParams namedtuple
    from the static configuration file

    :return: the small components filter parameters
    """
    if cfg is None:
        load_cfg()

    # get small components filter section
    small_cpn_filter_dict = cfg[compute_dsm_tag][cloud_filtering_tag][
                                small_cpnts_filter_tag]
    if small_cpn_filter_dict is None:
        return None

    small_cpn_filter_params = filtering.SmallComponentsFilterParams(
                                            *small_cpn_filter_dict.values())
    return small_cpn_filter_params


def get_statistical_outliers_filter_params()\
    -> filtering.StatisticalFilterParams:
    """
    Construct the filtering.StatisticalFilterParams namedtuple
    from the static configuration file

    :return: the statistical outliers filter parameters
    """
    if cfg is None:
        load_cfg()

    # get statistical filter section
    stat_filter_dict =\
        cfg[compute_dsm_tag][cloud_filtering_tag][stat_outliers_filter_tag]
    if stat_filter_dict is None:
        return None

    stat_filter_params = filtering.StatisticalFilterParams(
                                        *stat_filter_dict.values())
    return stat_filter_params

def get_color_image_encoding() -> dtype:
    """
    Get the encoding for the color image

    :returns: dtype for color image
    """
    if cfg is None:
        load_cfg()

    color_image_encoding = cfg[compute_dsm_tag][output_tag][
                                        color_image_encoding_tag]

    return dtype(color_image_encoding)
