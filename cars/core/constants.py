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
CARS Constants module
"""

# general
ROW = "row"  # cannot be changed because of PANDORA input format
COL = "col"  # cannot be changed because of PANDORA input format
BAND = "band"
BAND_IM = "band_im"
BAND_CLASSIF = "band_classif"
BAND_PERFORMANCE_MAP = "band_performance_map"
BAND_FILLING = "filling_type"
BAND_SOURCE_PC = "source_point_cloud"
RIO_TAG_PERFORMANCE_MAP_CLASSES = "rio_tag_performance_map_classes"
X = "x"
Y = "y"
Z = "z"
Z_INF = "intervals_z_inf"
Z_SUP = "intervals_z_sup"
RESOLUTION = "resolution"
EPI_FULL_SIZE = "full_epipolar_size"
ROI = "roi"
ROI_WITH_MARGINS = "roi_with_margins"
EPSG = "epsg"
DISPARITY = "disparity"
PC_EPSG = "point_cloud_epsg"

BAND_NAMES = "band_names"
NBITS = "nbits"
EPSG_WSG84 = 4326
# stereo keys
STEREO_REF = "ref"
STEREO_SEC = "sec"

# epipolar image dataset
EPI_IMAGE = "im"  # has to be synchronized with the PANDORA input format
EPI_MSK = "msk"  # has to be synchronized with the PANDORA input format
EPI_COLOR = "color"
EPI_CLASSIFICATION = "classif"
EPI_FILLING = "filling"
EPI_CONFIDENCE_KEY_ROOT = "confidence"
EPI_PERFORMANCE_MAP = "performance_map"
EPI_DENOISING_INFO_KEY_ROOT = "denoising"
EPI_MARGINS = "margins"
EPI_DISP_MIN = "disp_min"
EPI_DISP_MAX = "disp_max"
EPI_VALID_PIXELS = "valid_pixels"
EPI_NO_DATA_MASK = "no_data_mask"
EPI_NO_DATA_IMG = "no_data_img"
EPI_TRANSFORM = "transform"
EPI_CRS = "crs"
EPI_GROUND_TRUTH = "epi_ground_truth"
SENSOR_GROUND_TRUTH = "sensor_ground_truth"

# points cloud fields (xarray Dataset and pandas Dataframe)
POINT_CLOUD_CORR_MSK = "corr_msk"
POINT_CLOUD_MSK = "mask"
POINT_CLOUD_CLR_KEY_ROOT = "color"
POINT_CLOUD_PERFORMANCE_MAP = "performance_map"
POINT_CLOUD_CONFIDENCE_KEY_ROOT = "confidence"
POINT_CLOUD_INTERVALS_KEY_ROOT = "intervals"
POINT_CLOUD_CLASSIF_KEY_ROOT = "classif"
POINT_CLOUD_FILLING_KEY_ROOT = "filling"
POINT_CLOUD_SOURCE_KEY_ROOT = "source_pc"
POINT_CLOUD_COORD_EPI_GEOM_I = "coord_epi_geom_i"
POINT_CLOUD_COORD_EPI_GEOM_J = "coord_epi_geom_j"
POINT_CLOUD_ID_IM_EPI = "id_im_epi"
POINT_CLOUD_GLOBAL_ID = "global_id"
POINT_CLOUD_MATCHES = "point_cloud_matches"

# raster fields (xarray Dataset)
RASTER_HGT = "hgt"
RASTER_HGT_INF = "hgt_inf"
RASTER_HGT_SUP = "hgt_sup"
RASTER_WEIGHTS_SUM = "weights_sum"
RASTER_COLOR_IMG = "img"
RASTER_MSK = "raster_msk"
RASTER_CLASSIF = "raster_classif"
RASTER_NB_PTS = "n_pts"
RASTER_NB_PTS_IN_CELL = "pts_in_cell"
RASTER_HGT_MEAN = "hgt_mean"
RASTER_HGT_STD_DEV = "hgt_stdev"
RASTER_HGT_INF_MEAN = "hgt_inf_mean"
RASTER_HGT_INF_STD_DEV = "hgt_inf_stdev"
RASTER_HGT_SUP_MEAN = "hgt_sup_mean"
RASTER_HGT_SUP_STD_DEV = "hgt_sup_stdev"
RASTER_BAND_MEAN = "band_mean"
RASTER_BAND_STD_DEV = "band_stdev"
RASTER_CONFIDENCE = "confidence"
RASTER_PERFORMANCE_MAP = "performance_map"
RASTER_PERFORMANCE_MAP_RAW = "performance_map_raw"
RASTER_SOURCE_PC = "source_pc"
RASTER_FILLING = "filling"
# Geometry constants
DISP_MODE = "disp"
MATCHES_MODE = "matches"

# DSM index
INDEX_DSM_ALT = "dsm"
INDEX_DSM_COLOR = "color"
INDEX_DSM_MASK = "mask"
INDEX_DSM_CLASSIFICATION = "classification"
INDEX_DSM_PERFORMANCE_MAP = "performance_map"
INDEX_DSM_CONTRIBUTING_PAIR = "contributing_pair"
INDEX_DSM_FILLING = "filling"

# depth map index
INDEX_DEPTH_MAP_X = "x"
INDEX_DEPTH_MAP_Y = "y"
INDEX_DEPTH_MAP_Z = "z"
INDEX_DEPTH_MAP_COLOR = "color"
INDEX_DEPTH_MAP_MASK = "mask"
INDEX_DEPTH_MAP_CLASSIFICATION = "classification"
INDEX_DEPTH_MAP_PERFORMANCE_MAP = "performance_map"
INDEX_DEPTH_MAP_FILLING = "filling"
INDEX_DEPTH_MAP_EPSG = "epsg"

# dsms inputs index
DSM_CLASSIF = "classification"
DSM_ALT = "dsm"
DSM_ALT_INF = "dsm_inf"
DSM_ALT_SUP = "dsm_sup"
DSM_WEIGHTS_SUM = "weights"
DSM_MSK = "mask"
DSM_NB_PTS = "dsm_n_pts"
DSM_NB_PTS_IN_CELL = "dsm_pts_in_cell"
DSM_MEAN = "dsm_mean"
DSM_STD_DEV = "dsm_std"
DSM_INF_MEAN = "dsm_inf_mean"
DSM_INF_STD = "dsm_inf_std"
DSM_SUP_MEAN = "dsm_sup_mean"
DSM_SUP_STD = "dsm_sup_std"
DSM_CONFIDENCE_AMBIGUITY = "confidence_from_ambiguity"
DSM_CONFIDENCE_RISK_MIN = "confidence_from_risk_min"
DSM_CONFIDENCE_RISK_MAX = "confidence_from_risk_max"
DSM_PERFORMANCE_MAP = "performance_map"
DSM_SOURCE_PC = "source_pc"
DSM_FILLING = "filling"
DSM_COLOR = "color"
