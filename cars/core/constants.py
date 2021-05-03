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
Cars Constants module
"""

# general
ROW = 'row' # cannot be changed because of PANDORA input format
COL = 'col' # cannot be changed because of PANDORA input format
BAND = 'band'
X = 'x'
Y = 'y'
Z = 'z'
RESOLUTION = 'resolution'
EPI_FULL_SIZE = 'full_epipolar_size'
ROI = 'roi'
ROI_WITH_MARGINS = 'roi_with_margins'
EPSG = 'epsg'

# stereo keys
STEREO_REF = 'ref'
STEREO_SEC = 'sec'

# epipolar image dataset
EPI_IMAGE = 'im' # has to be synchronized with the PANDORA input format
EPI_MSK = 'msk' # has to be synchronized with the PANDORA input format
EPI_MARGINS = 'margins'
EPI_DISP_MIN = 'disp_min'
EPI_DISP_MAX = 'disp_max'

# disparity dataset
DISP_MAP = 'disp'
DISP_MSK = 'disp_msk'
DISP_MSK_INVALID_REF = 'msk_invalid_ref'
DISP_MSK_INVALID_SEC = 'msk_invalid_sec'
DISP_MSK_MASKED_REF = 'msk_masked_ref'
DISP_MSK_MASKED_SEC = 'msk_masked_sec'
DISP_MSK_OCCLUSION = 'msk_occlusion'
DISP_MSK_FALSE_MATCH = 'msk_false_match'
DISP_MSK_INSIDE_SEC_ROI = 'msk_inside_sec_roi'
DISP_MSK_DISP_TO_0 = 'msk_disp_to_0'

# points cloud fields (xarray Dataset and pandas Dataframe)
POINTS_CLOUD_CORR_MSK = 'corr_msk'
POINTS_CLOUD_MSK = 'msk'
POINTS_CLOUD_VALID_DATA = 'data_valid'
POINTS_CLOUD_CLR_KEY_ROOT = 'clr'
POINTS_CLOUD_COORD_EPI_GEOM_I = 'coord_epi_geom_i'
POINTS_CLOUD_COORD_EPI_GEOM_J = 'coord_epi_geom_j'
POINTS_CLOUD_IDX_IM_EPI = 'idx_im_epi'

# raster fields (xarray Dataset)
RASTER_HGT = 'hgt'
RASTER_COLOR_IMG = 'img'
RASTER_MSK = 'raster_msk'
RASTER_NB_PTS = 'n_pts'
RASTER_NB_PTS_IN_CELL = 'pts_in_cell'
RASTER_HGT_MEAN = 'hgt_mean'
RASTER_HGT_STD_DEV = 'hgt_stdev'
RASTER_BAND_MEAN = 'band_mean'
RASTER_BAND_STD_DEV = 'band_stdev'
