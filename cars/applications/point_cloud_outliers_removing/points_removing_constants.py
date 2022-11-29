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
file contains all the constants used in cloud fusion module
"""


CLOUD_OUTLIER_REMOVING_RUN_TAG = "points_removing_constants_run"

# Params
METHOD = "method"


# small components
SMALL_COMPONENTS_FILTER = "small_components_filter_activated"
SC_ON_GROUND_MARGIN = "on_ground_margin"
SC_CONNECTION_DISTANCE = "connection_distance"
SC_NB_POINTS_THRESHOLD = "nb_points_threshold"
SC_CLUSTERS_DISTANCES_THRESHOLD = "clusters_distance_threshold"


# statistical outlier
STATISTICAL_OUTLIER = "statistical_outliers_filter_activated"
SO_K = "k"
SO_STD_DEV_FACTOR = "std_dev_factor"
