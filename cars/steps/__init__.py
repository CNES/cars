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
CARS steps module init file
"""

from collections import namedtuple

from pkg_resources import iter_entry_points

# CARS imports
# TODO: remove the following import if the core/geometry/otb_geometry.py
# file is removed from CARS
from cars.core.geometry.otb_geometry import OTBGeometry  # noqa

for entry_point in iter_entry_points(group="geometryLoader"):
    entry_point.load()


# TODO: find a new emplacement for the points cloud filterings parameters
# structures (see issue https://gitlab.cnes.fr/cars/cars/-/issues/323)
# ##### Parameters structures ######

# Cloud small components filtering parameters :
# ---------------------------------------------
#   * on_ground_margin:
#           margin added to the on ground region to not filter points clusters
#           that were incomplete because they were on the edges
#
#   * pts_connection_dist:
#           distance to use to consider that two points are connected
#
#   * nb_pts_threshold:
#           points clusters that have less than this number of points
#           will be filtered
#
#   * dist_between_clusters:
#           distance to use to consider that two points clusters
#           are far from each other or not.
#       If a small points cluster is near to another one, it won't be filtered.
#          (None = deactivated)
#
#   * construct_removed_elt_msk:
#           if True, the removed points mask will be added to the cloud datasets
#           in input of the simple_rasterization_dataset
#
#   * mask_value:
#           value to use to identify the removed points in the mask
#
SmallComponentsFilterParams = namedtuple(
    "SmallComponentsFilterParams",
    [
        "on_ground_margin",
        "connection_val",
        "nb_pts_threshold",
        "clusters_distance_threshold",
        "filtered_elt_msk",
        "msk_value",
    ],
)

# Cloud statistical filtering :
# ------------------------------
#   * k:
#       number of neighbors
#
#   * stdev_factor:
#       factor to apply in the distance threshold computation
#
#   * construct_removed_elt_msk:
#       if True, the removed points mask will be added to the cloud datasets
#       in input of the simple_rasterization_dataset
#
#   * mask_value:
#       value to use to identify the removed points in the mask
#
StatisticalFilterParams = namedtuple(
    "StatisticalFilterParams",
    ["k", "std_dev_factor", "filtered_elt_msk", "msk_value"],
)
