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
this module contains tools for the dsm filling applications
"""
import numpy as np


def project(points, matrix):
    """
    Projects a np.array of (n, 2) points using any transformation
    matrix of  shape (3,3), taking into account homogeneous coordinates.
    """
    pts_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    tr_pts_homo = np.dot(matrix, pts_homo.T).T
    tr_points = tr_pts_homo[:, :2] / tr_pts_homo[:, 2][:, np.newaxis]

    return tr_points
