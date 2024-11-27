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
CARS application module init file
"""

# Imports needed in order to register application for Application factory
from . import dem_generation  # noqa: F401
from . import dense_match_filling  # noqa: F401
from . import dense_matching  # noqa: F401
from . import dsm_filling  # noqa: F401
from . import grid_generation  # noqa: F401
from . import hole_detection  # noqa: F401
from . import point_cloud_denoising  # noqa: F401
from . import point_cloud_fusion  # noqa: F401
from . import point_cloud_outlier_removal  # noqa: F401
from . import rasterization  # noqa: F401
from . import resampling  # noqa: F401
from . import sparse_matching  # noqa: F401
from . import triangulation  # noqa: F401
