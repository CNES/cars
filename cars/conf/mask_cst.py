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
CARS mask module
"""

# Standard imports


# CARS imports

# Specific values
# 0 = valid pixels
# 255 = value used as no data during the rectification in the epipolar geometry
VALID_VALUE = 0
NO_DATA_IN_EPIPOLAR_RECTIFICATION = 255
PROTECTED_VALUES = [NO_DATA_IN_EPIPOLAR_RECTIFICATION]
