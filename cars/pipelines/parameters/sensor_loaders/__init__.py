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

from .basic_classif_loader import BasicClassifSensorLoader  # noqa: F401
from .basic_image_loader import BasicImageSensorLoader  # noqa: F401
from .pivot_classif_loader import PivotClassifSensorLoader  # noqa: F401
from .pivot_image_loader import PivotImageSensorLoader  # noqa: F401
