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
this module contains the ram tools
"""

import psutil


def get_available_ram():
    """
    Get available ram

    :return : available ram in Mb
    """
    ram = psutil.virtual_memory()
    available_ram_mb = ram.available / (1024 * 1024)
    return available_ram_mb


def get_total_ram():
    """
    Get total ram

    :return : available ram in Mb
    """
    ram = psutil.virtual_memory()
    total_ram_mb = ram.available / (1024 * 1024)
    return total_ram_mb
