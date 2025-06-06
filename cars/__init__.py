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
Cars module init file
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cars")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

# Standard imports
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


__author__ = "CNES"
__email__ = "cars@cnes.fr"

# Force the use of CARS dask configuration
dask_config_path = os.path.join(
    os.path.dirname(__file__), "orchestrator", "cluster", "dask_config"
)
if not os.path.isdir(dask_config_path):
    raise NotADirectoryError("Wrong dask config path")
os.environ["DASK_CONFIG"] = str(dask_config_path)

# Force monothread for child workers
os.environ["PANDORA_NUMBA_PARALLEL"] = str(False)
os.environ["PANDORA_NUMBA_CACHE"] = str(False)
os.environ["SHARELOC_NUMBA_PARALLEL"] = str(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["GDAL_NUM_THREADS"] = "1"

# Limit GDAL cache per worker to 500MB
os.environ["GDAL_CACHEMAX"] = "500"


def import_plugins() -> None:
    """
    Load all the registered entry points
    :return: None
    """
    for entry_point in entry_points(group="cars.plugins"):
        entry_point.load()


import_plugins()
