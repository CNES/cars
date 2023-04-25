#!/usr/bin/env python
# coding: utf8
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
Test module for notebook helpers
"""

# Standard imports
from __future__ import absolute_import

import importlib.util
import os
import sys

# Third party imports
import pytest

from .helpers import cars_path

# Import notebook_helpers
spec_notebook_helpers = importlib.util.spec_from_file_location(
    "notebook_helpers",
    os.path.join(cars_path(), "tutorials/notebook_helpers.py"),
)
notebook_helpers = importlib.util.module_from_spec(spec_notebook_helpers)
sys.modules["notebook_helpers"] = notebook_helpers
spec_notebook_helpers.loader.exec_module(notebook_helpers)


@pytest.mark.notebook_tests
def test_set_dask_config():
    """
    Test set_dask_config
    """
    notebook_helpers.set_dask_config()
