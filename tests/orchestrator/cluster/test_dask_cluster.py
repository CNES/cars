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
Test module for cars/orchestrator/cluster/abstract_dask_cluster.py
"""

# Standard imports
import os
import tempfile

# Third party imports
import pytest
import yaml

# CARS imports
from cars.orchestrator.cluster import abstract_dask_cluster

# CARS Tests imports
from ...helpers import temporary_dir


@pytest.mark.unit_tests
def test_write_yaml_config():
    """
    Test save used dask config
    """
    file_root_name = "test"
    cfg_yaml = {"key1": 2, "key2": {"key3": "string1", "key4": [1, 2, 4]}}
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        abstract_dask_cluster.write_yaml_config(
            cfg_yaml, directory, file_root_name
        )

        # test file existence and content
        file_path = os.path.join(directory, file_root_name + ".yaml")

        assert os.path.exists(file_path)

        with open(file_path, encoding="utf-8") as file:
            cfg_yaml_from_file = yaml.load(file, Loader=yaml.FullLoader)

            assert cfg_yaml == cfg_yaml_from_file
