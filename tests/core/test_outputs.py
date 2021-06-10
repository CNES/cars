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
Test module for cars/core/outputs.py
"""

# Standard imports
import os
import tempfile

# Third party imports
import fiona
import pytest
import xarray as xr
import yaml
from shapely.geometry import Polygon, shape

# CARS imports
from cars.core import outputs

# CARS Tests imports
from ..helpers import absolute_data_path, temporary_dir


@pytest.mark.unit_tests
def test_write_vector():
    """
    Test if write_vector function works with testing Polygons
    """
    polys = [
        Polygon([(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)]),
        Polygon([(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0)]),
    ]

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        path_to_file = os.path.join(directory, "test.gpkg")
        outputs.write_vector(polys, path_to_file, 4326)

        assert os.path.exists(path_to_file)

        nb_feat = 0
        for feat in fiona.open(path_to_file):
            poly = shape(feat["geometry"])
            nb_feat += 1
            assert poly in polys

        assert nb_feat == 2


@pytest.mark.unit_tests
def test_write_ply():
    """
    Test write ply file
    """
    points = xr.open_dataset(
        absolute_data_path("input/intermediate_results/points_ref.nc")
    )
    outputs.write_ply(os.path.join(temporary_dir(), "test.ply"), points)


@pytest.mark.unit_tests
def test_write_dask_config():
    """
    Test write used dask config
    TODO: move to cluster/test_dask_mode with refacto
    """
    file_root_name = "test"
    cfg_dask = {"key1": 2, "key2": {"key3": "string1", "key4": [1, 2, 4]}}
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        outputs.write_dask_config(cfg_dask, directory, file_root_name)

        # test file existence and content
        file_path = os.path.join(directory, file_root_name + ".yaml")

        assert os.path.exists(file_path)

        with open(file_path) as file:
            cfg_dask_from_file = yaml.load(file, Loader=yaml.FullLoader)

            assert cfg_dask == cfg_dask_from_file
