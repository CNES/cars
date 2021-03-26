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
Utils testing generic module:
contains global shared generic functions for tests/*.py
"""

import os
import rasterio as rio
import numpy as np

def cars_path():
    """
    Return root of cars source directory
    One level down from tests
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def absolute_data_path(data_path):
    """
    Return a full absolute path to test data
    environment variable.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_folder, data_path)


def temporary_dir():
    """
    Returns path to temporary dir from CARS_TEST_TEMPORARY_DIR environment
    variable. Defaults to /tmp
    """
    if "CARS_TEST_TEMPORARY_DIR" not in os.environ:
        # return default tmp dir
        return "/tmp"
    # return env defined tmp dir
    return os.environ["CARS_TEST_TEMPORARY_DIR"]


def assert_same_images(actual, expected, rtol=0, atol=0):
    """
    Compare two image files with assertion:
    * same height, width, transform, crs
    * assert_allclose() on numpy buffers
    """
    with rio.open(actual) as rio_actual:
        with rio.open(expected) as rio_expected:
            np.testing.assert_equal(rio_actual.width, rio_expected.width)
            np.testing.assert_equal(rio_actual.height, rio_expected.height)
            assert rio_actual.transform == rio_expected.transform
            assert rio_actual.crs == rio_expected.crs
            assert rio_actual.nodata == rio_expected.nodata
            np.testing.assert_allclose(
                rio_actual.read(), rio_expected.read(), rtol=rtol, atol=atol)


def assert_same_datasets(actual, expected, rtol=0, atol=0):
    """
    Compare two datasets:
    """
    assert list(actual.attrs.keys()).sort() == list(
        expected.attrs.keys()).sort()
    for key in expected.attrs.keys():
        if isinstance(expected.attrs[key], np.ndarray):
            np.testing.assert_allclose(actual.attrs[key],
                                       expected.attrs[key])
        else:
            assert actual.attrs[key] == expected.attrs[key]
    assert actual.dims == expected.dims
    assert list(actual.coords.keys()).sort() == list(
        expected.coords.keys()).sort()
    for key in expected.coords.keys():
        np.testing.assert_allclose(actual.coords[key].values,
                                   expected.coords[key].values)
    assert list(actual.data_vars.keys()).sort() == list(
        expected.data_vars.keys()).sort()
    for key in expected.data_vars.keys():
        np.testing.assert_allclose(actual[key].values,
                                   expected[key].values,
                                   rtol=rtol,
                                   atol=atol)
