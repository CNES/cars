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
Test module for cars/data_structure/cars_dataset.py
"""

# Standard imports
from __future__ import absolute_import

import os
import tempfile

import numpy as np

# Third party imports
import pytest

# CARS imports
from cars.data_structures import cars_dataset

# CARS Tests import
from tests.helpers import temporary_dir


@pytest.mark.unit_tests
def test_save_load_numpy_array():
    """
    Test save_numpy_array and load_numpy_array
    """

    array = np.ones((5, 5))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        arr_path = os.path.join(directory, "array")

        # save numpy array
        cars_dataset.save_numpy_array(array, arr_path)

        # load numpy array
        new_array = cars_dataset.load_numpy_array(arr_path)

        # assert same array
        np.testing.assert_allclose(array, new_array)


@pytest.mark.unit_tests
def test_save_load_dict():
    """
    Test save_dict and load_dict
    """

    to_save_dict = {"A": "B", "C": 2}

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        dict_path = os.path.join(directory, "dict")

        # save numpy array
        cars_dataset.save_dict(to_save_dict, dict_path)

        # load numpy array
        new_dict = cars_dataset.load_dict(dict_path)

        # assert same dict
        assert to_save_dict == new_dict


@pytest.mark.unit_tests
def test_create_none():
    """
    Test create_none
    """

    nb_row = 2
    nb_col = 3

    none_mat = cars_dataset.create_none(nb_row, nb_col)

    assert nb_row == len(none_mat)
    assert nb_col == len(none_mat[0])
    assert none_mat[0][0] is None


@pytest.mark.unit_tests
def test_overlap_array_to_dict():
    """
    Test overlap_array_to_dict
    """

    overlap = [1, 2, 3, 4]

    dict_overlap = {"up": 1, "down": 2, "left": 3, "right": 4}

    out_dict_overlap = cars_dataset.overlap_array_to_dict(overlap)

    assert dict_overlap == out_dict_overlap


@pytest.mark.unit_tests
def test_window_array_to_dict():
    """
    Test window_array_to_dict
    """

    window = [0, 100, 3, 56]

    # test with no overlap
    dict_window = {
        "row_min": 0,
        "row_max": 100,
        "col_min": 3,
        "col_max": 56,
    }

    out_dict_window = cars_dataset.window_array_to_dict(window)

    assert dict_window == out_dict_window

    # test with overlap

    overlap = [1, 2, 3, 4]
    dict_window = {
        "row_min": -1,
        "row_max": 102,
        "col_min": 0,
        "col_max": 60,
    }

    out_dict_window = cars_dataset.window_array_to_dict(window, overlap=overlap)

    assert dict_window == out_dict_window


@pytest.mark.unit_tests
def test_separate_dicts():
    """
    Test separate_dicts
    """

    full_dict = {"a": 1, "b": 2, "c": "F", "d": [1, 2, 3]}

    list_tag = ["b", "d"]

    expected_dict1 = {"a": 1, "c": "F"}

    expected_dict2 = {"b": 2, "d": [1, 2, 3]}

    out_dict1, out_dict2 = cars_dataset.separate_dicts(full_dict, list_tag)

    assert expected_dict1 == out_dict1
    assert expected_dict2 == out_dict2


@pytest.mark.unit_tests
def test_rio_profile_to_dict_profile():
    """
    Test rio_profile_to_dict_profile and rio_profile_to_dict_profile
    """

    dict_profile = {
        "driver": "GTiff",
        "interleave": "band",
        "nodata": 0,
        "dtype": "uint8",
        "count": 3,
        "crs": 3005,
        "transform": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }

    rio_profile = cars_dataset.dict_profile_to_rio_profile(dict_profile)

    new_dict_profile = cars_dataset.rio_profile_to_dict_profile(rio_profile)

    assert dict_profile == new_dict_profile
