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
Test module for cars/pipelines/parameters/dsm_inputs.py
"""

import copy

import pytest

from cars.pipelines.default.default_pipeline import overide_pipeline_conf


@pytest.mark.unit_tests
def test_overide_pipeline_conf_basic_merge():
    """Test basic merge of two dictionaries with key + id"""
    conf = {"key1": "value1", "key2": {"nested_key1": "nested_value1"}}
    overiding_conf = {
        "key3": "value3",
        "key2": {"nested_key2": "nested_value2"},
    }

    result = overide_pipeline_conf(conf, overiding_conf)

    expected = {
        "key1": "value1",
        "key2": {
            "nested_key1": "nested_value1",
            "nested_key2": "nested_value2",
        },
        "key3": "value3",
    }

    assert result == expected


@pytest.mark.unit_tests
def test_overide_pipeline_conf_preserve_original_keys():
    """Test that original keys are preserved"""
    conf = {
        "key_001": "original_value",
        "key_002": {"sub_key_001": "sub_value"},
    }
    overiding_conf = {"key_003": "new_value"}

    result = overide_pipeline_conf(conf, overiding_conf)

    assert "key_001" in result
    assert "key_002" in result
    assert result["key_001"] == "original_value"
    assert result["key_002"]["sub_key_001"] == "sub_value"
    assert result["key_003"] == "new_value"


@pytest.mark.unit_tests
def test_overide_pipeline_conf_override_existing_values():
    """
    Test overriding existing values with key + id pattern
    """
    conf = {
        "resolution_1": {"param_001": "old_value"},
        "resolution_2": {"param_002": "keep_value"},
    }
    overiding_conf = {
        "resolution_1": {"param_001": "new_value", "param_003": "added_value"}
    }

    result = overide_pipeline_conf(conf, overiding_conf)

    expected = {
        "resolution_1": {"param_001": "new_value", "param_003": "added_value"},
        "resolution_2": {"param_002": "keep_value"},
    }

    assert result == expected


@pytest.mark.unit_tests
def test_overide_pipeline_conf_deep_nested_merge():
    """
    Test deep nested dictionary merge with numbered keys
    """
    conf = {
        "app_001": {
            "config_001": {"setting_001": "value1", "setting_002": "value2"}
        }
    }
    overiding_conf = {
        "app_001": {
            "config_001": {
                "setting_002": "overridden_value2",
                "setting_003": "value3",
            },
            "config_002": {"new_setting": "new_value"},
        }
    }

    result = overide_pipeline_conf(conf, overiding_conf)

    expected = {
        "app_001": {
            "config_001": {
                "setting_001": "value1",
                "setting_002": "overridden_value2",
                "setting_003": "value3",
            },
            "config_002": {"new_setting": "new_value"},
        }
    }

    assert result == expected


@pytest.mark.unit_tests
def test_overide_pipeline_conf_empty_dictionaries():
    """
    Test with empty dictionaries
    """
    conf = {}
    overiding_conf = {"key_001": "value"}

    result = overide_pipeline_conf(conf, overiding_conf)
    assert result == {"key_001": "value"}

    conf = {"key_001": "value"}
    overiding_conf = {}

    result = overide_pipeline_conf(conf, overiding_conf)
    assert result == {"key_001": "value"}


@pytest.mark.unit_tests
def test_overide_pipeline_conf_immutability():
    """
    Test that original dictionaries are not modified
    """
    conf = {"key_001": {"nested": "original"}}
    overiding_conf = {"key_001": {"new_nested": "new"}}

    original_conf = copy.deepcopy(conf)
    original_overiding = copy.deepcopy(overiding_conf)

    overide_pipeline_conf(conf, overiding_conf)

    assert conf == original_conf
    assert overiding_conf == original_overiding


@pytest.mark.unit_tests
def test_overide_pipeline_conf_mixed_types():
    """
    Test with mixed data types and numbered keys
    """
    conf = {
        "item_001": "string",
        "item_002": 42,
        "item_003": ["list", "values"],
        "item_004": {"nested": True},
    }
    overiding_conf = {
        "item_002": 100,
        "item_004": {"nested": False, "added": "value"},
        "item_005": None,
    }

    result = overide_pipeline_conf(conf, overiding_conf)

    expected = {
        "item_001": "string",
        "item_002": 100,
        "item_003": ["list", "values"],
        "item_004": {"nested": False, "added": "value"},
        "item_005": None,
    }

    assert result == expected
