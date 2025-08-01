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
Test module for cars/pipelines/pipeline_template
"""


# Third party imports
import pytest

# CARS imports
import cars.pipelines.pipeline_template as pipeline_temp


@pytest.mark.unit_tests
def test_merge_resolution_conf_rec():
    """ "
    Test merge_pipeline_conf

    """

    conf1 = {"a": "toto", "b": {"c": {"d": "titi"}, "e": "tutu"}}

    conf2 = {"b": {"e": "tota"}, "g": "tato"}

    updated_conf = conf1.copy()

    pipeline_temp._merge_resolution_conf_rec(  # pylint: disable=W0212
        updated_conf, conf2
    )

    vt_conf = {"a": "toto", "b": {"c": {"d": "titi"}, "e": "tutu"}, "g": "tato"}

    assert updated_conf == vt_conf
