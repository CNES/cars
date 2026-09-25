#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
Unit tests for point cloud refinement configuration.
"""

import pytest

from cars.applications.point_cloud_refinement.normals_guided_app import (
    NormalsGuidedPointCloudRefinement,
)


@pytest.mark.unit_tests
def test_point_cloud_refinement_config_activated_auto_resolves_to_default():
    """
    Test activated accepts auto and resolves to the default value.
    """

    app = NormalsGuidedPointCloudRefinement({"activated": "auto"})

    assert app.activated == "auto"

    app.default_value_for_auto_configuration()

    assert app.activated is True
