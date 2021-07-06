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
Test module for cars.plugins.triangulation.abstract
"""
import pytest

from cars.plugins.triangulation.abstract import (  # isort:skip;
    # pylint: disable=wrong-import-order
    AbstractTriangulation,
)

from ..dummy_plugins import (  # noqa; isort:skip; pylint: disable=unused-import
    NoDispTriangulationMethodClass,
    NoMatchesTriangulationMethodClass,
    NoTriangulationMethodClass,
)


@pytest.mark.unit_tests
def test_missing_abstract_methods():
    """
    Test cars triangulation abstract class
    """
    with pytest.raises(Exception) as error:
        AbstractTriangulation(  # pylint: disable=abstract-class-instantiated
            "NoDispTriangulationMethodClass"
        )
        assert (
            str(error.value) == "Can't instantiate abstract class"
            " NoDispTriangulationMethodClass with "
            "abstract methods triangulate, triangulate_matches"
        )

    with pytest.raises(Exception) as error:
        AbstractTriangulation(  # pylint: disable=abstract-class-instantiated
            "NoMatchesTriangulationMethodClass"
        )
        assert (
            str(error.value) == "Can't instantiate abstract class"
            " NoMatchesTriangulationMethodClass with "
            "abstract methods triangulate, triangulate_matches"
        )

    with pytest.raises(Exception) as error:
        AbstractTriangulation(  # pylint: disable=abstract-class-instantiated
            "NoTriangulationMethodClass"
        )
        assert (
            str(error.value) == "Can't instantiate abstract class"
            " NoTriangulationMethodClass with "
            "abstract methods triangulate, triangulate_matches"
        )


@pytest.mark.unit_tests
def test_wrong_class_name():
    """
    Test cars triangulation abstract class
    """
    with pytest.raises(Exception) as error:
        AbstractTriangulation(  # pylint: disable=abstract-class-instantiated
            "test"
        )
        assert (
            str(error.value) == "No triangulation plugin named test registered"
        )
