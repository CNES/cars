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
dummy CARS abstract classes tests
"""
from cars.core.geometry import AbstractGeometry


@AbstractGeometry.register_subclass("NoDispTriangulationMethodClass")
# pylint: disable=abstract-method
class NoDispTriangulationMethodClass(AbstractGeometry):
    """
    geometry class with no triangulate method
    """

    @staticmethod
    def triangulate_matches(
        matches,
        grid1,
        grid2,
        img1,
        img2,
        min_elev1,
        max_elev1,
        min_elev2,
        max_elev2,
    ):
        """
        test func
        """


@AbstractGeometry.register_subclass("NoMatchesTriangulationMethodClass")
# pylint: disable=abstract-method
class NoMatchesTriangulationMethodClass(AbstractGeometry):
    """
    geometry class with no triangulate_matches method
    """

    @staticmethod
    def triangulate(
        data,
        roi_key,
        grid1,
        grid2,
        img1,
        img2,
        min_elev1,
        max_elev1,
        min_elev2,
        max_elev2,
    ):
        """
        test func
        """


@AbstractGeometry.register_subclass("NoTriangulationMethodClass")
# pylint: disable=abstract-method
class NoTriangulationMethodClass(AbstractGeometry):
    """
    geometry class without any abstract method
    """

    def wrong_func(self):
        """
        test func
        """
