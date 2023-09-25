#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
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
Test module for cars/core/geometry/shareloc_geometry.py
"""

import numpy as np

# Third party imports
import pytest

from cars.core.geometry.abstract_geometry import AbstractGeometry

# CARS imports
from cars.core.geometry.shareloc_geometry import RPC_TYPE

# CARS Tests imports
from ...helpers import absolute_data_path, get_geoid_path


@pytest.mark.unit_tests
def test_dir_loc_rpc():
    """
    Test direct localization with RPC
    """
    sensor = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel = {"path": geomodel_path, "model_type": RPC_TYPE}

    dem = absolute_data_path("input/phr_ventoux/srtm/N44E005.hgt")
    geoid = get_geoid_path()

    geo_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=dem, geoid=geoid
        )
    )

    lat, lon, alt = geo_loader.direct_loc(
        sensor, geomodel, np.array([0, 5, 10]), np.array([0, 6, 12])
    )
    current = np.array([lat, lon, alt])
    reference = np.array(
        [
            [5.193442, 44.20802983, 504.01215811],
            [5.1934094, 44.20805591, 503.55024397],
            [5.19347457, 44.20800369, 504.43611393],
        ],
        np.dtype(np.float64),
    )
    np.testing.assert_allclose(current, reference)
