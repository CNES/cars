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

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=dem, geoid=geoid
        )
    )

    lat, lon, alt = geo_plugin.direct_loc(
        sensor, geomodel, np.array([0, 5, 10]), np.array([0, 6, 12])
    )
    current = np.array([lat, lon, alt])
    reference = np.array(
        [
            [44.20805591, 44.20802983, 44.20800369],
            [5.1934094, 5.193442, 5.19347457],
            [503.55024397, 504.01215811, 504.43611393],
        ],
        np.dtype(np.float64),
    )
    np.testing.assert_allclose(current, reference)


@pytest.mark.unit_tests
def test_inverse_loc_rpc():
    """
    Test inverse localization with RPC
    """
    sensor = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel = {"path": geomodel_path, "model_type": RPC_TYPE}

    dem = absolute_data_path("input/phr_ventoux/srtm/N44E005.hgt")
    geoid = get_geoid_path()

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=dem, geoid=geoid
        )
    )

    inputs_lat = np.array(
        [44.20805591, 44.20802983, 44.20800369],
    )
    inputs_lon = np.array([5.1934094, 5.193442, 5.19347457])
    inputs_z = np.array([503.55024397, 504.01215811, 504.43611393])

    row, col, alti = geo_plugin.inverse_loc(
        sensor, geomodel, inputs_lat, inputs_lon, z_coord=inputs_z
    )

    reference_row = np.array([0, 5, 10])
    reference_col = np.array([0, 6, 12])
    np.testing.assert_allclose(row, reference_row, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(col, reference_col, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(alti, inputs_z, rtol=0.01, atol=0.01)


@pytest.mark.unit_tests
def test_get_roi():
    """
    Test direct localization with RPC
    """
    sensor1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel1_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel1 = {"path": geomodel1_path, "model_type": RPC_TYPE}

    sensor2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    geomodel2_path = absolute_data_path("input/phr_ventoux/right_image.geom")
    geomodel2 = {"path": geomodel2_path, "model_type": RPC_TYPE}

    dem = absolute_data_path("input/phr_ventoux/srtm/N44E005.hgt")
    geoid = get_geoid_path()

    # Uses 0 margin with linar interpolator
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            {"plugin_name": "SharelocGeometry", "interpolator": "linear"},
            dem=dem,
            geoid=geoid,
        )
    )
    pairs_for_roi = [(sensor1, geomodel1, sensor2, geomodel2)]
    roi = geo_plugin.get_roi(pairs_for_roi, 4326, margin=0.005)
    ref_roi = [
        44.19920812310502,
        5.187107532543532,
        44.21309529125914,
        5.202048185183154,
    ]
    np.testing.assert_allclose(roi, ref_roi)

    # Uses a 5 pixel margin on rectification grid, with cubic interpolator
    geo_plugin_with_margin_on_grid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry",
            dem=dem,
            geoid=geoid,
        )
    )
    pairs_for_roi = [(sensor1, geomodel1, sensor2, geomodel2)]
    roi = geo_plugin_with_margin_on_grid.get_roi(
        pairs_for_roi, 4326, margin=0.005
    )

    ref_roi = [
        44.198651,
        5.185954,
        44.21382,
        5.203201,
    ]
    # Returned ROI is the footprint of the rectification
    # It takes into account the 5 pixels margin
    np.testing.assert_allclose(roi, ref_roi)


@pytest.mark.unit_tests
def test_sensors_arrangement_left_right():
    """
    Test sensors arrangement detection
    """
    sensor1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel1_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel1 = {"path": geomodel1_path, "model_type": RPC_TYPE}
    grid1 = absolute_data_path("input/phr_ventoux/left_epi_grid.tif")

    sensor2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    geomodel2_path = absolute_data_path("input/phr_ventoux/right_image.geom")
    geomodel2 = {"path": geomodel2_path, "model_type": RPC_TYPE}
    grid2 = absolute_data_path("input/phr_ventoux/corrected_right_epi_grid.tif")

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry"
        )
    )

    assert geo_plugin.sensors_arrangement_left_right(
        sensor1, sensor2, geomodel1, geomodel2, grid1, grid2
    )
    assert not geo_plugin.sensors_arrangement_left_right(
        sensor2, sensor1, geomodel2, geomodel1, grid2, grid1
    )
