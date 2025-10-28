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
    sensor1_path = absolute_data_path("input/phr_ventoux/left_image.tif")
    sensor1 = {"bands": {"b0": {"path": sensor1_path}}}
    geomodel1_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel1 = {"path": geomodel1_path, "model_type": RPC_TYPE}

    sensor2_path = absolute_data_path("input/phr_ventoux/right_image.tif")
    sensor2 = {"bands": {"b0": {"path": sensor2_path}}}
    geomodel2_path = absolute_data_path("input/phr_ventoux/right_image.geom")
    geomodel2 = {"path": geomodel2_path, "model_type": RPC_TYPE}

    # Uses 0 margin with linear interpolator
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
        )
    )
    pairs_for_roi = [(sensor1, geomodel1, sensor2, geomodel2)]
    roi = geo_plugin.get_roi(pairs_for_roi, 4326, constant_margin=0.005)
    ref_roi = [
        5.187108,
        44.199474,
        5.202048,
        44.212998,
    ]
    np.testing.assert_allclose(roi, ref_roi)

    # Uses a 5 pixel margin on rectification grid, with cubic interpolator
    geo_plugin_with_margin_on_grid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry"
        )
    )
    pairs_for_roi = [(sensor1, geomodel1, sensor2, geomodel2)]
    roi = geo_plugin_with_margin_on_grid.get_roi(
        pairs_for_roi, 4326, constant_margin=0.005
    )

    ref_roi = [
        5.185954,
        44.198651,
        5.203201,
        44.21382,
    ]
    # Returned ROI is the footprint of the rectification
    # It takes into account the 5 pixels margin
    np.testing.assert_allclose(roi, ref_roi)


@pytest.mark.unit_tests
def test_exception_roi_outside_dtm():
    """
    Test when the roi is outside the dtm
    """

    sensor1_path = absolute_data_path("input/phr_ventoux/left_image.tif")
    sensor1 = {"bands": {"b0": {"path": sensor1_path}}}
    geomodel1_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel1 = {"path": geomodel1_path, "model_type": RPC_TYPE}

    sensor2_path = absolute_data_path("input/phr_ventoux/right_image.tif")
    sensor2 = {"bands": {"b0": {"path": sensor2_path}}}
    geomodel2_path = absolute_data_path("input/phr_ventoux/right_image.geom")
    geomodel2 = {"path": geomodel2_path, "model_type": RPC_TYPE}

    pairs_for_roi = [(sensor1, geomodel1, sensor2, geomodel2)]

    dem = absolute_data_path("input/phr_gizeh/srtm_dir/N29E031_KHEOPS.tif")
    geoid = get_geoid_path()

    with pytest.raises(RuntimeError) as excinfo:
        _ = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry",
            dem=dem,
            geoid=geoid,
            default_alt=0,
            pairs_for_roi=pairs_for_roi,
            scaling_coeff=1,
        )

    assert str(excinfo.value) == (
        "The extent of the roi lies outside "
        "the extent of the initial elevation : the roi bounds are "
        "[5.156013852323939, 44.15169289345662, 5.235306512180193, "
        "44.256780712026995] while the dtm bounds are "
        "[31.099861111111114, 29.950138888888887, "
        "31.149861111111115, 30.000138888888888]"
    )
