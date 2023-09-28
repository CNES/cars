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
Cars tests/points_cloud_outliers_removing  file
"""

# Third party imports
import numpy as np
import pytest

# CARS imports
from cars.applications.point_cloud_outliers_removing import (
    outlier_removing_tools,
)

# CARS Tests imports


@pytest.mark.unit_tests
def test_detect_small_components():
    """
    Create fake cloud to process and test detect_small_components
    """
    x_coord = np.zeros((5, 5))
    x_coord[4, 4] = 20
    x_coord[0, 4] = 19.55
    x_coord[0, 3] = 19.10
    y_coord = np.zeros((5, 5))

    z_coord = np.zeros((5, 5))
    z_coord[0:2, 0:2] = 10
    z_coord[1, 1] = 12

    cloud_arr = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(  # noqa: B905
                x_coord, y_coord, z_coord
            )
        ],
        axis=0,
    )

    indexes_to_filter = outlier_removing_tools.detect_small_components(
        cloud_arr, 0.5, 10, 2
    )
    assert sorted(indexes_to_filter) == [3, 4, 24]

    # test without the second level of filtering
    indexes_to_filter = outlier_removing_tools.detect_small_components(
        cloud_arr, 0.5, 10, None
    )
    assert sorted(indexes_to_filter) == [0, 1, 3, 4, 5, 6, 24]


@pytest.mark.unit_tests
def test_detect_statistical_outliers():
    """
    Create fake cloud to process and test detect_statistical_outliers
    """
    x_coord = np.zeros((5, 6))
    off = 0
    for line in range(5):
        # x[line,:] = np.arange(off, off+(line+1)*5, line+1)
        last_val = off + 5
        x_coord[line, :5] = np.arange(off, last_val)
        off += (line + 2 + 1) * 5

        # outlier
        x_coord[line, 5] = (off + last_val - 1) / 2

    y_coord = np.zeros((5, 6))
    z_coord = np.zeros((5, 6))

    ref_cloud = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(  # noqa: B905
                x_coord, y_coord, z_coord
            )
        ],
        axis=0,
    )

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 0.0, use_median=False
    )
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 1.0, use_median=False
    )
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 2.0, use_median=False
    )
    assert sorted(removed_elt_pos) == [23, 29]

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 1.0, use_median=True
    )
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 7.0, use_median=True
    )
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = outlier_removing_tools.detect_statistical_outliers(
        ref_cloud, 4, 15.0, use_median=True
    )
    assert sorted(removed_elt_pos) == [23, 29]
