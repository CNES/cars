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
Unit tests for DSM merging statistics.
"""

import numpy as np
import pytest

from cars.applications.dsm_merging.weighted_fusion_app import (
    aggregate_overlap_stats,
    compute_overlap_stats,
)

pytestmark = pytest.mark.unit_tests


@pytest.mark.parametrize(
    "previous_data, current_data, previous_weights, current_weights, expected",
    [
        pytest.param(
            np.array([[100.0, 105.0], [110.0, 120.0]]),
            np.array([[102.0, 107.0], [112.0, 122.0]]),
            np.ones((2, 2)),
            np.ones((2, 2)),
            {
                "compared_pixels": 4,
                "mean": 2.0,
                "median": 2.0,
                "std": 0.0,
            },
            id="constant_difference",
        ),
        pytest.param(
            np.array(
                [
                    [100.0, 101.0, 102.0],
                    [103.0, np.nan, -9999.0],
                ]
            ),
            np.array(
                [
                    [101.0, 103.0, 999.0],
                    [103.0, 5.0, 103.0],
                ]
            ),
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            {
                "compared_pixels": 2,
                "mean": 1.5,
                "median": 1.5,
                "std": 0.5,
            },
            id="invalid_pixels",
        ),
        pytest.param(
            np.array([[100.0, 101.0]]),
            np.array([[102.0, 103.0]]),
            np.zeros((1, 2)),
            np.ones((1, 2)),
            None,
            id="no_overlap",
        ),
    ],
)
def test_compute_overlap_stats(
    previous_data,
    current_data,
    previous_weights,
    current_weights,
    expected,
):
    """
    Test overlap statistics with valid, invalid and non-overlapping pixels.
    """

    stats = compute_overlap_stats(
        previous_data,
        current_data,
        previous_weights,
        current_weights,
        previous_nodata=-9999,
        current_nodata=-9999,
    )

    if expected is None:
        assert stats is None
    else:
        assert stats == pytest.approx(expected)


@pytest.mark.parametrize(
    "tiles",
    [
        pytest.param(
            [
                np.array([0.0, 0.0, 10.0]),
                np.array([20.0, 20.0]),
            ],
            id="multiple_tiles",
        ),
        pytest.param(
            [
                np.array([1.0, 2.0, 3.0, 4.0]),
            ],
            id="single_tile",
        ),
    ],
)
def test_aggregate_overlap_stats(tiles):
    """
    Test global aggregation of overlap statistics.

    Mean and standard deviation are exact.
    Median is the weighted mean of tile medians.
    """

    tile_stats = [
        {
            "compared_pixels": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }
        for values in tiles
    ]

    stats = aggregate_overlap_stats(tile_stats)

    all_values = np.concatenate(tiles)

    expected_median = sum(
        len(values) * np.median(values) for values in tiles
    ) / len(all_values)

    assert stats["compared_pixels"] == len(all_values)

    assert stats["mean"] == pytest.approx(np.mean(all_values))
    assert stats["std"] == pytest.approx(np.std(all_values))
    assert stats["median"] == pytest.approx(expected_median)

    if len(tiles) > 1:
        # Weighted tile medians differ from the exact global median
        assert stats["median"] != np.median(all_values)


def test_aggregate_overlap_stats_empty():
    """
    Test aggregation without overlapping pixels.
    """

    assert aggregate_overlap_stats([]) is None
