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
Unit tests for depth to Z fusion.
"""

import numpy as np
import pytest
import xarray as xr

from cars.applications.depth_to_z_fusion.anisotropic_depth_to_z_app import (
    AnisotropicDepthToZFusion,
)
from cars.applications.depth_to_z_fusion.depth_to_z_fusion_algo import (
    build_masks,
    extract_scalar_layer,
    fit_depth_to_z_tile,
)
from cars.core import constants as cst


def build_test_tile(z_values):
    """
    Build a minimal point cloud tile with depth map and tile ids.
    """

    x_values = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    y_values = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    depth_values = np.ones((1, 3, 3), dtype=np.float32)
    tile_ids = np.zeros((1, 3, 3), dtype=np.int32)

    return xr.Dataset(
        data_vars={
            cst.X: ((cst.ROW, cst.COL), x_values),
            cst.Y: ((cst.ROW, cst.COL), y_values),
            cst.Z: ((cst.ROW, cst.COL), z_values.astype(np.float32)),
            cst.EPI_EDGES_DEPTH_MAP: (
                (cst.BAND_EDGES_DEPTH_MAP, cst.ROW, cst.COL),
                depth_values,
            ),
            cst.EPI_EDGES_TILE_ID: (
                (cst.BAND_EDGES_TILE_ID, cst.ROW, cst.COL),
                tile_ids,
            ),
        }
    )


@pytest.mark.unit_tests
def test_depth_to_z_fusion_config():
    """
    Test configuration check for anisotropic depth to Z fusion.
    """

    conf = {
        "method": "anisotropic",
        "activated": False,
        "save_intermediate_data": False,
        "lambda_data": 0.5,
        "depth_sigma": 1.0,
        "depth_sigma_auto": False,
        "detail_scale": 0.0,
        "cross_tile_weight": 0.0,
        "fill_values": "invalid",
        "iterations": 100,
        "tol": 1.0e-4,
    }
    _ = AnisotropicDepthToZFusion(conf)


@pytest.mark.unit_tests
def test_depth_to_z_fusion_config_activated_true():
    """
    Test configuration check with activated skip flag.
    """

    conf = {
        "method": "anisotropic",
        "activated": True,
        "save_intermediate_data": False,
        "lambda_data": 0.5,
        "depth_sigma": 1.0,
        "depth_sigma_auto": False,
        "detail_scale": 0.0,
        "cross_tile_weight": 0.0,
        "fill_values": "invalid",
        "iterations": 100,
        "tol": 1.0e-4,
    }
    _ = AnisotropicDepthToZFusion(conf)


@pytest.mark.unit_tests
def test_depth_to_z_fusion_config_activated_auto_resolves_to_default():
    """
    Test activated accepts auto and resolves to the default value.
    """

    app = AnisotropicDepthToZFusion({"activated": "auto"})

    assert app.activated == "auto"

    app.default_value_for_auto_configuration()

    assert app.activated is True


@pytest.mark.unit_tests
def test_fit_depth_to_z_tile_creates_fit_map_and_fills_invalid_z():
    """
    Test that anisotropic fusion outputs fit_depth_map, keeps valid Z unchanged,
    and fills invalid Z with fitted values.
    """

    tile = build_test_tile(
        np.array(
            [[0.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )

    result = fit_depth_to_z_tile(
        tile,
        lambda_data=0.5,
        depth_sigma=1.0,
        depth_sigma_auto=False,
        detail_scale=0.0,
        cross_tile_weight=0.0,
        fill_values="invalid",
        iterations=200,
        tol=1.0e-5,
    )

    input_z = tile[cst.Z].values
    output_z = result[cst.Z].values
    valid_mask = np.isfinite(input_z)
    invalid_mask = ~valid_mask

    assert np.allclose(
        output_z[valid_mask], input_z[valid_mask], equal_nan=True
    )
    assert np.all(np.isfinite(output_z[invalid_mask]))
    assert np.allclose(
        output_z[invalid_mask],
        result["fit_depth_map"].values[invalid_mask],
        equal_nan=True,
    )
    assert "fit_depth_map" in result
    assert np.all(np.isfinite(result["fit_depth_map"].values))
    assert np.allclose(result[cst.X].values, tile[cst.X].values)
    assert np.allclose(result[cst.Y].values, tile[cst.Y].values)
    assert cst.EPI_FILLING in result
    assert cst.FILLING_DEPTH_TO_Z in result.coords[cst.BAND_FILLING].values
    depth_to_z_filling = result[cst.EPI_FILLING].sel(
        **{cst.BAND_FILLING: cst.FILLING_DEPTH_TO_Z}
    )
    assert np.array_equal(depth_to_z_filling.values, invalid_mask)
    assert "depth_to_z_residual" in result
    assert "depth_to_z_weight_map" in result
    assert np.all(np.isfinite(result["depth_to_z_weight_map"].values))


@pytest.mark.unit_tests
def test_fit_depth_to_z_tile_fill_all_z_with_fit():
    """
    Test fill_values='all' mode overwrites all Z values on depth domain.
    """

    tile = build_test_tile(
        np.array(
            [[0.0, 1.0, 2.0], [3.0, np.nan, 5.0], [6.0, 7.0, 8.0]],
            dtype=np.float32,
        )
    )

    result = fit_depth_to_z_tile(
        tile,
        lambda_data=0.5,
        depth_sigma=1.0,
        depth_sigma_auto=False,
        detail_scale=0.0,
        cross_tile_weight=0.0,
        fill_values="all",
        iterations=200,
        tol=1.0e-5,
    )

    fit_map = result["fit_depth_map"].values
    out_z = result[cst.Z].values
    filling_mask = (
        result[cst.EPI_FILLING]
        .sel(**{cst.BAND_FILLING: cst.FILLING_DEPTH_TO_Z})
        .values
    )
    domain_mask = np.isfinite(fit_map)

    assert np.allclose(out_z[domain_mask], fit_map[domain_mask], equal_nan=True)
    assert np.array_equal(
        filling_mask[domain_mask],
        np.ones(np.count_nonzero(domain_mask), dtype=bool),
    )


@pytest.mark.unit_tests
def test_fit_depth_to_z_tile_fill_none_keeps_z_and_marks_no_filling():
    """
    Test fill_values=None keeps original Z values and does not mark filling.
    """

    tile = build_test_tile(
        np.array(
            [[0.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )

    result = fit_depth_to_z_tile(
        tile,
        lambda_data=0.5,
        depth_sigma=1.0,
        depth_sigma_auto=False,
        detail_scale=0.0,
        cross_tile_weight=0.0,
        fill_values=None,
        iterations=200,
        tol=1.0e-5,
    )

    assert np.allclose(result[cst.Z].values, tile[cst.Z].values, equal_nan=True)
    filling_mask = (
        result[cst.EPI_FILLING]
        .sel(**{cst.BAND_FILLING: cst.FILLING_DEPTH_TO_Z})
        .values
    )
    assert not np.any(filling_mask)


@pytest.mark.unit_tests
def test_fit_depth_to_z_tile_excludes_invalid_mask_pixels_from_anchors():
    """
    Test that pixels flagged by EPI_INVALIDITY_MASK (e.g. by outlier removal)
    are excluded from the anchor mask even when their Z is finite.
    Outlier removal fills Z rather than NaN-ing it, so without checking the
    invalidity mask those pixels would be incorrectly used as data anchors.
    """

    # All Z values are finite (simulating outlier-removal Z-fill behaviour)
    z_base = np.array(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
        dtype=np.float32,
    )
    tile = build_test_tile(z_base)

    # Flag the centre pixel as removed by outlier removal
    # EPI_INVALIDITY_MASK shape: (n_bands, H, W); band 1 = 1 means removed
    invalidity = np.zeros((2, 3, 3), dtype=np.int16)
    invalidity[1, 1, 1] = 1  # centre pixel flagged
    tile[cst.EPI_INVALIDITY_MASK] = xr.DataArray(
        invalidity,
        dims=[cst.BAND_INVALIDITY_MASK, cst.ROW, cst.COL],
    )

    z_map = tile[cst.Z].values.astype(np.float32)
    depth = extract_scalar_layer(tile[cst.EPI_EDGES_DEPTH_MAP])
    tile_id_arr = extract_scalar_layer(tile[cst.EPI_EDGES_TILE_ID])
    inv = tile[cst.EPI_INVALIDITY_MASK].values

    _, anchor_mask = build_masks(depth, z_map, tile_id_arr, inv)

    # Centre pixel must NOT be in anchor mask despite having finite Z
    assert not anchor_mask[1, 1], (
        "Outlier-removed pixel (finite Z, invalidity_mask=1) must be "
        "excluded from anchor_mask"
    )
    # All other pixels should still be anchors
    assert anchor_mask.sum() == 8
