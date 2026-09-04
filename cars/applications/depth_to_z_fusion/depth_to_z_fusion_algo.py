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
Depth to Z fusion algorithms.
"""

# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from cars.core import constants as cst

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_REQUIRED_SCALAR_LAYERS = (cst.X, cst.Y, cst.Z)


def has_required_bands(tile: xr.Dataset) -> bool:
    """
    Check required data fields for depth to Z fusion.
    """

    if tile is None:
        return False

    if not all(key in tile for key in _REQUIRED_SCALAR_LAYERS):
        return False

    return cst.EPI_EDGES_DEPTH_MAP in tile and cst.EPI_EDGES_TILE_ID in tile


def extract_scalar_layer(data_array: xr.DataArray) -> np.ndarray | None:
    """
    Extract a 2D scalar array from a 2D or 3D xarray layer.
    """

    values = data_array.values
    if values.ndim == 2:
        return values.astype(np.float32)
    if values.ndim == 3 and values.shape[0] > 0:
        return values[0].astype(np.float32)
    return None


def build_masks(
    depth: np.ndarray,
    z_map: np.ndarray,
    tile_id: np.ndarray,
    invalidity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build domain and anchor masks.

    Domain mask: where depth is defined and tile id is defined with a valid
    non-nodata value.
    Anchor mask: subset of domain where Z is defined and not flagged as
    invalid by the outlier removal step.

    Outlier removal does not NaN-fill Z; instead it sets band values in
    EPI_INVALIDITY_MASK to 1.  Without checking this mask, removed pixels
    (which carry an interpolated, unreliable Z) would be included as stereo
    data-attachment anchors, polluting the Jacobi solve.
    """

    domain_mask = (
        np.isfinite(depth)
        & np.isfinite(tile_id)
        & (tile_id != cst.EPI_EDGES_TILE_ID_NODATA)
    )
    anchor_mask = domain_mask & np.isfinite(z_map)

    if invalidity is not None:
        # invalidity shape: (n_bands, H, W) or (H, W)
        if invalidity.ndim == 3:
            flagged = np.any(invalidity != 0, axis=0)
        else:
            flagged = invalidity != 0
        anchor_mask &= ~flagged

    return domain_mask, anchor_mask


def _slices(height: int, width: int, delta_i: int, delta_j: int):
    """
    Slice pairs for a pixel and its neighbour.
    """

    src_i0 = max(0, -delta_i)
    src_i1 = min(height, height - delta_i)
    src_j0 = max(0, -delta_j)
    src_j1 = min(width, width - delta_j)
    return (
        src_i0,
        src_i1,
        src_j0,
        src_j1,
    ), (
        src_i0 + delta_i,
        src_i1 + delta_i,
        src_j0 + delta_j,
        src_j1 + delta_j,
    )


def estimate_global_sigma(depth: np.ndarray, valid: np.ndarray) -> float:
    """
    Estimate a robust global sigma from the depth IQR.
    """

    valid_depth = depth[valid]
    if valid_depth.size == 0:
        return 1.0e-9

    q25, q75 = np.percentile(valid_depth.astype(np.float64), [25.0, 75.0])
    return max(0.1 * float(q75 - q25) / 1.35, 1.0e-9)


def auto_sigma_per_tile(
    depth: np.ndarray,
    tile_id: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Estimate one depth-difference sigma per tile.
    """

    height, width = depth.shape
    unique_tiles = np.unique(tile_id[valid])
    diffs_by_tile = {tile_value: [] for tile_value in unique_tiles}

    for delta_i, delta_j in ((1, 0), (0, 1)):
        src_slices, dst_slices = _slices(height, width, delta_i, delta_j)
        src_i0, src_i1, src_j0, src_j1 = src_slices
        dst_i0, dst_i1, dst_j0, dst_j1 = dst_slices
        use = (
            valid[src_i0:src_i1, src_j0:src_j1]
            & valid[dst_i0:dst_i1, dst_j0:dst_j1]
            & (
                tile_id[src_i0:src_i1, src_j0:src_j1]
                == tile_id[dst_i0:dst_i1, dst_j0:dst_j1]
            )
        )
        src_tiles = tile_id[src_i0:src_i1, src_j0:src_j1][use]
        diffs = np.abs(
            depth[src_i0:src_i1, src_j0:src_j1][use].astype(np.float64)
            - depth[dst_i0:dst_i1, dst_j0:dst_j1][use].astype(np.float64)
        )
        for tile_value in unique_tiles:
            tile_mask = src_tiles == tile_value
            if np.any(tile_mask):
                diffs_by_tile[tile_value].append(diffs[tile_mask])

    sigma_per_tile = {}
    for tile_value in unique_tiles:
        parts = diffs_by_tile[tile_value]
        if parts:
            sigma_per_tile[tile_value] = max(
                float(np.median(np.concatenate(parts))), 1.0e-9
            )
        else:
            sigma_per_tile[tile_value] = 1.0e-9
    return sigma_per_tile


def build_edge_weights(
    depth: np.ndarray,
    tile_id: np.ndarray,
    valid: np.ndarray,
    sigma_d: float | dict,
    cross_tile_weight: float,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute per-direction depth-guided edge weights.
    """

    height, width = depth.shape
    depth_float = depth.astype(np.float32)
    weights = {}

    if isinstance(sigma_d, dict):
        sigma_map = np.ones((height, width), dtype=np.float64)
        for tile_value, sigma_value in sigma_d.items():
            sigma_map[tile_id == tile_value] = max(sigma_value, 1.0e-9)
        inv2s2_map = (1.0 / (2.0 * sigma_map**2)).astype(np.float32)
        inv2s2_scalar = None
    else:
        inv2s2_map = None
        inv2s2_scalar = 1.0 / (2.0 * max(sigma_d, 1.0e-9) ** 2)

    for delta_i, delta_j in _NEIGHBOURS:
        src_slices, dst_slices = _slices(height, width, delta_i, delta_j)
        src_i0, src_i1, src_j0, src_j1 = src_slices
        dst_i0, dst_i1, dst_j0, dst_j1 = dst_slices
        weight_map = np.zeros((height, width), dtype=np.float32)
        both_valid = (
            valid[src_i0:src_i1, src_j0:src_j1]
            & valid[dst_i0:dst_i1, dst_j0:dst_j1]
        )
        same_tile = (
            tile_id[src_i0:src_i1, src_j0:src_j1]
            == tile_id[dst_i0:dst_i1, dst_j0:dst_j1]
        )

        intra = both_valid & same_tile
        if np.any(intra):
            diff_sq = (
                depth_float[src_i0:src_i1, src_j0:src_j1][intra].astype(
                    np.float64
                )
                - depth_float[dst_i0:dst_i1, dst_j0:dst_j1][intra].astype(
                    np.float64
                )
            ) ** 2
            if inv2s2_map is not None:
                inv2s2 = inv2s2_map[src_i0:src_i1, src_j0:src_j1][intra]
            else:
                inv2s2 = inv2s2_scalar
            weight_map[src_i0:src_i1, src_j0:src_j1][intra] = np.exp(
                -diff_sq * inv2s2
            ).astype(np.float32)

        cross = both_valid & ~same_tile
        if cross_tile_weight > 0.0 and np.any(cross):
            weight_map[src_i0:src_i1, src_j0:src_j1][cross] = cross_tile_weight

        weights[(delta_i, delta_j)] = weight_map

    return weights


def _slope_from_pairs(
    delta_depth: np.ndarray,
    delta_z: np.ndarray,
    min_pairs: int,
) -> float | None:
    """
    Through-origin depth->Z slope from neighbour differences, MAD-clipped
    """

    if delta_depth.size < min_pairs:
        return None
    d_depth = delta_depth.astype(np.float64)
    d_z = delta_z.astype(np.float64)

    def _fit(diff_depth: np.ndarray, diff_z: np.ndarray) -> float | None:
        """
        Fit slope through origin from neighbour differences
        """
        denom = float(np.dot(diff_depth, diff_depth))
        if denom < 1.0e-6:
            return None
        return float(np.dot(diff_depth, diff_z) / denom)

    slope = _fit(d_depth, d_z)
    if slope is None:
        return None

    # Reject outlier pairs (stereo blunders, depth artifacts) then refit.
    residual = d_z - slope * d_depth
    mad = float(np.median(np.abs(residual - np.median(residual))))
    if mad > 0.0:
        keep = np.abs(residual) <= 3.0 * 1.4826 * mad
        if keep.sum() >= min_pairs:
            slope = _fit(d_depth[keep], d_z[keep]) or slope
    return slope


def _anchor_pair_differences(
    depth: np.ndarray,
    z_map: np.ndarray,
    anchor_mask: np.ndarray,
    tile_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect ``(dD, dZ, source tile id)`` over adjacent same-tile anchor pairs.
    """

    depth64 = depth.astype(np.float64)
    z64 = z_map.astype(np.float64)
    d_depth_parts, d_z_parts, tile_parts = [], [], []

    for axis in (0, 1):
        if axis == 0:
            pair = (
                anchor_mask[:-1, :]
                & anchor_mask[1:, :]
                & (tile_id[:-1, :] == tile_id[1:, :])
            )
            d_depth_parts.append((depth64[1:, :] - depth64[:-1, :])[pair])
            d_z_parts.append((z64[1:, :] - z64[:-1, :])[pair])
            tile_parts.append(tile_id[:-1, :][pair])
        else:
            pair = (
                anchor_mask[:, :-1]
                & anchor_mask[:, 1:]
                & (tile_id[:, :-1] == tile_id[:, 1:])
            )
            d_depth_parts.append((depth64[:, 1:] - depth64[:, :-1])[pair])
            d_z_parts.append((z64[:, 1:] - z64[:, :-1])[pair])
            tile_parts.append(tile_id[:, :-1][pair])

    return (
        np.concatenate(d_depth_parts),
        np.concatenate(d_z_parts),
        np.concatenate(tile_parts),
    )


def estimate_depth_to_z_scale(
    depth: np.ndarray,
    z_map: np.ndarray,
    anchor_mask: np.ndarray,
    tile_id: np.ndarray,
    min_pairs: int = 64,
    min_spread_ratio: float = 0.25,
    max_slope_ratio: float = 3.0,
) -> tuple[dict[float, float], float]:
    """
    Estimate the signed depth->Z slope ``a`` in ``Z ~= a * depth + b`` per tile
    Returns a ``{tile_id: a}`` dict plus the robust global slope fallback
    """

    all_d_depth, all_d_z, all_tile = _anchor_pair_differences(
        depth, z_map, anchor_mask, tile_id
    )
    global_slope = _slope_from_pairs(all_d_depth, all_d_z, min_pairs)
    if global_slope is None or global_slope == 0.0:
        return {}, global_slope or 0.0

    global_spread = float(np.sqrt(np.mean(all_d_depth**2)))
    slope_sign = np.sign(global_slope)
    slope_low = abs(global_slope) / max_slope_ratio
    slope_high = abs(global_slope) * max_slope_ratio

    scale_per_tile: dict[float, float] = {}
    for tile_value in np.unique(tile_id[anchor_mask]):
        in_tile = all_tile == tile_value
        tile_d_depth = all_d_depth[in_tile]
        tile_d_z = all_d_z[in_tile]
        spread = (
            float(np.sqrt(np.mean(tile_d_depth**2)))
            if tile_d_depth.size
            else 0.0
        )
        slope = None
        if spread >= min_spread_ratio * global_spread:
            slope = _slope_from_pairs(tile_d_depth, tile_d_z, min_pairs)
        if slope is None or np.sign(slope) != slope_sign:
            scale_per_tile[tile_value] = global_slope
        else:
            scale_per_tile[tile_value] = float(
                slope_sign * min(max(abs(slope), slope_low), slope_high)
            )
    return scale_per_tile, global_slope


def build_slope_map(
    tile_id: np.ndarray,
    scale_per_tile: dict[float, float],
    global_slope: float,
) -> np.ndarray:
    """
    Expand the per-tile depth->Z slope into a per-pixel ``a`` map.
    """

    slope_map = np.full(tile_id.shape, global_slope, dtype=np.float32)
    for tile_value, slope in scale_per_tile.items():
        slope_map[tile_id == tile_value] = np.float32(slope)
    return slope_map


def smooth_fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """
    Multiscale push-pull fill of a smooth scalar field over unknown pixels.
    """

    value_pyramid: list[np.ndarray] = [
        np.where(known, values.astype(np.float64), 0.0)
    ]
    weight_pyramid: list[np.ndarray] = [known.astype(np.float64)]
    shapes: list[tuple[int, int]] = [values.shape]

    # PULL: coarsen weighted values until the grid is tiny.
    while min(value_pyramid[-1].shape) > 2:
        value_top = value_pyramid[-1]
        weight_top = weight_pyramid[-1]
        height, width = value_top.shape
        pad_h, pad_w = height % 2, width % 2
        weighted = np.pad(value_top * weight_top, ((0, pad_h), (0, pad_w)))
        weight = np.pad(weight_top, ((0, pad_h), (0, pad_w)))
        block_shape = (weighted.shape[0] // 2, 2, weighted.shape[1] // 2, 2)
        weighted_sum = weighted.reshape(block_shape).sum(axis=(1, 3))
        weight_sum = weight.reshape(block_shape).sum(axis=(1, 3))
        coarse_value = np.where(
            weight_sum > 0.0,
            weighted_sum / np.maximum(weight_sum, 1.0e-12),
            0.0,
        )
        value_pyramid.append(coarse_value)
        weight_pyramid.append(np.clip(weight_sum / 4.0, 0.0, 1.0))
        shapes.append((height, width))

    # PUSH: interpolate coarse estimates back down, blending in known values.
    filled = value_pyramid[-1]
    for level in range(len(value_pyramid) - 2, -1, -1):
        target = shapes[level + 1]
        upsampled = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)
        upsampled = upsampled[: target[0], : target[1]]
        weight = weight_pyramid[level]
        filled = weight * value_pyramid[level] + (1.0 - weight) * upsampled
        # Light smoothing removes the block structure of nearest-neighbour
        # upsampling, keeping the fill smooth across scales.
        filled = ndimage.uniform_filter(filled, size=3, mode="nearest")

    return np.where(known, values, filled.astype(values.dtype))


def fuse_anisotropic(
    z_map: np.ndarray,
    domain_mask: np.ndarray,
    anchor_mask: np.ndarray,
    lambda_data: float,
    edge_weights: dict[tuple[int, int], np.ndarray],
    iterations: int,
    tol: float,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Run Jacobi iterations for depth-guided anisotropic Z fusion.
    """

    height, width = z_map.shape
    z_float = np.where(anchor_mask, z_map, 0.0).astype(np.float32)
    lambda_per_pixel = lambda_data * anchor_mask.astype(np.float32)
    lambda_z = lambda_per_pixel * z_float

    z_mean = (
        float(np.mean(z_float[anchor_mask])) if np.any(anchor_mask) else 0.0
    )
    if init is not None:
        fused = init.astype(np.float32).copy()
    else:
        fused = np.full(z_map.shape, z_mean, dtype=np.float32)
    fused[anchor_mask] = z_float[anchor_mask]

    # The edge weights (hence the neighbour weight sum and the solve
    # denominator) are constant across sweeps
    sum_weights = np.zeros((height, width), dtype=np.float32)
    for delta_i, delta_j in _NEIGHBOURS:
        src_slices, _ = _slices(height, width, delta_i, delta_j)
        src_i0, src_i1, src_j0, src_j1 = src_slices
        sum_weights[src_i0:src_i1, src_j0:src_j1] += edge_weights[
            (delta_i, delta_j)
        ][src_i0:src_i1, src_j0:src_j1]

    denominator = lambda_per_pixel + sum_weights
    has_denom = domain_mask & (denominator > 1.0e-12)
    inv_denom = np.zeros((height, width), dtype=np.float32)
    inv_denom[has_denom] = 1.0 / denominator[has_denom]

    last_delta = np.inf
    for iteration in range(1, iterations + 1):
        accumulator = np.zeros((height, width), dtype=np.float32)

        for delta_i, delta_j in _NEIGHBOURS:
            src_slices, dst_slices = _slices(height, width, delta_i, delta_j)
            src_i0, src_i1, src_j0, src_j1 = src_slices
            dst_i0, dst_i1, dst_j0, dst_j1 = dst_slices
            accumulator[src_i0:src_i1, src_j0:src_j1] += (
                edge_weights[(delta_i, delta_j)][src_i0:src_i1, src_j0:src_j1]
                * fused[dst_i0:dst_i1, dst_j0:dst_j1]
            )

        updated = (lambda_z + accumulator) * inv_denom
        next_fused = np.where(has_denom, updated, fused)

        # Non-updated pixels are unchanged, so the full-array max abs diff
        # equals the domain max without any boolean indexing.
        delta = float(np.max(np.abs(next_fused - fused)))
        fused = next_fused
        last_delta = delta
        if delta < tol:
            return fused, iteration, last_delta

    return fused, iterations, last_delta


def mean_weight_map(
    edge_weights: dict[tuple[int, int], np.ndarray],
    domain_mask: np.ndarray,
    tile_id: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel average edge weight from same-tile neighbors only.

    Cross-tile links are excluded from this diagnostic map to avoid imprinting
    the tile grid (which can create regular square-like artifacts when
    cross_tile_weight is small or zero).
    """

    height, width = domain_mask.shape
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for delta_i, delta_j in _NEIGHBOURS:
        weight_map = edge_weights[(delta_i, delta_j)]
        src_slices, dst_slices = _slices(height, width, delta_i, delta_j)
        src_i0, src_i1, src_j0, src_j1 = src_slices
        dst_i0, dst_i1, dst_j0, dst_j1 = dst_slices
        pair_valid = (
            domain_mask[src_i0:src_i1, src_j0:src_j1]
            & domain_mask[dst_i0:dst_i1, dst_j0:dst_j1]
        )
        same_tile = (
            tile_id[src_i0:src_i1, src_j0:src_j1]
            == tile_id[dst_i0:dst_i1, dst_j0:dst_j1]
        )
        pair_valid &= same_tile
        if not np.any(pair_valid):
            continue

        total[src_i0:src_i1, src_j0:src_j1][pair_valid] += weight_map[
            src_i0:src_i1, src_j0:src_j1
        ][pair_valid]
        count[src_i0:src_i1, src_j0:src_j1][pair_valid] += 1.0

    mean_map = np.full((height, width), np.nan, dtype=np.float32)
    valid_count = count > 0
    mean_map[valid_count] = total[valid_count] / count[valid_count]
    mean_map[domain_mask & ~valid_count] = 0.0
    return mean_map


def update_depth_to_z_filling(
    dataset: xr.Dataset,
    fill_mask: np.ndarray,
) -> None:
    """
    Merge depth-to-z provenance into EPI_FILLING in place.

    If EPI_FILLING already exists, update the "depth_to_z" band or append it.
    Otherwise, create EPI_FILLING with a single "depth_to_z" band.
    """

    if cst.EPI_FILLING in dataset and cst.BAND_FILLING in dataset.coords:
        filling = dataset[cst.EPI_FILLING].values.astype(bool, copy=True)
        band_values = [str(v) for v in dataset.coords[cst.BAND_FILLING].values]
        if cst.FILLING_DEPTH_TO_Z in band_values:
            band_index = band_values.index(cst.FILLING_DEPTH_TO_Z)
            filling[band_index, :, :] |= fill_mask
        else:
            filling = np.concatenate(
                [filling, fill_mask[np.newaxis, :, :]],
                axis=0,
            )
            band_values.append(cst.FILLING_DEPTH_TO_Z)
    else:
        filling = fill_mask[np.newaxis, :, :]
        band_values = [cst.FILLING_DEPTH_TO_Z]

    dataset[cst.EPI_FILLING] = xr.DataArray(
        filling,
        dims=[cst.BAND_FILLING, cst.ROW, cst.COL],
        coords={
            cst.BAND_FILLING: band_values,
            cst.ROW: dataset.coords[cst.ROW],
            cst.COL: dataset.coords[cst.COL],
        },
    )


def fit_depth_to_z_tile(
    tile: xr.Dataset,
    lambda_data: float,
    depth_sigma: float | None,
    depth_sigma_auto: bool,
    detail_scale: float,
    cross_tile_weight: float,
    fill_values: str | None,
    iterations: int,
    tol: float,
) -> xr.Dataset:
    """
    Fit monocular depth structure onto the Z map of one point-cloud tile.
    """

    if tile is None or not has_required_bands(tile):
        return tile

    result = tile.copy(deep=True)

    z_map = tile[cst.Z].values.astype(np.float32)
    depth = extract_scalar_layer(tile[cst.EPI_EDGES_DEPTH_MAP])
    tile_id = extract_scalar_layer(tile[cst.EPI_EDGES_TILE_ID])
    if depth is None or tile_id is None:
        return result

    invalidity = (
        tile[cst.EPI_INVALIDITY_MASK].values
        if cst.EPI_INVALIDITY_MASK in tile
        else None
    )
    domain_mask, anchor_mask = build_masks(depth, z_map, tile_id, invalidity)
    if not np.any(domain_mask):
        return result

    if depth_sigma is not None:
        sigma_d = depth_sigma
    elif depth_sigma_auto:
        sigma_d = auto_sigma_per_tile(depth, tile_id, domain_mask)
    else:
        sigma_d = estimate_global_sigma(depth, domain_mask)

    edge_weights = build_edge_weights(
        depth,
        tile_id,
        domain_mask,
        sigma_d,
        cross_tile_weight,
    )

    # Relief / drift decomposition
    relief_map = np.zeros(z_map.shape, dtype=np.float32)
    if detail_scale != 0.0:
        scale_per_tile, global_slope = estimate_depth_to_z_scale(
            depth,
            z_map,
            anchor_mask,
            tile_id,
        )
        slope_map = detail_scale * build_slope_map(
            tile_id, scale_per_tile, global_slope
        )
        relief_map = (slope_map * depth.astype(np.float32)).astype(np.float32)

    residual_map = np.where(anchor_mask, z_map - relief_map, 0.0).astype(
        np.float32
    )

    # Initial guess for the residual solve
    has_z = domain_mask & np.isfinite(z_map)
    residual_known = np.where(has_z, z_map - relief_map, 0.0).astype(np.float32)
    residual_init = smooth_fill(residual_known, has_z)

    fused_residual, _, _ = fuse_anisotropic(
        z_map=residual_map,
        domain_mask=domain_mask,
        anchor_mask=anchor_mask,
        lambda_data=lambda_data,
        edge_weights=edge_weights,
        iterations=iterations,
        tol=tol,
        init=residual_init,
    )
    fused_z = relief_map + fused_residual

    if np.any(anchor_mask):
        z_low, z_high = np.percentile(
            z_map[anchor_mask].astype(np.float64), [0.1, 99.9]
        )
        margin = float(z_high - z_low)
        np.clip(fused_z, z_low - margin, z_high + margin, out=fused_z)

    fit_depth_map = np.full(z_map.shape, np.nan, dtype=np.float32)
    fit_depth_map[domain_mask] = fused_z[domain_mask]
    result["fit_depth_map"] = xr.DataArray(
        fit_depth_map,
        dims=[cst.ROW, cst.COL],
    )

    # - None: keep valid stereo Z unchanged and do not fill any pixels
    # - invalid: fill invalid stereo Z pixels with fitted values
    # - all: replace all Z values with fitted values
    z_filled = z_map.copy()
    if fill_values == "all":
        fill_mask = domain_mask
        z_filled[fill_mask] = fit_depth_map[fill_mask]
    elif fill_values == "invalid":
        fill_mask = domain_mask & ~anchor_mask
        z_filled[fill_mask] = fit_depth_map[fill_mask]
    else:
        fill_mask = np.zeros(z_map.shape, dtype=bool)
    result[cst.Z] = xr.DataArray(
        z_filled,
        dims=[cst.ROW, cst.COL],
    )

    update_depth_to_z_filling(result, fill_mask)

    residual = np.full(z_map.shape, np.nan, dtype=np.float32)
    residual[anchor_mask] = fused_z[anchor_mask] - z_map[anchor_mask]
    weight_map = mean_weight_map(edge_weights, domain_mask, tile_id)

    result["depth_to_z_residual"] = xr.DataArray(
        residual,
        dims=[cst.ROW, cst.COL],
    )
    result["depth_to_z_weight_map"] = xr.DataArray(
        weight_map,
        dims=[cst.ROW, cst.COL],
    )

    return result
