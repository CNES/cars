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
Point cloud refinement algorithms.
"""

# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from cars.core import constants as cst
from cars.core import projection

_REQUIRED_SCALAR_LAYERS = (cst.X, cst.Y, cst.Z, cst.EPI_EDGES_TILE_ID)


def has_required_bands(tile: xr.Dataset) -> bool:
    """
    Check required data fields for normals-guided refinement.
    """

    if tile is None:
        return False

    if not all(key in tile for key in _REQUIRED_SCALAR_LAYERS):
        return False

    if cst.EPI_EDGES_NORMALS not in tile:
        return False

    normals = tile[cst.EPI_EDGES_NORMALS]
    if cst.BAND_EDGES_NORMALS not in normals.dims:
        return False

    return int(normals.sizes[cst.BAND_EDGES_NORMALS]) == 3


def normalize_normals(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize normal vectors and return validity mask.
    """

    nrm = np.linalg.norm(normals, axis=0)
    valid = np.all(np.isfinite(normals), axis=0) & (nrm > 0.0)
    out = np.zeros_like(normals, dtype=np.float64)
    out[:, valid] = normals[:, valid] / nrm[valid]
    return out, valid


def estimate_normals_from_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate surface normals from the XYZ grid using central differences.
    """

    points = np.stack([x, y, z], axis=-1)

    dx = np.zeros_like(points, dtype=np.float64)
    dy = np.zeros_like(points, dtype=np.float64)
    dx[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
    dy[1:-1, :, :] = points[2:, :, :] - points[:-2, :, :]

    estimated = np.cross(dx, dy)
    estimated_norm = np.linalg.norm(estimated, axis=-1)

    left_right = np.zeros_like(active, dtype=bool)
    left_right[:, 1:-1] = active[:, :-2] & active[:, 2:]
    up_down = np.zeros_like(active, dtype=bool)
    up_down[1:-1, :] = active[:-2, :] & active[2:, :]

    valid = active & left_right & up_down
    valid &= np.all(np.isfinite(estimated), axis=-1)
    valid &= estimated_norm > 0.0

    normals = np.zeros_like(points, dtype=np.float64)
    normals[valid] = estimated[valid] / estimated_norm[valid, np.newaxis]
    return np.moveaxis(normals, -1, 0), valid


def estimate_global_normal_rotation(
    target_normals: np.ndarray,
    estimated_normals: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Estimate a global rotation aligning target normals to estimated XYZ normals.
    """

    if np.count_nonzero(valid) < 3:
        return np.eye(3, dtype=np.float64)

    target_vectors = target_normals[:, valid].T
    reference_vectors = estimated_normals[:, valid].T

    target_norm = np.linalg.norm(target_vectors, axis=1)
    reference_norm = np.linalg.norm(reference_vectors, axis=1)
    valid_vectors = (target_norm > 0.0) & (reference_norm > 0.0)
    if np.count_nonzero(valid_vectors) < 3:
        return np.eye(3, dtype=np.float64)

    target_vectors = target_vectors[valid_vectors]
    reference_vectors = reference_vectors[valid_vectors]

    target_vectors = target_vectors / np.linalg.norm(
        target_vectors, axis=1, keepdims=True
    )
    reference_vectors = reference_vectors / np.linalg.norm(
        reference_vectors, axis=1, keepdims=True
    )

    alignment = np.sum(target_vectors * reference_vectors, axis=1)
    if float(np.median(alignment)) < 0.0:
        target_vectors = -target_vectors

    covariance = target_vectors.T @ reference_vectors
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    return rotation


def rotate_normals(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 rotation to a normal field.
    """

    flat_normals = normals.reshape(3, -1)
    rotated = rotation @ flat_normals
    return rotated.reshape(normals.shape)


def get_depth_to_z_fill_mask(tile: xr.Dataset) -> np.ndarray:
    """
    Return pixels filled by the depth-to-Z fusion.
    """

    shape = tile[cst.Z].shape
    if cst.EPI_FILLING not in tile or cst.BAND_FILLING not in tile.coords:
        return np.zeros(shape, dtype=bool)

    band_values = [str(v) for v in tile.coords[cst.BAND_FILLING].values]
    if cst.FILLING_DEPTH_TO_Z not in band_values:
        return np.zeros(shape, dtype=bool)

    band_index = band_values.index(cst.FILLING_DEPTH_TO_Z)
    return tile[cst.EPI_FILLING].values[band_index].astype(bool)


def build_valid_mask(
    tile: xr.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normals_valid: np.ndarray,
) -> np.ndarray:
    """
    Build refinement validity mask from finite values and masks.
    """

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & normals_valid

    if cst.EPI_INVALIDITY_MASK in tile:
        invalidity = tile[cst.EPI_INVALIDITY_MASK].values

        if invalidity.ndim == 3:
            invalidity_mask = invalidity != 0
            if (
                cst.BAND_INVALIDITY_MASK in tile[cst.EPI_INVALIDITY_MASK].dims
                and cst.BAND_INVALIDITY_MASK in tile.coords
            ):
                band_values = [
                    str(v) for v in tile.coords[cst.BAND_INVALIDITY_MASK].values
                ]
                if "edge" in band_values:
                    edge_band_idx = band_values.index("edge")
                    invalidity_mask[edge_band_idx, :, :] = False
            invalid_any = np.any(invalidity_mask, axis=0)
        else:
            invalid_any = invalidity != 0

        # Keep pixels filled by depth-to-Z fusion as valid
        invalid_any &= ~get_depth_to_z_fill_mask(tile)

        valid &= ~invalid_any

    return valid


def get_metric_xyz(
    tile: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get point coordinates in a metric frame for refinement computations.
    """

    x = tile[cst.X].values.astype(np.float64)
    y = tile[cst.Y].values.astype(np.float64)
    z = tile[cst.Z].values.astype(np.float64)

    epsg = tile.attrs.get(cst.EPSG)
    if epsg is None:
        return x, y, z

    if not CRS.from_epsg(int(epsg)).is_geographic:
        return x, y, z

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite_mask):
        return x, y, z

    x_metric = np.full(x.shape, np.nan, dtype=np.float64)
    y_metric = np.full(y.shape, np.nan, dtype=np.float64)
    z_metric = np.full(z.shape, np.nan, dtype=np.float64)

    xyz_metric = projection.point_cloud_conversion(
        np.stack(
            [
                x[finite_mask],
                y[finite_mask],
                z[finite_mask],
            ],
            axis=1,
        ),
        int(epsg),
        4978,
    )
    x_metric[finite_mask] = xyz_metric[:, 0]
    y_metric[finite_mask] = xyz_metric[:, 1]
    z_metric[finite_mask] = xyz_metric[:, 2]

    return x_metric, y_metric, z_metric


def convert_metric_xyz_to_output_coords(
    tile: xr.Dataset,
    x_metric: np.ndarray,
    y_metric: np.ndarray,
    z_metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert metric refined coordinates back to the original tile CRS.
    """

    epsg = tile.attrs.get(cst.EPSG)
    if epsg is None:
        return x_metric, y_metric, z_metric

    if not CRS.from_epsg(int(epsg)).is_geographic:
        return x_metric, y_metric, z_metric

    finite_mask = (
        np.isfinite(x_metric) & np.isfinite(y_metric) & np.isfinite(z_metric)
    )
    if not np.any(finite_mask):
        return x_metric, y_metric, z_metric

    x_out = np.full(x_metric.shape, np.nan, dtype=np.float64)
    y_out = np.full(y_metric.shape, np.nan, dtype=np.float64)
    z_out = np.full(z_metric.shape, np.nan, dtype=np.float64)

    xyz_out = projection.point_cloud_conversion(
        np.stack(
            [
                x_metric[finite_mask],
                y_metric[finite_mask],
                z_metric[finite_mask],
            ],
            axis=1,
        ),
        4978,
        int(epsg),
    )
    x_out[finite_mask] = xyz_out[:, 0]
    y_out[finite_mask] = xyz_out[:, 1]
    z_out[finite_mask] = xyz_out[:, 2]

    return x_out, y_out, z_out


def build_index_map(active: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Build sparse index map for active pixels.
    """

    idx = -np.ones(active.shape, dtype=np.int64)
    ids = np.flatnonzero(active.ravel())
    idx.ravel()[ids] = np.arange(ids.size, dtype=np.int64)
    return idx, int(ids.size)


def add_laplacian_equations(
    rows,
    cols,
    vals,
    rhs,
    eq_id,
    active,
    index_map,
    w_smooth,
):
    """
    Add smoothness equations using a 4-neighbor Laplacian.
    """

    ws = np.sqrt(max(w_smooth, 0.0))
    if ws == 0.0:
        return eq_id

    up = np.zeros_like(active)
    up[1:, :] = active[:-1, :]
    down = np.zeros_like(active)
    down[:-1, :] = active[1:, :]
    left = np.zeros_like(active)
    left[:, 1:] = active[:, :-1]
    right = np.zeros_like(active)
    right[:, :-1] = active[:, 1:]

    neigh_count = up.astype(np.int8)
    neigh_count += down.astype(np.int8)
    neigh_count += left.astype(np.int8)
    neigh_count += right.astype(np.int8)

    center_mask = active & (neigh_count > 0)
    n_eq = int(np.count_nonzero(center_mask))
    if n_eq == 0:
        return eq_id

    eq_map = -np.ones(active.shape, dtype=np.int64)
    eq_map[center_mask] = np.arange(eq_id, eq_id + n_eq, dtype=np.int64)

    center_eq = eq_map[center_mask]
    center_idx = index_map[center_mask]
    center_val = ws * neigh_count[center_mask].astype(np.float64)

    rows.append(center_eq)
    cols.append(center_idx)
    vals.append(center_val)

    up_idx = np.full(active.shape, -1, dtype=np.int64)
    up_idx[1:, :] = index_map[:-1, :]
    down_idx = np.full(active.shape, -1, dtype=np.int64)
    down_idx[:-1, :] = index_map[1:, :]
    left_idx = np.full(active.shape, -1, dtype=np.int64)
    left_idx[:, 1:] = index_map[:, :-1]
    right_idx = np.full(active.shape, -1, dtype=np.int64)
    right_idx[:, :-1] = index_map[:, 1:]

    for neigh_mask, neigh_idx in (
        (center_mask & up, up_idx),
        (center_mask & down, down_idx),
        (center_mask & left, left_idx),
        (center_mask & right, right_idx),
    ):
        if not np.any(neigh_mask):
            continue
        rows.append(eq_map[neigh_mask])
        cols.append(neigh_idx[neigh_mask])
        vals.append(np.full(np.count_nonzero(neigh_mask), -ws))

    rhs.append(np.zeros(n_eq, dtype=np.float64))
    eq_id += n_eq

    return eq_id


def add_data_attachment(
    rows, cols, vals, rhs, eq_id, active, index_map, w_data
):
    """
    Add Tikhonov data attachment equations.
    """

    wd = np.sqrt(max(w_data, 0.0))
    if wd == 0.0:
        return eq_id

    flat_idx = index_map[active]
    n_eq = flat_idx.size
    if n_eq == 0:
        return eq_id

    rows.append(np.arange(eq_id, eq_id + n_eq, dtype=np.int64))
    cols.append(flat_idx.astype(np.int64))
    vals.append(np.full(n_eq, wd, dtype=np.float64))
    rhs.append(np.zeros(n_eq, dtype=np.float64))
    eq_id += n_eq

    return eq_id


def add_guidance_equations(
    rows,
    cols,
    vals,
    rhs,
    eq_id,
    active,
    index_map,
    xyz,
    normals,
    update_dirs,
    w_guidance,
):
    """
    Add normals-guidance equations for horizontal and vertical neighbors.
    """

    wg = np.sqrt(max(w_guidance, 0.0))
    if wg == 0.0:
        return eq_id

    x, y, z = xyz

    def append_guidance_pairs(
        edge_mask,
        normal_center,
        update_center,
        update_neighbor,
        point_center,
        point_neighbor,
        center_idx,
        neighbor_idx,
        local_eq_id,
    ):
        """
        Add guidance equations to the system for a given set of neighbor pairs.
        """
        mask = edge_mask
        if not np.any(mask):
            return local_eq_id

        normal_vec = normal_center[:, mask]
        update_center_vec = update_center[:, mask]
        update_neighbor_vec = update_neighbor[:, mask]

        coeff_center = -np.sum(normal_vec * update_center_vec, axis=0)
        coeff_neighbor = np.sum(normal_vec * update_neighbor_vec, axis=0)
        nz = (coeff_center != 0.0) | (coeff_neighbor != 0.0)
        if not np.any(nz):
            return local_eq_id

        point_center_vec = point_center[:, mask][:, nz]
        point_neighbor_vec = point_neighbor[:, mask][:, nz]
        normal_vec = normal_vec[:, nz]
        coeff_center = coeff_center[nz]
        coeff_neighbor = coeff_neighbor[nz]
        center_ids = center_idx[mask][nz]
        neighbor_ids = neighbor_idx[mask][nz]

        rhs_values = -np.sum(
            normal_vec * (point_neighbor_vec - point_center_vec), axis=0
        )
        n_eq = rhs_values.size
        eq_ids = np.arange(local_eq_id, local_eq_id + n_eq, dtype=np.int64)

        rows_local = np.empty(2 * n_eq, dtype=np.int64)
        cols_local = np.empty(2 * n_eq, dtype=np.int64)
        vals_local = np.empty(2 * n_eq, dtype=np.float64)
        rows_local[0::2] = eq_ids
        rows_local[1::2] = eq_ids
        cols_local[0::2] = center_ids
        cols_local[1::2] = neighbor_ids
        vals_local[0::2] = wg * coeff_center
        vals_local[1::2] = wg * coeff_neighbor

        rows.append(rows_local)
        cols.append(cols_local)
        vals.append(vals_local)
        rhs.append(wg * rhs_values)

        return local_eq_id + n_eq

    horizontal_mask = active[:, :-1] & active[:, 1:]
    eq_id = append_guidance_pairs(
        horizontal_mask,
        normals[:, :, :-1],
        update_dirs[:, :, :-1],
        update_dirs[:, :, 1:],
        np.stack([x[:, :-1], y[:, :-1], z[:, :-1]], axis=0),
        np.stack([x[:, 1:], y[:, 1:], z[:, 1:]], axis=0),
        index_map[:, :-1],
        index_map[:, 1:],
        eq_id,
    )

    vertical_mask = active[:-1, :] & active[1:, :]
    eq_id = append_guidance_pairs(
        vertical_mask,
        normals[:, :-1, :],
        update_dirs[:, :-1, :],
        update_dirs[:, 1:, :],
        np.stack([x[:-1, :], y[:-1, :], z[:-1, :]], axis=0),
        np.stack([x[1:, :], y[1:, :], z[1:, :]], axis=0),
        index_map[:-1, :],
        index_map[1:, :],
        eq_id,
    )

    return eq_id


def solve_displacement(
    active,
    xyz,
    normals,
    update_dirs,
    w_guidance,
    w_smooth,
    w_data,
    damp,
    max_iter,
):
    """
    Solve displacement field with sparse least squares.
    """

    index_map, n_unknowns = build_index_map(active)
    if n_unknowns == 0:
        return np.zeros(active.shape, dtype=np.float64)

    rows = []
    cols = []
    vals = []
    rhs = []
    eq_id = 0

    eq_id = add_guidance_equations(
        rows,
        cols,
        vals,
        rhs,
        eq_id,
        active,
        index_map,
        xyz,
        normals,
        update_dirs,
        w_guidance,
    )
    eq_id = add_laplacian_equations(
        rows,
        cols,
        vals,
        rhs,
        eq_id,
        active,
        index_map,
        w_smooth,
    )
    eq_id = add_data_attachment(
        rows,
        cols,
        vals,
        rhs,
        eq_id,
        active,
        index_map,
        w_data,
    )

    if eq_id == 0:
        return np.zeros(active.shape, dtype=np.float64)

    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    vals_arr = np.concatenate(vals)
    rhs_vec = np.concatenate(rhs).astype(np.float64)

    # Scale columns to improve LSQR numerical stability.
    col_norm = np.sqrt(
        np.bincount(cols_arr, weights=vals_arr * vals_arr, minlength=n_unknowns)
    )
    col_norm[col_norm == 0] = 1.0
    vals_scaled = vals_arr / col_norm[cols_arr]
    matrix_scaled = csr_matrix(
        (vals_scaled, (rows_arr, cols_arr)), shape=(eq_id, n_unknowns)
    )

    solution = lsqr(matrix_scaled, rhs_vec, damp=damp, iter_lim=max_iter)[0]
    displacement = solution / col_norm

    disp_img = np.zeros(active.shape, dtype=np.float64)
    active_ids = np.flatnonzero(active.ravel())
    disp_img.ravel()[active_ids] = displacement
    return disp_img


def refine_point_cloud_tile(  # noqa: C901
    tile: xr.Dataset,
    w_guidance: float,
    w_smooth: float,
    w_data: float,
    max_displacement: float,
    update_xy: bool,
    use_metric: bool,
    damp: float,
    max_iter: int,
) -> xr.Dataset:
    """
    Refine one point-cloud tile using normals-guided displacement.
    """

    if tile is None or not has_required_bands(tile):
        return tile

    result = tile.copy(deep=True)

    x = tile[cst.X].values.astype(np.float64)
    y = tile[cst.Y].values.astype(np.float64)
    z = tile[cst.Z].values.astype(np.float64)

    if use_metric:
        x_solver, y_solver, z_solver = get_metric_xyz(tile)
    else:
        x_solver, y_solver, z_solver = x, y, z

    normals = tile[cst.EPI_EDGES_NORMALS].values.astype(np.float64)
    if normals.shape[0] != 3 or normals.shape[1:] != x.shape:
        return result

    normals_unit, normals_valid = normalize_normals(normals)

    valid = build_valid_mask(
        tile,
        x,
        y,
        z,
        normals_valid,
    )

    if not np.any(valid):
        return result

    estimated_normals, estimated_valid = estimate_normals_from_xyz(
        x_solver,
        y_solver,
        z_solver,
        valid,
    )
    rotation_mask = valid & estimated_valid
    if np.any(rotation_mask):
        rotation = estimate_global_normal_rotation(
            normals_unit,
            estimated_normals,
            rotation_mask,
        )
        normals_unit = rotate_normals(normals_unit, rotation)

    if update_xy:
        update_dirs = normals_unit.copy()
    else:
        update_dirs = np.zeros_like(normals_unit)
        update_dirs[2, :, :] = normals_unit[2, :, :]

    xyz = np.stack([x_solver, y_solver, z_solver], axis=0)
    displacement = solve_displacement(
        active=valid,
        xyz=xyz,
        normals=normals_unit,
        update_dirs=update_dirs,
        w_guidance=w_guidance,
        w_smooth=w_smooth,
        w_data=w_data,
        damp=damp,
        max_iter=max_iter,
    )

    displacement = np.clip(
        displacement,
        -max_displacement,
        max_displacement,
    )

    x_refined_solver = x_solver.copy()
    y_refined_solver = y_solver.copy()
    z_refined_solver = z_solver.copy()

    if update_xy:
        x_refined_solver[valid] += displacement[valid] * normals_unit[0, valid]
        y_refined_solver[valid] += displacement[valid] * normals_unit[1, valid]
    z_refined_solver[valid] += displacement[valid] * normals_unit[2, valid]

    if use_metric:
        x_refined, y_refined, z_refined = convert_metric_xyz_to_output_coords(
            tile,
            x_refined_solver,
            y_refined_solver,
            z_refined_solver,
        )
    else:
        x_refined, y_refined, z_refined = (
            x_refined_solver,
            y_refined_solver,
            z_refined_solver,
        )

    dx = x_refined - x
    dy = y_refined - y
    dz = z_refined - z

    result[cst.X].values = x_refined.astype(result[cst.X].dtype)
    result[cst.Y].values = y_refined.astype(result[cst.Y].dtype)
    result[cst.Z].values = z_refined.astype(result[cst.Z].dtype)

    displacement = np.stack([dx, dy, dz], axis=0).astype(np.float64)

    result.coords["band_displacement"] = np.array(["dx", "dy", "dz"])

    result["displacement"] = xr.DataArray(
        displacement,
        dims=["band_displacement", cst.ROW, cst.COL],
    )

    return result
