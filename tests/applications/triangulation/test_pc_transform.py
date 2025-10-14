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
# pylint: disable=too-many-lines
"""
Test module for cars/steps/pc_fusion_algo.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

from cars.applications.triangulation import pc_transform

# CARS imports
from cars.core import constants as cst

# CARS Tests imports
from tests.helpers import add_color


@pytest.mark.unit_tests
def test_depth_map_dataset_to_dataframe():
    """
    Tests several configurations of depth_map_dataset_to_dataframe function :
    - test only color
    - test with mask
    - test with color
    - test with coords and colors
    - test with coords (no colors)
    - test exception
    """
    epsg = 4326

    # test only color
    def get_cloud0_ds(with_msk, with_conf_intervals=False):
        """Return local test point cloud 10x10 dataset"""
        row = 10
        col = 10
        x_coord = np.arange(row * col)
        x_coord = x_coord.reshape((row, col))
        y_coord = x_coord + 1
        z_coord = y_coord + 1
        z_inf_coord = z_coord - 0.5
        z_sup_coord = z_coord + 1
        corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
        corr_msk[4, 4] = 0
        if with_msk:
            msk = np.full((row, col), fill_value=0, dtype=np.int16)
            msk[4, 6] = 255

        ds_values = {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINT_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        }

        if with_msk:
            ds_values[cst.EPI_MSK] = ([cst.ROW, cst.COL], msk)
        if with_conf_intervals:
            ds_values[cst.POINT_CLOUD_LAYER_INF_FROM_INTERVALS] = (
                [cst.ROW, cst.COL],
                z_inf_coord,
            )
            ds_values[cst.POINT_CLOUD_LAYER_SUP_FROM_INTERVALS] = (
                [cst.ROW, cst.COL],
                z_sup_coord,
            )

        cloud0 = xr.Dataset(
            ds_values,
            coords={
                cst.ROW: np.array(range(row)),
                cst.COL: np.array(range(col)),
            },
        )
        cloud0.attrs[cst.EPSG] = epsg
        cloud0.attrs[cst.EPI_MARGINS] = [0, 0, 0, 0]
        cloud0.attrs[cst.ROI] = [0, 0, col, row]

        return cloud0

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        get_cloud0_ds(with_msk=False),
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    ref_cloud0 = np.array(
        [
            [39.0, 40.0, 41.0],
            [40.0, 41.0, 42.0],
            [41.0, 42.0, 43.0],
            [42.0, 43.0, 44.0],
            [43.0, 44.0, 45.0],
            [45.0, 46.0, 47.0],
            [46.0, 47.0, 48.0],
            [47.0, 48.0, 49.0],
            [48.0, 49.0, 50.0],
            [49.0, 50.0, 51.0],
            [50.0, 51.0, 52.0],
        ]
    )

    ref_cloud = np.concatenate([ref_cloud0])

    assert np.allclose(cloud.values, ref_cloud)

    # test with mask

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        get_cloud0_ds(with_msk=True),
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    ref_cloud0_with_msk = np.array(
        [
            [39.0, 40.0, 41.0, 0.0],
            [40.0, 41.0, 42.0, 0.0],
            [41.0, 42.0, 43.0, 0.0],
            [42.0, 43.0, 44.0, 0.0],
            [43.0, 44.0, 45.0, 0.0],
            [45.0, 46.0, 47.0, 0.0],
            [46.0, 47.0, 48.0, 255.0],
            [47.0, 48.0, 49.0, 0.0],
            [48.0, 49.0, 50.0, 0.0],
            [49.0, 50.0, 51.0, 0.0],
            [50.0, 51.0, 52.0, 0.0],
        ]
    )

    assert np.allclose(cloud.values, ref_cloud0_with_msk)

    # test with color
    band = 3
    row = 10
    col = 10
    clr0 = np.zeros((band, row, col))
    clr0[0, :, :] = 10
    clr0[1, :, :] = 20
    clr0[2, :, :] = 30
    cloud_with_color0 = add_color(get_cloud0_ds(with_msk=False), clr0)

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        cloud_with_color0,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    ref_clr0 = np.zeros((11, 3))
    ref_clr0[:, 0] = 10
    ref_clr0[:, 1] = 20
    ref_clr0[:, 2] = 30
    ref_cloud_clr0 = np.concatenate([ref_cloud0, ref_clr0], axis=1)
    assert np.allclose(cloud.values, ref_cloud_clr0)

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # test with coords and colors
    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        cloud_with_color0,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=True,
    )

    ref_coords0 = np.array(
        [
            [3.0, 9.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [4.0, 2.0],
            [4.0, 3.0],
            [4.0, 5.0],
            [4.0, 6.0],
            [4.0, 7.0],
            [4.0, 8.0],
            [4.0, 9.0],
            [5.0, 0.0],
        ]
    )
    ref_cloud_clr_coords0 = np.concatenate(
        [ref_cloud0, ref_clr0, ref_coords0], axis=1
    )

    assert np.allclose(cloud, ref_cloud_clr_coords0)

    # test with coords (no colors)

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        get_cloud0_ds(with_msk=False),
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=True,
    )

    ref_cloud_coords0 = np.concatenate([ref_cloud0, ref_coords0], axis=1)

    assert np.allclose(cloud, ref_cloud_coords0)

    # Test with confidence intervals

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        get_cloud0_ds(with_msk=False, with_conf_intervals=True),
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    ref_cloud0 = np.array(
        [
            [39.0, 40.0, 41.0, 41.0 - 0.5, 41.0 + 1],
            [40.0, 41.0, 42.0, 42.0 - 0.5, 42.0 + 1],
            [41.0, 42.0, 43.0, 43.0 - 0.5, 43.0 + 1],
            [42.0, 43.0, 44.0, 44.0 - 0.5, 44.0 + 1],
            [43.0, 44.0, 45.0, 45.0 - 0.5, 45.0 + 1],
            [45.0, 46.0, 47.0, 47.0 - 0.5, 47.0 + 1],
            [46.0, 47.0, 48.0, 48.0 - 0.5, 48.0 + 1],
            [47.0, 48.0, 49.0, 49.0 - 0.5, 49.0 + 1],
            [48.0, 49.0, 50.0, 50.0 - 0.5, 50.0 + 1],
            [49.0, 50.0, 51.0, 51.0 - 0.5, 51.0 + 1],
            [50.0, 51.0, 52.0, 52.0 - 0.5, 52.0 + 1],
        ]
    )

    assert np.allclose(cloud.values, ref_cloud0)

    # Test with colors and confidence intervals
    band = 3
    row = 10
    col = 10
    clr0 = np.zeros((band, row, col))
    clr0[0, :, :] = 10
    clr0[1, :, :] = 20
    clr0[2, :, :] = 30
    cloud_with_color0 = add_color(
        get_cloud0_ds(with_msk=False, with_conf_intervals=True), clr0
    )

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = pc_transform.depth_map_dataset_to_dataframe(
        cloud_with_color0,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    ref_clr0 = np.zeros((11, 3))
    ref_clr0[:, 0] = 10
    ref_clr0[:, 1] = 20
    ref_clr0[:, 2] = 30
    ref_cloud_clr0 = np.concatenate([ref_cloud0, ref_clr0], axis=1)

    assert np.allclose(cloud.values, ref_cloud_clr0)


@pytest.mark.unit_tests
def test_filter_cloud():
    """
    Create fake cloud and test filter_cloud function
    """
    cloud_arr = np.arange(6 * 10)
    cloud_arr = cloud_arr.reshape((10, 6))
    cloud = pandas.DataFrame(
        cloud_arr,
        columns=[
            "x",
            "y",
            "z",
            "coord_epi_geom_i",
            "coord_epi_geom_j",
            "id_im_epi",
        ],
    )

    elt_to_remove = [0, 5]
    __, removed_elt_pos = pc_transform.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=True
    )

    # reference
    pos_arr = np.zeros((len(elt_to_remove), 3), dtype=np.int64)
    for elt_idx, elt_to_remove_item in enumerate(elt_to_remove):
        for last_column_idx in range(3):
            # 3 last elements of each lines
            pos_arr[elt_idx, last_column_idx] = int(
                6 * elt_to_remove_item + 3 + last_column_idx
            )

    ref_removed_elt_pos = pandas.DataFrame(
        pos_arr, columns=["coord_epi_geom_i", "coord_epi_geom_j", "id_im_epi"]
    )

    assert ref_removed_elt_pos.equals(removed_elt_pos)

    # test cases where removed_elt_pos should be None
    __, removed_elt_pos = pc_transform.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=False
    )

    assert removed_elt_pos is None

    cloud_arr = np.arange(3 * 10)
    cloud_arr = cloud_arr.reshape((10, 3))

    cloud = pandas.DataFrame(cloud_arr, columns=["x", "y", "z"])
    __, removed_elt_pos = pc_transform.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=True
    )

    assert removed_elt_pos is None
