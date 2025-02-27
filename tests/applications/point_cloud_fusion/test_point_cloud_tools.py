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
Test module for cars/steps/point_cloud_tools.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

from cars.applications.point_cloud_fusion import point_cloud_tools

# CARS imports
from cars.core import constants as cst

# CARS Tests imports
from tests.helpers import add_color, assert_same_datasets


@pytest.mark.unit_tests
def test_create_combined_dense_cloud():
    """
    Tests several configurations of create_combined_cloud function :
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

        return cloud0

    row = 10
    col = 10
    x_coord = np.full((row, col), fill_value=0, dtype=np.float64)
    y_coord = np.full((row, col), fill_value=1, dtype=np.float64)
    z_coord = np.full((row, col), fill_value=2, dtype=np.float64)
    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[6, 6] = 0

    cloud1 = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINT_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        },
        coords={cst.ROW: np.array(range(row)), cst.COL: np.array(range(col))},
    )
    cloud1.attrs[cst.EPSG] = epsg

    row = 5
    col = 5
    x_coord = np.full((row, col), fill_value=45, dtype=np.float64)
    y_coord = np.full((row, col), fill_value=45, dtype=np.float64)
    z_coord = np.full((row, col), fill_value=50, dtype=np.float64)
    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[2, 2] = 0

    cloud2 = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINT_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        },
        coords={cst.ROW: np.array(range(row)), cst.COL: np.array(range(col))},
    )
    cloud2.attrs[cst.EPSG] = epsg

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]
    cloud_id = list(range(3))

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
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
            [0.0, 39.0, 40.0, 41.0],
            [0.0, 40.0, 41.0, 42.0],
            [0.0, 41.0, 42.0, 43.0],
            [0.0, 42.0, 43.0, 44.0],
            [0.0, 43.0, 44.0, 45.0],
            [0.0, 45.0, 46.0, 47.0],
            [0.0, 46.0, 47.0, 48.0],
            [0.0, 47.0, 48.0, 49.0],
            [0.0, 48.0, 49.0, 50.0],
            [0.0, 49.0, 50.0, 51.0],
            [0.0, 50.0, 51.0, 52.0],
        ]
    )

    ref_cloud2 = np.zeros((row * col, 4))
    ref_cloud2[:, 0] = 2
    ref_cloud2[:, 1] = 45
    ref_cloud2[:, 2] = 45
    ref_cloud2[:, 3] = 50

    ref_cloud2 = np.delete(ref_cloud2, 2 * col + 2, 0)

    ref_cloud = np.concatenate([ref_cloud0, ref_cloud2])

    print(cloud.values)
    print(ref_cloud)
    assert np.allclose(cloud.values, ref_cloud)

    # test with mask
    cloud_list = [get_cloud0_ds(with_msk=True), cloud2]
    cloud_id = [0, 2]

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
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
            [0.0, 39.0, 40.0, 41.0, 0.0],
            [0.0, 40.0, 41.0, 42.0, 0.0],
            [0.0, 41.0, 42.0, 43.0, 0.0],
            [0.0, 42.0, 43.0, 44.0, 0.0],
            [0.0, 43.0, 44.0, 45.0, 0.0],
            [0.0, 45.0, 46.0, 47.0, 0.0],
            [0.0, 46.0, 47.0, 48.0, 255.0],
            [0.0, 47.0, 48.0, 49.0, 0.0],
            [0.0, 48.0, 49.0, 50.0, 0.0],
            [0.0, 49.0, 50.0, 51.0, 0.0],
            [0.0, 50.0, 51.0, 52.0, 0.0],
        ]
    )

    ref_cloud = np.concatenate(
        [
            ref_cloud0_with_msk,
            np.concatenate([ref_cloud2, np.zeros((row * col - 1, 1))], axis=1),
        ]
    )
    assert np.allclose(cloud.values, ref_cloud)

    # test with color
    band = 3
    row = 10
    col = 10
    clr0 = np.zeros((band, row, col))
    clr0[0, :, :] = 10
    clr0[1, :, :] = 20
    clr0[2, :, :] = 30
    cloud_with_color0 = add_color(get_cloud0_ds(with_msk=False), clr0)

    clr1 = np.full((band, row, col), fill_value=20)
    cloud_with_color1 = add_color(cloud1, clr1)

    row = 5
    col = 5
    clr2 = np.zeros((band, row, col))
    clr2[0, :, :] = np.arange(row * col).reshape((row, col))
    clr2[1, :, :] = clr2[0, :, :] + 1
    clr2[2, :, :] = clr2[1, :, :] + 1
    cloud_with_color2 = add_color(cloud2, clr2)

    cloud_list_with_color = [
        cloud_with_color0,
        cloud_with_color1,
        cloud_with_color2,
    ]
    cloud_id = list(range(3))

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list_with_color,
        cloud_id,
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

    ref_clr2 = np.zeros((row * col, 3))
    ref_clr2[:, 0] = np.arange(row * col)
    ref_clr2[:, 1] = ref_clr2[:, 0] + 1
    ref_clr2[:, 2] = ref_clr2[:, 1] + 1
    ref_clr2 = np.delete(ref_clr2, 2 * col + 2, 0)
    ref_cloud_clr2 = np.concatenate([ref_cloud2, ref_clr2], axis=1)

    ref_cloud_clr = np.concatenate([ref_cloud_clr0, ref_cloud_clr2])

    assert np.allclose(cloud.values, ref_cloud_clr)

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # test with coords and colors
    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list_with_color,
        cloud_id,
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
            [3.0, 9.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 5.0, 0.0],
            [4.0, 6.0, 0.0],
            [4.0, 7.0, 0.0],
            [4.0, 8.0, 0.0],
            [4.0, 9.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    ref_cloud_clr_coords0 = np.concatenate(
        [ref_cloud0, ref_clr0, ref_coords0], axis=1
    )

    ref_coords2 = np.zeros((row * col, 3))
    ref_coords2[:, 2] = 2
    for i in range(row):
        for j in range(col):
            ref_coords2[i * col + j, 0] = i
            ref_coords2[i * col + j, 1] = j
    ref_coords2 = np.delete(ref_coords2, 2 * col + 2, 0)
    ref_cloud_clr_coords2 = np.concatenate(
        [ref_cloud2, ref_clr2, ref_coords2], axis=1
    )

    ref_cloud_clr_coords = np.concatenate(
        [ref_cloud_clr_coords0, ref_cloud_clr_coords2]
    )
    assert np.allclose(cloud, ref_cloud_clr_coords)

    # test with coords (no colors)

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=True,
    )

    ref_cloud_coords0 = np.concatenate([ref_cloud0, ref_coords0], axis=1)
    ref_cloud_coords2 = np.concatenate([ref_cloud2, ref_coords2], axis=1)
    ref_cloud_coords = np.concatenate([ref_cloud_coords0, ref_cloud_coords2])

    assert np.allclose(cloud, ref_cloud_coords)

    # Test with confidence intervals
    row = 10
    col = 10
    x_coord = np.full((row, col), fill_value=0, dtype=np.float64)
    y_coord = np.full((row, col), fill_value=1, dtype=np.float64)
    z_coord = np.full((row, col), fill_value=2, dtype=np.float64)
    z_inf_coord = np.full((row, col), fill_value=0.5, dtype=np.float64)
    z_sup_coord = np.full((row, col), fill_value=2.5, dtype=np.float64)

    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[6, 6] = 0

    cloud1 = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINT_CLOUD_LAYER_INF_FROM_INTERVALS: (
                [cst.ROW, cst.COL],
                z_inf_coord,
            ),
            cst.POINT_CLOUD_LAYER_SUP_FROM_INTERVALS: (
                [cst.ROW, cst.COL],
                z_sup_coord,
            ),
            cst.POINT_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        },
        coords={cst.ROW: np.array(range(row)), cst.COL: np.array(range(col))},
    )
    cloud1.attrs[cst.EPSG] = epsg

    row = 5
    col = 5
    x_coord = np.full((row, col), fill_value=45, dtype=np.float64)
    y_coord = np.full((row, col), fill_value=45, dtype=np.float64)
    z_coord = np.full((row, col), fill_value=50, dtype=np.float64)
    z_inf_coord = np.full((row, col), fill_value=49.5, dtype=np.float64)
    z_sup_coord = np.full((row, col), fill_value=51, dtype=np.float64)
    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[2, 2] = 0

    cloud2 = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINT_CLOUD_LAYER_INF_FROM_INTERVALS: (
                [cst.ROW, cst.COL],
                z_inf_coord,
            ),
            cst.POINT_CLOUD_LAYER_SUP_FROM_INTERVALS: (
                [cst.ROW, cst.COL],
                z_sup_coord,
            ),
            cst.POINT_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        },
        coords={cst.ROW: np.array(range(row)), cst.COL: np.array(range(col))},
    )
    cloud2.attrs[cst.EPSG] = epsg

    cloud_list = [
        get_cloud0_ds(with_msk=False, with_conf_intervals=True),
        cloud1,
        cloud2,
    ]
    cloud_id = list(range(3))

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
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
            [0.0, 39.0, 40.0, 41.0, 41.0 - 0.5, 41.0 + 1],
            [0.0, 40.0, 41.0, 42.0, 42.0 - 0.5, 42.0 + 1],
            [0.0, 41.0, 42.0, 43.0, 43.0 - 0.5, 43.0 + 1],
            [0.0, 42.0, 43.0, 44.0, 44.0 - 0.5, 44.0 + 1],
            [0.0, 43.0, 44.0, 45.0, 45.0 - 0.5, 45.0 + 1],
            [0.0, 45.0, 46.0, 47.0, 47.0 - 0.5, 47.0 + 1],
            [0.0, 46.0, 47.0, 48.0, 48.0 - 0.5, 48.0 + 1],
            [0.0, 47.0, 48.0, 49.0, 49.0 - 0.5, 49.0 + 1],
            [0.0, 48.0, 49.0, 50.0, 50.0 - 0.5, 50.0 + 1],
            [0.0, 49.0, 50.0, 51.0, 51.0 - 0.5, 51.0 + 1],
            [0.0, 50.0, 51.0, 52.0, 52.0 - 0.5, 52.0 + 1],
        ]
    )

    ref_cloud2 = np.zeros((row * col, 6))
    ref_cloud2[:, 0] = 2
    ref_cloud2[:, 1] = 45
    ref_cloud2[:, 2] = 45
    ref_cloud2[:, 3] = 50
    ref_cloud2[:, 4] = 49.5
    ref_cloud2[:, 5] = 51

    ref_cloud2 = np.delete(ref_cloud2, 2 * col + 2, 0)

    ref_cloud = np.concatenate([ref_cloud0, ref_cloud2])
    assert np.allclose(cloud.values, ref_cloud)

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

    clr1 = np.full((band, row, col), fill_value=20)
    cloud_with_color1 = add_color(cloud1, clr1)

    row = 5
    col = 5
    clr2 = np.zeros((band, row, col))
    clr2[0, :, :] = np.arange(row * col).reshape((row, col))
    clr2[1, :, :] = clr2[0, :, :] + 1
    clr2[2, :, :] = clr2[1, :, :] + 1
    cloud_with_color2 = add_color(cloud2, clr2)

    cloud_list_with_color = [
        cloud_with_color0,
        cloud_with_color1,
        cloud_with_color2,
    ]
    cloud_id = list(range(3))

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list_with_color,
        cloud_id,
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

    ref_clr2 = np.zeros((row * col, 3))
    ref_clr2[:, 0] = np.arange(row * col)
    ref_clr2[:, 1] = ref_clr2[:, 0] + 1
    ref_clr2[:, 2] = ref_clr2[:, 1] + 1
    ref_clr2 = np.delete(ref_clr2, 2 * col + 2, 0)
    ref_cloud_clr2 = np.concatenate([ref_cloud2, ref_clr2], axis=1)

    ref_cloud_clr = np.concatenate([ref_cloud_clr0, ref_cloud_clr2])

    assert np.allclose(cloud.values, ref_cloud_clr)


@pytest.mark.unit_tests
def test_create_combined_sparse_cloud():
    """
    Tests several configurations of create_combined_cloud function :
    - test with mask
    - test with coords
    """
    epsg = 4326

    # test only color
    def get_cloud0_ds(with_msk):
        """Return local test point cloud 1x10 dataset"""
        number = 5
        x_coord = np.arange(number) + 40
        y_coord = x_coord + 1
        z_coord = y_coord + 1
        corr_msk = np.full(number, fill_value=255, dtype=np.int16)
        corr_msk[4] = 0
        msk = corr_msk
        if with_msk:
            msk = np.full(number, fill_value=0, dtype=np.int16)
            msk[3] = 255
            msk[4] = 255

        point_cloud_index = [cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_CORR_MSK]
        point_cloud_array = np.zeros((number, 4), dtype=np.float64)
        point_cloud_array[:, 0] = np.array(x_coord)
        point_cloud_array[:, 1] = np.array(y_coord)
        point_cloud_array[:, 2] = np.array(z_coord)
        point_cloud_array[:, 3] = msk
        cloud0 = pandas.DataFrame(point_cloud_array, columns=point_cloud_index)
        cloud0.attrs[cst.EPSG] = epsg
        return cloud0

    number = 7

    point_cloud_index = [cst.X, cst.Y, cst.Z, cst.POINT_CLOUD_CORR_MSK]
    point_cloud_array1 = np.zeros((number, 4), dtype=np.float64)
    point_cloud_array1[:, 0] = np.full(number, fill_value=0, dtype=np.float64)
    point_cloud_array1[:, 1] = np.full(number, fill_value=1, dtype=np.float64)
    point_cloud_array1[:, 2] = np.full(number, fill_value=2, dtype=np.float64)
    point_cloud_array1[:, 3] = np.full(number, fill_value=255, dtype=np.int16)
    point_cloud_array1[6, 3] = 0
    cloud1 = pandas.DataFrame(point_cloud_array1, columns=point_cloud_index)
    cloud1.attrs[cst.EPSG] = epsg
    number = 5
    point_cloud_array2 = np.zeros((number, 4), dtype=np.float64)
    point_cloud_array2[:, 0] = np.full(number, fill_value=45, dtype=np.float64)
    point_cloud_array2[:, 1] = np.full(number, fill_value=45, dtype=np.float64)
    point_cloud_array2[:, 2] = np.full(number, fill_value=50, dtype=np.float64)
    point_cloud_array2[:, 3] = np.full(number, fill_value=255, dtype=np.int16)
    point_cloud_array2[2, 3] = 0
    cloud2 = pandas.DataFrame(point_cloud_array2, columns=point_cloud_index)
    cloud2.attrs[cst.EPSG] = epsg
    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]
    cloud_id = list(range(3))

    # Compute margin
    on_ground_margin = 1
    resolution = 0.5
    radius = 1
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )
    # check against the reference value
    ref_point_cloud_index = [cst.X, cst.Y, cst.Z]
    number = 9
    ref_array2 = np.zeros((number, 3), dtype=np.float64)

    ref_x = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 45.0, 45.0, 45.0]
    ref_y = [41.0, 42.0, 43.0, 44.0, 45.0, 45.0, 45.0, 45.0, 45.0]
    ref_z = [42.0, 43.0, 44.0, 45.0, 46.0, 50.0, 50.0, 50.0, 50.0]
    ref_array2[:, 0] = np.asarray(ref_x, dtype=np.float64)
    ref_array2[:, 1] = np.asarray(ref_y, dtype=np.float64)
    ref_array2[:, 2] = np.asarray(ref_z, dtype=np.float64)

    ref_cloud = pandas.DataFrame(ref_array2, columns=ref_point_cloud_index)

    for key in ref_point_cloud_index:
        assert np.allclose(cloud[key].values, ref_cloud[key].values)

    # # test with coords
    # former :  xstart=40.0, ystart=50.0, xsize=20, ysize=25
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=True,
    )

    ref_point_cloud_index = [
        cst.X,
        cst.Y,
        cst.Z,
        cst.POINT_CLOUD_COORD_EPI_GEOM_I,
    ]
    number = 9
    ref_array2 = np.zeros((number, 4), dtype=np.float64)

    ref_x = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 45.0, 45.0, 45.0]
    ref_y = [41.0, 42.0, 43.0, 44.0, 45.0, 45.0, 45.0, 45.0, 45.0]
    ref_z = [42.0, 43.0, 44.0, 45.0, 46.0, 50.0, 50.0, 50.0, 50.0]
    ref_coord = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 3.0, 4.0]
    ref_array2[:, 0] = np.asarray(ref_x, dtype=np.float64)
    ref_array2[:, 1] = np.asarray(ref_y, dtype=np.float64)
    ref_array2[:, 2] = np.asarray(ref_z, dtype=np.float64)
    ref_array2[:, 3] = np.asarray(ref_coord, dtype=np.float64)
    ref_cloud2 = pandas.DataFrame(ref_array2, columns=ref_point_cloud_index)

    for key in ref_point_cloud_index:
        assert np.allclose(cloud[key].values, ref_cloud2[key].values)

    # test with msk
    cloud_list = [get_cloud0_ds(with_msk=True), cloud1, cloud2]
    cloud, epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        cloud_id,
        epsg,
        xmin=40.0,
        ymin=37.0,
        xmax=50.5,
        ymax=50,
        margin=used_margin,
        with_coords=False,
    )

    number = 6
    ref_array2 = np.zeros((number, 3), dtype=np.float64)
    ref_point_cloud_index = [cst.X, cst.Y, cst.Z]

    ref_x = [43.0, 44.0, 45.0, 45.0, 45.0, 45.0]
    ref_y = [44.0, 45.0, 45.0, 45.0, 45.0, 45.0]
    ref_z = [45.0, 46.0, 50.0, 50.0, 50.0, 50.0]
    ref_array2[:, 0] = np.asarray(ref_x, dtype=np.float64)
    ref_array2[:, 1] = np.asarray(ref_y, dtype=np.float64)
    ref_array2[:, 2] = np.asarray(ref_z, dtype=np.float64)

    ref_cloud = pandas.DataFrame(ref_array2, columns=ref_point_cloud_index)

    for key in ref_point_cloud_index:
        assert np.allclose(cloud[key].values, ref_cloud[key].values)


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
    __, removed_elt_pos = point_cloud_tools.filter_cloud(
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
    __, removed_elt_pos = point_cloud_tools.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=False
    )

    assert removed_elt_pos is None

    cloud_arr = np.arange(3 * 10)
    cloud_arr = cloud_arr.reshape((10, 3))

    cloud = pandas.DataFrame(cloud_arr, columns=["x", "y", "z"])
    __, removed_elt_pos = point_cloud_tools.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=True
    )

    assert removed_elt_pos is None


@pytest.mark.unit_tests
def test_add_cloud_filtering_msk():
    """
    Create fake cloud, msk, cfg to test add_cloud_filtering_msk function
    """
    nb_row = 5
    nb_col = 10
    rows = np.array(range(nb_row))
    cols = np.array(range(nb_col))

    ds0 = xr.Dataset({}, coords={"row": rows, "col": cols})
    ds1 = xr.Dataset({}, coords={"row": rows, "col": cols})

    pos_arr = np.array([[1, 2, 0], [2, 2, 1]])
    elt_remove = pandas.DataFrame(
        pos_arr, columns=["coord_epi_geom_i", "coord_epi_geom_j", "id_im_epi"]
    )

    point_cloud_tools.add_cloud_filtering_msk(
        [ds0, ds1], elt_remove, "mask", 255
    )

    mask0 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask0[1, 2] = 255
    mask1 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask1[2, 2] = 255
    ds0_ref = xr.Dataset(
        {"mask": (["row", "col"], mask0)}, coords={"row": rows, "col": cols}
    )
    ds1_ref = xr.Dataset(
        {"mask": (["row", "col"], mask1)}, coords={"row": rows, "col": cols}
    )

    assert_same_datasets(ds0_ref, ds0)
    assert_same_datasets(ds1_ref, ds1)

    # test exceptions
    np_pos = np.array([[1, 2, 2], [2, 2, 1]])
    elt_remove = pandas.DataFrame(
        np_pos,
        columns=["coord_epi_geom_i", "coord_epi_geom_j", "id_im_epi"],
    )
    with pytest.raises(Exception) as index_error:
        point_cloud_tools.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Index indicated in the elt_pos_infos "
        "pandas. DataFrame is not coherent with the clouds "
        "list given in input"
    )

    np_pos = np.array([[1, 2, -1], [2, 2, 1]])
    elt_remove = pandas.DataFrame(
        np_pos,
        columns=["coord_epi_geom_i", "coord_epi_geom_j", "id_im_epi"],
    )
    with pytest.raises(Exception) as index_error:
        point_cloud_tools.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Index indicated in the elt_pos_infos "
        "pandas. DataFrame is not coherent "
        "with the clouds list given in input"
    )

    np_pos = np.array([[11, 2, 0]])
    elt_remove = pandas.DataFrame(
        np_pos,
        columns=["coord_epi_geom_i", "coord_epi_geom_j", "id_im_epi"],
    )
    with pytest.raises(Exception) as index_error:
        point_cloud_tools.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Point at location (11,2) is not "
        "accessible in an image of size (5,10)"
    )
