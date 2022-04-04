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
Test module for cars/steps/points_cloud.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.steps import points_cloud

# CARS Tests imports
from ..helpers import assert_same_datasets


@pytest.mark.unit_tests
def test_create_combined_cloud():
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
    def get_cloud0_ds(with_msk):
        """Return local test point cloud 10x10 dataset"""
        row = 10
        col = 10
        x_coord = np.arange(row * col)
        x_coord = x_coord.reshape((row, col))
        y_coord = x_coord + 1
        z_coord = y_coord + 1
        corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
        corr_msk[4, 4] = 0
        if with_msk:
            msk = np.full((row, col), fill_value=0, dtype=np.int16)
            msk[4, 6] = 255

        ds_values = {
            cst.X: ([cst.ROW, cst.COL], x_coord),
            cst.Y: ([cst.ROW, cst.COL], y_coord),
            cst.Z: ([cst.ROW, cst.COL], z_coord),
            cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        }

        if with_msk:
            ds_values[cst.EPI_MSK] = ([cst.ROW, cst.COL], msk)

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
            cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
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
            cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk),
        },
        coords={cst.ROW: np.array(range(row)), cst.COL: np.array(range(col))},
    )
    cloud2.attrs[cst.EPSG] = epsg

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]

    cloud, epsg = points_cloud.create_combined_cloud(
        cloud_list,
        epsg,
        color_list=None,
        resolution=0.5,
        xstart=40.0,
        ystart=50.0,
        xsize=20,
        ysize=25,
        on_ground_margin=1,
        epipolar_border_margin=1,
        radius=1,
        with_coords=False,
    )

    ref_cloud0 = np.array(
        [
            [0.0, 39.0, 40.0, 41.0],
            [0.0, 40.0, 41.0, 42.0],
            [1.0, 41.0, 42.0, 43.0],
            [1.0, 42.0, 43.0, 44.0],
            [1.0, 43.0, 44.0, 45.0],
            [1.0, 45.0, 46.0, 47.0],
            [1.0, 46.0, 47.0, 48.0],
            [1.0, 47.0, 48.0, 49.0],
            [1.0, 48.0, 49.0, 50.0],
            [0.0, 49.0, 50.0, 51.0],
            [0.0, 50.0, 51.0, 52.0],
        ]
    )

    ref_cloud2 = np.zeros((row * col, 4))
    ref_cloud2[:, 1] = 45
    ref_cloud2[:, 2] = 45
    ref_cloud2[:, 3] = 50

    for i in range(1, col - 1):
        ref_cloud2[i * row + 1 : i * row + 4, 0] = 1
    ref_cloud2 = np.delete(ref_cloud2, 2 * col + 2, 0)

    ref_cloud = np.concatenate([ref_cloud0, ref_cloud2])

    assert np.allclose(cloud.values, ref_cloud)

    # test with mask
    cloud_list = [get_cloud0_ds(with_msk=True), cloud2]

    cloud, epsg = points_cloud.create_combined_cloud(
        cloud_list,
        epsg,
        color_list=None,
        resolution=0.5,
        xstart=40.0,
        ystart=50.0,
        xsize=20,
        ysize=25,
        on_ground_margin=1,
        epipolar_border_margin=1,
        radius=1,
        with_coords=False,
    )

    ref_cloud0_with_msk = np.array(
        [
            [0.0, 39.0, 40.0, 41.0, 0.0],
            [0.0, 40.0, 41.0, 42.0, 0.0],
            [1.0, 41.0, 42.0, 43.0, 0.0],
            [1.0, 42.0, 43.0, 44.0, 0.0],
            [1.0, 43.0, 44.0, 45.0, 0.0],
            [1.0, 45.0, 46.0, 47.0, 0.0],
            [1.0, 46.0, 47.0, 48.0, 255.0],
            [1.0, 47.0, 48.0, 49.0, 0.0],
            [1.0, 48.0, 49.0, 50.0, 0.0],
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
    clr0 = xr.Dataset(
        {cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr0)},
        coords={
            cst.BAND: np.array(range(band)),
            cst.ROW: np.array(range(row)),
            cst.COL: np.array(range(col)),
        },
    )

    clr1 = np.full((band, row, col), fill_value=20)
    clr1 = xr.Dataset(
        {cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr1)},
        coords={
            cst.BAND: np.array(range(band)),
            cst.ROW: np.array(range(row)),
            cst.COL: np.array(range(col)),
        },
    )

    row = 5
    col = 5
    clr2 = np.zeros((band, row, col))
    clr2[0, :, :] = np.arange(row * col).reshape((row, col))
    clr2[1, :, :] = clr2[0, :, :] + 1
    clr2[2, :, :] = clr2[1, :, :] + 1
    clr2 = xr.Dataset(
        {cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr2)},
        coords={
            cst.BAND: np.array(range(band)),
            cst.ROW: np.array(range(row)),
            cst.COL: np.array(range(col)),
        },
    )

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]
    clr_list = [clr0, clr1, clr2]

    cloud, epsg = points_cloud.create_combined_cloud(
        cloud_list,
        epsg,
        color_list=clr_list,
        resolution=0.5,
        xstart=40.0,
        ystart=50.0,
        xsize=20,
        ysize=25,
        on_ground_margin=1,
        epipolar_border_margin=1,
        radius=1,
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

    # test with coords and colors
    cloud, epsg = points_cloud.create_combined_cloud(
        cloud_list,
        epsg,
        color_list=clr_list,
        resolution=0.5,
        xstart=40.0,
        ystart=50.0,
        xsize=20,
        ysize=25,
        on_ground_margin=1,
        epipolar_border_margin=1,
        radius=1,
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
    cloud, epsg = points_cloud.create_combined_cloud(
        cloud_list,
        epsg,
        color_list=None,
        resolution=0.5,
        xstart=40.0,
        ystart=50.0,
        xsize=20,
        ysize=25,
        on_ground_margin=1,
        epipolar_border_margin=1,
        radius=1,
        with_coords=True,
    )

    ref_cloud_coords0 = np.concatenate([ref_cloud0, ref_coords0], axis=1)
    ref_cloud_coords2 = np.concatenate([ref_cloud2, ref_coords2], axis=1)
    ref_cloud_coords = np.concatenate([ref_cloud_coords0, ref_cloud_coords2])

    assert np.allclose(cloud, ref_cloud_coords)

    # test exception
    with pytest.raises(Exception) as test_error:
        points_cloud.create_combined_cloud(
            cloud_list,
            epsg,
            color_list=[clr0],
            resolution=0.5,
            xstart=40.0,
            ystart=50.0,
            xsize=20,
            ysize=25,
            on_ground_margin=1,
            epipolar_border_margin=1,
            radius=1,
            with_coords=True,
        )
    assert (
        str(test_error.value) == "There shall be as many cloud elements "
        "as color ones"
    )


@pytest.mark.unit_tests
def test_detect_small_components():
    """
    Create fake cloud to process and test detect_small_components
    """
    # Create random init fake cloud x,y,z coords structure.
    x_coord_init = np.zeros((5, 5))
    x_coord_init[4, 4] = 20
    x_coord_init[0, 4] = 19.55
    x_coord_init[0, 3] = 19.10

    y_coord_init = np.zeros((5, 5))

    z_coord_init = np.zeros((5, 5))
    z_coord_init[0:2, 0:2] = 10
    z_coord_init[1, 1] = 12

    cloud_arr = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(
                x_coord_init, y_coord_init, z_coord_init
            )
        ],
        axis=0,
    )

    indexes_to_filter = points_cloud.detect_small_components(
        cloud_arr, 0.5, 10, 2
    )
    assert sorted(indexes_to_filter) == [3, 4, 24]

    # test without the second level of filtering
    indexes_to_filter = points_cloud.detect_small_components(
        cloud_arr, 0.5, 10, None
    )
    assert sorted(indexes_to_filter) == [0, 1, 3, 4, 5, 6, 24]


@pytest.mark.unit_tests
def test_detect_statistical_outliers():
    """
    Create fake cloud to process and test detect_statistical_outliers
    """
    # Create fake cloud init 3D structure
    x_coord_init = np.zeros((5, 6))
    off = 0
    for line in range(5):
        # x[line,:] = np.arange(off, off+(line+1)*5, line+1)
        last_val = off + 5
        x_coord_init[line, :5] = np.arange(off, last_val)
        off += (line + 2 + 1) * 5

        # outlier
        x_coord_init[line, 5] = (off + last_val - 1) / 2

    y_coord_init = np.zeros((5, 6))
    z_coord_init = np.zeros((5, 6))

    ref_cloud = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(
                x_coord_init, y_coord_init, z_coord_init
            )
        ],
        axis=0,
    )

    removed_elt_pos = points_cloud.detect_statistical_outliers(
        ref_cloud, 4, 0.0
    )
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = points_cloud.detect_statistical_outliers(
        ref_cloud, 4, 1.0
    )
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = points_cloud.detect_statistical_outliers(
        ref_cloud, 4, 2.0
    )
    assert sorted(removed_elt_pos) == [23, 29]


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
            "idx_im_epi",
        ],
    )

    elt_to_remove = [0, 5]
    __, removed_elt_pos = points_cloud.filter_cloud(
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
        pos_arr, columns=["coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"]
    )

    assert ref_removed_elt_pos.equals(removed_elt_pos)

    # test cases where removed_elt_pos should be None
    __, removed_elt_pos = points_cloud.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=False
    )

    assert removed_elt_pos is None

    cloud_arr = np.arange(3 * 10)
    cloud_arr = cloud_arr.reshape((10, 3))

    cloud = pandas.DataFrame(cloud_arr, columns=["x", "y", "z"])
    __, removed_elt_pos = points_cloud.filter_cloud(
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
        pos_arr, columns=["coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"]
    )

    points_cloud.add_cloud_filtering_msk([ds0, ds1], elt_remove, "mask", 255)

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
    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[1, 2, 2], [2, 2, 1]])
        elt_remove = pandas.DataFrame(
            np_pos,
            columns=["coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"],
        )
        points_cloud.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Index indicated in the elt_pos_infos "
        "pandas. DataFrame is not coherent with the clouds "
        "list given in input"
    )

    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[1, 2, -1], [2, 2, 1]])
        elt_remove = pandas.DataFrame(
            np_pos,
            columns=["coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"],
        )
        points_cloud.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Index indicated in the elt_pos_infos "
        "pandas. DataFrame is not coherent "
        "with the clouds list given in input"
    )

    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[11, 2, 0]])
        elt_remove = pandas.DataFrame(
            np_pos,
            columns=["coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"],
        )
        points_cloud.add_cloud_filtering_msk(
            [ds0, ds1], elt_remove, "mask", 255
        )
    assert (
        str(index_error.value) == "Point at location (11,2) is not "
        "accessible in an image of size (5,10)"
    )
