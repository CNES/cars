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
Test module for cars/steps/rasterization.py
TODO: refactor in several files and remove too-many-lines
"""
# pylint: disable=too-many-lines

# Standard imports
from __future__ import absolute_import

import logging

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.steps import rasterization

# CARS Tests imports
from ..helpers import absolute_data_path, assert_same_datasets


@pytest.mark.unit_tests
def test_simple_rasterization_synthetic_case():
    """
    Test simple_rasterization on synthetic case
    """
    cloud = np.array(
        [
            [1.0, 0.5, 10.5, 0],
            [1.0, 1.5, 10.5, 1],
            [1.0, 2.5, 10.5, 2],
            [1.0, 0.5, 11.5, 3],
            [1.0, 1.5, 11.5, 4],
            [1.0, 2.5, 11.5, 5],
        ]
    )

    pd_cloud = pandas.DataFrame(
        cloud, columns=[cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z]
    )
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, 0.3, 0)

    # Test with radius = 0 and fixed grid
    np.testing.assert_equal(
        raster[cst.RASTER_HGT].values[..., None],
        [[[3], [4], [5]], [[0], [1], [2]]],
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS].values, np.ones((2, 3))
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS_IN_CELL].values, np.ones((2, 3))
    )
    np.testing.assert_array_equal(
        np.ravel(np.flipud(raster[cst.RASTER_HGT_MEAN].values)), pd_cloud[cst.Z]
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_HGT_STD_DEV].values, np.zeros((2, 3))
    )

    # Test with grid evaluated by function
    xstart, ystart, xsize, ysize = rasterization.compute_xy_starts_and_sizes(
        1, pd_cloud
    )
    raster = rasterization.rasterize(
        pd_cloud, 1, None, xstart, ystart, xsize, ysize, 0.3, 0
    )
    np.testing.assert_equal(
        raster[cst.RASTER_HGT].values[..., None],
        [[[3], [4], [5]], [[0], [1], [2]]],
    )

    # Test with fixed grid, radius =  1 and sigma = inf
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, np.inf, 1)
    np.testing.assert_equal(
        raster[cst.RASTER_HGT].values[..., None],
        [[[2], [2.5], [3]], [[2], [2.5], [3]]],
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_HGT_MEAN].values, [[2.0, 2.5, 3.0], [2.0, 2.5, 3.0]]
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS_IN_CELL].values, np.ones((2, 3), dtype=int)
    )
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS].values, [[4, 6, 4], [4, 6, 4]]
    )


@pytest.mark.unit_tests
def test_simple_rasterization_single():
    """
    Test simple rasterization from test cloud ref_single_cloud_in_df.nc
    """

    resolution = 0.5

    cloud_xr = xr.open_dataset(
        absolute_data_path(
            "input/rasterization_input/ref_single_cloud_in_df.nc"
        )
    )
    cloud_df = cloud_xr.to_dataframe()

    xstart, ystart, xsize, ysize = rasterization.compute_xy_starts_and_sizes(
        resolution, cloud_df
    )
    raster = rasterization.rasterize(
        cloud_df,
        resolution,
        32630,
        xstart,
        ystart,
        xsize,
        ysize,
        0.3,
        3,
        hgt_no_data=np.nan,
        color_no_data=np.nan,
    )

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/ref_simple_rasterization.nc'),
    # )

    ref_rasterized = xr.open_dataset(
        absolute_data_path("ref_output/ref_simple_rasterization.nc")
    )

    assert_same_datasets(raster, ref_rasterized, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.unit_tests
def test_simple_rasterization_dataset_1():
    """
    Test simple rasterization dataset from test cloud cloud1_ref_epsg_32630.nc
    Configuration 1 : random xstart, ystart, xsize, ysize values
    """

    cloud = xr.open_dataset(
        absolute_data_path("input/rasterization_input/cloud1_ref_epsg_32630.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    xstart = 1154790
    ystart = 4927552
    xsize = 114
    ysize = 112
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [cloud],
        resolution,
        32630,
        [color],
        xstart,
        ystart,
        xsize,
        ysize,
        0.3,
        3,
    )

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_res_ref_1.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_res_ref_1.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.unit_tests
def test_simple_rasterization_dataset_2():
    """
    Test simple rasterization dataset from test cloud cloud1_ref_epsg_32630.nc
    Configuration 2 : no xstart, ystart, xsize, ysize values
    """

    cloud = xr.open_dataset(
        absolute_data_path("input/rasterization_input/cloud1_ref_epsg_32630.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    xstart = None
    ystart = None
    xsize = None
    ysize = None
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [cloud],
        resolution,
        32630,
        [color],
        xstart,
        ystart,
        xsize,
        ysize,
        0.3,
        3,
    )

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_res_ref_2.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_res_ref_2.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.unit_tests
def test_simple_rasterization_multiple_datasets():
    """
    Test simple_rasterization_dataset with a list of datasets
    """
    cloud = xr.open_dataset(
        absolute_data_path("input/rasterization_input/cloud1_ref_epsg_32630.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    utm1 = cloud.isel(row=range(0, 60))
    utm2 = cloud.isel(row=range(60, 120))

    color1 = color.isel(row=range(0, 60))
    color2 = color.isel(row=range(60, 120))

    xstart = 1154790
    ystart = 4927552
    xsize = 114
    ysize = 112
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [utm1, utm2],
        resolution,
        32630,
        [color1, color2],
        xstart,
        ystart,
        xsize,
        ysize,
        0.3,
        3,
    )

    # Uncomment to update reference
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_multiple_res_ref.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_multiple_res_ref.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.0e-10, rtol=1.0e-10)


# Mask interpolation tests


@pytest.fixture(scope="module")
def mask_interp_inputs():  # pylint: disable=redefined-outer-name
    """
    pytest fixture local function to ease rasterization tests readibility:
    Returns a mask interpolated cloud.
    """
    row = 4
    col = 5
    resolution = 1.0

    # simple mask all 100 and one 0
    x_coord, y_coord = np.meshgrid(
        np.linspace(0, col - 1, col), np.linspace(0, row - 1, row)
    )

    msk = np.full((row, col), fill_value=100, dtype=np.float64)
    msk[0, 0] = 0

    cloud = np.zeros((row * col, 3), dtype=np.float64)
    cloud[:, 0] = x_coord.reshape((row * col))
    cloud[:, 1] = y_coord.reshape((row * col))
    cloud[:, 2] = msk.reshape((row * col))

    data_valid = np.ones((row * col), dtype=bool)

    grid_points = rasterization.compute_grid_points(
        -0.5, row - 0.5, col, row, resolution
    )

    mask_interp_cloud = {
        cst.ROW: row,
        cst.COL: col,
        cst.RESOLUTION: resolution,
        "cloud": cloud,
        "msk": msk,
        "grid_points": grid_points,
        cst.POINTS_CLOUD_VALID_DATA: data_valid,
    }

    return mask_interp_cloud


@pytest.mark.unit_tests
def test_mask_interp_case1(
    mask_interp_inputs,
):  # pylint: disable=redefined-outer-name
    """
    case 1 - simple mask all 100 and one 0
    """
    worker_logger = logging.getLogger("distributed.worker")

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    grid_points = mask_interp_inputs["grid_points"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # create panda dataframe and search for neighbors
    cloud_pd = pandas.DataFrame(
        cloud, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK]
    )

    neighbors_id, start_ids, n_count = rasterization.get_flatten_neighbors(
        grid_points, cloud_pd, radius, resolution, worker_logger
    )

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud,
        data_valid.astype(bool),
        neighbors_id,
        start_ids,
        n_count,
        grid_points,
        sigma,
        nodata_val,
        undefined_val,
    )

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case2(
    mask_interp_inputs,
):  # pylint: disable=redefined-outer-name
    """
    case 2 - add several points from a second class aiming the same terrain cell
    """
    worker_logger = logging.getLogger("distributed.worker")

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    grid_points = mask_interp_inputs["grid_points"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # add several points from a second class aiming the same terrain cell
    tgt_terrain_cell_x_coord = 1
    tgt_terrain_cell_y_coord = 1
    row_additional_pts_nb = 3
    col_additional_pts_nb = 3
    additional_pts_x_coords, additional_pts_y_coords = np.meshgrid(
        np.linspace(
            tgt_terrain_cell_x_coord - 0.3,
            tgt_terrain_cell_x_coord + 0.3,
            col_additional_pts_nb,
        ),
        np.linspace(
            tgt_terrain_cell_y_coord - 0.3,
            tgt_terrain_cell_y_coord + 0.3,
            row_additional_pts_nb,
        ),
    )

    additional_pts_msk = np.full(
        (row_additional_pts_nb * col_additional_pts_nb),
        fill_value=200,
        dtype=np.float64,
    )

    cloud_case2 = np.zeros(
        (row_additional_pts_nb * col_additional_pts_nb, 3), dtype=np.float64
    )
    cloud_case2[:, 0] = additional_pts_x_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb)
    )
    cloud_case2[:, 1] = additional_pts_y_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb)
    )
    cloud_case2[:, 2] = additional_pts_msk
    cloud_case2 = np.concatenate((cloud, cloud_case2), axis=0)

    data_valid_case2 = np.concatenate(
        (
            data_valid,
            np.ones(
                (row_additional_pts_nb * col_additional_pts_nb), dtype=bool
            ),
        ),
        axis=0,
    )

    # create panda dataframe and search for neighbors
    cloud_pd_case2 = pandas.DataFrame(
        cloud_case2, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK]
    )

    neighbors_id, start_ids, n_count = rasterization.get_flatten_neighbors(
        grid_points, cloud_pd_case2, radius, resolution, worker_logger
    )

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case2,
        data_valid_case2,
        neighbors_id,
        start_ids,
        n_count,
        grid_points,
        sigma,
        nodata_val,
        undefined_val,
    )

    ref_msk = np.copy(msk)
    ref_msk[tgt_terrain_cell_y_coord, tgt_terrain_cell_x_coord] = 200

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case3(
    mask_interp_inputs,
):  # pylint: disable=redefined-outer-name
    """
    case 3 - only two points from different classes at the same position in a
    single cell
    """
    worker_logger = logging.getLogger("distributed.worker")

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    grid_points = mask_interp_inputs["grid_points"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # only two points from different classes at the same position
    # in a single cell
    cloud_case3 = np.array([[2.0, 2.0, 200.0]], dtype=np.float64)
    cloud_case3 = np.concatenate((cloud, cloud_case3), axis=0)

    cloud_pd_case3 = pandas.DataFrame(
        cloud_case3, columns=[cst.X, cst.Y, "left_mask"]
    )

    data_valid_case3 = np.concatenate(
        (data_valid, np.ones((1), dtype=bool)), axis=0
    )

    neighbors_id, start_ids, n_count = rasterization.get_flatten_neighbors(
        grid_points, cloud_pd_case3, radius, resolution, worker_logger
    )

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case3,
        data_valid_case3,
        neighbors_id,
        start_ids,
        n_count,
        grid_points,
        sigma,
        nodata_val,
        undefined_val,
    )

    ref_msk = np.copy(msk)
    ref_msk[2, 2] = undefined_val

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case4(
    mask_interp_inputs,
):  # pylint: disable=redefined-outer-name
    """
    case 4 - no data cell
    """
    worker_logger = logging.getLogger("distributed.worker")

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    grid_points = mask_interp_inputs["grid_points"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # no data cell
    cloud_case4 = np.copy(cloud)
    cloud_case4 = np.delete(cloud_case4, 1, 0)

    data_valid_case4 = np.copy(data_valid)
    data_valid_case4 = np.delete(data_valid_case4, 1, 0)

    # create panda dataframe and search for neighbors
    cloud_pd_case4 = pandas.DataFrame(
        cloud_case4, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK]
    )

    neighbors_id, start_ids, n_count = rasterization.get_flatten_neighbors(
        grid_points, cloud_pd_case4, radius, resolution, worker_logger
    )

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case4,
        data_valid_case4,
        neighbors_id,
        start_ids,
        n_count,
        grid_points,
        sigma,
        nodata_val,
        undefined_val,
    )

    ref_msk = np.copy(msk)
    ref_msk[int(cloud[1, 1]), int(cloud[1, 0])] = nodata_val

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case5(
    mask_interp_inputs,
):  # pylint: disable=redefined-outer-name
    """
    case 5 - add several points equal to 0 aiming the same terrain cell
    """
    worker_logger = logging.getLogger("distributed.worker")

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    grid_points = mask_interp_inputs["grid_points"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # add several points equal to 0 aiming the same terrain cell
    tgt_terrain_cell_x_coord = 1
    tgt_terrain_cell_y_coord = 1
    row_additional_pts_nb = 3
    col_additional_pts_nb = 3
    additional_pts_x_coords, additional_pts_y_coords = np.meshgrid(
        np.linspace(
            tgt_terrain_cell_x_coord - 0.3,
            tgt_terrain_cell_x_coord + 0.3,
            col_additional_pts_nb,
        ),
        np.linspace(
            tgt_terrain_cell_y_coord - 0.3,
            tgt_terrain_cell_y_coord + 0.3,
            row_additional_pts_nb,
        ),
    )

    additional_pts_msk = np.full(
        (row_additional_pts_nb * col_additional_pts_nb),
        fill_value=0,
        dtype=np.float64,
    )

    cloud_case5 = np.zeros(
        (row_additional_pts_nb * col_additional_pts_nb, 3), dtype=np.float64
    )
    cloud_case5[:, 0] = additional_pts_x_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb)
    )
    cloud_case5[:, 1] = additional_pts_y_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb)
    )
    cloud_case5[:, 2] = additional_pts_msk
    cloud_case5 = np.concatenate((cloud, cloud_case5), axis=0)

    data_valid_case5 = np.concatenate(
        (
            data_valid,
            np.ones(
                (row_additional_pts_nb * col_additional_pts_nb), dtype=bool
            ),
        ),
        axis=0,
    )

    # create panda dataframe and search for neighbors
    cloud_pd_case5 = pandas.DataFrame(
        cloud_case5, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK]
    )

    neighbors_id, start_ids, n_count = rasterization.get_flatten_neighbors(
        grid_points, cloud_pd_case5, radius, resolution, worker_logger
    )

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case5,
        data_valid_case5,
        neighbors_id,
        start_ids,
        n_count,
        grid_points,
        sigma,
        nodata_val,
        undefined_val,
    )

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(msk, res)
