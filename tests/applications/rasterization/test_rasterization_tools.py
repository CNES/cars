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
Test module for
    cars/application/points_cloud_rasterization/rasterization_tools.py
TODO: refactor with code source
"""
# pylint: disable= C0302

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

from cars.applications.point_cloud_fusion import mapping_to_terrain_tiles
from cars.applications.rasterization import rasterization_tools

# CARS imports
from cars.core import constants as cst

# CARS Tests imports
from tests.helpers import absolute_data_path, add_color, assert_same_datasets


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
    raster = rasterization_tools.rasterize(
        pd_cloud, 1, None, 0, 12, 3, 2, 0.3, 0
    )

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
    (
        xstart,
        ystart,
        xsize,
        ysize,
    ) = rasterization_tools.compute_xy_starts_and_sizes(1, pd_cloud)
    raster = rasterization_tools.rasterize(
        pd_cloud, 1, None, xstart, ystart, xsize, ysize, 0.3, 0
    )
    np.testing.assert_equal(
        raster[cst.RASTER_HGT].values[..., None],
        [[[3], [4], [5]], [[0], [1], [2]]],
    )

    # Test with fixed grid, radius =  1 and sigma = inf
    raster = rasterization_tools.rasterize(
        pd_cloud, 1, None, 0, 12, 3, 2, np.inf, 1
    )
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

    (
        xstart,
        ystart,
        xsize,
        ysize,
    ) = rasterization_tools.compute_xy_starts_and_sizes(resolution, cloud_df)

    raster = rasterization_tools.rasterize(
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

    # equals to :
    xmin = xstart
    xmax = xstart + (xsize + 1) * resolution
    ymin = ystart - (ysize + 1) * resolution
    ymax = ystart

    epsg = 32630
    sigma = 0.3
    radius = 3

    # Compute margin
    on_ground_margin = 0
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    # combine datasets
    cloud = add_color(cloud, color[cst.EPI_IMAGE].values)

    cloud = mapping_to_terrain_tiles.compute_point_cloud_wrapper(
        [cloud],
        [None],
        epsg,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        margins=used_margin,
        save_pc_as_laz=False,
        save_pc_as_csv=False,
        saving_info=None,
    )

    # TODO test from here -> dump cloud as test data input

    raster = rasterization_tools.simple_rasterization_dataset_wrapper(
        cloud,
        resolution,
        epsg,
        xstart=xstart,
        ystart=ystart,
        xsize=xsize,
        ysize=ysize,
        sigma=sigma,
        radius=radius,
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

    # equals to :
    xmin = None
    xmax = None
    ymin = None
    ymax = None

    # combine datasets
    cloud = add_color(cloud, color[cst.EPI_IMAGE].values)

    epsg = 32630
    sigma = 0.3
    radius = 3

    # Compute margin
    on_ground_margin = 0
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    cloud = mapping_to_terrain_tiles.compute_point_cloud_wrapper(
        [cloud],
        [None],
        epsg,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        margins=used_margin,
        save_pc_as_laz=False,
        save_pc_as_csv=False,
        saving_info=None,
    )

    # TODO test from here -> dump cloud as test data input

    raster = rasterization_tools.simple_rasterization_dataset_wrapper(
        cloud,
        resolution,
        epsg,
        xstart=xstart,
        ystart=ystart,
        xsize=xsize,
        ysize=ysize,
        sigma=sigma,
        radius=radius,
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
    Test simple_rasterization_dataset_wrapper with a list of datasets
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

    # Combine datasets

    utm1 = add_color(utm1, color1[cst.EPI_IMAGE].values)
    utm2 = add_color(utm2, color2[cst.EPI_IMAGE].values)

    resolution = 0.5

    xstart = 1154790
    ystart = 4927552
    xsize = 114
    ysize = 112
    # equals to :
    xmin = xstart
    xmax = xstart + (xsize + 1) * resolution
    ymin = ystart - (ysize + 1) * resolution
    ymax = ystart

    epsg = 32630
    sigma = 0.3
    radius = 3

    # Compute margin
    on_ground_margin = 0
    # Former computation of merged margin
    used_margin = (on_ground_margin + radius + 1) * resolution

    cloud = mapping_to_terrain_tiles.compute_point_cloud_wrapper(
        [utm1, utm2],
        [None, None],
        epsg,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        margins=used_margin,
        save_pc_as_laz=False,
        save_pc_as_csv=False,
        saving_info=None,
    )

    # TODO test from here -> dump cloud as test data input
    raster = rasterization_tools.simple_rasterization_dataset_wrapper(
        cloud,
        resolution,
        epsg,
        xstart=xstart,
        ystart=ystart,
        xsize=xsize,
        ysize=ysize,
        sigma=sigma,
        radius=radius,
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
    pytest fixture local function to ease rasterization tests readability:
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

    mask_interp_cloud = {
        cst.ROW: row,
        cst.COL: col,
        cst.RESOLUTION: resolution,
        "cloud": cloud,
        "msk": msk,
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

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs["cloud"]
    msk = mask_interp_inputs["msk"]
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1

    # create panda dataframe
    cloud_pd = pandas.DataFrame(
        cloud, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK]
    )

    # test mask_interp function
    (
        __,
        __,
        __,
        __,
        __,
        res,
        __,
        __,
        __,
    ) = rasterization_tools.compute_vector_raster_and_stats(
        cloud_pd,
        data_valid,
        -0.5,
        row - 0.5,
        col,
        row,
        resolution,
        sigma,
        radius,
    )

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(msk, res)
