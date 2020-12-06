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

from __future__ import absolute_import

import pytest
import pandas

import numpy as np
import xarray as xr

from utils import absolute_data_path
from utils import assert_same_datasets

from cars import rasterization
from cars import constants as cst

@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    epsg = rasterization.get_utm_zone_as_epsg_code(1.442299, 43.600764)
    assert epsg == 32631


@pytest.mark.unit_tests
def test_create_combined_cloud():
    epsg = 4326

    # test only color
    def get_cloud0_ds(with_msk):
        row = 10
        col = 10
        x = np.arange(row*col)
        x = x.reshape((row, col))
        y = x + 1
        z = y + 1
        corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
        corr_msk[4, 4] = 0
        if with_msk:
            msk = np.full((row, col), fill_value=0, dtype=np.int16)
            msk[4, 6] = 255

        ds_values = {
            cst.X: ([cst.ROW, cst.COL], x),
            cst.Y: ([cst.ROW, cst.COL], y),
            cst.Z: ([cst.ROW, cst.COL], z),
            cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL], corr_msk)
        }

        if with_msk:
            ds_values[cst.EPI_MSK] = ([cst.ROW, cst.COL], msk)

        cloud0 = xr.Dataset(ds_values, coords={cst.ROW: np.array(range(row)),
                                               cst.COL: np.array(range(col))})
        cloud0.attrs[cst.EPSG] = epsg

        return cloud0

    row = 10
    col = 10
    x = np.full((row, col), fill_value=0, dtype=np.float)
    y = np.full((row, col), fill_value=1, dtype=np.float)
    z = np.full((row, col), fill_value=2, dtype=np.float)
    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[6, 6] = 0

    cloud1 = xr.Dataset({cst.X: ([cst.ROW, cst.COL], x),
                         cst.Y: ([cst.ROW, cst.COL], y),
                         cst.Z: ([cst.ROW, cst.COL], z),
                         cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL],
                                                     corr_msk)},
                        coords={cst.ROW: np.array(range(row)),
                                cst.COL: np.array(range(col))})
    cloud1.attrs[cst.EPSG] = epsg

    row = 5
    col = 5
    x = np.full((row, col), fill_value=45, dtype=np.float)
    y = np.full((row, col), fill_value=45, dtype=np.float)
    z = np.full((row, col), fill_value=50, dtype=np.float)
    corr_msk = np.full((row, col), fill_value=255, dtype=np.int16)
    corr_msk[2, 2] = 0

    cloud2 = xr.Dataset({cst.X: ([cst.ROW, cst.COL], x),
                         cst.Y: ([cst.ROW, cst.COL], y),
                         cst.Z: ([cst.ROW, cst.COL], z),
                         cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL],
                                                     corr_msk)},
                        coords={cst.ROW: np.array(range(row)),
                                cst.COL: np.array(range(col))})
    cloud2.attrs[cst.EPSG] = epsg

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]

    cloud, epsg = rasterization.create_combined_cloud(
        cloud_list, epsg, color_list=None, resolution=0.5, xstart=40.0,
        ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
        epipolar_border_margin=1, radius=1, with_coords=False)

    ref_cloud0 = np.array([[0., 39., 40., 41.],
                           [0., 40., 41., 42.],
                           [1., 41., 42., 43.],
                           [1., 42., 43., 44.],
                           [1., 43., 44., 45.],
                           [1., 45., 46., 47.],
                           [1., 46., 47., 48.],
                           [1., 47., 48., 49.],
                           [1., 48., 49., 50.],
                           [0., 49., 50., 51.],
                           [0., 50., 51., 52.]])

    ref_cloud2 = np.zeros((row*col, 4))
    ref_cloud2[:, 1] = 45
    ref_cloud2[:, 2] = 45
    ref_cloud2[:, 3] = 50

    for i in range(1, col - 1):
        ref_cloud2[i*row+1:i*row+4, 0] = 1
    ref_cloud2 = np.delete(ref_cloud2, 2 * col + 2, 0)

    ref_cloud = np.concatenate([ref_cloud0, ref_cloud2])

    assert np.allclose(cloud.values, ref_cloud)

    # test with mask
    cloud_list = [get_cloud0_ds(with_msk=True), cloud2]

    cloud, epsg = rasterization.create_combined_cloud(
        cloud_list, epsg, color_list=None, resolution=0.5, xstart=40.0,
        ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
        epipolar_border_margin=1, radius=1, with_coords=False)

    ref_cloud0_with_msk = np.array([[0., 39., 40., 41., 0.],
                                    [0., 40., 41., 42., 0.],
                                    [1., 41., 42., 43., 0.],
                                    [1., 42., 43., 44., 0.],
                                    [1., 43., 44., 45., 0.],
                                    [1., 45., 46., 47., 0.],
                                    [1., 46., 47., 48., 255.],
                                    [1., 47., 48., 49., 0.],
                                    [1., 48., 49., 50., 0.],
                                    [0., 49., 50., 51., 0.],
                                    [0., 50., 51., 52., 0.]])

    ref_cloud = np.concatenate([ref_cloud0_with_msk,
                                np.concatenate(
                                    [ref_cloud2, np.zeros((row*col-1, 1))],
                                    axis=1)])
    assert np.allclose(cloud.values, ref_cloud)

    # test with color
    band = 3
    row = 10
    col = 10
    clr0 = np.zeros((band, row, col))
    clr0[0, :, :] = 10
    clr0[1, :, :] = 20
    clr0[2, :, :] = 30
    clr0 = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr0)},
                      coords={
                          cst.BAND: np.array(range(band)),
                          cst.ROW: np.array(range(row)),
                          cst.COL: np.array(range(col))})

    clr1 = np.full((band, row, col), fill_value=20)
    clr1 = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr1)},
                      coords={
                          cst.BAND: np.array(range(band)),
                          cst.ROW: np.array(range(row)),
                          cst.COL: np.array(range(col))})

    row = 5
    col = 5
    clr2 = np.zeros((band, row, col))
    clr2[0, :, :] = np.arange(row*col).reshape((row, col))
    clr2[1, :, :] = clr2[0, :, :] + 1
    clr2[2, :, :] = clr2[1, :, :] + 1
    clr2 = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], clr2)},
                      coords={cst.BAND: np.array(range(band)),
                              cst.ROW: np.array(range(row)),
                              cst.COL: np.array(range(col))})

    cloud_list = [get_cloud0_ds(with_msk=False), cloud1, cloud2]
    clr_list = [clr0, clr1, clr2]

    cloud, epsg = rasterization.\
        create_combined_cloud(
            cloud_list, epsg, color_list=clr_list, resolution=0.5, xstart=40.0,
            ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
            epipolar_border_margin=1, radius=1, with_coords=False)

    ref_clr0 = np.zeros((11, 3))
    ref_clr0[:, 0] = 10
    ref_clr0[:, 1] = 20
    ref_clr0[:, 2] = 30
    ref_cloud_clr0 = np.concatenate([ref_cloud0, ref_clr0], axis=1)

    ref_clr2 = np.zeros((row*col, 3))
    ref_clr2[:, 0] = np.arange(row*col)
    ref_clr2[:, 1] = ref_clr2[:, 0]+1
    ref_clr2[:, 2] = ref_clr2[:, 1]+1
    ref_clr2 = np.delete(ref_clr2, 2*col+2, 0)
    ref_cloud_clr2 = np.concatenate([ref_cloud2, ref_clr2], axis=1)

    ref_cloud_clr = np.concatenate([ref_cloud_clr0, ref_cloud_clr2])

    assert np.allclose(cloud.values, ref_cloud_clr)

    # test with coords and colors
    cloud, epsg = rasterization.\
        create_combined_cloud(
            cloud_list, epsg,
            color_list=clr_list, resolution=0.5, xstart=40.0,
            ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
            epipolar_border_margin=1, radius=1, with_coords=True)

    ref_coords0 = np.array([[3.,  9., 0.],
                            [4.,  0., 0.],
                            [4.,  1., 0.],
                            [4.,  2., 0.],
                            [4.,  3., 0.],
                            [4.,  5., 0.],
                            [4.,  6., 0.],
                            [4.,  7., 0.],
                            [4.,  8., 0.],
                            [4.,  9., 0.],
                            [5.,  0., 0.]])
    ref_cloud_clr_coords0 = np.concatenate(
        [ref_cloud0, ref_clr0, ref_coords0], axis=1)

    ref_coords2 = np.zeros((row*col, 3))
    ref_coords2[:, 2] = 2
    for i in range(row):
        for j in range(col):
            ref_coords2[i*col+j, 0] = i
            ref_coords2[i*col+j, 1] = j
    ref_coords2 = np.delete(ref_coords2, 2 * col + 2, 0)
    ref_cloud_clr_coords2 = np.concatenate(
        [ref_cloud2, ref_clr2, ref_coords2], axis=1)

    ref_cloud_clr_coords = np.concatenate(
        [ref_cloud_clr_coords0, ref_cloud_clr_coords2])

    assert np.allclose(cloud, ref_cloud_clr_coords)

    # test with coords (no colors)
    cloud, epsg = rasterization.create_combined_cloud(
        cloud_list, epsg, color_list=None, resolution=0.5, xstart=40.0,
        ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
        epipolar_border_margin=1, radius=1, with_coords=True)

    ref_cloud_coords0 = np.concatenate([ref_cloud0, ref_coords0], axis=1)
    ref_cloud_coords2 = np.concatenate([ref_cloud2, ref_coords2], axis=1)
    ref_cloud_coords = np.concatenate([ref_cloud_coords0, ref_cloud_coords2])

    assert np.allclose(cloud, ref_cloud_coords)

    # test exception
    with pytest.raises(Exception) as e:
        rasterization.create_combined_cloud(
            cloud_list, epsg, color_list=[clr0], resolution=0.5, xstart=40.0,
            ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
            epipolar_border_margin=1, radius=1, with_coords=True)
        assert str(e) == 'There shall be as many cloud elements as color ones'


@pytest.mark.unit_tests
def test_simple_rasterization_synthetic_case():
    """
    Test simple_rasterization on synthetic case
    """
    cloud = np.array([[1., 0.5, 10.5, 0],
                      [1., 1.5, 10.5, 1],
                      [1., 2.5, 10.5, 2],
                      [1., 0.5, 11.5, 3],
                      [1., 1.5, 11.5, 4],
                      [1., 2.5, 11.5, 5]])

    pd_cloud = pandas.DataFrame(
        cloud, columns=[cst.POINTS_CLOUD_VALID_DATA, cst.X, cst.Y, cst.Z])
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, 0.3, 0)

    # Test with radius = 0 and fixed grid
    np.testing.assert_equal(
        raster[cst.RASTER_HGT].values[..., None],
        [[[3], [4], [5]], [[0], [1], [2]]])
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS].values,
        np.ones((2, 3)))
    np.testing.assert_array_equal(
        raster[cst.RASTER_NB_PTS_IN_CELL].values,
        np.ones((2, 3)))
    np.testing.assert_array_equal(
        np.ravel(np.flipud(raster[cst.RASTER_HGT_MEAN].values)),
        pd_cloud[cst.Z])
    np.testing.assert_array_equal(
        raster[cst.RASTER_HGT_STD_DEV].values,
        np.zeros((2, 3)))

    # Test with grid evaluated by function
    xstart, ystart, xsize, ysize = \
        rasterization.compute_xy_starts_and_sizes(1, pd_cloud)
    raster = rasterization.rasterize(
        pd_cloud, 1, None, xstart, ystart, xsize, ysize, 0.3, 0)
    np.testing.assert_equal(raster[cst.RASTER_HGT].values[..., None],
                            [[[3], [4], [5]], [[0], [1], [2]]])

    # Test with fixed grid, radius =  1 and sigma = inf
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, np.inf, 1)
    np.testing.assert_equal(raster[cst.RASTER_HGT].values[..., None],
                            [[[2], [2.5], [3]], [[2], [2.5], [3]]])
    np.testing.assert_array_equal(raster[cst.RASTER_HGT_MEAN].values,
                                  [[2.0, 2.5, 3.0], [2.0, 2.5, 3.0]])
    np.testing.assert_array_equal(raster[cst.RASTER_NB_PTS_IN_CELL].values,
                                  np.ones((2, 3), dtype=int))
    np.testing.assert_array_equal(raster[cst.RASTER_NB_PTS].values,
                                  [[4, 6, 4], [4, 6, 4]])


@pytest.mark.unit_tests
def test_simple_rasterization_single():

    resolution = 0.5

    cloud_xr = xr.open_dataset(
        absolute_data_path(
            'input/rasterization_input/ref_single_cloud_in_df.nc')
    )
    cloud_df = cloud_xr.to_dataframe()

    xstart, ystart, xsize, ysize = \
        rasterization.compute_xy_starts_and_sizes(resolution, cloud_df)
    raster = rasterization.\
        rasterize(
            cloud_df, resolution, 32630, xstart, ystart, xsize, ysize, 0.3, 3,
            hgt_no_data=np.nan, color_no_data=np.nan)

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/ref_simple_rasterization.nc'),
    # )

    ref_rasterized = xr.open_dataset(
        absolute_data_path("ref_output/ref_simple_rasterization.nc")
    )

    assert_same_datasets(raster, ref_rasterized, atol=1.e-10, rtol=1.e-10)


@pytest.mark.unit_tests
def test_simple_rasterization_dataset_1():

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
        [cloud], resolution, 32630, [color],
        xstart, ystart, xsize, ysize, 0.3, 3
    )

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_res_ref_1.nc'),
    # )

    raster_ref = xr.open_dataset(absolute_data_path(
        "ref_output/rasterization_res_ref_1.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.e-10, rtol=1.e-10)


@pytest.mark.unit_tests
def test_simple_rasterization_dataset_2():

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
        [cloud], resolution, 32630, [color],
        xstart, ystart, xsize, ysize, 0.3, 3
    )

    # Uncomment to update references
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_res_ref_2.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_res_ref_2.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.e-10, rtol=1.e-10)


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
        [utm1, utm2], resolution, 32630, [color1, color2],
        xstart, ystart, xsize, ysize, 0.3, 3
    )

    # Uncomment to update reference
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_multiple_res_ref.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_multiple_res_ref.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.e-10, rtol=1.e-10)


# Mask interpolation tests

@pytest.fixture(scope='module')
def mask_interp_inputs():
    row = 4
    col = 5
    resolution = 1.0

    # simple mask all 100 and one 0
    x, y = np.meshgrid(np.linspace(0, col - 1, col),
                       np.linspace(0, row - 1, row))

    msk = np.full((row, col), fill_value=100, dtype=np.float)
    msk[0, 0] = 0

    cloud = np.zeros((row * col, 3), dtype=np.float)
    cloud[:, 0] = x.reshape((row * col))
    cloud[:, 1] = y.reshape((row * col))
    cloud[:, 2] = msk.reshape((row * col))

    data_valid = np.ones((row * col), dtype=np.bool)

    grid_points = \
        rasterization.compute_grid_points(-0.5, row - 0.5, col, row, resolution)

    mask_interp_cloud = {
        cst.ROW: row,
        cst.COL: col,
        cst.RESOLUTION: resolution,
        'cloud': cloud,
        'msk': msk,
        'grid_points': grid_points,
        cst.POINTS_CLOUD_VALID_DATA: data_valid
    }

    return mask_interp_cloud


@pytest.mark.unit_tests
def test_mask_interp_case1(mask_interp_inputs):
    """
    case 1 - simple mask all 100 and one 0
    """
    import logging
    worker_logger = logging.getLogger('distributed.worker')

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs['cloud']
    msk = mask_interp_inputs['msk']
    grid_points = mask_interp_inputs['grid_points']
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # create panda dataframe and search for neighbors
    cloud_pd = pandas.DataFrame(cloud,
                                columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK])

    neighbors_id, start_ids, n_count = \
        rasterization.get_flatten_neighbors(
            grid_points, cloud_pd,
            radius, resolution, worker_logger)

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud, data_valid.astype(np.bool), neighbors_id, start_ids, n_count,
        grid_points, sigma, nodata_val, undefined_val)

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case2(mask_interp_inputs):
    """
    case 2 - add several points from a second class aiming the same terrain cell
    """
    import logging
    worker_logger = logging.getLogger('distributed.worker')

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs['cloud']
    msk = mask_interp_inputs['msk']
    grid_points = mask_interp_inputs['grid_points']
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
        np.linspace(tgt_terrain_cell_x_coord-0.3,
                    tgt_terrain_cell_x_coord+0.3,
                    col_additional_pts_nb),
        np.linspace(tgt_terrain_cell_y_coord-0.3,
                    tgt_terrain_cell_y_coord+0.3,
                    row_additional_pts_nb))

    additional_pts_msk = np.full(
        (row_additional_pts_nb * col_additional_pts_nb),
        fill_value=200, dtype=np.float)

    cloud_case2 = np.zeros(
        (row_additional_pts_nb * col_additional_pts_nb, 3),
        dtype=np.float)
    cloud_case2[:, 0] = additional_pts_x_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb))
    cloud_case2[:, 1] = additional_pts_y_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb))
    cloud_case2[:, 2] = additional_pts_msk
    cloud_case2 = np.concatenate((cloud, cloud_case2), axis=0)

    data_valid_case2 = \
        np.concatenate((data_valid,
                        np.ones(
                            (row_additional_pts_nb * col_additional_pts_nb),
                            dtype=np.bool)), axis=0)

    # create panda dataframe and search for neighbors
    cloud_pd_case2 = pandas.DataFrame(
        cloud_case2, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK])

    neighbors_id, start_ids, n_count = \
        rasterization.get_flatten_neighbors(
            grid_points, cloud_pd_case2, radius, resolution, worker_logger)

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case2, data_valid_case2,
        neighbors_id, start_ids, n_count,
        grid_points, sigma, nodata_val, undefined_val)

    ref_msk = np.copy(msk)
    ref_msk[tgt_terrain_cell_y_coord, tgt_terrain_cell_x_coord] = 200

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)

@pytest.mark.unit_tests
def test_mask_interp_case3(mask_interp_inputs):
    """
    case 3 - only two points from different classes at the same position in a
    single cell
    """
    import logging
    worker_logger = logging.getLogger('distributed.worker')

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs['cloud']
    msk = mask_interp_inputs['msk']
    grid_points = mask_interp_inputs['grid_points']
    data_valid = mask_interp_inputs[cst.POINTS_CLOUD_VALID_DATA]
    radius = 0
    sigma = 1
    undefined_val = 254
    nodata_val = 255

    # only two points from different classes at the same position
    # in a single cell
    cloud_case3 = np.array([[2., 2., 200.]], dtype=np.float)
    cloud_case3 = np.concatenate((cloud, cloud_case3), axis=0)

    cloud_pd_case3 = pandas.DataFrame(
        cloud_case3, columns=[cst.X, cst.Y, 'left_mask'])

    data_valid_case3 = np.concatenate((data_valid,
                                       np.ones((1),
                                       dtype=np.bool)), axis=0)

    neighbors_id, start_ids, n_count = \
        rasterization.get_flatten_neighbors(
            grid_points, cloud_pd_case3,
            radius, resolution, worker_logger)

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case3, data_valid_case3,
        neighbors_id, start_ids, n_count,
        grid_points, sigma, nodata_val, undefined_val)

    ref_msk = np.copy(msk)
    ref_msk[2, 2] = undefined_val

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case4(mask_interp_inputs):
    """
    case 4 - no data cell
    """
    import logging
    worker_logger = logging.getLogger('distributed.worker')

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs['cloud']
    msk = mask_interp_inputs['msk']
    grid_points = mask_interp_inputs['grid_points']
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
        cloud_case4, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK])

    neighbors_id, start_ids, n_count = \
        rasterization.get_flatten_neighbors(
            grid_points, cloud_pd_case4, radius, resolution, worker_logger)

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case4, data_valid_case4,
        neighbors_id, start_ids, n_count,
        grid_points, sigma, nodata_val, undefined_val)

    ref_msk = np.copy(msk)
    ref_msk[int(cloud[1, 1]), int(cloud[1, 0])] = nodata_val

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(ref_msk, res)


@pytest.mark.unit_tests
def test_mask_interp_case5(mask_interp_inputs):
    """
    case 5 - add several points equal to 0 aiming the same terrain cell
    """
    import logging
    worker_logger = logging.getLogger('distributed.worker')

    # read fixture inputs and set parameters
    row = mask_interp_inputs[cst.ROW]
    col = mask_interp_inputs[cst.COL]
    resolution = mask_interp_inputs[cst.RESOLUTION]
    cloud = mask_interp_inputs['cloud']
    msk = mask_interp_inputs['msk']
    grid_points = mask_interp_inputs['grid_points']
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
        np.linspace(tgt_terrain_cell_x_coord - 0.3,
                    tgt_terrain_cell_x_coord + 0.3,
                    col_additional_pts_nb),
        np.linspace(tgt_terrain_cell_y_coord - 0.3,
                    tgt_terrain_cell_y_coord + 0.3,
                    row_additional_pts_nb))

    additional_pts_msk = np.full(
        (row_additional_pts_nb * col_additional_pts_nb),
        fill_value=0, dtype=np.float)

    cloud_case5 = np.zeros(
        (row_additional_pts_nb * col_additional_pts_nb, 3), dtype=np.float)
    cloud_case5[:, 0] = additional_pts_x_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb))
    cloud_case5[:, 1] = additional_pts_y_coords.reshape(
        (row_additional_pts_nb * col_additional_pts_nb))
    cloud_case5[:, 2] = additional_pts_msk
    cloud_case5 = np.concatenate((cloud, cloud_case5), axis=0)

    data_valid_case5 = \
        np.concatenate((data_valid,
                        np.ones(
                            (row_additional_pts_nb * col_additional_pts_nb),
                            dtype=np.bool)), axis=0)

    # create panda dataframe and search for neighbors
    cloud_pd_case5 = pandas.DataFrame(
        cloud_case5, columns=[cst.X, cst.Y, cst.POINTS_CLOUD_MSK])

    neighbors_id, start_ids, n_count = \
        rasterization.get_flatten_neighbors(
            grid_points, cloud_pd_case5, radius, resolution, worker_logger)

    # test mask_interp function
    res = rasterization.mask_interp(
        cloud_case5, data_valid_case5,
        neighbors_id, start_ids, n_count,
        grid_points, sigma, nodata_val, undefined_val)

    res = res.reshape((row, col))
    res = res[::-1, :]

    assert np.allclose(msk, res)
