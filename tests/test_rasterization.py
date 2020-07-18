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

import pytest
import pandas

import numpy as np
import xarray as xr

from cars import rasterization, projection
from utils import absolute_data_path, temporary_dir, assert_same_images, assert_same_datasets


@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    epsg = rasterization.get_utm_zone_as_epsg_code(1.442299, 43.600764)
    assert epsg == 32631


@pytest.mark.unit_tests
def test_create_combined_cloud():

    # test only color
    epsg = 4326
    row = 10
    col = 10
    x = np.arange(row*col)
    x = x.reshape((row, col))
    y = x + 1
    z = y + 1
    msk = np.full((row, col), fill_value=255, dtype=np.int16)
    msk[4, 4] = 0

    cloud0 = xr.Dataset({'x': (['row', 'col'], x),
                         'y': (['row', 'col'], y),
                         'z': (['row', 'col'], z),
                         'msk': (['row', 'col'], msk)},
                        coords={'row': np.array(range(row)), 'col': np.array(range(col))})
    cloud0.attrs['epsg'] = epsg

    x = np.full((row, col), fill_value=0, dtype=np.float)
    y = np.full((row, col), fill_value=1, dtype=np.float)
    z = np.full((row, col), fill_value=2, dtype=np.float)
    msk = np.full((row, col), fill_value=255, dtype=np.int16)
    msk[6, 6] = 0

    cloud1 = xr.Dataset({'x': (['row', 'col'], x),
                         'y': (['row', 'col'], y),
                         'z': (['row', 'col'], z),
                         'msk': (['row', 'col'], msk)},
                        coords={'row': np.array(range(row)), 'col': np.array(range(col))})
    cloud1.attrs['epsg'] = epsg

    row = 5
    col = 5
    x = np.full((row, col), fill_value=45, dtype=np.float)
    y = np.full((row, col), fill_value=45, dtype=np.float)
    z = np.full((row, col), fill_value=50, dtype=np.float)
    msk = np.full((row, col), fill_value=255, dtype=np.int16)
    msk[2, 2] = 0

    cloud2 = xr.Dataset({'x': (['row', 'col'], x),
                         'y': (['row', 'col'], y),
                         'z': (['row', 'col'], z),
                         'msk': (['row', 'col'], msk)},
                        coords={'row': np.array(range(row)), 'col': np.array(range(col))})
    cloud2.attrs['epsg'] = epsg

    cloud_list = [cloud0, cloud1, cloud2]

    cloud, epsg = rasterization.create_combined_cloud(cloud_list, epsg, color_list=None, resolution=0.5, xstart=40.0,
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

    # test with color
    band = 3
    row = 10
    col = 10
    clr0 = np.zeros((band, row, col))
    clr0[0, :, :] = 10
    clr0[1, :, :] = 20
    clr0[2, :, :] = 30
    clr0 = xr.Dataset({'im': (['band', 'row', 'col'], clr0)},
                      coords={'band': np.array(range(band)), 'row': np.array(range(row)), 'col': np.array(range(col))})

    clr1 = np.full((band, row, col), fill_value=20)
    clr1 = xr.Dataset({'im': (['band', 'row', 'col'], clr1)},
                      coords={'band': np.array(range(band)), 'row': np.array(range(row)), 'col': np.array(range(col))})

    row = 5
    col = 5
    clr2 = np.zeros((band, row, col))
    clr2[0, :, :] = np.arange(row*col).reshape((row, col))
    clr2[1, :, :] = clr2[0, :, :] + 1
    clr2[2, :, :] = clr2[1, :, :] + 1
    clr2 = xr.Dataset({'im': (['band', 'row', 'col'], clr2)},
                      coords={'band': np.array(range(band)), 'row': np.array(range(row)), 'col': np.array(range(col))})

    clr_list = [clr0, clr1, clr2]

    cloud, epsg = rasterization.create_combined_cloud(cloud_list, epsg, color_list=clr_list, resolution=0.5, xstart=40.0,
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
    cloud, epsg = rasterization.create_combined_cloud(cloud_list, epsg, color_list=clr_list, resolution=0.5, xstart=40.0,
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
    ref_cloud_clr_coords0 = np.concatenate([ref_cloud0, ref_clr0, ref_coords0], axis=1)

    ref_coords2 = np.zeros((row*col, 3))
    ref_coords2[:, 2] = 2
    for i in range(row):
        for j in range(col):
            ref_coords2[i*col+j, 0] = i
            ref_coords2[i*col+j, 1] = j
    ref_coords2 = np.delete(ref_coords2, 2 * col + 2, 0)
    ref_cloud_clr_coords2 = np.concatenate([ref_cloud2, ref_clr2, ref_coords2], axis=1)

    ref_cloud_clr_coords = np.concatenate([ref_cloud_clr_coords0, ref_cloud_clr_coords2])

    assert np.allclose(cloud, ref_cloud_clr_coords)

    # test with coords (no colors)
    cloud, epsg = rasterization.create_combined_cloud(cloud_list, epsg, color_list=None, resolution=0.5, xstart=40.0,
                                                ystart=50.0, xsize=20, ysize=25, on_ground_margin=1,
                                                epipolar_border_margin=1, radius=1, with_coords=True)

    ref_cloud_coords0 = np.concatenate([ref_cloud0, ref_coords0], axis=1)
    ref_cloud_coords2 = np.concatenate([ref_cloud2, ref_coords2], axis=1)
    ref_cloud_coords = np.concatenate([ref_cloud_coords0, ref_cloud_coords2])

    assert np.allclose(cloud, ref_cloud_coords)

    # test exception
    with pytest.raises(Exception) as e:
        rasterization.create_combined_cloud(cloud_list, epsg, color_list=[clr0], resolution=0.5, xstart=40.0,
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

    pd_cloud = pandas.DataFrame(cloud, columns=['data_valid', 'x', 'y', 'z'])
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, 0.3, 0)

    # Test with radius = 0 and fixed grid
    np.testing.assert_equal(raster.hgt.values[..., None],
                            [[[3], [4], [5]], [[0], [1], [2]]])
    np.testing.assert_array_equal(raster.n_pts.values, np.ones((2, 3)))
    np.testing.assert_array_equal(raster.pts_in_cell.values, np.ones((2, 3)))
    np.testing.assert_array_equal(np.ravel(np.flipud(raster.hgt_mean.values)),
                                  pd_cloud['z'])
    np.testing.assert_array_equal(raster.hgt_stdev.values,
                                  np.zeros((2, 3)))

    # Test with grid evaluated by function
    xstart, ystart, xsize, ysize = rasterization.compute_xy_starts_and_sizes(1, pd_cloud)
    raster = rasterization.rasterize(pd_cloud, 1, None, xstart, ystart, xsize, ysize, 0.3, 0)
    np.testing.assert_equal(raster.hgt.values[..., None],
                            [[[3], [4], [5]], [[0], [1], [2]]])

    # Test with fixed grid, radius =  1 and sigma = inf
    raster = rasterization.rasterize(pd_cloud, 1, None, 0, 12, 3, 2, np.inf, 1)
    np.testing.assert_equal(raster.hgt.values[..., None],
                            [[[2], [2.5], [3]], [[2], [2.5], [3]]])
    np.testing.assert_array_equal(raster.hgt_mean.values,
                                  [[2.0, 2.5, 3.0], [2.0, 2.5, 3.0]])
    np.testing.assert_array_equal(raster.pts_in_cell,
                                  np.ones((2, 3), dtype=int))
    np.testing.assert_array_equal(raster.n_pts,
                                  [[4, 6, 4], [4, 6, 4]])


@pytest.mark.unit_tests
def test_simple_rasterization_single():

    resolution = 0.5

    cloud_xr = xr.open_dataset(
        absolute_data_path('input/rasterization_input/ref_single_cloud_in_df.nc')
    )
    cloud_df = cloud_xr.to_dataframe()

    xstart, ystart, xsize, ysize = rasterization.compute_xy_starts_and_sizes(resolution, cloud_df)
    raster = rasterization.rasterize(cloud_df, resolution, 32630, xstart, ystart, xsize, ysize, 0.3, 3)

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
        absolute_data_path("input/intermediate_results/cloud1_ref.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    utm = projection.points_cloud_conversion_dataset(cloud, 32630)

    xstart = 1154790
    ystart = 4927552
    xsize = 114
    ysize = 112
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [utm], resolution, 32630, [color], xstart, ystart, xsize, ysize, 0.3, 3
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
        absolute_data_path("input/intermediate_results/cloud1_ref.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    utm = projection.points_cloud_conversion_dataset(cloud, 32630)

    xstart = None
    ystart = None
    xsize = None
    ysize = None
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [utm], resolution, 32630, [color], xstart, ystart, xsize, ysize, 0.3, 3
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
        absolute_data_path("input/intermediate_results/cloud1_ref.nc")
    )
    color = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_clr.nc")
    )

    utm = projection.points_cloud_conversion_dataset(cloud, 32630)

    utm1 = utm.isel(row=range(0, 60))
    utm2 = utm.isel(row=range(60, 120))

    color1 = color.isel(row=range(0, 60))
    color2 = color.isel(row=range(60, 120))

    xstart = 1154790
    ystart = 4927552
    xsize = 114
    ysize = 112
    resolution = 0.5

    raster = rasterization.simple_rasterization_dataset(
        [utm1, utm2], resolution, 32630, [color1, color2], xstart, ystart, xsize,
        ysize, 0.3, 3
    )

    # Uncomment to update reference
    # raster.to_netcdf(
    #     absolute_data_path('ref_output/rasterization_multiple_res_ref.nc'),
    # )

    raster_ref = xr.open_dataset(
        absolute_data_path("ref_output/rasterization_multiple_res_ref.nc")
    )
    assert_same_datasets(raster, raster_ref, atol=1.e-10, rtol=1.e-10)
