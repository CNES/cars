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
Test module for cars/filtering.py
"""

from __future__ import absolute_import
import pytest
import pandas

import numpy as np
import xarray as xr

from cars.lib.steps import points_cloud
from .utils import assert_same_datasets



@pytest.mark.unit_tests
def test_detect_small_components():
    """
    Create fake cloud to process and test detect_small_components
    """
    x_coord = np.zeros((5, 5))
    x_coord[4, 4] = 20
    x_coord[0, 4] = 19.55
    x_coord[0, 3] = 19.10
    y_coord = np.zeros((5, 5))

    z_coord = np.zeros((5, 5))
    z_coord[0:2, 0:2] = 10
    z_coord[1, 1] = 12

    cloud_arr = np.concatenate(
                [np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
                for x_coord, y_coord, z_coord in zip(x_coord, y_coord, z_coord)
                ], axis=0)

    indexes_to_filter = points_cloud.detect_small_components(
        cloud_arr, 0.5, 10, 2)
    assert sorted(indexes_to_filter ) == [3, 4, 24]

    # test without the second level of filtering
    indexes_to_filter = points_cloud.detect_small_components(
        cloud_arr, 0.5, 10, None)
    assert sorted(indexes_to_filter) == [0, 1, 3, 4, 5, 6, 24]


@pytest.mark.unit_tests
def test_detect_statistical_outliers():
    """
    Create fake cloud to process and test detect_statistical_outliers
    """
    x_coord = np.zeros((5,6))
    off = 0
    for line in range(5):
        #x[line,:] = np.arange(off, off+(line+1)*5, line+1)
        last_val = off+5
        x_coord[line,:5] = np.arange(off, last_val)
        off += (line + 2+1) * 5

        # outlier
        x_coord[line, 5] = (off+last_val-1)/2

    y_coord = np.zeros((5,6))
    z_coord = np.zeros((5,6))

    ref_cloud = np.concatenate(
                [np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
                for x_coord, y_coord, z_coord in zip(x_coord, y_coord, z_coord)
                ], axis=0)

    removed_elt_pos = points_cloud.detect_statistical_outliers(ref_cloud,
                                                               4, 0.0)
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = points_cloud.detect_statistical_outliers(ref_cloud,
                                                               4, 1.0)
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = points_cloud.detect_statistical_outliers(ref_cloud,
                                                               4, 2.0)
    assert sorted(removed_elt_pos) == [23, 29]


@pytest.mark.unit_tests
def test_filter_cloud():
    """
    Create fake cloud and test filter_cloud function
    """
    cloud_arr = np.arange(6*10)
    cloud_arr = cloud_arr.reshape((10, 6))
    cloud = pandas.DataFrame(cloud_arr,
                             columns=['x', 'y', 'z',
                             'coord_epi_geom_i',
                             'coord_epi_geom_j',
                             'idx_im_epi'])

    elt_to_remove = [0, 5]
    __, removed_elt_pos = points_cloud.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=True)

    # reference
    pos_arr = np.zeros((len(elt_to_remove), 3), dtype=np.int)
    for elt_idx, elt_to_remove_item in enumerate(elt_to_remove):
        for last_column_idx in range(3):
            # 3 last elements of each lines
            pos_arr[elt_idx, last_column_idx] = \
                                    int(6*elt_to_remove_item+3+last_column_idx)

    ref_removed_elt_pos = pandas.DataFrame(pos_arr,
                                           columns=['coord_epi_geom_i',
                                                    'coord_epi_geom_j',
                                                    'idx_im_epi'])

    assert ref_removed_elt_pos.equals(removed_elt_pos)

    # test cases where removed_elt_pos should be None
    __, removed_elt_pos = points_cloud.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=False)

    assert removed_elt_pos is None

    cloud_arr = np.arange(3 * 10)
    cloud_arr = cloud_arr.reshape((10, 3))

    cloud = pandas.DataFrame(cloud_arr, columns=['x', 'y', 'z'])
    __, removed_elt_pos = points_cloud.filter_cloud(
        cloud, elt_to_remove, filtered_elt_pos=True)

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

    ds0 = xr.Dataset({}, coords={'row': rows, 'col': cols})
    ds1 = xr.Dataset({}, coords={'row': rows, 'col': cols})

    pos_arr = np.array([[1, 2, 0],
                       [2, 2, 1]])
    elt_remove = pandas.DataFrame(pos_arr,
                                  columns=['coord_epi_geom_i',
                                           'coord_epi_geom_j',
                                           'idx_im_epi'])

    points_cloud.add_cloud_filtering_msk([ds0, ds1],elt_remove, 'mask', 255)

    mask0 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask0[1, 2] = 255
    mask1 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask1[2, 2] = 255
    ds0_ref = xr.Dataset({'mask': (['row', 'col'], mask0)},
                         coords={'row': rows, 'col': cols})
    ds1_ref = xr.Dataset({'mask': (['row', 'col'], mask1)},
                         coords={'row': rows, 'col': cols})

    assert_same_datasets(ds0_ref, ds0)
    assert_same_datasets(ds1_ref, ds1)

    # test exceptions
    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[1, 2, 2],
                           [2, 2, 1]])
        elt_remove = pandas.DataFrame(np_pos,
                                      columns=['coord_epi_geom_i',
                                               'coord_epi_geom_j',
                                               'idx_im_epi'])
        points_cloud.add_cloud_filtering_msk([ds0, ds1],
                                             elt_remove,
                                             'mask',
                                             255)
        assert str(index_error) == 'Index indicated in the elt_pos_infos '\
                         'pandas.DataFrame is not coherent with the clouds '\
                         'list given in input'

    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[1, 2, -1],
                           [2, 2, 1]])
        elt_remove = pandas.DataFrame(np_pos,
                                      columns=['coord_epi_geom_i',
                                               'coord_epi_geom_j',
                                               'idx_im_epi'])
        points_cloud.add_cloud_filtering_msk([ds0, ds1],
                                             elt_remove,
                                             'mask',
                                             255)
        assert str(index_error) == 'Index indicated in the elt_pos_infos '\
                         'pandas.DataFrame is not coherent ' \
                         'with the clouds list given in input'

    with pytest.raises(Exception) as index_error:
        np_pos = np.array([[11, 2, 0]])
        elt_remove = pandas.DataFrame(np_pos,
                                      columns=['coord_epi_geom_i',
                                               'coord_epi_geom_j',
                                               'idx_im_epi'])
        points_cloud.add_cloud_filtering_msk([ds0, ds1],
                                             elt_remove,
                                             'mask',
                                             255)
        assert str(index_error) == 'Point at location (11,2) is not '\
                                    'accessible in an image of size (5,10)'
