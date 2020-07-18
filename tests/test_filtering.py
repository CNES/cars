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

from cars import filtering
from utils import assert_same_datasets


@pytest.mark.unit_tests
def test_detect_small_components():
    # create fake cloud to process
    x = np.zeros((5, 5))
    x[4, 4] = 20
    x[0, 4] = 19.55
    x[0, 3] = 19.10
    y = np.zeros((5, 5))

    z = np.zeros((5, 5))
    z[0:2, 0:2] = 10
    z[1, 1] = 12

    cloud_arr = np.concatenate([np.stack((x, y, z), axis=-1).reshape(-1, 3)
                            for x, y, z in zip(x, y, z)
                            ], axis=0)

    indexes_to_filter = filtering.detect_small_components(cloud_arr, 0.5, 10, 2)
    assert sorted(indexes_to_filter ) == [3, 4, 24]

    # test without the second level of filtering
    indexes_to_filter = filtering.detect_small_components(cloud_arr, 0.5, 10, None)
    assert sorted(indexes_to_filter) == [0, 1, 3, 4, 5, 6, 24]


@pytest.mark.unit_tests
def test_detect_statistical_outliers():
    # create fake cloud to process
    x = np.zeros((5,6))
    off = 0
    for line in range(5):
        #x[line,:] = np.arange(off, off+(line+1)*5, line+1)
        last_val = off+5
        x[line,:5] = np.arange(off, last_val)
        off += (line + 2+1) * 5

        # outlier
        x[line, 5] = (off+last_val-1)/2

    y = np.zeros((5,6))
    z = np.zeros((5,6))

    ref_cloud = np.concatenate([np.stack((x, y, z), axis=-1).reshape(-1, 3)
                                for x, y, z in zip(x, y, z)
                                ], axis=0)

    removed_elt_pos = filtering.detect_statistical_outliers(ref_cloud, 4, 0.0)
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = filtering.detect_statistical_outliers(ref_cloud, 4, 1.0)
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = filtering.detect_statistical_outliers(ref_cloud, 4, 2.0)
    assert sorted(removed_elt_pos) == [23, 29]


@pytest.mark.unit_tests
def test_filter_cloud():
    cloud_arr = np.arange(6*10)
    cloud_arr = cloud_arr.reshape((10, 6))
    cloud = pandas.DataFrame(cloud_arr, columns=['x', 'y', 'z', 'coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])

    elt_to_remove = [0, 5]
    filtered_cloud, removed_elt_pos = filtering.filter_cloud(cloud, elt_to_remove, filtered_elt_pos=True)

    # reference
    pos_arr = np.zeros((len(elt_to_remove), 3), dtype=np.int)
    for i in range(len(elt_to_remove)):
        for j in range(3):
            # 3 last elements of each lines
            pos_arr[i, j] = int(6*elt_to_remove[i]+3+j)

    ref_removed_elt_pos = pandas.DataFrame(pos_arr, columns=['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])

    assert ref_removed_elt_pos.equals(removed_elt_pos)

    # test cases where removed_elt_pos should be None
    filtered_cloud, removed_elt_pos = filtering.filter_cloud(cloud, elt_to_remove, filtered_elt_pos=False)

    assert removed_elt_pos is None

    cloud_arr = np.arange(3 * 10)
    cloud_arr = cloud_arr.reshape((10, 3))

    cloud = pandas.DataFrame(cloud_arr, columns=['x', 'y', 'z'])
    filtered_cloud, removed_elt_pos = filtering.filter_cloud(cloud, elt_to_remove, filtered_elt_pos=True)

    assert removed_elt_pos is None


@pytest.mark.unit_tests
def test_add_cloud_filtering_msk():
    nb_row = 5
    nb_col = 10
    rows = np.array(range(nb_row))
    cols = np.array(range(nb_col))

    ds0 = xr.Dataset({}, coords={'row': rows, 'col': cols})
    ds1 = xr.Dataset({}, coords={'row': rows, 'col': cols})

    pos_arr = np.array([[1, 2, 0],
                       [2, 2, 1]])
    elt_remove = pandas.DataFrame(pos_arr, columns=['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])

    filtering.add_cloud_filtering_msk([ds0, ds1],elt_remove, 'mask', 255)

    mask0 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask0[1, 2] = 255
    mask1 = np.zeros((nb_row, nb_col), dtype=np.uint16)
    mask1[2, 2] = 255
    ds0_ref = xr.Dataset({'mask': (['row', 'col'], mask0)}, coords={'row': rows, 'col': cols})
    ds1_ref = xr.Dataset({'mask': (['row', 'col'], mask1)}, coords={'row': rows, 'col': cols})

    assert_same_datasets(ds0_ref, ds0)
    assert_same_datasets(ds1_ref, ds1)

    # test exceptions
    with pytest.raises(Exception) as e:
        np_pos = np.array([[1, 2, 2],
                           [2, 2, 1]])
        elt_remove = pandas.DataFrame(np_pos, columns=['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])
        filtering.add_cloud_filtering_msk([ds0, ds1], elt_remove, 'mask', 255)
        assert str(e) == 'Index indicated in the elt_pos_infos pandas.DataFrame is not coherent ' \
                         'with the clouds list given in input'

    with pytest.raises(Exception) as e:
        np_pos = np.array([[1, 2, -1],
                           [2, 2, 1]])
        elt_remove = pandas.DataFrame(np_pos, columns=['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])
        filtering.add_cloud_filtering_msk([ds0, ds1], elt_remove, 'mask', 255)
        assert str(e) == 'Index indicated in the elt_pos_infos pandas.DataFrame is not coherent ' \
                         'with the clouds list given in input'

    with pytest.raises(Exception) as e:
        np_pos = np.array([[11, 2, 0]])
        elt_remove = pandas.DataFrame(np_pos, columns=['coord_epi_geom_i', 'coord_epi_geom_j', 'idx_im_epi'])
        filtering.add_cloud_filtering_msk([ds0, ds1], elt_remove, 'mask', 255)
        assert str(e) == 'Point at location (11,2) is not accessible in an image of size (5,10)'
