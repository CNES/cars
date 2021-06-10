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
Test module for cars/steps/triangulation.py
Important : Uses conftest.py for shared pytest fixtures
"""

# Third party imports
import numpy as np
import pytest
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.core.inputs import read_geoid_file
from cars.steps import triangulation

# CARS Tests imports
from ..helpers import (
    absolute_data_path,
    assert_same_datasets,
    otb_geoid_file_set,
    otb_geoid_file_unset,
)


@pytest.mark.unit_tests
def test_triangulation_1(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test triangulation ventoux dataset
    """
    disp1_ref = xr.open_dataset(
        absolute_data_path("input/intermediate_results/disp1_ref.nc")
    )
    point_cloud_dict = triangulation.triangulate(
        images_and_grids_conf, disp1_ref, None
    )

    assert point_cloud_dict[cst.STEREO_REF][cst.X].shape == (120, 110)

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(
    # absolute_data_path("ref_output/triangulation1_ref.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/triangulation1_ref.nc")
    )
    assert_same_datasets(point_cloud_dict[cst.STEREO_REF], ref, atol=1.0e-3)


@pytest.mark.unit_tests
def test_triangulate_matches(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test triangulate_matches function from images_and_grids_conf
    """

    matches = np.array([[0.0, 0.0, 0.0, 0.0]])

    llh = triangulation.triangulate_matches(images_and_grids_conf, matches)

    # Check properties
    assert llh.dims == {cst.ROW: 1, cst.COL: 1}
    np.testing.assert_almost_equal(llh.x.values[0], 5.1973629)
    np.testing.assert_almost_equal(llh.y.values[0], 44.2079813)
    np.testing.assert_almost_equal(llh.z.values[0], 511.4383088)
    assert llh[cst.POINTS_CLOUD_CORR_MSK].values[0] == 255
    assert cst.EPSG in llh.attrs
    assert llh.attrs[cst.EPSG] == 4326


@pytest.mark.unit_tests
def test_geoid_offset():
    """
    Returns test result of reference and computed geoid comparison
    """
    # ref file contains 32x32 points issued from proj 6.2
    ref_file = absolute_data_path("ref_output/egm96_15_ref_hgt.nc")

    geoid_ref = xr.open_dataset(ref_file)

    # create a zero elevation Dataset with the same geodetic coordinates
    points = xr.Dataset(
        {
            cst.X: geoid_ref.x,
            cst.Y: geoid_ref.y,
            cst.Z: ((cst.ROW, cst.COL), np.zeros_like(geoid_ref.z)),
        }
    )

    # Set the geoid file from code source
    otb_geoid_file_set()

    geoid = read_geoid_file()

    computed_geoid = triangulation.geoid_offset(points, geoid)

    assert np.allclose(
        geoid_ref.z.values, computed_geoid.z.values, atol=1e-3, rtol=1e-12
    )

    # Unset geoid for the test to be standalone
    otb_geoid_file_unset()
