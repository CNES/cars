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
Test module for cars/steps/triangulation_tools.py
Important : Uses conftest.py for shared pytest fixtures
"""

# Third party imports
import numpy as np
import pandas
import pytest
import xarray as xr

from cars.applications.triangulation import triangulation_tools

# CARS imports
from cars.core import constants as cst

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_datasets,
    get_geoid_path,
    get_geometry_plugin,
)


@pytest.mark.unit_tests
def test_triangulation_ventoux_shareloc(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test triangulation ventoux dataset
    """
    disp1_ref = xr.open_dataset(
        absolute_data_path("input/intermediate_results/disp1_ref.nc")
    )
    sensor1 = images_and_grids_conf["input"]["img1"]
    sensor2 = images_and_grids_conf["input"]["img2"]
    geomodel1 = {
        "path": images_and_grids_conf["input"]["model1"],
        "model_type": images_and_grids_conf["input"]["model_type1"],
    }
    geomodel2 = {
        "path": images_and_grids_conf["input"]["model2"],
        "model_type": images_and_grids_conf["input"]["model_type2"],
    }
    grid_left = images_and_grids_conf["preprocessing"]["output"][
        "left_epipolar_grid"
    ]
    grid_right = images_and_grids_conf["preprocessing"]["output"][
        "right_epipolar_grid"
    ]

    point_cloud_dict = triangulation_tools.triangulate(
        get_geometry_plugin(),
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        disp1_ref,
    )

    assert point_cloud_dict[cst.STEREO_REF][cst.X].shape == (120, 110)

    # Uncomment to update baseline
    # point_cloud_dict[cst.STEREO_REF].to_netcdf(
    # absolute_data_path("ref_output/triangulation1_ref.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/triangulation1_ref.nc")
    )
    assert_same_datasets(point_cloud_dict[cst.STEREO_REF], ref, atol=1.0e-3)


@pytest.mark.unit_tests
def test_triangulate_matches_shareloc(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test triangulate_matches function from images_and_grids_conf
    """

    matches = np.array([[0.0, 0.0, 0.0, 0.0]])
    sensor1 = images_and_grids_conf["input"]["img1"]
    sensor2 = images_and_grids_conf["input"]["img2"]
    geomodel1 = {
        "path": images_and_grids_conf["input"]["model1"],
        "model_type": images_and_grids_conf["input"]["model_type1"],
    }
    geomodel2 = {
        "path": images_and_grids_conf["input"]["model2"],
        "model_type": images_and_grids_conf["input"]["model_type2"],
    }
    grid_left = images_and_grids_conf["preprocessing"]["output"][
        "left_epipolar_grid"
    ]
    grid_right = images_and_grids_conf["preprocessing"]["output"][
        "right_epipolar_grid"
    ]

    llh = triangulation_tools.triangulate_matches(
        get_geometry_plugin(),
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        matches,
    )
    # Check properties
    pandas.testing.assert_index_equal(
        llh.columns, pandas.Index(["x", "y", "z", "disparity", "corr_msk"])
    )
    # put decimal values to 10 to know if modifications are done.
    # for long/lat, 10**(-8) have been checked
    np.testing.assert_almost_equal(llh.x[0], 5.197378451485809, decimal=10)
    np.testing.assert_almost_equal(llh.y[0], 44.20798042552038, decimal=10)
    # for altitude, 10**(-3) have been checked
    np.testing.assert_almost_equal(llh.z[0], 512.8074492644519, decimal=10)
    # np.testing.assert_almost_equal(llh.z[0], 511.4383088)
    assert llh[cst.DISPARITY][0] == 0.0
    assert llh[cst.POINT_CLOUD_CORR_MSK][0] == 255


@pytest.mark.parametrize(
    "geoid_path",
    [
        get_geoid_path(),  # default geoid (.gdr ENVI-HDR file)
        absolute_data_path("input/geoid/egm96_15.tif"),
    ],
)
@pytest.mark.unit_tests
def test_geoid_offset_from_xarray(geoid_path):
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
    computed_geoid = triangulation_tools.geoid_offset(points, geoid_path)

    # Note: the two versions of the egm96 geoid file are slightly different,
    # the tolerance used should be sufficient here.
    assert np.allclose(
        geoid_ref.z.values, computed_geoid.z.values, atol=1e-3, rtol=1e-12
    )


@pytest.mark.unit_tests
def test_geoid_offset_from_pandas():
    """
    Returns test result of reference and computed geoid comparison
    """
    # ref file contains 32x32 points issued from proj 6.2
    ref_file = absolute_data_path("ref_output/egm96_15_ref_hgt.nc")

    geoid_ref = xr.open_dataset(ref_file)

    # create a zero elevation Dataset with the same geodetic coordinates
    data = {
        cst.X: np.ravel(geoid_ref.x),
        cst.Y: np.ravel(geoid_ref.y),
        cst.Z: np.zeros_like(np.ravel(geoid_ref.x)),
    }
    points = pandas.DataFrame(data=data)

    computed_geoid = triangulation_tools.geoid_offset(points, get_geoid_path())

    # Test outside border where it is nan
    assert np.allclose(
        np.ravel(geoid_ref.z.values)[~np.isnan(computed_geoid.z.values)],
        computed_geoid.z.values[~np.isnan(computed_geoid.z.values)],
        atol=1e-1,
        rtol=1e-1,
    )


@pytest.mark.unit_tests
def test_triangulation_intervals_shareloc(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test triangulation with disparity intervals on ventoux dataset
    """
    disp1_ref = xr.open_dataset(
        absolute_data_path("input/intermediate_results/disp1_ref.nc")
    )
    disp1_ref["confidence_from_interval_bounds_inf.int"] = disp1_ref["disp"] - 1
    disp1_ref["confidence_from_interval_bounds_sup.int"] = disp1_ref["disp"] + 1

    sensor1 = images_and_grids_conf["input"]["img1"]
    sensor2 = images_and_grids_conf["input"]["img2"]
    geomodel1 = {"path": images_and_grids_conf["input"]["model1"]}
    geomodel2 = {"path": images_and_grids_conf["input"]["model2"]}
    grid_left = images_and_grids_conf["preprocessing"]["output"][
        "left_epipolar_grid"
    ]
    grid_right = images_and_grids_conf["preprocessing"]["output"][
        "right_epipolar_grid"
    ]

    point_cloud_dict = triangulation_tools.triangulate(
        get_geometry_plugin(),
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        disp1_ref,
    )

    point_cloud_dict_inf = triangulation_tools.triangulate(
        get_geometry_plugin(),
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        disp1_ref,
        disp_key="confidence_from_interval_bounds_inf.int",
    )

    point_cloud_dict_sup = triangulation_tools.triangulate(
        get_geometry_plugin(),
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        disp1_ref,
        disp_key="confidence_from_interval_bounds_sup.int",
    )

    point_cloud_dict[cst.STEREO_REF][cst.Z_INF] = point_cloud_dict_inf[
        cst.STEREO_REF
    ][cst.Z]
    point_cloud_dict[cst.STEREO_REF][cst.Z_SUP] = point_cloud_dict_sup[
        cst.STEREO_REF
    ][cst.Z]

    assert point_cloud_dict[cst.STEREO_REF][cst.X].shape == (120, 110)

    # Uncomment to update baseline
    # point_cloud_dict[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/triangulation1_ref_intervals.nc"))
    ref = xr.open_dataset(
        absolute_data_path("ref_output/triangulation1_ref_intervals.nc")
    )
    assert_same_datasets(point_cloud_dict[cst.STEREO_REF], ref, atol=1.0e-3)
