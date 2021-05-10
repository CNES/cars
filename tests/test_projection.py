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
Test module for cars/projection.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import pandas
import pytest
from shapely.geometry import Polygon

# CARS imports
from cars.core import inputs, projection

# CARS Tests imports
from .utils import absolute_data_path


@pytest.mark.unit_tests
def test_point_cloud_conversion():
    """
    Create fake and right cloud and test points_cloud_conversion function
    """
    llh = np.load(absolute_data_path("input/rasterization_input/llh.npy"))

    # points_cloud_conversion expects a list of points
    llh = np.reshape(llh, (-1, 3))
    utm = projection.points_cloud_conversion(llh, 4326, 32630)

    assert len(utm) == llh.shape[0]

    utm_ref = np.load(absolute_data_path("ref_output/utm_cloud.npy"))
    np.testing.assert_allclose(utm, utm_ref)


@pytest.mark.unit_tests
def test_point_cloud_conversion_dataframe():
    """
    Create fake and right point cloud and test points_cloud_conversion_dataframe
    """
    llh = np.load(absolute_data_path("input/rasterization_input/llh.npy"))

    # points_cloud_conversion expects a list of points
    llh = np.reshape(llh, (-1, 3))
    utm_df = pandas.DataFrame(llh, columns=["x", "y", "z"])
    projection.points_cloud_conversion_dataframe(utm_df, 4326, 32630)

    assert len(utm_df.loc[:, ["x", "y", "z"]].values) == llh.shape[0]

    utm_ref = np.load(absolute_data_path("ref_output/utm_cloud.npy"))
    np.testing.assert_allclose(utm_df.loc[:, ["x", "y", "z"]].values, utm_ref)


@pytest.mark.unit_tests
def test_compute_dem_intersection_with_poly():
    """
    Test compute_dem_intersection_with_poly with right and fake configs
    """
    # test 100% coverage
    inter_poly, inter_epsg = inputs.read_vector(
        absolute_data_path("input/utils_input/envelopes_intersection.gpkg")
    )

    dem_inter_poly, cover = projection.compute_dem_intersection_with_poly(
        absolute_data_path("input/phr_ventoux/srtm"), inter_poly, inter_epsg
    )
    assert dem_inter_poly == inter_poly
    assert cover == 100.0

    # test partial coverage over with several srtm tiles with no data holes
    inter_poly = Polygon(
        [(4.8, 44.2), (4.8, 44.3), (6.2, 44.3), (6.2, 44.2), (4.8, 44.2)]
    )
    dem_inter_poly, cover = projection.compute_dem_intersection_with_poly(
        absolute_data_path("input/utils_input/srtm_with_hole"),
        inter_poly,
        inter_epsg,
    )

    ref_dem_inter_poly = Polygon(
        [
            (4.999583333333334, 44.2),
            (4.999583333333334, 44.3),
            (6.2, 44.3),
            (6.2, 44.2),
            (4.999583333333334, 44.2),
        ]
    )

    assert dem_inter_poly.exterior == ref_dem_inter_poly.exterior
    assert len(list(dem_inter_poly.interiors)) == 6
    assert cover == 85.72172619047616

    # test no coverage
    inter_poly = Polygon(
        [(1.5, 2.0), (1.5, 2.1), (1.8, 2.1), (1.8, 2.0), (1.5, 2.0)]
    )

    with pytest.raises(Exception) as intersect_error:
        dem_inter_poly, cover = projection.compute_dem_intersection_with_poly(
            absolute_data_path("input/phr_ventoux/srtm"), inter_poly, inter_epsg
        )
    assert (
        str(intersect_error.value) == "The input DEM does not intersect "
        "the useful zone"
    )
