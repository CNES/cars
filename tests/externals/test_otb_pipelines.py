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
Test module for cars/externals/otb_pipelines.py
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import pytest
import xarray as xr

# CARS imports
from cars.externals import otb_pipelines

# CARS Tests imports
from ..helpers import absolute_data_path, assert_same_datasets, get_geoid_path


@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    epsg = otb_pipelines.get_utm_zone_as_epsg_code(1.442299, 43.600764)
    assert epsg == 32631


@pytest.mark.unit_tests
def test_read_lowres_dem():
    """
    Test read_lowres_dem function
    """
    dem = absolute_data_path("input/phr_ventoux/srtm")
    startx = 5.193458
    starty = 44.206671
    sizex = 100
    sizey = 100

    srtm_ds = otb_pipelines.read_lowres_dem(
        startx, starty, sizex, sizey, dem=dem, geoid=get_geoid_path()
    )

    # Uncomment to update baseline
    # srtm_ds.to_netcdf(absolute_data_path("ref_output/srtm_xt.nc"))

    srtm_ds_ref = xr.open_dataset(absolute_data_path("ref_output/srtm_xt.nc"))
    assert_same_datasets(srtm_ds, srtm_ds_ref)
