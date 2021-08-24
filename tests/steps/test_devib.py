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
Test module for cars/steps/devib.py
"""

# Standard imports
from __future__ import absolute_import

import pickle

# Third party imports
import numpy as np
import pytest
import xarray as xr

# CARS imports
from cars.steps import devib

# CARS Tests imports
from ..helpers import absolute_data_path


@pytest.mark.unit_tests
def test_lowres_initial_dem_splines_fit():
    """
    Test lowres_initial_dem_splines_fit
    """
    lowres_dsm_from_matches = xr.open_dataset(
        absolute_data_path("input/splines_fit_input/lowres_dsm_from_matches.nc")
    )
    lowres_initial_dem = xr.open_dataset(
        absolute_data_path("input/splines_fit_input/lowres_initial_dem.nc")
    )

    origin = [
        float(lowres_dsm_from_matches.x[0].values),
        float(lowres_dsm_from_matches.y[0].values),
    ]
    vec = [0, 1]

    splines = devib.lowres_initial_dem_splines_fit(
        lowres_dsm_from_matches, lowres_initial_dem, origin, vec
    )

    # Uncomment to update reference
    # with open(absolute_data_path(
    #                   "ref_output/splines_ref.pck"),'wb') as splines_files:
    #     pickle.dump(splines, splines_file)

    with open(
        absolute_data_path("ref_output/splines_ref.pck"), "rb"
    ) as splines_file:
        ref_splines = pickle.load(splines_file)
        np.testing.assert_allclose(
            splines.get_coeffs(), ref_splines.get_coeffs()
        )
        np.testing.assert_allclose(splines.get_knots(), ref_splines.get_knots())
