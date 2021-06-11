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
Test module for cars/pipelines/wrappers.py
Important : Uses conftest.py for shared pytest fixtures
"""


# Third party imports
import pytest
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.pipelines import wrappers

# CARS Tests imports
from ..helpers import absolute_data_path, assert_same_datasets, create_corr_conf


@pytest.mark.unit_tests
def test_images_pair_to_3d_points(
    images_and_grids_conf,
    color1_conf,  # pylint: disable=redefined-outer-name
    no_data_conf,
    disparities_conf,  # pylint: disable=redefined-outer-name
    epipolar_origins_spacings_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test images_pair_to_3d_points on ventoux dataset (epipolar geometry)
    with Pandora
    Note: Fixtures parameters are set and shared in conftest.py
    """
    # With nodata and color
    configuration = images_and_grids_conf
    configuration["input"].update(color1_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_origins_spacings_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        disparities_conf["preprocessing"]["output"]
    )

    region = [420, 200, 530, 320]
    # Pandora configuration
    corr_cfg = create_corr_conf()

    cloud, __ = wrappers.images_pair_to_3d_points(
        configuration,
        region,
        corr_cfg,
        disp_min=-13,
        disp_max=14,
        add_msk_info=True,
    )

    # Uncomment to update baseline
    # cloud[cst.STEREO_REF].to_netcdf(
    # absolute_data_path("ref_output/cloud1_ref_pandora.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/cloud1_ref_pandora.nc")
    )
    assert_same_datasets(cloud[cst.STEREO_REF], ref, atol=1.0e-3)
