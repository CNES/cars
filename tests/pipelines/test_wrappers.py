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


import pickle

# Third party imports
import pytest
from pandora.img_tools import get_metadata
from pandora.margins import Margins

# CARS imports
from cars.applications.dense_matching import (
    dense_matching_wrappers as dense_match_wrappers,
)
from cars.applications.dense_matching.census_mccnn_sgm_app import (
    CensusMccnnSgm,
    compute_disparity_wrapper,
)
from cars.applications.resampling.bicubic_resampling_app import (
    generate_epipolar_images_wrapper,
)
from cars.applications.triangulation.line_of_sight_intersection_app import (
    triangulation_wrapper,
)
from cars.conf import input_parameters as in_params

# CARS Tests imports
from ..helpers import (
    absolute_data_path,
    assert_same_datasets,
    corr_conf_defaut,
    create_corr_conf,
    get_geometry_plugin,
)


@pytest.mark.unit_tests
def test_epipolar_pipeline(
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

    img1 = configuration["input"][in_params.IMG1_TAG]
    img2 = configuration["input"][in_params.IMG2_TAG]
    geomodel1 = {
        "path": configuration["input"][in_params.MODEL1_TAG],
        "model_type": configuration["input"][in_params.MODEL1_TYPE_TAG],
    }
    geomodel2 = {
        "path": configuration["input"][in_params.MODEL2_TAG],
        "model_type": configuration["input"][in_params.MODEL2_TYPE_TAG],
    }
    grid1 = {
        "path": configuration["preprocessing"]["output"]["left_epipolar_grid"]
    }
    grid2 = {
        "path": configuration["preprocessing"]["output"]["right_epipolar_grid"]
    }
    nodata1 = configuration["input"].get(in_params.NODATA1_TAG, None)
    nodata2 = configuration["input"].get(in_params.NODATA2_TAG, None)
    mask1 = configuration["input"].get(in_params.MASK1_TAG, None)
    mask2 = configuration["input"].get(in_params.MASK2_TAG, None)

    region = [420, 200, 530, 320]
    # Pandora configuration
    corr_cfg = corr_conf_defaut()
    left_input = get_metadata(img1)
    right_input = get_metadata(img2)
    left_input = left_input.assign_coords(band_im=["b0"])
    right_input = right_input.assign_coords(band_im=["b0"])
    corr_cfg = create_corr_conf(corr_cfg, left_input, right_input)

    global_disp_min = -19
    global_disp_max = 15

    # cumulative margins between sgm (40) and matching_cost (ws/2)
    margins = Margins(42, 42, 42, 42)

    initial_margins = dense_match_wrappers.get_margins(
        margins, global_disp_min, global_disp_max
    )
    pandora_margins = initial_margins["left_margin"].values

    # overlaps are directly margins because current region is in the center
    overlaps = {
        "left": pandora_margins[0],
        "up": pandora_margins[1],
        "right": pandora_margins[2],
        "down": pandora_margins[3],
    }

    window = {
        "col_min": region[0],
        "row_min": region[1],
        "col_max": region[2],
        "row_max": region[3],
    }

    # retrieves some data
    epipolar_size_x = configuration["preprocessing"]["output"][
        "epipolar_size_x"
    ]
    epipolar_size_y = configuration["preprocessing"]["output"][
        "epipolar_size_y"
    ]

    left_imgs = {img1: {"band_name": ["b0"], "band_id": [1]}}
    img2 = configuration["input"][in_params.IMG2_TAG]
    right_imgs = {img2: {"band_name": ["b0"], "band_id": [1]}}

    left_image, right_image = generate_epipolar_images_wrapper(
        overlaps,
        overlaps,
        window,
        epipolar_size_x,
        epipolar_size_y,
        left_imgs,
        right_imgs,
        grid1,
        grid2,
        interpolator_image="bicubic",
        interpolator_classif="nearest",
        interpolator_mask="nearest",
        used_disp_min=global_disp_min,
        used_disp_max=global_disp_max,
        mask1=mask1,
        mask2=mask2,
        nodata1=nodata1,
        nodata2=nodata2,
    )

    right_grid = {
        "epipolar_size_x": epipolar_size_x,
        "epipolar_size_y": epipolar_size_y,
        "disp_to_alt_ratio": None,
    }
    dense_matching_app = CensusMccnnSgm()
    # Overide margin
    dense_matching_app.disparity_margin = 0
    disp_range_grid = dense_matching_app.generate_disparity_grids(
        None,
        right_grid,
        None,
        dmin=global_disp_min,
        dmax=global_disp_max,
        pair_folder=None,
    )

    disp_map = compute_disparity_wrapper(
        left_image,
        right_image,
        corr_cfg,
        "b0",
        disp_range_grid,
        texture_bands=[0],
    )

    epipolar_point_cloud, _ = triangulation_wrapper(
        disp_map,
        img1,
        img2,
        geomodel1,
        geomodel2,
        grid1,
        grid2,
        get_geometry_plugin(
            conf={"plugin_name": "SharelocGeometry", "interpolator": "linear"}
        ),
        32636,
    )

    # Uncomment to update reference
    # with open(
    #     absolute_data_path("ref_output/cloud1_ref_pandora"), "wb"
    # ) as epipolar_point_cloud_file:
    #     pickle.dump(epipolar_point_cloud, epipolar_point_cloud_file)

    with open(
        absolute_data_path("ref_output/cloud1_ref_pandora"), "rb"
    ) as epipolar_point_cloud_file:
        ref_pc = pickle.load(epipolar_point_cloud_file)
        assert_same_datasets(epipolar_point_cloud, ref_pc, atol=1.0e-3)
