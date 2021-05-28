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
Test module for cars/stereo.py
TODO: refactor in several files and remove too-many-lines
"""
# pylint: disable=too-many-lines

import json
import math
from copy import deepcopy

import numpy as np
import pandora
import pytest
import xarray as xr
from pandora.check_json import (
    check_pipeline_section,
    concat_conf,
    get_config_pipeline,
)
from pandora.state_machine import PandoraMachine

from cars.core import constants as cst
from cars.core import tiling
from cars.core.inputs import read_geoid_file
from cars.externals.matching.correlator_configuration.corr_conf import (
    check_input_section_custom_cars,
    get_config_input_custom_cars,
)
from cars.pipelines import wrappers
from cars.steps import triangulation
from cars.steps.epi_rectif import resampling
from cars.steps.matching import dense_matching

from .utils import (
    absolute_data_path,
    assert_same_datasets,
    otb_geoid_file_set,
    otb_geoid_file_unset,
)

# Local testing stereo function pytest fixtures
# Ease following stereo tests readibility


@pytest.fixture(scope="module")
def images_and_grids_conf():  # pylint: disable=redefined-outer-name
    """
    Returns images (img1 and img2) and grids (left, right) configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["images_and_grids"]

    for tag in ["img1", "img2"]:
        configuration["input"][tag] = absolute_data_path(
            configuration["input"][tag]
        )

    for tag in ["left_epipolar_grid", "right_epipolar_grid"]:
        configuration["preprocessing"]["output"][tag] = absolute_data_path(
            configuration["preprocessing"]["output"][tag]
        )

    return configuration


@pytest.fixture(scope="module")
def color1_conf():  # pylint: disable=redefined-outer-name
    """
    Returns color1 configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["color1"]

    configuration["input"]["color1"] = absolute_data_path(
        configuration["input"]["color1"]
    )

    return configuration


@pytest.fixture(scope="module")
def color_pxs_conf():  # pylint: disable=redefined-outer-name
    """
    Returns color_pxs configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["color_pxs"]

    configuration["input"]["color1"] = absolute_data_path(
        configuration["input"]["color1"]
    )

    return configuration


@pytest.fixture(scope="module")
def no_data_conf():  # pylint: disable=redefined-outer-name
    """
    Returns no data configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["no_data"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_sizes_conf():  # pylint: disable=redefined-outer-name
    """
    Returns epipolar size configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["epipolar_sizes"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_origins_spacings_conf():  # pylint: disable=redefined-outer-name
    """
    Returns epipolar spacing configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["epipolar_origins_spacings"]

    return configuration


@pytest.fixture(scope="module")
def disparities_conf():  # pylint: disable=redefined-outer-name
    """
    Returns disparities configuration
    """
    json_path = "input/stereo_input/tests_configurations.json"
    with open(absolute_data_path(json_path), "r") as json_file:
        json_dict = json.load(json_file)
        configuration = json_dict["disparities"]

    return configuration


def create_corr_conf():
    """
    Create correlator configuration for stereo testing
    """
    user_cfg = {}
    user_cfg["input"] = {}
    user_cfg["pipeline"] = {}
    user_cfg["pipeline"]["right_disp_map"] = {}
    user_cfg["pipeline"]["right_disp_map"]["method"] = "accurate"
    user_cfg["pipeline"]["matching_cost"] = {}
    user_cfg["pipeline"]["matching_cost"]["matching_cost_method"] = "census"
    user_cfg["pipeline"]["matching_cost"]["window_size"] = 5
    user_cfg["pipeline"]["matching_cost"]["subpix"] = 1
    user_cfg["pipeline"]["optimization"] = {}
    user_cfg["pipeline"]["optimization"]["optimization_method"] = "sgm"
    user_cfg["pipeline"]["optimization"]["P1"] = 8
    user_cfg["pipeline"]["optimization"]["P2"] = 32
    user_cfg["pipeline"]["optimization"]["p2_method"] = "constant"
    user_cfg["pipeline"]["optimization"]["penalty_method"] = "sgm_penalty"
    user_cfg["pipeline"]["optimization"]["overcounting"] = False
    user_cfg["pipeline"]["optimization"]["min_cost_paths"] = False
    user_cfg["pipeline"]["disparity"] = {}
    user_cfg["pipeline"]["disparity"]["disparity_method"] = "wta"
    user_cfg["pipeline"]["disparity"]["invalid_disparity"] = 0
    user_cfg["pipeline"]["refinement"] = {}
    user_cfg["pipeline"]["refinement"]["refinement_method"] = "vfit"
    user_cfg["pipeline"]["filter"] = {}
    user_cfg["pipeline"]["filter"]["filter_method"] = "median"
    user_cfg["pipeline"]["filter"]["filter_size"] = 3
    user_cfg["pipeline"]["validation"] = {}
    user_cfg["pipeline"]["validation"]["validation_method"] = "cross_checking"
    user_cfg["pipeline"]["validation"]["cross_checking_threshold"] = 1.0
    # Import plugins before checking configuration
    pandora.import_plugin()
    # Check configuration and update the configuration with default values
    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()
    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, pandora_machine)
    # check a part of input section
    user_cfg_input = get_config_input_custom_cars(user_cfg)
    cfg_input = check_input_section_custom_cars(user_cfg_input)
    # concatenate updated config
    cfg = concat_conf([cfg_input, cfg_pipeline])
    return cfg


@pytest.mark.unit_tests
def test_optimal_tile_size():
    """
    Test optimal_tile_size function
    """
    disp = 61
    mem = 313

    res = dense_matching.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=1000, otb_max_ram_hint=mem
    )

    assert res == 400

    res = dense_matching.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=0, max_tile_size=300, otb_max_ram_hint=mem
    )

    assert res == 300

    res = dense_matching.optimal_tile_size_pandora_plugin_libsgm(
        0, disp, min_tile_size=500, max_tile_size=1000, otb_max_ram_hint=mem
    )

    assert res == 500

    # Test case where default tile size is returned
    assert (
        dense_matching.optimal_tile_size_pandora_plugin_libsgm(
            -1000,
            1000,
            min_tile_size=0,
            max_tile_size=1000,
            otb_max_ram_hint=100,
            tile_size_rounding=33,
        )
        == 33
    )


@pytest.mark.unit_tests
def test_resample_image():
    """
    Test resample image method
    """
    region = [387, 180, 564, 340]

    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    nodata = 0
    grid = absolute_data_path("input/stereo_input/left_epipolar_grid.tif")
    epipolar_size_x = 612
    epipolar_size_y = 612

    test_dataset = resampling.resample_image(
        img,
        grid,
        [epipolar_size_x, epipolar_size_y],
        region=region,
        nodata=nodata,
    )

    # For convenience we use same reference as test_epipolar_rectify_images_1
    ref_dataset = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_left.nc")
    )

    # We need to remove attributes that are not generated by resample_image
    # method
    ref_dataset.attrs.pop(cst.ROI, None)
    ref_dataset.attrs.pop(cst.EPI_MARGINS, None)
    ref_dataset.attrs.pop(cst.EPI_DISP_MIN, None)
    ref_dataset.attrs.pop(cst.EPI_DISP_MAX, None)
    ref_dataset.attrs["region"] = ref_dataset.attrs[cst.ROI_WITH_MARGINS]
    ref_dataset.attrs.pop(cst.ROI_WITH_MARGINS, None)

    assert_same_datasets(test_dataset, ref_dataset)


@pytest.mark.unit_tests
def test_epipolar_rectify_images_1(
    images_and_grids_conf,
    color1_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,  # pylint: disable=redefined-outer-name
    epipolar_origins_spacings_conf,  # pylint: disable=redefined-outer-name
    no_data_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test epipolar_rectify_image on ventoux dataset (epipolar geometry)
    with nodata and color
    """
    configuration = images_and_grids_conf
    configuration["input"].update(color1_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_origins_spacings_conf["preprocessing"]["output"]
    )

    region = [420, 200, 530, 320]
    col = np.arange(4)
    margin = xr.Dataset(
        {"left_margin": (["col"], np.array([33, 20, 34, 20]))},
        coords={"col": col},
    )
    margin["right_margin"] = xr.DataArray(
        np.array([33, 20, 34, 20]), dims=["col"]
    )

    margin.attrs[cst.EPI_DISP_MIN] = -13
    margin.attrs[cst.EPI_DISP_MAX] = 14

    # Rectify images
    left, right, clr = resampling.epipolar_rectify_images(
        configuration, region, margin
    )

    print("\nleft dataset: {}".format(left))
    print("right dataset: {}".format(right))
    print("clr dataset: {}".format(clr))

    # Uncomment to update baseline
    # left.to_netcdf(absolute_data_path("ref_output/data1_ref_left.nc"))

    left_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_left.nc")
    )
    assert_same_datasets(left, left_ref)

    # Uncomment to update baseline
    # right.to_netcdf(absolute_data_path("ref_output/data1_ref_right.nc"))

    right_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_right.nc")
    )
    assert_same_datasets(right, right_ref)

    # Uncomment to update baseline
    # clr.to_netcdf(absolute_data_path("ref_output/data1_ref_clr.nc"))

    clr_ref = xr.open_dataset(absolute_data_path("ref_output/data1_ref_clr.nc"))
    assert_same_datasets(clr, clr_ref)


@pytest.mark.unit_tests
def test_epipolar_rectify_images_3(
    images_and_grids_conf,  # pylint: disable=redefined-outer-name
    color_pxs_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,  # pylint: disable=redefined-outer-name
    epipolar_origins_spacings_conf,  # pylint: disable=redefined-outer-name
    no_data_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test epipolar_rectify_image on ventoux dataset (epipolar geometry)
    with nodata and color as a p+xs fusion
    """
    configuration = images_and_grids_conf
    configuration["input"].update(color_pxs_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_origins_spacings_conf["preprocessing"]["output"]
    )

    region = [420, 200, 530, 320]
    col = np.arange(4)
    margin = xr.Dataset(
        {"left_margin": (["col"], np.array([33, 20, 34, 20]))},
        coords={"col": col},
    )
    margin["right_margin"] = xr.DataArray(
        np.array([33, 20, 34, 20]), dims=["col"]
    )

    margin.attrs[cst.EPI_DISP_MIN] = -13
    margin.attrs[cst.EPI_DISP_MAX] = 14

    # Rectify images
    left, right, clr = resampling.epipolar_rectify_images(
        configuration, region, margin
    )

    print("\nleft dataset: {}".format(left))
    print("right dataset: {}".format(right))
    print("clr dataset: {}".format(clr))

    left_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_left.nc")
    )
    assert_same_datasets(left, left_ref)

    right_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_right.nc")
    )
    assert_same_datasets(right, right_ref)

    # Uncomment to update baseline
    # clr.to_netcdf(absolute_data_path("ref_output/data3_ref_clr_4bands.nc"))

    clr_ref = xr.open_dataset(
        absolute_data_path("ref_output/data3_ref_clr_4bands.nc")
    )
    assert_same_datasets(clr, clr_ref)


@pytest.mark.unit_tests
def test_compute_disparity_1(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching.compute_disparity(
        left_input,
        right_input,
        images_and_grids_conf,
        corr_cfg,
        disp_min,
        disp_max,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([387, 180, 564, 340]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp1_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    sec = xr.open_dataset(absolute_data_path("ref_output/disp1_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_3(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on paca dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data3_ref_right.nc")
    )
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -43
    disp_max = 41

    output = dense_matching.compute_disparity(
        left_input,
        right_input,
        images_and_grids_conf,
        corr_cfg,
        disp_min,
        disp_max,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (90, 90)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (90, 90)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (170, 254)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (170, 254)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI],
        np.array([16500, 23160, 16590, 23250]),
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([16417, 23120, 16671, 23290]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp3_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp3_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp3_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    sec = xr.open_dataset(absolute_data_path("ref_output/disp3_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_ref(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_left_masked.nc"
        )
    )
    right_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_right.nc")
    )
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching.compute_disparity(
        left_input,
        right_input,
        images_and_grids_conf,
        corr_cfg,
        disp_min,
        disp_max,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora_msk_ref.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora_msk_ref.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_ref.nc")
    )
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)

    sec = xr.open_dataset(
        absolute_data_path("ref_output/disp1_sec_pandora_msk_ref.nc")
    )
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)

    # test multi-classes left mask
    left_input[cst.EPI_MSK].values[10, 10] = 1  # valid class
    left_input[cst.EPI_MSK].values[10, 140] = 2  # nonvalid class
    conf = deepcopy(images_and_grids_conf)
    conf["input"]["mask1_classes"] = absolute_data_path(
        "input/intermediate_results/data1_ref_left_mask_classes.json"
    )

    output = dense_matching.compute_disparity(
        left_input,
        right_input,
        conf,
        corr_cfg,
        disp_min,
        disp_max,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )

    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_sec(
    images_and_grids_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(
        absolute_data_path("input/intermediate_results/data1_ref_left.nc")
    )
    right_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_right_masked.nc"
        )
    )
    conf = deepcopy(images_and_grids_conf)
    conf["input"]["mask2_classes"] = absolute_data_path(
        "input/intermediate_results/data1_ref_right_mask_classes.json"
    )

    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = dense_matching.compute_disparity(
        left_input,
        right_input,
        conf,
        corr_cfg,
        disp_min,
        disp_max,
        verbose=True,
    )

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(
        output[cst.STEREO_REF].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI], np.array([420, 200, 530, 320])
    )
    np.testing.assert_allclose(
        output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
        np.array([387, 180, 564, 340]),
    )

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path(
    # "ref_output/disp1_ref_pandora_msk_sec.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path(
    # "ref_output/disp1_sec_pandora_msk_sec.nc"))

    ref = xr.open_dataset(
        absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc")
    )
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.0e-6)

    sec = xr.open_dataset(
        absolute_data_path("ref_output/disp1_sec_pandora_msk_sec.nc")
    )
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.0e-6)


@pytest.mark.unit_tests
def test_compute_mask_to_use_in_pandora():
    """
    Test compute_mask_to_use_in_pandora with a cloud "data1_ref_right_masked.nc"
    """

    right_input = xr.open_dataset(
        absolute_data_path(
            "input/intermediate_results/data1_ref_right_masked.nc"
        )
    )

    test_mask = dense_matching.compute_mask_to_use_in_pandora(
        right_input, cst.EPI_MSK, [100]
    )

    ref_msk = np.copy(right_input[cst.EPI_MSK].values)
    ref_msk.astype(np.int16)
    ref_msk[np.where(right_input[cst.EPI_MSK].values == 100, True, False)] = 1

    assert np.allclose(test_mask, ref_msk)


@pytest.mark.unit_tests
def test_create_inside_sec_roi_mask():
    """
    Test create_inside_sec_roi_mask fonction
    """
    # create fake dataset with margins values and image dimension
    dataset_left_margin = 1
    dataset_right_margin = 2
    dataset_top_margin = 1
    dataset_bottom_margin = 1
    dataset_nb_col = 6
    dataset_nb_row = 5
    test_dataset = xr.Dataset(
        {
            cst.EPI_IMAGE: (
                [cst.ROW, cst.COL],
                np.ones((dataset_nb_row, dataset_nb_col), dtype=bool),
            )
        }
    )
    test_dataset.attrs[cst.EPI_MARGINS] = np.array(
        [
            dataset_left_margin,
            dataset_top_margin,
            dataset_right_margin,
            dataset_bottom_margin,
        ]
    )

    # create fake disp map and mask
    disp_nb_col = 7
    disp_nb_row = 6
    disp_value = 0.5
    disp = np.full((disp_nb_row, disp_nb_col), disp_value)
    mask = np.full((disp_nb_row, disp_nb_col), 255, dtype=np.int16)

    # add an invalid pixel in the useful zone
    mask[1, 1] = 0

    msk = dense_matching.create_inside_sec_roi_mask(disp, mask, test_dataset)

    # create reference data
    ref_msk = np.zeros((disp_nb_row, disp_nb_col), dtype=np.int16)
    ref_msk[
        math.ceil(disp_value) : disp_nb_row - dataset_bottom_margin,
        math.ceil(disp_value) : disp_nb_col
        - dataset_right_margin
        - math.floor(disp_value),
    ] = 255
    ref_msk[1, 1] = 0

    assert np.allclose(msk, ref_msk)


@pytest.mark.unit_tests
def test_estimate_color_from_disparity():
    """
    Test estimate_color_from_disparity function
    """
    # create fake disp map and mask
    margins = [1, 1, 1, 1]
    disp_nb_col = 12
    disp_nb_row = 10
    disp_value = 0.75
    disp_row = np.array(range(disp_nb_row))
    disp_col = np.array(range(disp_nb_col))

    disp = np.full((disp_nb_row, disp_nb_col), disp_value)

    mask = np.full((disp_nb_row - 2, disp_nb_col - 3), 255, dtype=np.int16)
    mask = np.pad(mask, ((1, 1), (1, 2)), constant_values=0)
    mask[2, 2] = 0

    disp_dataset = xr.Dataset(
        {
            cst.DISP_MAP: ([cst.ROW, cst.COL], np.copy(disp)),
            cst.DISP_MSK: ([cst.ROW, cst.COL], np.copy(mask)),
        },
        coords={cst.ROW: disp_row, cst.COL: disp_col},
    )

    disp_dataset.attrs[cst.ROI] = [
        margins[0],
        margins[1],
        disp_nb_col - margins[2],
        disp_nb_row - margins[3],
    ]
    disp_dataset.attrs[cst.ROI_WITH_MARGINS] = [0, 0, disp_nb_col, disp_nb_row]
    disp_dataset.attrs[cst.EPI_FULL_SIZE] = [100, 100]

    # create fake color image
    clr_nb_col = 10
    clr_nb_row = 8
    clr_nb_band = 3
    clr_row = np.array(range(clr_nb_row))
    clr_col = np.array(range(clr_nb_col))

    clr = np.ones((clr_nb_band, clr_nb_row, clr_nb_col), dtype=np.float)
    val = np.arange(clr_nb_row * clr_nb_col)
    val = val.reshape(clr_nb_row, clr_nb_col)
    for band in range(clr_nb_band):
        clr[band, :, :] = val

    clr_mask = np.full((clr_nb_row, clr_nb_col), 255, dtype=np.int16)
    clr_mask[4, 4] = 0
    clr_dataset = xr.Dataset(
        {
            cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], np.copy(clr)),
            cst.EPI_MSK: ([cst.ROW, cst.COL], np.copy(clr_mask)),
        },
        coords={
            cst.BAND: range(clr_nb_band),
            cst.ROW: clr_row,
            cst.COL: clr_col,
        },
    )

    # create fake secondary dataset
    sec_margins = [1, 1, 1, 1]
    sec_dataset = xr.Dataset()
    sec_dataset.attrs[cst.EPI_MARGINS] = np.array(sec_margins)

    # interpolate color
    interp_clr_dataset = dense_matching.estimate_color_from_disparity(
        disp_dataset, sec_dataset, clr_dataset
    )

    # reference
    ref_mask = mask.astype(np.bool)
    ref_mask[margins[0] + 4, margins[2] + 4 - math.ceil(disp_value)] = False
    ref_mask = ~ref_mask

    assert np.allclose(
        ref_mask, np.isnan(interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :])
    )

    ref_data = np.ceil(disp)
    ref_data[margins[1] : -margins[3], margins[0] : -margins[2]] += val
    ref_data[ref_mask] = 0

    interp_clr_msk = np.isnan(interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :])
    interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :][interp_clr_msk] = 0

    assert np.allclose(
        ref_data, interp_clr_dataset[cst.EPI_IMAGE].values[0, :, :]
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


@pytest.mark.unit_tests
def test_terrain_region_to_epipolar(
    images_and_grids_conf,  # pylint: disable=redefined-outer-name
    disparities_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,
):  # pylint: disable=redefined-outer-name
    """
    Test transform to epipolar method
    """
    configuration = images_and_grids_conf
    configuration["preprocessing"]["output"].update(
        disparities_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )

    region = [5.1952, 44.205, 5.2, 44.208]
    out_region = tiling.terrain_region_to_epipolar(region, configuration)
    assert out_region == [0.0, 0.0, 612.0, 400.0]
