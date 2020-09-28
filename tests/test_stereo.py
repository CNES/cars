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
import json
import math
import numpy as np
import xarray as xr

import pandora
from pandora.JSON_checker import get_config_pipeline, check_pipeline_section,\
                                 get_config_image, check_image_section,\
                                 concat_conf

from utils import absolute_data_path, assert_same_datasets

from cars import stereo
from cars import constants as cst
from cars.utils import read_geoid_file


@pytest.fixture(scope="module")
def images_and_grids_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["images_and_grids"]

    for tag in ['img1', 'img2']:
        configuration['input'][tag] = absolute_data_path(configuration['input'][tag])

    for tag in ['left_epipolar_grid', 'right_epipolar_grid']:
        configuration['preprocessing']['output'][tag] = \
            absolute_data_path(configuration['preprocessing']['output'][tag])

    return configuration


@pytest.fixture(scope="module")
def color1_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["color1"]

    configuration['input']['color1'] = absolute_data_path(configuration['input']['color1'])

    return configuration


@pytest.fixture(scope="module")
def color_pxs_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["color_pxs"]

    configuration['input']['color1'] = absolute_data_path(configuration['input']['color1'])

    return configuration


@pytest.fixture(scope="module")
def no_data_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["no_data"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_sizes_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["epipolar_sizes"]

    return configuration


@pytest.fixture(scope="module")
def epipolar_origins_spacings_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["epipolar_origins_spacings"]

    return configuration


@pytest.fixture(scope="module")
def disparities_conf():
    json_path = 'input/stereo_input/tests_configurations.json'
    with open(absolute_data_path(json_path), 'r') as f:
        json_dict = json.load(f)
        configuration = json_dict["disparities"]

    return configuration


def create_corr_conf():
    user_cfg = dict()
    user_cfg['image'] = {}
    user_cfg['image']['valid_pixels'] = 0
    user_cfg['image']['no_data'] = 255
    user_cfg["stereo"] = {}
    user_cfg["stereo"]["stereo_method"] = "census"
    user_cfg["stereo"]["window_size"] = 5
    user_cfg["stereo"]["subpix"] = 1
    user_cfg["aggregation"] = {}
    user_cfg["aggregation"]["aggregation_method"] = "none"
    user_cfg["optimization"] = {}
    user_cfg["optimization"]["optimization_method"] = "sgm"
    user_cfg["optimization"]["P1"] = 8
    user_cfg["optimization"]["P2"] = 32
    user_cfg["optimization"]["p2_method"] = "constant"
    user_cfg["optimization"]["penalty_method"] = "sgm_penalty"
    user_cfg["optimization"]["overcounting"] = False
    user_cfg["optimization"]["min_cost_paths"] = False
    user_cfg["refinement"] = {}
    user_cfg["refinement"]["refinement_method"] = "vfit"
    user_cfg["filter"] = {}
    user_cfg["filter"]["filter_method"] = "median"
    user_cfg["filter"]["filter_size"] = 3
    user_cfg["validation"] = {}
    user_cfg["validation"]["validation_method"] = "cross_checking"
    user_cfg["validation"]["cross_checking_threshold"] = 1.0
    user_cfg["validation"]["right_left_mode"] = "accurate"
    user_cfg["validation"]["interpolated_disparity"] = "none"
    # Import plugins before checking confifuration
    pandora.import_plugin()
    # Check configuration and update the configuration with default values
    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline)
    # check image
    user_cfg_image = get_config_image(user_cfg)
    cfg_image = check_image_section(user_cfg_image)
    # concatenate updated config
    cfg = concat_conf([cfg_image, cfg_pipeline])
    return cfg


@pytest.mark.unit_tests
def test_optimal_tile_size():
    """
    Test optimal_tile_size function
    """
    row = 450
    col = 375
    disp = 61  
    mem = 313

    res = stereo.optimal_tile_size_pandora_plugin_libsgm(0, disp, mem)
    assert res == 411

    # Test case where default tile size is returned
    assert stereo.optimal_tile_size_pandora_plugin_libsgm(-1000, 1000,
                                                          100, tile_size_rounding=33) == 33


@pytest.mark.unit_tests
def test_resample_image():
    """
    Test resample image method
    """
    region = [387, 180, 564, 340]

    img = absolute_data_path('input/phr_ventoux/left_image.tif')
    nodata = 0
    grid = absolute_data_path('input/stereo_input/left_epipolar_grid.tif')
    epipolar_size_x = 612
    epipolar_size_y = 612

    ds = stereo.resample_image(
        img, grid, [
            epipolar_size_x, epipolar_size_y], region=region, nodata=nodata)

    # For convenience we use same reference as test_epipolar_rectify_images_1
    ref_ds = xr.open_dataset(absolute_data_path(
        "ref_output/data1_ref_left.nc"))

    # We need to remove attributes that are not generated by resample_image
    # method
    ref_ds.attrs.pop(cst.ROI, None)
    ref_ds.attrs.pop(cst.EPI_MARGINS, None)
    ref_ds.attrs.pop(cst.EPI_DISP_MIN, None)
    ref_ds.attrs.pop(cst.EPI_DISP_MAX, None)
    ref_ds.attrs['region'] = ref_ds.attrs[cst.ROI_WITH_MARGINS]
    ref_ds.attrs.pop(cst.ROI_WITH_MARGINS, None)

    assert_same_datasets(ds, ref_ds)


@pytest.mark.unit_tests
def test_epipolar_rectify_images_1(images_and_grids_conf, color1_conf, epipolar_sizes_conf,
                                   epipolar_origins_spacings_conf, no_data_conf):
    """
    Test epipolar_rectify_image on ventoux dataset (epipolar geometry)
    with nodata and color
    """
    configuration = images_and_grids_conf
    configuration["input"].update(color1_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(epipolar_sizes_conf["preprocessing"]["output"])
    configuration["preprocessing"]["output"].update(epipolar_origins_spacings_conf["preprocessing"]["output"])

    region = [420, 200, 530, 320]
    margin = xr.DataArray(np.array([[33, 20, 34, 20], [33, 20, 34, 20]], dtype=int),
                          coords=[['ref_margin', 'sec_margin'],
                                  ['left', 'up', 'right', 'down']],
                          dims=['image', 'corner'])
    margin.name = cst.EPI_MARGINS
    margin.attrs[cst.EPI_DISP_MIN] = -13
    margin.attrs[cst.EPI_DISP_MAX] = 14

    # Rectify images
    left, right, clr = stereo.epipolar_rectify_images(configuration,
                                                      region,
                                                      margin)

    print("\nleft dataset: {}".format(left))
    print("right dataset: {}".format(right))
    print("clr dataset: {}".format(clr))

    # Uncomment to update baseline
    # left.to_netcdf(absolute_data_path("ref_output/data1_ref_left.nc"))

    left_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_left.nc"))
    assert_same_datasets(left, left_ref)

    # Uncomment to update baseline
    # right.to_netcdf(absolute_data_path("ref_output/data1_ref_right.nc"))

    right_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_right.nc"))
    assert_same_datasets(right, right_ref)

    # Uncomment to update baseline
    # clr.to_netcdf(absolute_data_path("ref_output/data1_ref_clr.nc"))

    clr_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_clr.nc"))
    assert_same_datasets(clr, clr_ref)


@pytest.mark.unit_tests
def test_epipolar_rectify_images_3(images_and_grids_conf, color_pxs_conf, epipolar_sizes_conf,
                                   epipolar_origins_spacings_conf, no_data_conf):
    """
    Test epipolar_rectify_image on ventoux dataset (epipolar geometry)
    with nodata and color as a p+xs fusion
    """
    configuration = images_and_grids_conf
    configuration["input"].update(color_pxs_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(epipolar_sizes_conf["preprocessing"]["output"])
    configuration["preprocessing"]["output"].update(epipolar_origins_spacings_conf["preprocessing"]["output"])

    region = [420, 200, 530, 320]
    margin = xr.DataArray(np.array([[33, 20, 34, 20], [33, 20, 34, 20]], dtype=int),
                          coords=[['ref_margin', 'sec_margin'],
                                  ['left', 'up', 'right', 'down']],
                          dims=['image', 'corner'])
    margin.name = cst.EPI_MARGINS
    margin.attrs[cst.EPI_DISP_MIN] = -13
    margin.attrs[cst.EPI_DISP_MAX] = 14

    # Rectify images
    left, right, clr = stereo.epipolar_rectify_images(configuration,
                                                      region,
                                                      margin)

    print("\nleft dataset: {}".format(left))
    print("right dataset: {}".format(right))
    print("clr dataset: {}".format(clr))

    left_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_left.nc"))
    assert_same_datasets(left, left_ref)

    right_ref = xr.open_dataset(
        absolute_data_path("ref_output/data1_ref_right.nc"))
    assert_same_datasets(right, right_ref)

    # Uncomment to update baseline
    # clr.to_netcdf(absolute_data_path("ref_output/data3_ref_clr_4bands.nc"))

    clr_ref = xr.open_dataset(absolute_data_path(
        "ref_output/data3_ref_clr_4bands.nc"))
    assert_same_datasets(clr, clr_ref)


@pytest.mark.unit_tests
def test_compute_disparity_1():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(absolute_data_path(
        "input/intermediate_results/data1_ref_left.nc"))
    right_input = xr.open_dataset(absolute_data_path(
        "input/intermediate_results/data1_ref_right.nc"))
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = stereo.compute_disparity(left_input,
                                      right_input,
                                      corr_cfg,
                                      disp_min,
                                      disp_max)

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(output[cst.STEREO_REF].attrs[cst.ROI],
                               np.array([420, 200, 530, 320]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI],
                               np.array([420, 200, 530, 320]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
                               np.array([387, 180, 564, 340]))

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/disp1_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path("ref_output/disp1_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path(
        "ref_output/disp1_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.e-6)
    sec = xr.open_dataset(absolute_data_path(
        "ref_output/disp1_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.e-6)


@pytest.mark.unit_tests
def test_compute_disparity_3():
    """
    Test compute_disparity on paca dataset with pandora
    """
    left_input = xr.open_dataset(absolute_data_path(
        "input/intermediate_results/data3_ref_left.nc"))
    right_input = xr.open_dataset(absolute_data_path(
        "input/intermediate_results/data3_ref_right.nc"))
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -43
    disp_max = 41

    output = stereo.compute_disparity(left_input,
                                      right_input,
                                      corr_cfg,
                                      disp_min,
                                      disp_max)

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (90, 90)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (90, 90)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (170, 254)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (170, 254)

    np.testing.assert_allclose(output[cst.STEREO_REF].attrs[cst.ROI],
                               np.array([16500, 23160, 16590, 23250]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI],
                               np.array([16500, 23160, 16590, 23250]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
                               np.array([16417, 23120, 16671, 23290]))

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/disp3_ref_pandora.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path("ref_output/disp3_sec_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path(
        "ref_output/disp3_ref_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=5.e-6)
    sec = xr.open_dataset(absolute_data_path(
        "ref_output/disp3_sec_pandora.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_ref():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(absolute_data_path("input/intermediate_results/data1_ref_left_masked.nc"))
    right_input = xr.open_dataset(absolute_data_path("input/intermediate_results/data1_ref_right.nc"))
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = stereo.compute_disparity(left_input,
                                      right_input,
                                      corr_cfg,
                                      disp_min,
                                      disp_max,
                                      verbose = True)

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)

    np.testing.assert_allclose(output[cst.STEREO_REF].attrs[cst.ROI],
                               np.array([420, 200, 530, 320]))

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/disp1_ref_pandora_msk_ref.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path("ref_output/disp1_sec_pandora_msk_ref.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp1_ref_pandora_msk_ref.nc"))
    assert_same_datasets(output[cst.STEREO_REF],ref,atol=5.e-6)

    sec = xr.open_dataset(absolute_data_path("ref_output/disp1_sec_pandora_msk_ref.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.e-6)


@pytest.mark.unit_tests
def test_compute_disparity_1_msk_sec():
    """
    Test compute_disparity on ventoux dataset with pandora
    """
    left_input = xr.open_dataset(absolute_data_path("input/intermediate_results/data1_ref_left.nc"))
    right_input = xr.open_dataset(absolute_data_path("input/intermediate_results/data1_ref_right_masked.nc"))
    # Pandora configuration
    corr_cfg = create_corr_conf()

    disp_min = -13
    disp_max = 14

    output = stereo.compute_disparity(left_input,
                                      right_input,
                                      corr_cfg,
                                      disp_min,
                                      disp_max,
                                      verbose = True)

    assert output[cst.STEREO_REF][cst.DISP_MAP].shape == (120, 110)
    assert output[cst.STEREO_REF][cst.DISP_MSK].shape == (120, 110)
    assert output[cst.STEREO_SEC][cst.DISP_MAP].shape == (160, 177)
    assert output[cst.STEREO_SEC][cst.DISP_MSK].shape == (160, 177)

    np.testing.assert_allclose(output[cst.STEREO_REF].attrs[cst.ROI],
                               np.array([420, 200, 530, 320]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI],
                               np.array([420, 200, 530, 320]))
    np.testing.assert_allclose(output[cst.STEREO_SEC].attrs[cst.ROI_WITH_MARGINS],
                               np.array([387, 180, 564, 340]))

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc"))
    # output[cst.STEREO_SEC].to_netcdf(absolute_data_path("ref_output/disp1_sec_pandora_msk_sec.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/disp1_ref_pandora_msk_sec.nc"))
    assert_same_datasets(output[cst.STEREO_REF],ref,atol=5.e-6)

    sec = xr.open_dataset(absolute_data_path("ref_output/disp1_sec_pandora_msk_sec.nc"))
    assert_same_datasets(output[cst.STEREO_SEC], sec, atol=5.e-6)


@pytest.mark.unit_tests
def test_create_inside_sec_roi_mask():
    """
    Test create_inside_sec_roi_mask fonction
    """
    # create fake dataset with margins values and image dimension
    ds_left_margin = 1
    ds_right_margin = 2
    ds_top_margin = 1
    ds_bottom_margin = 1
    ds_nb_col = 6
    ds_nb_row = 5
    ds = xr.Dataset({cst.EPI_IMAGE: ([cst.ROW, cst.COL], np.ones((ds_nb_row, ds_nb_col), dtype=bool))})
    ds.attrs[cst.EPI_MARGINS] = np.array([ds_left_margin, ds_top_margin, ds_right_margin, ds_bottom_margin])

    # create fake disp map and mask
    disp_nb_col = 7
    disp_nb_row = 6
    disp_value = 0.5
    disp = np.full((disp_nb_row, disp_nb_col),disp_value)
    mask = np.full((disp_nb_row, disp_nb_col), 255, dtype=np.int16)

    # add an invalid pixel in the useful zone
    mask[1, 1] = 0

    msk = stereo.create_inside_sec_roi_mask(disp, mask, ds)

    # create reference data
    ref_msk = np.zeros((disp_nb_row, disp_nb_col), dtype=np.int16)
    ref_msk[math.ceil(disp_value):disp_nb_row-ds_bottom_margin,
            math.ceil(disp_value):disp_nb_col-ds_right_margin-math.floor(disp_value)] = 255
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

    mask = np.full((disp_nb_row-2, disp_nb_col-3), 255, dtype=np.int16)
    mask = np.pad(mask, ((1, 1), (1, 2)), constant_values=0)
    mask[2,2] = 0

    disp_ds = xr.Dataset({cst.DISP_MAP: ([cst.ROW, cst.COL], np.copy(disp)),
                          cst.DISP_MSK: ([cst.ROW, cst.COL], np.copy(mask))},
                         coords={cst.ROW: disp_row, cst.COL: disp_col})

    disp_ds.attrs[cst.ROI] = [margins[0], margins[1], disp_nb_col-margins[2], disp_nb_row-margins[3]]
    disp_ds.attrs[cst.ROI_WITH_MARGINS] = [0, 0, disp_nb_col, disp_nb_row]
    disp_ds.attrs[cst.EPI_FULL_SIZE] = [100, 100]

    # create fake color image
    clr_nb_col = 10
    clr_nb_row = 8
    clr_nb_band = 3
    clr_row = np.array(range(clr_nb_row))
    clr_col = np.array(range(clr_nb_col))

    clr = np.ones((clr_nb_band, clr_nb_row, clr_nb_col), dtype=np.float)
    val = np.arange(clr_nb_row*clr_nb_col)
    val = val.reshape(clr_nb_row, clr_nb_col)
    for band in range(clr_nb_band):
        clr[band, :, :] = val

    clr_mask = np.full((clr_nb_row, clr_nb_col), 255, dtype=np.int16)
    clr_mask[4, 4] = 0
    clr_ds = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL], np.copy(clr)),
                         cst.EPI_MSK: ([cst.ROW, cst.COL], np.copy(clr_mask))},
                        coords={cst.BAND: range(clr_nb_band), cst.ROW: clr_row, cst.COL: clr_col})

    # create fake secondary dataset
    sec_margins = [1, 1, 1, 1]
    sec_ds = xr.Dataset()
    sec_ds.attrs[cst.EPI_MARGINS] = np.array(sec_margins)

    # interpolate color
    interp_clr_ds = stereo.estimate_color_from_disparity(disp_ds, sec_ds, clr_ds)

    # reference
    ref_mask = mask.astype(np.bool)
    ref_mask[margins[0] + 4, margins[2] + 4 - math.ceil(disp_value)] = False
    ref_mask=~ref_mask

    assert np.allclose(ref_mask, np.isnan(interp_clr_ds[cst.EPI_IMAGE].values[0,:,:]))

    ref_data = np.ceil(disp)
    ref_data[margins[1]:-margins[3], margins[0]:-margins[2]] += val
    ref_data[ref_mask] = 0

    interp_clr_msk = np.isnan(interp_clr_ds[cst.EPI_IMAGE].values[0, :, :])
    interp_clr_ds[cst.EPI_IMAGE].values[0, :, :][interp_clr_msk] = 0

    assert np.allclose(ref_data, interp_clr_ds[cst.EPI_IMAGE].values[0,:,:])


@pytest.mark.unit_tests
def test_triangulation_1(images_and_grids_conf):
    """
    Test triangulation ventoux dataset
    """
    input = xr.open_dataset(absolute_data_path(
        "input/intermediate_results/disp1_ref.nc"))
    output = stereo.triangulate(images_and_grids_conf, input, None)

    assert output[cst.STEREO_REF][cst.X].shape == (120, 110)

    # Uncomment to update baseline
    # output[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/triangulation1_ref.nc"))

    ref = xr.open_dataset(absolute_data_path(
        "ref_output/triangulation1_ref.nc"))
    assert_same_datasets(output[cst.STEREO_REF], ref, atol=1.e-3)


@pytest.mark.unit_tests
def test_triangulate_matches(images_and_grids_conf):

    matches = np.array([[0.,0.,0.,0.]])

    llh = stereo.triangulate_matches(images_and_grids_conf, matches)

    # Check properties
    assert(llh.dims == {cst.ROW: 1, cst.COL: 1})
    np.testing.assert_almost_equal(llh.x.values[0],5.1973629)
    np.testing.assert_almost_equal(llh.y.values[0],44.2079813)
    np.testing.assert_almost_equal(llh.z.values[0],511.4383088)
    assert(llh[cst.POINTS_CLOUD_CORR_MSK].values[0] == 255)
    assert(cst.EPSG in llh.attrs)
    assert(llh.attrs[cst.EPSG] == 4326)


@pytest.mark.unit_tests
def test_images_pair_to_3d_points(images_and_grids_conf, color1_conf, no_data_conf, disparities_conf,
                                  epipolar_origins_spacings_conf, epipolar_sizes_conf):
    """
    Test images_pair_to_3d_points on ventoux dataset (epipolar geometry)
    with Pandora
    """
    # With nodata and color
    configuration = images_and_grids_conf
    configuration["input"].update(color1_conf["input"])
    configuration["input"].update(no_data_conf["input"])
    configuration["preprocessing"]["output"].update(epipolar_sizes_conf["preprocessing"]["output"])
    configuration["preprocessing"]["output"].update(epipolar_origins_spacings_conf["preprocessing"]["output"])
    configuration["preprocessing"]["output"].update(disparities_conf["preprocessing"]["output"])

    region = [420, 200, 530, 320]
    # Pandora configuration
    corr_cfg = create_corr_conf()

    cloud, color = stereo.images_pair_to_3d_points(configuration,
                                                   region,
                                                   corr_cfg,
                                                   disp_min=-13,
                                                   disp_max=14,
                                                   add_msk_info=True)

    # Uncomment to update baseline
    # cloud[cst.STEREO_REF].to_netcdf(absolute_data_path("ref_output/cloud1_ref_pandora.nc"))

    ref = xr.open_dataset(absolute_data_path("ref_output/cloud1_ref_pandora.nc"))
    assert_same_datasets(cloud[cst.STEREO_REF],ref,atol=1.e-3)


@pytest.mark.unit_tests
def test_geoid_offset():
    # ref file contains 32x32 points issued from proj 6.2
    ref_file = absolute_data_path('ref_output/egm96_15_ref_hgt.nc')

    geoid_ref = xr.open_dataset(ref_file)

    # create a zero elevation Dataset with the same geodetic coordinates
    points = xr.Dataset({
        cst.X: geoid_ref.x, cst.Y: geoid_ref.y,
        cst.Z: ((cst.ROW, cst.COL), np.zeros_like(geoid_ref.z))
    })

    geoid = read_geoid_file()

    computed_geoid = stereo.geoid_offset(points, geoid)

    assert(np.allclose(geoid_ref.z.values, computed_geoid.z.values,
                       atol=1e-3, rtol=1e-12))


@pytest.mark.unit_tests
def test_transform_terrain_region_to_epipolar(images_and_grids_conf, disparities_conf, epipolar_sizes_conf):
    """
    Test transform to epipolar method
    """
    configuration = images_and_grids_conf
    configuration["preprocessing"]["output"].update(disparities_conf["preprocessing"]["output"])
    configuration["preprocessing"]["output"].update(epipolar_sizes_conf["preprocessing"]["output"])

    region = [5.1952, 44.205, 5.2, 44.208]
    out_region = stereo.transform_terrain_region_to_epipolar(region, configuration)
    assert out_region == [0.0, 0.0, 612.0, 400.0]


@pytest.mark.unit_tests
def test_get_elevation_range_from_metadata():
    """
    Test the get_elevation_range_from_metadata function
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    
    (min_elev, max_elev) = stereo.get_elevation_range_from_metadata(img)

    assert(min_elev == 632.5)
    assert(max_elev == 1517.5)
