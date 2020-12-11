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
Test module for cars/preprocessing.py
"""

from __future__ import absolute_import
import tempfile
import os
import pickle
from typing import Tuple

import pytest

import numpy as np
import rasterio as rio
import xarray as xr

from utils import absolute_data_path, temporary_dir, assert_same_datasets
from cars import preprocessing
from cars import stereo
from cars import pipelines
from cars import constants as cst


def generate_epipolar_grids(
        img1: str,
        img2: str,
        srtm_dir: str=None,
        default_alt: float=None,
        epi_step: float=30)\
    -> Tuple[xr.Dataset, xr.Dataset, int, int, float]:
    """
    Generate epipolar resampling grids
    as xarray.Dataset from a pair of images and srtm_dir

    :param img1: Path to the left image
    :param img2: Path to right image
    :param srtm_dir: Path to folder containing SRTM tiles
    :param default_alt: Default altitude above ellipsoid
    :epi_step: Step of the resampling grid
    :return: Tuple containing :
        left_grid_dataset, right_grid_dataset containing the resampling grids
        epipolar_size_x, epipolar_size_y epipolar grids size
        baseline  : (resolution * B/H)
    """
    # Launch OTB pipeline to get stero grids
    grid1, grid2, __, __, epipolar_size_x, epipolar_size_y, baseline = \
        pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem=srtm_dir, default_alt=default_alt, epi_step=epi_step)

    col = np.array(range(0, grid1.shape[0] * epi_step, epi_step))
    row = np.array(range(0, grid1.shape[1] * epi_step, epi_step))

    left_grid_dataset = xr.Dataset({cst.X: ([cst.ROW, cst.COL],
                                             grid1[:, :, 0]),
                                    cst.Y: ([cst.ROW, cst.COL],
                                             grid1[:, :, 1])},
                                   coords={cst.ROW: row,
                                           cst.COL: col},
                                   attrs={"epi_step": epi_step,
                                          "epipolar_size_x": epipolar_size_x,
                                          "epipolar_size_y": epipolar_size_y})

    right_grid_dataset = xr.Dataset({cst.X: ([cst.ROW, cst.COL],
                                              grid2[:, :, 0]),
                                     cst.Y: ([cst.ROW, cst.COL],
                                              grid2[:, :, 1])},
                                    coords={cst.ROW: row,
                                            cst.COL: col},
                                    attrs={"epi_step": epi_step,
                                           "epipolar_size_x": epipolar_size_x,
                                           "epipolar_size_y": epipolar_size_y})
    return left_grid_dataset, right_grid_dataset,\
           epipolar_size_x, epipolar_size_y,\
           baseline


@pytest.mark.unit_tests
def test_generate_epipolar_grids():
    """
    Test generate_epipolar_grids method
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    left_grid, right_grid, size_x, size_y, baseline = \
        generate_epipolar_grids(
            img1, img2, dem)

    assert size_x == 612
    assert size_y == 612
    assert baseline == 0.7039416432380676

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path("ref_output/left_grid.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid.nc"))
    assert_same_datasets(left_grid, left_grid_ref)

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path("ref_output/right_grid.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid.nc"))
    assert_same_datasets(right_grid, right_grid_ref)


@pytest.mark.unit_tests
def test_generate_epipolar_grids_default_alt():
    """
    Test generate_epipolar_grids method
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = None
    default_alt = 500

    left_grid, right_grid, size_x, size_y, baseline = \
        generate_epipolar_grids(
            img1, img2, dem, default_alt)

    assert size_x == 612
    assert size_y == 612
    assert baseline == 0.7039446234703064

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path(
    # "ref_output/left_grid_default_alt.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid_default_alt.nc"))
    assert_same_datasets(left_grid, left_grid_ref)

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path(
    # "ref_output/right_grid_default_alt.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid_default_alt.nc"))
    assert_same_datasets(right_grid, right_grid_ref)


@pytest.mark.unit_tests
def test_dataset_matching():
    """
    Test dataset_matching method
    """
    region = [200, 250, 320, 400]
    img1 = absolute_data_path("input/phr_reunion/left_image.tif")
    img2 = absolute_data_path("input/phr_reunion/right_image.tif")
    mask1 = absolute_data_path("input/phr_reunion/left_mask.tif")
    mask2 = absolute_data_path("input/phr_reunion/right_mask.tif")
    nodata1 = 0
    nodata2 = 0
    grid1 = absolute_data_path(
        "input/preprocessing_input/left_epipolar_grid_reunion.tif")
    grid2 = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_reunion.tif")

    epipolar_size_x = 596
    epipolar_size_y = 596

    left = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)
    right = stereo.resample_image(
        img2, grid2, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata2, mask=mask2)

    matches = preprocessing.dataset_matching(left, right)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/matches.npy"), matches)

    matches_ref = np.load(absolute_data_path("ref_output/matches.npy"))
    np.testing.assert_allclose(matches, matches_ref)

    # Case with no matches
    region = [0, 0, 2, 2]

    left = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)
    right = stereo.resample_image(
        img1, grid1, [
            epipolar_size_x, epipolar_size_y],
            region=region, nodata=nodata1, mask=mask1)

    matches = preprocessing.dataset_matching(left, right)

    assert matches.shape == (0, 4)


@pytest.mark.unit_tests
def test_remove_epipolar_outliers():
    """
    Test remove epipolar outliers function
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_reunion.npy")

    matches = np.load(matches_file)

    matches_filtered = preprocessing.remove_epipolar_outliers(matches)

    nb_filtered_points = matches.shape[0] - matches_filtered.shape[0]
    assert nb_filtered_points == 2


@pytest.mark.unit_tests
def test_compute_disparity_range():
    """
    Test compute disparity range function
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_reunion.npy")

    matches = np.load(matches_file)

    matches_filtered = preprocessing.remove_epipolar_outliers(matches)
    dispmin, dispmax = preprocessing.compute_disparity_range(matches_filtered)

    assert dispmin == -3.1239416122436525
    assert dispmax == 3.820396270751972


@pytest.mark.unit_tests
def test_correct_right_grid():
    """
    Call right grid correction method and check outputs properties
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_ventoux.npy")
    grid_file = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_uncorrected_ventoux.tif")
    origin = [0, 0]
    spacing = [30, 30]

    matches = np.load(matches_file)
    matches = np.array(matches)

    matches_filtered = preprocessing.remove_epipolar_outliers(matches)

    with rio.open(grid_file) as rio_grid:
        grid = rio_grid.read()
        grid = np.transpose(grid, (1, 2, 0))

        corrected_grid, corrected_matches, in_stats, out_stats = \
            preprocessing.correct_right_grid(
                matches_filtered, grid, origin, spacing)

        # Uncomment to update ref
        # np.save(absolute_data_path("ref_output/corrected_right_grid.npy"),
        # corrected_grid)
        corrected_grid_ref = np.load(
            absolute_data_path("ref_output/corrected_right_grid.npy"))
        np.testing.assert_allclose(corrected_grid, corrected_grid_ref,
                                   atol=0.05, rtol=1.0e-6)

        assert corrected_grid.shape == grid.shape

        # Assert that we improved all stats
        assert abs(
            out_stats["mean_epipolar_error"][0]) < abs(
            in_stats["mean_epipolar_error"][0])
        assert abs(
            out_stats["mean_epipolar_error"][1]) < abs(
            in_stats["mean_epipolar_error"][1])
        assert abs(
            out_stats["median_epipolar_error"][0]) < abs(
            in_stats["median_epipolar_error"][0])
        assert abs(
            out_stats["median_epipolar_error"][1]) < abs(
            in_stats["median_epipolar_error"][1])
        assert out_stats["std_epipolar_error"][0] \
             < in_stats["std_epipolar_error"][0]
        assert out_stats["std_epipolar_error"][1] \
             < in_stats["std_epipolar_error"][1]
        assert out_stats["rms_epipolar_error"] \
             < in_stats["rms_epipolar_error"]
        assert out_stats["rmsd_epipolar_error"] \
             < in_stats["rmsd_epipolar_error"]

        # Assert absolute performances

        assert abs(out_stats["median_epipolar_error"][0]) < 0.1
        assert abs(out_stats["median_epipolar_error"][1]) < 0.1

        assert abs(out_stats["mean_epipolar_error"][0]) < 0.1
        assert abs(out_stats["mean_epipolar_error"][1]) < 0.1
        assert out_stats["rms_epipolar_error"] < 0.5

        # Assert corrected matches are corrected
        assert np.fabs(
            np.mean(corrected_matches[:, 1] - corrected_matches[:, 3])) < 0.1


@pytest.mark.unit_tests
def test_image_envelope():
    """
    Test image_envelope function
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        shp = os.path.join(directory, "envelope.gpkg")
        preprocessing.image_envelope(img, shp, dem)
        assert os.path.isfile(shp)


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

    srtm_ds = preprocessing.read_lowres_dem(
        startx, starty, sizex, sizey, dem=dem)

    # Uncomment to update baseline
    #srtm_ds.to_netcdf(absolute_data_path("ref_output/srtm_xt.nc"))

    srtm_ds_ref = xr.open_dataset(
        absolute_data_path("ref_output/srtm_xt.nc"))
    assert_same_datasets(srtm_ds, srtm_ds_ref)

    print(srtm_ds)


@pytest.mark.unit_tests
def test_get_time_ground_direction():
    """
    Test the get_time_ground_direction
    """

    # Force use of DEM if test is ran standalone
    dem = absolute_data_path("input/phr_ventoux/srtm")

    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    vec = preprocessing.get_time_ground_direction(img,dem=dem)

    assert vec[0] == -0.03760314420222626
    assert vec[1] == 0.9992927516729553

@pytest.mark.unit_tests
def test_get_ground_angles():
    """
    Test the get_ground_angles function
    """

    left_img = absolute_data_path("input/phr_ventoux/left_image.tif")
    right_img = absolute_data_path("input/phr_ventoux/right_image.tif")

    angles = preprocessing.get_ground_angles(left_img, right_img)
    angles = np.asarray(angles) # transform tuple to array

    np.testing.assert_allclose(angles,
    [19.48120732, 81.18985592, 189.98986491, 78.61360403, 20.12773114],
    rtol=1e-01)


@pytest.mark.unit_tests
def test_project_coordinates_on_line():
    """
    Test project_coordinates_on_line
    """
    origin=[0,0]
    vec = [0.5, 0.5]

    x_coord = np.array([1,2,3])
    y_coord = np.array([1,2,3])

    coords = preprocessing.project_coordinates_on_line(x_coord, y_coord,
                                                                origin, vec)

    np.testing.assert_allclose(coords, [1.41421356, 2.82842712, 4.24264069])


@pytest.mark.unit_tests
def test_lowres_initial_dem_splines_fit():
    """
    Test lowres_initial_dem_splines_fit
    """
    lowres_dsm_from_matches = xr.open_dataset(absolute_data_path(
        "input/splines_fit_input/lowres_dsm_from_matches.nc"))
    lowres_initial_dem = xr.open_dataset(absolute_data_path(
        "input/splines_fit_input/lowres_initial_dem.nc"))

    origin = [float(lowres_dsm_from_matches.x[0].values),
              float(lowres_dsm_from_matches.y[0].values)]
    vec = [0,1]

    splines = preprocessing.lowres_initial_dem_splines_fit(
        lowres_dsm_from_matches, lowres_initial_dem, origin, vec)


    # Uncomment to update reference
    # with open(absolute_data_path(
    #                   "ref_output/splines_ref.pck"),'wb') as splines_files:
    #     pickle.dump(splines, splines_file)

    with open(absolute_data_path(
                        "ref_output/splines_ref.pck"),'rb') as splines_file:
        ref_splines = pickle.load(splines_file)
        np.testing.assert_allclose(splines.get_coeffs(),
                                   ref_splines.get_coeffs())
        np.testing.assert_allclose(splines.get_knots(),
                                   ref_splines.get_knots())
