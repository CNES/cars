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

import os
import tempfile
from shutil import copy2
from typing import Tuple

# Third party imports
import numpy as np
import otbApplication
import pytest
import rasterio as rio
import xarray as xr

# CARS imports
from cars.core import constants as cst
from cars.externals import otb_pipelines

# CARS Tests imports
from ..helpers import (
    absolute_data_path,
    assert_same_datasets,
    otb_geoid_file_set,
    otb_geoid_file_unset,
    temporary_dir,
)


@pytest.mark.unit_tests
def test_build_stereorectification_grid_pipeline():
    """
    Test if the pipeline is correctly built and produces consistent grids
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45
    # Set the geoid file from code source
    otb_geoid_file_set()

    # Launch otb stereorectification grid pipeline
    (
        left_grid_np,
        right_grid_np,
        left_grid_origin,
        left_grid_spacing,
        epipolar_size_x,
        epipolar_size_y,
        disp_to_alt_ratio,
    ) = otb_pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem, epi_step=step
    )

    assert epipolar_size_x == 612
    assert epipolar_size_y == 612
    assert left_grid_np.shape == (15, 15, 2)
    assert left_grid_origin[0] == 0
    assert left_grid_origin[1] == 0
    assert left_grid_spacing[0] == step
    assert left_grid_spacing[1] == step
    assert np.isclose(disp_to_alt_ratio, 0.7, 0.01)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/left_grid.npy"), left_grid_np)

    left_grid_np_reference = np.load(
        absolute_data_path("ref_output/left_grid.npy")
    )
    np.testing.assert_allclose(left_grid_np, left_grid_np_reference)

    assert right_grid_np.shape == (15, 15, 2)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/right_grid.npy"), right_grid_np)

    right_grid_np_reference = np.load(
        absolute_data_path("ref_output/right_grid.npy")
    )
    np.testing.assert_allclose(right_grid_np, right_grid_np_reference)

    # unset otb geoid file
    otb_geoid_file_unset()


@pytest.mark.unit_tests
def test_build_stereorectification_grid_pipeline_scaled_inputs():
    """
    test different pixel sizes in input images
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45

    # Set the geoid file from code source
    otb_geoid_file_set()

    # reference
    (
        _,
        _,
        _,
        _,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
    ) = otb_pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem, epi_step=step
    )

    # define negative scale transform
    def create_negative_transform(srs_img, dst_img, reverse_x, reverse_y):
        """
        Reverse transform on x or y axis if reverse_x or reverse_y are activated
        :param srs_img:
        :type srs_img: str
        :param dst_img:
        :type dst_img: str
        :param reverse_x:
        :type srs_img: bool
        :param reverse_y:
        :type srs_img: bool
        :return:
        """
        with rio.open(srs_img, "r") as rio_former_dst:
            former_array = rio_former_dst.read(1)
            former_transform = rio_former_dst.transform
            # modify transform
            x_fact = 1
            y_fact = 1
            x_size = 0
            y_size = 0

            if reverse_x:
                x_fact = -1
                x_size = former_array.shape[0] * abs(former_transform[0])
            if reverse_y:
                y_fact = -1
                y_size = former_array.shape[1] * abs(former_transform[4])
            new_transform = rio.Affine(
                x_fact * former_transform[0],
                former_transform[1],
                x_size + former_transform[2],
                former_transform[3],
                y_fact * former_transform[4],
                y_size + former_transform[5],
            )

            with rio.open(
                dst_img,
                "w",
                driver="GTiff",
                height=former_array.shape[0],
                width=former_array.shape[1],
                count=1,
                dtype=former_array.dtype,
                crs=rio_former_dst.crs,
                transform=new_transform,
            ) as rio_dst:
                rio_dst.write(former_array, 1)

    # define generic test
    def test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex,
        scaley,
    ):
        """
        Test that epipolar image size and disp_to_alt_ratio remain unchanged
        when scaling the input images

        tested combinations:
        - scaled img1 and scaled img2
        - img1 and scaled img2
        - scaled img1 and img2
        """

        # create otb app to rescale input images
        app = otbApplication.Registry.CreateApplication(
            "RigidTransformResample"
        )

        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
            # manage negative scaling
            negative_scale_x = scalex < 0
            negative_scale_y = scaley < 0

            # rescale inputs
            img1_transform = os.path.join(directory, "img1_transform.tif")
            img2_transform = os.path.join(directory, "img2_transform.tif")

            if negative_scale_x or negative_scale_y:
                # create new images
                img1_geom = img1.replace(".tif", ".geom")
                img2_geom = img2.replace(".tif", ".geom")
                img1_reversed = os.path.join(directory, "img1_reversed.tif")
                img2_reversed = os.path.join(directory, "img2_reversed.tif")
                img1_reversed_geom = os.path.join(
                    directory, "img1_reversed.geom"
                )
                img2_reversed_geom = os.path.join(
                    directory, "img2_reversed.geom"
                )
                copy2(img1_geom, img1_reversed_geom)
                copy2(img2_geom, img2_reversed_geom)
                create_negative_transform(
                    img1, img1_reversed, negative_scale_x, negative_scale_y
                )
                create_negative_transform(
                    img2, img2_reversed, negative_scale_x, negative_scale_y
                )
                img1 = img1_reversed
                img2 = img2_reversed

            app.SetParameterString("in", img1)
            app.SetParameterString("transform.type", "id")
            app.SetParameterFloat("transform.type.id.scalex", abs(scalex))
            app.SetParameterFloat("transform.type.id.scaley", abs(scaley))
            app.SetParameterString("out", img1_transform)
            app.ExecuteAndWriteOutput()

            app.SetParameterString("in", img2)
            app.SetParameterString("out", img2_transform)
            app.ExecuteAndWriteOutput()

            with rio.open(img1_transform, "r") as rio_dst:
                pixel_size_x, pixel_size_y = (
                    rio_dst.transform[0],
                    rio_dst.transform[4],
                )
                assert pixel_size_x == 1 / scalex
                assert pixel_size_y == 1 / scaley

            with rio.open(img2_transform, "r") as rio_dst:
                pixel_size_x, pixel_size_y = (
                    rio_dst.transform[0],
                    rio_dst.transform[4],
                )
                assert pixel_size_x == 1 / scalex
                assert pixel_size_y == 1 / scaley

            # img1_transform / img2_transform
            (
                _,
                _,
                _,
                _,
                epipolar_size_x,
                epipolar_size_y,
                disp_to_alt_ratio,
            ) = otb_pipelines.build_stereorectification_grid_pipeline(
                img1_transform, img2_transform, dem, epi_step=step
            )

            assert epipolar_size_x == ref_epipolar_size_x
            assert epipolar_size_y == ref_epipolar_size_y
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

            # img1_transform / img2
            (
                _,
                _,
                _,
                _,
                epipolar_size_x,
                epipolar_size_y,
                disp_to_alt_ratio,
            ) = otb_pipelines.build_stereorectification_grid_pipeline(
                img1_transform, img2, dem, epi_step=step
            )

            assert epipolar_size_x == ref_epipolar_size_x
            assert epipolar_size_y == ref_epipolar_size_y
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

            # img1 / img2_transform
            (
                _,
                _,
                _,
                _,
                epipolar_size_x,
                epipolar_size_y,
                disp_to_alt_ratio,
            ) = otb_pipelines.build_stereorectification_grid_pipeline(
                img1, img2_transform, dem, epi_step=step
            )

            assert epipolar_size_x == ref_epipolar_size_x
            assert epipolar_size_y == ref_epipolar_size_y
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

    # test with scalex= 2, scaley=2
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=2.0,
        scaley=2.0,
    )
    # test with scalex= 2, scaley=3
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=2.0,
        scaley=3.0,
    )
    # test with scalex= 0.5, scaley=0.5
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=1 / 2.0,
        scaley=1 / 2.0,
    )
    # test with scalex= 0.5, scaley=0.25
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=1 / 2.0,
        scaley=1 / 4.0,
    )

    # test with scalex= 1, scaley=-1
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=1.0,
        scaley=-1.0,
    )

    # test with scalex= -1, scaley=1
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=-1.0,
        scaley=1.0,
    )

    # test with scalex= -1, scaley=-2
    test_with_scaled_inputs(
        img1,
        img2,
        dem,
        step,
        ref_epipolar_size_x,
        ref_epipolar_size_y,
        ref_disp_to_alt_ratio,
        scalex=-1.0,
        scaley=-2.0,
    )

    # unset otb geoid file
    otb_geoid_file_unset()


@pytest.mark.unit_tests
def test_build_extract_roi_application():
    """
    Test that input region is correctly use to build the roi extraction
    application
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    region = [100, 200, 300, 400]

    app = otb_pipelines.build_extract_roi_application(img, region)

    assert app.GetParameterInt("startx") == region[0]
    assert app.GetParameterInt("starty") == region[1]
    assert app.GetParameterInt("sizex") == region[2] - region[0]
    assert app.GetParameterInt("sizey") == region[3] - region[1]


@pytest.mark.unit_tests
def test_build_mask_pipeline():
    """
    Test that the pipeline is correctly built
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    nodata = 0
    mask = absolute_data_path("input/phr_reunion/left_mask.tif")
    roi = [100, 200, 300, 400]
    out_np = otb_pipelines.build_mask_pipeline(
        img, grid, nodata, mask, 2387, 2387, roi
    )

    assert out_np.shape == (200, 200)


@pytest.mark.unit_tests
def test_build_image_resampling_pipeline():
    """
    Test that the pipeline is correctly built
    """
    img = absolute_data_path("input/phr_reunion/left_image.tif")
    grid = absolute_data_path("input/pipelines_input/left_epipolar_grid.tif")
    roi = [100, 200, 300, 400]
    out_np = otb_pipelines.build_image_resampling_pipeline(
        img, grid, 2387, 2387, roi
    )

    assert out_np.shape == (200, 200, 1)


@pytest.mark.unit_tests
def test_get_utm_zone_as_epsg_code():
    """
    Test if a point in Toulouse gives the correct EPSG code
    """
    epsg = otb_pipelines.get_utm_zone_as_epsg_code(1.442299, 43.600764)
    assert epsg == 32631


@pytest.mark.unit_tests
def test_image_envelope():
    """
    Test image_envelope function
    """
    img = absolute_data_path("input/phr_ventoux/left_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        shp = os.path.join(directory, "envelope.gpkg")
        otb_pipelines.image_envelope(img, shp, dem)
        assert os.path.isfile(shp)


def generate_epipolar_grids(
    img1: str,
    img2: str,
    srtm_dir: str = None,
    default_alt: float = None,
    epi_step: float = 30,
) -> Tuple[xr.Dataset, xr.Dataset, int, int, float]:
    """
    Generate epipolar resampling grids
    as xarray.Dataset from a pair of images and srtm_dir

    TODO: move in cars src code in grids.py as generic call for pipelines ?

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
    # Set the geoid file from code source
    otb_geoid_file_set()

    # Launch OTB pipeline to get stero grids
    (
        grid1,
        grid2,
        __,
        __,
        epipolar_size_x,
        epipolar_size_y,
        baseline,
    ) = otb_pipelines.build_stereorectification_grid_pipeline(
        img1, img2, dem=srtm_dir, default_alt=default_alt, epi_step=epi_step
    )

    col = np.array(range(0, grid1.shape[0] * epi_step, epi_step))
    row = np.array(range(0, grid1.shape[1] * epi_step, epi_step))

    left_grid_dataset = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], grid1[:, :, 0]),
            cst.Y: ([cst.ROW, cst.COL], grid1[:, :, 1]),
        },
        coords={cst.ROW: row, cst.COL: col},
        attrs={
            "epi_step": epi_step,
            "epipolar_size_x": epipolar_size_x,
            "epipolar_size_y": epipolar_size_y,
        },
    )

    right_grid_dataset = xr.Dataset(
        {
            cst.X: ([cst.ROW, cst.COL], grid2[:, :, 0]),
            cst.Y: ([cst.ROW, cst.COL], grid2[:, :, 1]),
        },
        coords={cst.ROW: row, cst.COL: col},
        attrs={
            "epi_step": epi_step,
            "epipolar_size_x": epipolar_size_x,
            "epipolar_size_y": epipolar_size_y,
        },
    )

    # Unset geoid for the test to be standalone
    otb_geoid_file_unset()

    return (
        left_grid_dataset,
        right_grid_dataset,
        epipolar_size_x,
        epipolar_size_y,
        baseline,
    )


@pytest.mark.unit_tests
def test_generate_epipolar_grids():
    """
    Test generate_epipolar_grids method
    """
    img1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    img2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    left_grid, right_grid, size_x, size_y, baseline = generate_epipolar_grids(
        img1, img2, dem
    )

    assert size_x == 612
    assert size_y == 612
    assert baseline == 0.7039416432380676

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path("ref_output/left_grid.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid.nc")
    )
    assert_same_datasets(left_grid, left_grid_ref)

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path("ref_output/right_grid.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid.nc")
    )
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

    left_grid, right_grid, size_x, size_y, baseline = generate_epipolar_grids(
        img1, img2, dem, default_alt
    )

    assert size_x == 612
    assert size_y == 612
    assert baseline == 0.7039446234703064

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path(
    # "ref_output/left_grid_default_alt.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid_default_alt.nc")
    )
    assert_same_datasets(left_grid, left_grid_ref)

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path(
    # "ref_output/right_grid_default_alt.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid_default_alt.nc")
    )
    assert_same_datasets(right_grid, right_grid_ref)


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
        startx, starty, sizex, sizey, dem=dem
    )

    # Uncomment to update baseline
    # srtm_ds.to_netcdf(absolute_data_path("ref_output/srtm_xt.nc"))

    srtm_ds_ref = xr.open_dataset(absolute_data_path("ref_output/srtm_xt.nc"))
    assert_same_datasets(srtm_ds, srtm_ds_ref)
