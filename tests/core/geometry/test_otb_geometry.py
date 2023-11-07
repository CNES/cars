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
Test module for cars/core/geometry/otb_geometry.py
Work only with CARS and OTB install. CI without OTB fails also.
"""

import os
import tempfile
from shutil import copy2

# Third party imports
import numpy as np
import pytest
import rasterio as rio

from cars.core.geometry.abstract_geometry import AbstractGeometry

# CARS imports
from cars.core.inputs import read_vector

# CARS Tests imports
from ...helpers import absolute_data_path, get_geoid_path, temporary_dir


@pytest.mark.unit_tests
def test_generate_epipolar_grids_otb():
    """
    Test if the pipeline is correctly built and produces consistent grids
    OTB only
    """
    sensor1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    sensor2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    geomodel1 = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel2 = absolute_data_path("input/phr_ventoux/right_image.geom")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem, geoid=get_geoid_path()
        )
    )

    # test with geoid
    (
        left_grid_as_array,
        right_grid_as_array,
        origin,
        spacing,
        epipolar_size,
        disp_to_alt_ratio,
    ) = geo_plugin.generate_epipolar_grids(
        sensor1, sensor2, geomodel1, geomodel2, epipolar_step=step
    )

    assert epipolar_size == [612, 612]
    assert left_grid_as_array.shape == (15, 15, 2)
    assert origin[0] == 0
    assert origin[1] == 0
    assert spacing[0] == step
    assert spacing[1] == step
    assert np.isclose(disp_to_alt_ratio, 1 / 0.7, 0.01)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/left_grid.npy"),
    #         left_grid_as_array)

    left_grid_np_reference = np.load(
        absolute_data_path("ref_output/left_grid.npy")
    )
    np.testing.assert_allclose(left_grid_as_array, left_grid_np_reference)

    assert right_grid_as_array.shape == (15, 15, 2)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/right_grid.npy"),
    #         right_grid_as_array)

    right_grid_np_reference = np.load(
        absolute_data_path("ref_output/right_grid.npy")
    )
    np.testing.assert_allclose(right_grid_as_array, right_grid_np_reference)

    # test without geoid
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem
        )
    )

    (
        left_grid_as_array,
        right_grid_as_array,
        origin,
        spacing,
        epipolar_size,
        disp_to_alt_ratio,
    ) = geo_plugin.generate_epipolar_grids(
        sensor1, sensor2, geomodel1, geomodel2, epipolar_step=step
    )

    assert epipolar_size == [612, 612]
    assert left_grid_as_array.shape == (15, 15, 2)
    assert origin[0] == 0
    assert origin[1] == 0
    assert spacing[0] == step
    assert spacing[1] == step
    assert np.isclose(disp_to_alt_ratio, 1 / 0.7, 0.01)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/left_grid_no_geoid.npy"),
    #         left_grid_as_array)

    left_grid_np_reference = np.load(
        absolute_data_path("ref_output/left_grid_no_geoid.npy")
    )
    np.testing.assert_allclose(left_grid_as_array, left_grid_np_reference)

    assert right_grid_as_array.shape == (15, 15, 2)

    # Uncomment to update baseline
    # np.save(absolute_data_path("ref_output/right_grid_no_geoid.npy"),
    #         right_grid_as_array)

    right_grid_np_reference = np.load(
        absolute_data_path("ref_output/right_grid_no_geoid.npy")
    )
    np.testing.assert_allclose(right_grid_as_array, right_grid_np_reference)


@pytest.mark.unit_tests
def test_generate_epipolar_grids_scaled_inputs_otb():
    """
    test different pixel sizes in input images
    """
    import otbApplication  # pylint: disable=import-error, C0415

    def rigid_transform_resample(
        img: str, scale_x: float, scale_y: float, img_transformed: str
    ):
        """
        Execute RigidTransformResample OTB application

        :param img: path to the image to transform
        :param scale_x: scale factor to apply along x axis
        :param scale_y: scale factor to apply along y axis
        :param img_transformed: output image path
        """

        # create otb app to rescale input images
        app = otbApplication.Registry.CreateApplication(
            "RigidTransformResample"
        )

        app.SetParameterString("in", img)
        app.SetParameterString("transform.type", "id")
        app.SetParameterFloat("transform.type.id.scalex", abs(scale_x))
        app.SetParameterFloat("transform.type.id.scaley", abs(scale_y))
        app.SetParameterString("out", img_transformed)
        app.ExecuteAndWriteOutput()

    sensor1 = absolute_data_path("input/phr_ventoux/left_image.tif")
    sensor2 = absolute_data_path("input/phr_ventoux/right_image.tif")
    geomodel1 = absolute_data_path("input/phr_ventoux/left_image.geom")
    geomodel2 = absolute_data_path("input/phr_ventoux/right_image.geom")
    dem = absolute_data_path("input/phr_ventoux/srtm")
    step = 45

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem, geoid=get_geoid_path()
        )
    )

    # reference
    (
        _,
        _,
        _,
        _,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
    ) = geo_plugin.generate_epipolar_grids(
        sensor1, sensor2, geomodel1, geomodel2, epipolar_step=step
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
        ref_epipolar_size,
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

        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
            # link geomodels
            img1_geom = img1.replace(".tif", ".geom")
            img2_geom = img2.replace(".tif", ".geom")
            # manage negative scaling
            negative_scale_x = scalex < 0
            negative_scale_y = scaley < 0

            # rescale inputs
            img1_transform = os.path.join(directory, "img1_transform.tif")
            img2_transform = os.path.join(directory, "img2_transform.tif")
            img1_transform_geom = os.path.join(directory, "img1_transform.geom")
            img2_transform_geom = os.path.join(directory, "img2_transform.geom")

            if negative_scale_x or negative_scale_y:
                # create new images
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

            rigid_transform_resample(img1, scalex, scaley, img1_transform)
            rigid_transform_resample(img2, scalex, scaley, img2_transform)

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

            geo_plugin = (
                AbstractGeometry(  # pylint: disable=abstract-class-instantiated
                    "OTBGeometry", dem=dem, geoid=get_geoid_path()
                )
            )

            # img1_transform / img2_transform
            (
                _,
                _,
                _,
                _,
                epipolar_size,
                disp_to_alt_ratio,
            ) = geo_plugin.generate_epipolar_grids(
                img1_transform,
                img2_transform,
                img1_transform_geom,
                img2_transform_geom,
                epipolar_step=step,
            )

            assert epipolar_size == ref_epipolar_size
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

            # img1_transform / img2
            (
                _,
                _,
                _,
                _,
                epipolar_size,
                disp_to_alt_ratio,
            ) = geo_plugin.generate_epipolar_grids(
                img1_transform,
                img2,
                img1_transform_geom,
                img2_geom,
                epipolar_step=step,
            )

            assert epipolar_size == ref_epipolar_size
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

            # img1 / img2_transform
            (
                _,
                _,
                _,
                _,
                epipolar_size,
                disp_to_alt_ratio,
            ) = geo_plugin.generate_epipolar_grids(
                img1,
                img2_transform,
                img1_geom,
                img2_transform_geom,
                epipolar_step=step,
            )

            assert epipolar_size == ref_epipolar_size
            assert abs(disp_to_alt_ratio - ref_disp_to_alt_ratio) < 1e-06

    # test with scalex= 2, scaley=2
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=2.0,
        scaley=2.0,
    )
    # test with scalex= 2, scaley=3
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=2.0,
        scaley=3.0,
    )
    # test with scalex= 0.5, scaley=0.5
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=1 / 2.0,
        scaley=1 / 2.0,
    )
    # test with scalex= 0.5, scaley=0.25
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=1 / 2.0,
        scaley=1 / 4.0,
    )

    # test with scalex= 1, scaley=-1
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=1.0,
        scaley=-1.0,
    )

    # test with scalex= -1, scaley=1
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=-1.0,
        scaley=1.0,
    )

    # test with scalex= -1, scaley=-2
    test_with_scaled_inputs(
        sensor1,
        sensor2,
        dem,
        step,
        ref_epipolar_size,
        ref_disp_to_alt_ratio,
        scalex=-1.0,
        scaley=-2.0,
    )


@pytest.mark.unit_tests
def test_image_envelope_otb():
    """
    Test image_envelope function
    """
    sensor = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel = absolute_data_path("input/phr_ventoux/left_image.geom")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry"
        )
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        shp = os.path.join(directory, "envelope.gpkg")

        geo_plugin.image_envelope(
            sensor,
            geomodel,
            shp,
        )

        assert os.path.isfile(shp)
        poly, epsg = read_vector(shp)
        assert epsg == 4326
        assert list(poly.exterior.coords) == [
            (5.193078704737658, 44.207395924036454),
            (5.19624740770905, 44.20744798617616),
            (5.196301073780709, 44.205180098419376),
            (5.193132520720433, 44.20512807053881),
            (5.193078704737658, 44.207395924036454),
        ]

    # test with dem + geoid
    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem, geoid=get_geoid_path()
        )
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        shp = os.path.join(directory, "envelope.gpkg")

        geo_plugin.image_envelope(sensor, geomodel, shp)

        assert os.path.isfile(shp)
        poly, epsg = read_vector(shp)
        assert epsg == 4326
        assert list(poly.exterior.coords) == [
            (5.193406138843349, 44.20805805252155),
            (5.1965650939582435, 44.20809526197842),
            (5.196654349708835, 44.205901416036546),
            (5.193485218293437, 44.205842790578764),
            (5.193406138843349, 44.20805805252155),
        ]

    # test image_envelope function defined in AbstractGeometry using the
    # OTBGeometry direct_loc

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        shp = os.path.join(directory, "envelope.shp")
        super(type(geo_plugin), geo_plugin).image_envelope(
            sensor,
            geomodel,
            shp,
        )

        assert os.path.isfile(shp)

        poly, epsg = read_vector(shp)
        assert epsg == 4326
        assert list(poly.exterior.coords) == [
            (5.193406138843349, 44.20805805252155),
            (5.1965650939582435, 44.20809526197842),
            (5.196654349708835, 44.205901416036546),
            (5.193485218293437, 44.205842790578764),
            (5.193406138843349, 44.20805805252155),
        ]


@pytest.mark.unit_tests
def test_check_consistency_otb():
    """
    Test otb_can_open() with different geom configurations
    """

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry"
        )
    )

    # existing
    existing_with_geom = absolute_data_path("input/phr_ventoux/left_image.tif")
    # existing with no geom file
    existing_no_geom = absolute_data_path("input/utils_input/im1.tif")
    # not existing
    not_existing = "/stuff/dummy_file.doe"
    # dummy geomodel
    dummy_geomodel = {"path": "dummy"}

    geo_plugin.check_product_consistency(existing_with_geom, dummy_geomodel)
    with pytest.raises(RuntimeError):
        geo_plugin.check_product_consistency(existing_no_geom, dummy_geomodel)
    with pytest.raises(RuntimeError):
        geo_plugin.check_product_consistency(not_existing, dummy_geomodel)


@pytest.mark.unit_tests
def test_dir_loc_multipoint_otb():
    """
    Test direct localization multipoint
    """
    sensor = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel = absolute_data_path("input/phr_ventoux/left_image.geom")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem
        )
    )

    lat, lon, alt = geo_plugin.direct_loc(
        sensor,
        geomodel,
        np.array([0, 5, 10]),
        np.array([0, 6, 12]),
    )
    current = np.array([lat, lon, alt])
    reference = np.array(
        [
            [44.20805589, 44.2080298, 44.20800366],
            [5.19340938, 5.19344198, 5.19347456],
            [503.5501949, 504.01208942, 504.43602082],
        ],
        np.dtype(np.float64),
    )
    np.testing.assert_allclose(current, reference)


@pytest.mark.unit_tests
def test_inverse_loc_multipoint_otb():
    """
    Test inverse localization multipoint
    """
    sensor = absolute_data_path("input/phr_ventoux/left_image.tif")
    geomodel = absolute_data_path("input/phr_ventoux/left_image.geom")
    dem = absolute_data_path("input/phr_ventoux/srtm")

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "OTBGeometry", dem=dem
        )
    )

    inputs_lat = np.array(
        [44.20805589, 44.2080298, 44.20800366],
    )
    inputs_lon = np.array([5.19340938, 5.19344198, 5.19347456])
    inputs_z = np.array([503.5501949, 504.01208942, 504.43602082])

    col, row, alti = geo_plugin.inverse_loc(
        sensor, geomodel, inputs_lat, inputs_lon, z_coord=inputs_z
    )

    reference_col = np.array([0, 5, 10])
    reference_row = np.array([0, 6, 12])
    np.testing.assert_allclose(row, reference_row, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(col, reference_col, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(alti, inputs_z, rtol=0.01, atol=0.01)
