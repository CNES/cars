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
Test module for cars.core.geometry
"""
import numpy as np
import pytest

import cars.core.constants as cst
from cars.applications.application import Application
from cars.core.geometry.shareloc_geometry import RPC_TYPE

from ...helpers import absolute_data_path
from .dummy_abstract_classes import (  # noqa; pylint: disable=unused-import
    NoMethodClass,
)

from cars.core.geometry.abstract_geometry import (  # noqa;  isort:skip; pylint: disable=wrong-import-order
    AbstractGeometry,
)


@pytest.fixture
def epipolar_coords():
    """
    inputs for the test_matches_to_sensor_coords and
    test_sensor_position_from_grid tests
    """
    left_epipolar_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ]
    )

    right_epipolar_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [6.0, 1.0],
            [8.0, 1.0],
            [10.0, 1.0],
            [8.0, 2.0],
            [10.0, 2.0],
            [12.0, 2.0],
            [14.0, 2.0],
        ]
    )

    out_dict = {
        "left": left_epipolar_coords,
        "right": right_epipolar_coords,
    }

    return out_dict


@pytest.fixture
def ref_sensor_coords():
    """
    expected results for the test_matches_to_sensor_coords,
    test_disp_to_sensor_coords and test_sensor_position_from_grid tests
    """
    left_sensor_coords = np.array(
        [
            [-5737.38623047, -1539.64440918],
            [-5738.12719727, -1540.61027832],
            [-5738.86816406, -1541.57614746],
            [-5739.60913086, -1542.5420166],
            [-5736.42036133, -1540.38536784],
            [-5737.16132867, -1541.35123711],
            [-5737.90229601, -1542.31710639],
            [-5738.64326335, -1543.28297567],
            [-5735.45449219, -1541.1263265],
            [-5736.19546007, -1542.09219591],
            [-5736.93642795, -1543.05806532],
            [-5737.67739583, -1544.02393473],
        ]
    )
    right_sensor_coords = np.array(
        [
            [-5737.38623047, -1539.64440918],
            [-5738.12719727, -1540.61027832],
            [-5738.86816406, -1541.57614746],
            [-5739.60913086, -1542.5420166],
            [-5739.38423069, -1544.24884494],
            [-5740.86616536, -1546.1805835],
            [-5742.34810004, -1548.11232205],
            [-5743.83003472, -1550.0440606],
            [-5741.38223524, -1548.85328179],
            [-5742.86417101, -1550.78502062],
            [-5744.34610677, -1552.71675944],
            [-5745.82804253, -1554.64849826],
        ]
    )

    out_dict = {
        "left": left_sensor_coords,
        "right": right_sensor_coords,
    }
    return out_dict


@pytest.mark.unit_tests
def test_missing_abstract_methods():
    """
    Test cars geometry abstract class
    """
    with pytest.raises(Exception) as error:
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "NoMethodClass"
        )
    print(str(error.value))
    assert (
        str(error.value) == "Can't instantiate abstract class "
        "NoMethodClass with abstract methods check_product_consistency,"
        " direct_loc, generate_epipolar_grids, inverse_loc, triangulate"
    )


@pytest.mark.unit_tests
def test_wrong_class_name():
    """
    Test cars geometry abstract class
    """
    with pytest.raises(KeyError) as error:
        AbstractGeometry("test")  # pylint: disable=abstract-class-instantiated
    assert str(error.value) == "'No geometry plugin named test registered'"


@pytest.mark.unit_tests
def test_wrong_class_name_with_int():
    """
    Test cars geometry abstract class
    """
    with pytest.raises(RuntimeError) as error:
        AbstractGeometry(3)  # pylint: disable=abstract-class-instantiated
    assert str(error.value) == "Not a supported type"


@pytest.mark.unit_tests
def test_wrong_class_name_with_dict():
    """
    Test cars geometry abstract class
    """
    with pytest.raises(KeyError) as error:
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            {"plugin_name": "test"}
        )  # pylint: disable=abstract-class-instantiated
    assert str(error.value) == "'No geometry plugin named test registered'"


@pytest.mark.unit_tests
def test_correct_class_name_with():
    """
    Test cars geometry abstract class
    """
    AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry"}
    )
    AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {  # pylint: disable=abstract-class-instantiated
            "plugin_name": "SharelocGeometry",
            "interpolator": "cubic",
        }
    )
    AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {
            "plugin_name": "SharelocGeometry",
            "interpolator": "linear",
        }
    )
    AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        "SharelocGeometry"
    )


@pytest.mark.unit_tests
def test_sensor_position_from_grid(
    epipolar_coords, ref_sensor_coords
):  # pylint: disable=redefined-outer-name
    """
    Test sensor_position_from_grid
    """
    grid = absolute_data_path("input/abstract_geometry_input/grid.tif")

    coords = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
    ).sensor_position_from_grid(grid, epipolar_coords["left"])
    assert np.allclose(ref_sensor_coords["left"], coords)

    coords = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
    ).sensor_position_from_grid(grid, epipolar_coords["right"])
    assert np.allclose(ref_sensor_coords["right"], coords)


@pytest.mark.unit_tests
def test_disp_to_sensor_coords(
    ref_sensor_coords,
):  # pylint: disable=redefined-outer-name
    """
    Test matching_data_to_sensor_coords with the cst.DISP_MODE
    """
    grid1 = absolute_data_path("input/abstract_geometry_input/grid.tif")
    grid2 = absolute_data_path("input/abstract_geometry_input/grid.tif")

    nb_row = 3
    nb_col = 4
    disp_map = np.arange(nb_row * nb_col)
    disp_map = disp_map.reshape((nb_row, nb_col))
    disp_msk = np.full((nb_row, nb_col), fill_value=255)
    disp_msk[0, :] = 0

    (
        sensor_pos_left,
        sensor_pos_right,
    ) = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
    ).matches_to_sensor_coords(
        grid1, grid2, disp_map, cst.DISP_MODE, matches_msk=disp_msk
    )

    ref_sensor_pos_left_x = np.copy(ref_sensor_coords["left"][:, 0])
    ref_sensor_pos_left_y = np.copy(ref_sensor_coords["left"][:, 1])
    ref_sensor_pos_left_x = ref_sensor_pos_left_x.reshape((nb_row, nb_col))
    ref_sensor_pos_left_y = ref_sensor_pos_left_y.reshape((nb_row, nb_col))
    ref_sensor_pos_left_x[np.where(disp_msk != 255)] = np.nan
    ref_sensor_pos_left_y[np.where(disp_msk != 255)] = np.nan

    assert np.allclose(
        sensor_pos_left[:, :, 0], ref_sensor_pos_left_x, equal_nan=True
    )
    assert np.allclose(
        sensor_pos_left[:, :, 1], ref_sensor_pos_left_y, equal_nan=True
    )

    ref_sensor_pos_right_x = np.copy(ref_sensor_coords["right"][:, 0])
    ref_sensor_pos_right_y = np.copy(ref_sensor_coords["right"][:, 1])
    ref_sensor_pos_right_x = ref_sensor_pos_right_x.reshape((nb_row, nb_col))
    ref_sensor_pos_right_y = ref_sensor_pos_right_y.reshape((nb_row, nb_col))
    ref_sensor_pos_right_x[np.where(disp_msk != 255)] = np.nan
    ref_sensor_pos_right_y[np.where(disp_msk != 255)] = np.nan

    assert np.allclose(
        sensor_pos_right[:, :, 0], ref_sensor_pos_right_x, equal_nan=True
    )
    assert np.allclose(
        sensor_pos_right[:, :, 1], ref_sensor_pos_right_y, equal_nan=True
    )

    # test with a cropped disparity map (ul_corner is expressed as (X,Y))
    ul_corner_crop = (1, 2)
    disp_map = disp_map[ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col]
    disp_msk = disp_msk[ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col]

    (
        sensor_pos_left,
        sensor_pos_right,
    ) = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
    ).matches_to_sensor_coords(
        grid1,
        grid2,
        disp_map,
        cst.DISP_MODE,
        matches_msk=disp_msk,
        ul_matches_shift=ul_corner_crop,
    )

    ref_sensor_pos_left_x = ref_sensor_pos_left_x[
        ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col
    ]
    ref_sensor_pos_left_y = ref_sensor_pos_left_y[
        ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col
    ]

    assert np.allclose(
        sensor_pos_left[:, :, 0], ref_sensor_pos_left_x, equal_nan=True
    )
    assert np.allclose(
        sensor_pos_left[:, :, 1], ref_sensor_pos_left_y, equal_nan=True
    )

    ref_sensor_pos_right_x = ref_sensor_pos_right_x[
        ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col
    ]
    ref_sensor_pos_right_y = ref_sensor_pos_right_y[
        ul_corner_crop[1] : nb_row, ul_corner_crop[0] : nb_col
    ]
    assert np.allclose(
        sensor_pos_right[:, :, 0], ref_sensor_pos_right_x, equal_nan=True
    )
    assert np.allclose(
        sensor_pos_right[:, :, 1], ref_sensor_pos_right_y, equal_nan=True
    )


@pytest.mark.unit_tests
def test_matches_to_sensor_coords(
    epipolar_coords, ref_sensor_coords
):  # pylint: disable=redefined-outer-name
    """
    Test matching_data_to_sensor_coords with the cst.MATCHES_MODE
    """
    grid1 = absolute_data_path("input/abstract_geometry_input/grid.tif")
    grid2 = absolute_data_path("input/abstract_geometry_input/grid.tif")

    matches = np.hstack([epipolar_coords["left"], epipolar_coords["right"]])

    (
        sensor_pos_left,
        sensor_pos_right,
    ) = AbstractGeometry(  # pylint: disable=abstract-class-instantiated
        {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
    ).matches_to_sensor_coords(
        grid1, grid2, matches, cst.MATCHES_MODE
    )

    assert np.allclose(sensor_pos_left, ref_sensor_coords["left"])
    assert np.allclose(sensor_pos_right, ref_sensor_coords["right"])


@pytest.mark.unit_tests
def test_epipolar_position_from_grid():
    """
    Test epipolar_position_from_grid
    """

    sensor_left = {
        "image": absolute_data_path("input/phr_ventoux/left_image.tif"),
        "geomodel": {
            "path": absolute_data_path("input/phr_ventoux/left_image.geom"),
            "model_type": RPC_TYPE,
        },
    }

    sensor_right = {
        "image": absolute_data_path("input/phr_ventoux/right_image.tif"),
        "geomodel": {
            "path": absolute_data_path("input/phr_ventoux/right_image.geom"),
            "model_type": RPC_TYPE,
        },
    }

    geo_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry"
        )
    )

    epipolar_grid_generation_application = Application("grid_generation")
    (
        grid_left,
        _,
    ) = epipolar_grid_generation_application.run(
        sensor_left, sensor_right, geo_loader
    )

    epi_pos = np.array([[2, 2], [2, 300], [2, 580], [300, 300], [600, 300]])

    sensor_pos = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
        ).sensor_position_from_grid(grid_left, epi_pos)
    )

    new_epi_pos = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            {"plugin_name": "SharelocGeometry", "interpolator": "linear"}
        ).epipolar_position_from_grid(grid_left, sensor_pos, step=30)
    )

    assert np.allclose(new_epi_pos, epi_pos)
    np.testing.assert_allclose(new_epi_pos, epi_pos, rtol=0.01, atol=0.01)
