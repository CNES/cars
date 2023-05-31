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
Test module for cars/steps/epi_rectif/test_grids.py
"""

# Standard imports
from __future__ import absolute_import

import os
import pickle
import tempfile

# Third party imports
import numpy as np
import pytest
import rasterio as rio
import xarray as xr

from cars import __version__
from cars.applications.application import Application
from cars.applications.grid_generation import grid_correction, grids
from cars.applications.sparse_matching import sparse_matching_tools

# CARS imports
from cars.conf import input_parameters
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    assert_same_carsdatasets,
    get_geoid_path,
    get_geometry_loader,
    temporary_dir,
)


@pytest.mark.unit_tests
def test_correct_right_grid():
    """
    Call right grid correction method and check outputs properties
    """
    matches_file = absolute_data_path(
        "input/preprocessing_input/matches_ventoux.npy"
    )
    grid_file = absolute_data_path(
        "input/preprocessing_input/right_epipolar_grid_uncorrected_ventoux.tif"
    )
    origin = [0, 0]
    spacing = [30, 30]

    matches = np.load(matches_file)
    matches = np.array(matches)

    matches_filtered = sparse_matching_tools.remove_epipolar_outliers(matches)

    with rio.open(grid_file) as rio_grid:
        grid = rio_grid.read()
        grid = np.transpose(grid, (1, 2, 0))

        # Estimate grid correction
        # Create fake cars dataset with grid
        grid_right = cars_dataset.CarsDataset("arrays")
        grid_right.tiling_grid = np.array(
            [[[0, grid.shape[0], 0, grid.shape[1]]]]
        )
        grid_right[0, 0] = grid
        grid_right.attributes["grid_origin"] = origin
        grid_right.attributes["grid_spacing"] = spacing

        (
            grid_correction_coef,
            corrected_matches,
            _,
            _,
            in_stats,
            out_stats,
        ) = grid_correction.estimate_right_grid_correction(
            matches_filtered, grid_right
        )

        # Correct grid right
        corrected_grid_cars_ds = grid_correction.correct_grid(
            grid_right, grid_correction_coef
        )
        corrected_grid = corrected_grid_cars_ds[0, 0]

        # Uncomment to update ref
        # np.save(absolute_data_path("ref_output/corrected_right_grid.npy"),
        # corrected_grid)
        corrected_grid_ref = np.load(
            absolute_data_path("ref_output/corrected_right_grid.npy")
        )
        np.testing.assert_allclose(
            corrected_grid, corrected_grid_ref, atol=0.05, rtol=1.0e-6
        )

        assert corrected_grid.shape == grid.shape

        # Assert that we improved all stats
        assert abs(out_stats["mean_epipolar_error"][0]) < abs(
            in_stats["mean_epipolar_error"][0]
        )
        assert abs(out_stats["mean_epipolar_error"][1]) < abs(
            in_stats["mean_epipolar_error"][1]
        )
        assert abs(out_stats["median_epipolar_error"][0]) < abs(
            in_stats["median_epipolar_error"][0]
        )
        assert abs(out_stats["median_epipolar_error"][1]) < abs(
            in_stats["median_epipolar_error"][1]
        )
        assert (
            out_stats["std_epipolar_error"][0]
            < in_stats["std_epipolar_error"][0]
        )
        assert (
            out_stats["std_epipolar_error"][1]
            < in_stats["std_epipolar_error"][1]
        )
        assert out_stats["rms_epipolar_error"] < in_stats["rms_epipolar_error"]
        assert (
            out_stats["rmsd_epipolar_error"] < in_stats["rmsd_epipolar_error"]
        )

        # Assert absolute performances

        assert abs(out_stats["median_epipolar_error"][0]) < 0.1
        assert abs(out_stats["median_epipolar_error"][1]) < 0.1

        assert abs(out_stats["mean_epipolar_error"][0]) < 0.1
        assert abs(out_stats["mean_epipolar_error"][1]) < 0.1
        assert out_stats["rms_epipolar_error"] < 0.5

        # Assert corrected matches are corrected
        assert (
            np.fabs(np.mean(corrected_matches[:, 1] - corrected_matches[:, 3]))
            < 0.1
        )


@pytest.mark.unit_tests
def test_generate_epipolar_grids_default_alt_otb():
    """
    Test generate_epipolar_grids method with default alt and no dem with OTB
    """
    conf = {
        input_parameters.IMG1_TAG: absolute_data_path(
            "input/phr_ventoux/left_image.tif"
        ),
        input_parameters.IMG2_TAG: absolute_data_path(
            "input/phr_ventoux/right_image.tif"
        ),
    }
    dem = None
    default_alt = 500

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        conf,
        "OTBGeometry",
        dem,
        default_alt=default_alt,
        epipolar_step=30,
        geoid=get_geoid_path(),
    )

    assert epi_size == [612, 612]
    assert baseline == 1 / 0.7039446234703064

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path(
    # "ref_output/left_grid_default_alt.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid_default_alt.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path(
    # "ref_output/right_grid_default_alt.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid_default_alt.nc")
    )
    assert np.allclose(right_grid_ref["x"].values, right_grid[:, :, 0])
    assert np.allclose(right_grid_ref["y"].values, right_grid[:, :, 1])


@pytest.mark.unit_tests
def test_generate_epipolar_grids_default_alt_shareloc(images_and_grids_conf):
    """
    Test generate_epipolar_grids method with default alt and no dem with OTB
    """
    # Retrieve information from configuration
    conf = images_and_grids_conf[input_parameters.INPUT_SECTION_TAG]
    dem = None
    default_alt = 500

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        conf,
        "SharelocGeometry",
        dem,
        default_alt=default_alt,
        epipolar_step=30,
        geoid=get_geoid_path(),
    )

    assert epi_size == [612, 612]

    # test baseline: 1/(disp to alt ratio), adapted from OTB.
    # see OTB StereoRectificationGridGenerator output for meaning
    # difference between shareloc and OTB : sum is not done exactly the same
    # but precision result to 10**-5 is enough for baseline
    # put exact values to know if modifications are done.
    # put decimal values to 10 to know if modifications are done.
    np.testing.assert_almost_equal(baseline, 1.420566289522033, decimal=10)

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path(
    # "ref_output/left_grid_default_alt.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid_default_alt.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path(
    # "ref_output/right_grid_default_alt.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid_default_alt.nc")
    )
    assert np.allclose(right_grid_ref["x"].values, right_grid[:, :, 0])
    assert np.allclose(right_grid_ref["y"].values, right_grid[:, :, 1])


@pytest.mark.unit_tests
def test_generate_epipolar_grids_otb():
    """
    Test generate_epipolar_grids method
    """
    conf = {
        input_parameters.IMG1_TAG: absolute_data_path(
            "input/phr_ventoux/left_image.tif"
        ),
        input_parameters.IMG2_TAG: absolute_data_path(
            "input/phr_ventoux/right_image.tif"
        ),
    }
    dem = absolute_data_path("input/phr_ventoux/srtm")

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        conf,
        "OTBGeometry",
        dem,
        default_alt=None,
        epipolar_step=30,
        geoid=get_geoid_path(),
    )

    assert epi_size == [612, 612]
    assert baseline == 1 / 0.7039416432380676

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path("ref_output/left_grid.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path("ref_output/right_grid.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid.nc")
    )
    assert np.allclose(right_grid_ref["x"].values, right_grid[:, :, 0])
    assert np.allclose(right_grid_ref["y"].values, right_grid[:, :, 1])


@pytest.mark.unit_tests
def test_generate_epipolar_grids_shareloc(images_and_grids_conf):
    """
    Test generate_epipolar_grids method
    """
    # Retrieve information from configuration
    conf = images_and_grids_conf[input_parameters.INPUT_SECTION_TAG]

    # use a file and not a directory ! (shareloc != OTB)
    dem = absolute_data_path("input/phr_ventoux/srtm/N44E005.hgt")

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        conf,
        "SharelocGeometry",
        dem,
        default_alt=None,
        epipolar_step=30,
        geoid=get_geoid_path(),
    )

    assert epi_size == [612, 612]

    # test baseline: 1/(disp to alt ratio) adapted from OTB.
    # see OTB StereoRectificationGridGenerator output for meaning
    # difference between shareloc and OTB : sum is not done exactly the same
    # but precision result to 10**-5 is enough for baseline (shareloc vs OTB)
    # put decimal values to 10 to know if modifications are done.
    np.testing.assert_almost_equal(baseline, 1.420571770533341, decimal=10)

    # Uncomment to update baseline
    # left_grid.to_netcdf(absolute_data_path("ref_output/left_grid.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # right_grid.to_netcdf(absolute_data_path("ref_output/right_grid.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid.nc")
    )
    assert np.allclose(right_grid_ref["x"].values, right_grid[:, :, 0])
    assert np.allclose(right_grid_ref["y"].values, right_grid[:, :, 1])


@pytest.mark.unit_tests
@pytest.mark.parametrize("save_reference", [False])
@pytest.mark.parametrize(
    "input_file,ref_file",
    [
        (
            "grid_generation_gizeh_ROI_no_color",
            "grid_generation_gizeh_ROI_ref_no_color",
        ),
        (
            "grid_generation_gizeh_ROI_color",
            "grid_generation_gizeh_ROI_ref_color",
        ),
    ],
)
def test_grid_generation(save_reference, input_file, ref_file):
    """
    Grid generation application test
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf = {}
        conf["out_dir"] = directory
        input_relative_path = os.path.join("input", "test_application")
        input_path = absolute_data_path(input_relative_path)
        # Triangulation
        epipolar_grid_generation_application = Application(
            "grid_generation", cfg=conf.get("grid_generation", {})
        )
        orchestrator_conf = {
            "mode": "sequential",
            "max_ram_per_worker": 40,
            "profiling": {"mode": "time", "activated": True},
        }

        with orchestrator.Orchestrator(
            orchestrator_conf=orchestrator_conf
        ) as cars_orchestrator:
            # initialize out_json
            cars_orchestrator.update_out_info({"version": __version__})
            # load dictionary of cardatasets
            with open(
                absolute_data_path(
                    os.path.join(input_relative_path, input_file)
                ),
                "rb",
            ) as file:
                # load pickle data
                data = pickle.load(file)
                adapt_path_for_test_dir(data, input_path, input_relative_path)
                package_path = os.path.dirname(__file__)
                # Run grid generation
                (
                    grid_left,
                    grid_right,
                ) = epipolar_grid_generation_application.run(
                    data["sensor_image_left"],
                    data["sensor_image_right"],
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(directory, "pair_0"),
                    srtm_dir=os.path.join(input_path, "srtm_dir"),
                    default_alt=0,
                    geoid_path=os.path.join(
                        package_path,
                        "..",
                        "..",
                        "..",
                        "cars",
                        "conf",
                        "geoid",
                        "egm96.grd",
                    ),
                )
                ref_data_path = absolute_data_path(
                    os.path.join(
                        "ref_output_application",
                        "grid_generation",
                        ref_file,
                    )
                )
                # serialize reference data if needed
                if save_reference:
                    serialize_grid_ref_data(
                        grid_left, grid_right, ref_data_path
                    )
                # load reference output data
                with open(ref_data_path, "rb") as file:
                    ref_data = pickle.load(file)
                    assert_same_carsdatasets(
                        cast_swigobj_grid(grid_left),
                        ref_data["grid_left"],
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )
                    assert_same_carsdatasets(
                        cast_swigobj_grid(grid_right),
                        ref_data["grid_right"],
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )


def adapt_path_for_test_dir(data, input_path, input_relative_path):
    """
    Adapt path of source for the test dir
    """
    for primary_key in data:
        for key in data[primary_key]:
            if isinstance(data[primary_key][key], str):
                if input_relative_path in data[primary_key][key]:
                    basename = os.path.basename(data[primary_key][key])
                    data[primary_key][key] = os.path.join(input_path, basename)


def serialize_grid_ref_data(grid_left, grid_right, ref_data_path):
    """
    Serialize reference data if needed with pickle
    """
    # cast C++ SwigObject to serializable(pickable) object
    cast_swigobj_grid(grid_left)
    cast_swigobj_grid(grid_right)
    data_dict = {"grid_left": grid_left, "grid_right": grid_right}
    with open(ref_data_path, "wb") as file:
        pickle.dump(data_dict, file)


def cast_swigobj_grid(grid):
    """
    cast swig object attribute of the grid carsdataset
    """
    grid.attributes["grid_spacing"] = list(grid.attributes["grid_spacing"])
    grid.attributes["grid_origin"] = list(grid.attributes["grid_origin"])
    return grid


def test_terrain_region_to_epipolar(
    images_and_grids_conf,  # pylint: disable=redefined-outer-name
    epipolar_sizes_conf,
    disparities_conf,  # pylint: disable=redefined-outer-name
):  # pylint: disable=redefined-outer-name
    """
    Test terrain_region_to_epipolar
    """
    configuration = images_and_grids_conf
    configuration["preprocessing"]["output"].update(
        disparities_conf["preprocessing"]["output"]
    )
    configuration["preprocessing"]["output"].update(
        epipolar_sizes_conf["preprocessing"]["output"]
    )

    # fill constants with final dsm footprint
    terrain_region = [675248, 4897075, 675460.5, 4897173]
    disp_min, disp_max = -20, 15
    epsg = 32631

    epipolar_sizes = epipolar_sizes_conf["preprocessing"]["output"]

    epipolar_region = grids.terrain_region_to_epipolar(
        terrain_region,
        configuration,
        get_geometry_loader(),
        epsg=epsg,
        disp_min=disp_min,
        disp_max=disp_max,
        tile_size=100,
        epipolar_size_x=epipolar_sizes["epipolar_size_x"],
        epipolar_size_y=epipolar_sizes["epipolar_size_y"],
    )

    epipolar_region_ref = [0, 600, 300, 612]

    assert epipolar_region == epipolar_region_ref
