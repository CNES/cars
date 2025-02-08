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
    get_geometry_plugin,
    temporary_dir,
)


def generate_grid_xr_dataset(grid_np):
    """
    Transform numpy grid to dataset grid
    """
    rows = np.arange(0, grid_np.shape[0])
    cols = np.arange(0, grid_np.shape[1])
    dataset_grid = xr.Dataset(
        {
            "x": xr.DataArray(
                data=grid_np[:, :, 0],
                dims=["row", "col"],
                coords={"row": rows, "col": cols},
            ),
            "y": xr.DataArray(
                data=grid_np[:, :, 1],
                dims=["row", "col"],
                coords={"row": rows, "col": cols},
            ),
        }
    )

    return dataset_grid


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
        #  corrected_grid)
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
def test_generate_epipolar_grids_default_alt_shareloc(images_and_grids_conf):
    """
    Test generate_epipolar_grids method with default alt and no dem with
        Shareloc
    """
    # Retrieve information from configuration
    conf = images_and_grids_conf[input_parameters.INPUT_SECTION_TAG]
    default_alt = 500
    sensor1 = conf["img1"]
    sensor2 = conf["img2"]
    geomodel1 = {"path": conf["model1"], "model_type": conf["model_type1"]}
    geomodel2 = {"path": conf["model2"], "model_type": conf["model_type2"]}

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        get_geometry_plugin(default_alt=default_alt),
        epipolar_step=30,
    )

    assert epi_size == [612, 612]

    # test baseline: 1/(disp to alt ratio), adapted from Shareloc.
    # but precision result to 10**-5 is enough for baseline
    # put exact values to know if modifications are done.
    # put decimal values to 10 to know if modifications are done.
    np.testing.assert_almost_equal(baseline, 1.4205663917758247, decimal=10)

    # Uncomment to update baseline
    # generate_grid_xr_dataset(left_grid).to_netcdf(absolute_data_path(
    #  "ref_output/left_grid_default_alt.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid_default_alt.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # generate_grid_xr_dataset(right_grid).to_netcdf(absolute_data_path(
    # "ref_output/right_grid_default_alt.nc"))

    right_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/right_grid_default_alt.nc")
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
    sensor1 = conf["img1"]
    sensor2 = conf["img2"]
    geomodel1 = {"path": conf["model1"], "model_type": conf["model_type1"]}
    geomodel2 = {"path": conf["model2"], "model_type": conf["model_type2"]}

    # use a file and not a directory !
    dem = absolute_data_path("input/phr_ventoux/srtm/N44E005.hgt")

    (
        left_grid,
        right_grid,
        _,
        _,
        epi_size,
        baseline,
    ) = grids.generate_epipolar_grids(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        get_geometry_plugin(dem=dem),
        epipolar_step=30,
    )

    assert epi_size == [612, 612]

    # test baseline: 1/(disp to alt ratio) adapted from Shareloc.
    # but precision result to 10**-5 is enough for baseline
    # put decimal values to 10 to know if modifications are done.
    np.testing.assert_almost_equal(baseline, 1.4205731358019482, decimal=10)

    # Uncomment to update baseline
    # generate_grid_xr_dataset(left_grid).to_netcdf(
    # absolute_data_path("ref_output/left_grid.nc"))

    left_grid_ref = xr.open_dataset(
        absolute_data_path("ref_output/left_grid.nc")
    )
    assert np.allclose(left_grid_ref["x"].values, left_grid[:, :, 0])
    assert np.allclose(left_grid_ref["y"].values, left_grid[:, :, 1])

    # Uncomment to update baseline
    # generate_grid_xr_dataset(right_grid).to_netcdf(
    # absolute_data_path("ref_output/right_grid.nc"))

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
            "grid_generation_gizeh_ROI_no_color.pickle",
            "grid_generation_gizeh_ROI_ref_no_color",
        ),
        (
            "grid_generation_gizeh_ROI_color.pickle",
            "grid_generation_gizeh_ROI_ref_color",
        ),
    ],
)
def test_grid_generation(save_reference, input_file, ref_file):
    """
    Grid generation application test
    works with  Shareloc
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
            "profiling": {"mode": "cars_profiling"},
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
                # Run grid generation
                geometry_plugin = get_geometry_plugin(
                    dem=os.path.join(
                        input_path, "srtm_dir", "N29E031_KHEOPS.tif"
                    ),
                    default_alt=0,
                )
                (
                    grid_left,
                    grid_right,
                ) = epipolar_grid_generation_application.run(
                    data["sensor_image_left"],
                    data["sensor_image_right"],
                    geometry_plugin,
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(directory, "pair_0"),
                )

                ref_file = ref_file + "_shareloc"

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
        for key2 in data[primary_key]:
            if isinstance(data[primary_key][key2], str):
                if input_relative_path in data[primary_key][key2]:
                    basename = os.path.basename(data[primary_key][key2])
                    data[primary_key][key2] = os.path.join(input_path, basename)
            # adapt for third level for geomodel path (quick dirty fix)
            if (
                key2 == "geomodel"
                and input_relative_path in data[primary_key][key2]["path"]
            ):
                basename = os.path.basename(
                    data[primary_key]["geomodel"]["path"]
                )
                data[primary_key]["geomodel"]["path"] = os.path.join(
                    input_path, basename
                )


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

    sensor1 = configuration["input"]["img1"]
    sensor2 = configuration["input"]["img2"]
    geomodel1 = {"path": configuration["input"]["model1"]}
    geomodel2 = {"path": configuration["input"]["model2"]}
    grid_left = configuration["preprocessing"]["output"]["left_epipolar_grid"]
    grid_right = configuration["preprocessing"]["output"]["right_epipolar_grid"]

    epipolar_region = grids.terrain_region_to_epipolar(
        terrain_region,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        get_geometry_plugin(),
        epsg=epsg,
        disp_min=disp_min,
        disp_max=disp_max,
        tile_size=100,
        epipolar_size_x=epipolar_sizes["epipolar_size_x"],
        epipolar_size_y=epipolar_sizes["epipolar_size_y"],
    )

    epipolar_region_ref = [0, 600, 300, 612]

    assert epipolar_region == epipolar_region_ref
