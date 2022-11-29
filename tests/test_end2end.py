#!/usr/bin/env python # pylint: disable=too-many-lines
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
Test module End to End:
Prepare and Compute DSM run user tests through pipelines run() functions
TODO: Cars_cli is not tested
TODO: Refactor in several files and remove too-many-lines
"""

# Standard imports
from __future__ import absolute_import

import json
import math
import os
import shutil
import tempfile
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

# Third party imports
import pyproj
import pytest
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform

# CARS imports
from cars.conf.input_parameters import read_input_parameters
from cars.pipelines.sensor_to_full_resolution_dsm import (
    sensor_to_full_resolution_dsm_pipeline as pipeline_full_res,
)
from cars.pipelines.sensor_to_low_resolution_dsm import (
    sensor_to_low_resolution_dsm_pipeline as pipeline_low_res,
)

# CARS Tests imports
from .helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    temporary_dir,
)


@pytest.mark.end2end_tests
def test_end2end_ventoux_unique():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run low resolution pipeline
        _, input_config_low_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_low_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_low_res["applications"].update(application_config)

        low_res_pipeline = pipeline_low_res.SensorToLowResolutionDsmPipeline(
            input_config_low_res
        )
        low_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as json_file:
            out_json = json.load(json_file)
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_x"
                ]
                == 612
            )
            assert (
                out_json["applications"]["left_right"]["grid_generation_run"][
                    "epipolar_size_y"
                ]
                == 612
            )
            assert (
                -20
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["minimum_disparity"]
                < -18
            )
            assert (
                14
                < out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["maximum_disparity"]
                < 15
            )

            # check matches file exists
            assert os.path.isfile(
                out_json["applications"]["left_right"][
                    "disparity_range_computation_run"
                ]["matches"]
            )

        # Check used_conf for low res

        gt_used_conf_orchestrator = {
            "orchestrator": {
                "mode": "local_dask",
                "walltime": "00:10:00",
                "nb_workers": 4,
                "profiling": {
                    "activated": False,
                    "mode": "time",
                    "loop_testing": False,
                },
                "use_memory_logger": True,
                "config_name": "unknown",
            }
        }

        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensor_to_low_resolution_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["sparse_matching"]["disparity_margin"]
                == 0.25
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            # check used_conf reentry
            _ = pipeline_low_res.SensorToLowResolutionDsmPipeline(used_conf)
        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # full resolution pipeline
        input_config_full_res = input_config_low_res.copy()
        # update applications
        full_res_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)
        # update epsg
        input_config_full_res["inputs"]["epsg"] = 32631

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # Check used_conf for full res
        used_conf_path = os.path.join(out_dir, "used_conf.json")

        # check used_conf file exists
        assert os.path.isfile(used_conf_path)

        with open(used_conf_path, "r", encoding="utf-8") as json_file:
            used_conf = json.load(json_file)
            # check used_conf inputs conf exists
            assert "inputs" in used_conf
            assert "sensors" in used_conf["inputs"]
            # check used_conf pipeline
            assert used_conf["pipeline"] == "sensor_to_full_resolution_dsm"
            # check used_conf sparse_matching configuration
            assert (
                used_conf["applications"]["point_cloud_rasterization"]["sigma"]
                == 0.3
            )
            # check used_conf orchestrator conf is the same as gt
            assert (
                used_conf["orchestrator"]
                == gt_used_conf_orchestrator["orchestrator"]
            )
            # check used_conf reentry
            _ = pipeline_full_res.SensorToFullResolutionDsmPipeline(used_conf)
        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux.tif"))
        # copy2(
        #     os.path.join(out_dir, "ambiguity.tif"),
        #     absolute_data_path("ref_output/ambiguity_end2end_ventoux.tif"),
        # )
        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "ambiguity.tif"),
            absolute_data_path("ref_output/ambiguity_end2end_ventoux.tif"),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    input_json = absolute_data_path(
        "input/phr_ventoux/input_without_color.json"
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        # Run low resolution pipeline
        _, input_config_low_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_low_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_low_res["applications"].update(application_config)

        low_res_pipeline = pipeline_low_res.SensorToLowResolutionDsmPipeline(
            input_config_low_res
        )
        low_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # full resolution pipeline
        input_config_full_res = input_config_low_res.copy()
        # update applications
        full_res_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)
        # update epsg
        input_config_full_res["inputs"]["epsg"] = 32631

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test we have the same results with multiprocessing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")
        # Run low resolution pipeline
        _, input_config_low_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_low_resolution_dsm",
            "mp",
            orchestrator_parameters={"nb_workers": 4},
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_low_res["applications"].update(application_config)

        low_res_pipeline = pipeline_low_res.SensorToLowResolutionDsmPipeline(
            input_config_low_res
        )
        low_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # full resolution pipeline
        input_config_full_res = input_config_low_res.copy()
        # update applications
        full_res_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)
        # update epsg
        input_config_full_res["inputs"]["epsg"] = 32631

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_prepare_ventoux_bias():
    """
    Dask prepare with bias geoms
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        input_json = absolute_data_path("input/phr_ventoux/input_bias.json")
        # Run low resolution pipeline
        _, input_config_low_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_low_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "epipolar_error_maximum_bias": 50.0,
                "elevation_delta_lower_bound": -120.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_low_res["applications"].update(application_config)

        low_res_pipeline = pipeline_low_res.SensorToLowResolutionDsmPipeline(
            input_config_low_res
        )
        low_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # Check preproc properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -86
            assert out_disp_compute["minimum_disparity"] < -84
            assert out_disp_compute["maximum_disparity"] > -46
            assert out_disp_compute["maximum_disparity"] < -44

            # check matches file exists
            assert os.path.isfile(out_disp_compute["matches"])


@pytest.mark.end2end_tests
def test_end2end_ventoux_with_color():
    """
    End to end processing with p+xs fusion
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    input_json = read_input_parameters(
        absolute_data_path(
            "input/phr_ventoux/preproc_input_with_pxs_fusion.json"
        )
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_pxs_fusion.json"
        )
        # Run low resolution pipeline
        _, input_config_low_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_low_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        application_config = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
        }

        input_config_low_res["applications"].update(application_config)

        low_res_pipeline = pipeline_low_res.SensorToLowResolutionDsmPipeline(
            input_config_low_res
        )
        low_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -20
            assert out_disp_compute["minimum_disparity"] < -18
            assert out_disp_compute["maximum_disparity"] > 14
            assert out_disp_compute["maximum_disparity"] < 15

            assert os.path.isfile(out_disp_compute["matches"])

        # Run full res dsm pipeline
        # clean outdir
        shutil.rmtree(out_dir, ignore_errors=False, onerror=None)

        # full resolution pipeline
        input_config_full_res = input_config_low_res.copy()
        # update applications
        full_res_applications = {
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": 0.5,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)
        # update epsg
        input_config_full_res["inputs"]["epsg"] = 32631

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_low_res["output"]["out_dir"]

        # Uncomment the following instruction to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path("ref_output/clr_end2end_ventoux_4bands.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_4bands.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_compute_dsm_with_roi_ventoux():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        input_json = absolute_data_path(
            "input/phr_ventoux/input_with_pxs_fusion.json"
        )
        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        # Update roi
        roi = [5.194, 44.2059, 5.195, 44.2064]
        roi_epsg = 4326
        input_config_full_res["inputs"]["roi"] = (roi, roi_epsg)

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_end2end_ventoux_with_roi.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #      absolute_data_path(
        #      "ref_output/clr_end2end_ventoux_with_roi.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_with_roi.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux_with_roi.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

        # check final bounding box
        # create reference
        [roi_xmin, roi_ymin, roi_xmax, roi_ymax] = roi
        roi_poly = Polygon(
            [
                (roi_xmin, roi_ymin),
                (roi_xmax, roi_ymin),
                (roi_xmax, roi_ymax),
                (roi_xmin, roi_ymax),
                (roi_xmin, roi_ymin),
            ]
        )

        project = pyproj.Transformer.from_proj(
            pyproj.Proj(init="epsg:{}".format(roi_epsg)),
            pyproj.Proj(init="epsg:{}".format(final_epsg)),
        )
        ref_roi_poly = transform(project.transform, roi_poly)

        [ref_xmin, ref_ymin, ref_xmax, ref_ymax] = ref_roi_poly.bounds

        # retrieve bounding box of computed dsm
        data = rasterio.open(os.path.join(out_dir, "dsm.tif"))
        xmin = min(data.bounds.left, data.bounds.right)
        ymin = min(data.bounds.bottom, data.bounds.top)
        xmax = max(data.bounds.left, data.bounds.right)
        ymax = max(data.bounds.bottom, data.bounds.top)

        assert math.floor(ref_xmin / resolution) * resolution == xmin
        assert math.ceil(ref_xmax / resolution) * resolution == xmax
        assert math.floor(ref_ymin / resolution) * resolution == ymin
        assert math.ceil(ref_ymax / resolution) * resolution == ymax


@pytest.mark.end2end_tests
def test_compute_dsm_with_snap_to_img1():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "snap_to_img1": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #     absolute_data_path(
        #    "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #     absolute_data_path(
        #     "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path(
                "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path(
                "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"
            ),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_quality_stats():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "write_stats": True,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -20
            assert out_disp_compute["minimum_disparity"] < -18
            assert out_disp_compute["maximum_disparity"] > 14
            assert out_disp_compute["maximum_disparity"] < 15

            assert os.path.isfile(out_disp_compute["matches"])

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'dsm_mean.tif'),
        #      absolute_data_path("ref_output/dsm_mean_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'dsm_std.tif'),
        #      absolute_data_path("ref_output/dsm_std_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'dsm_n_pts.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_n_pts_end2end_ventoux.tif"))
        # copy2(os.path.join(out_dir, 'dsm_pts_in_cell.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_pts_in_cell_end2end_ventoux.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_mean.tif"),
            absolute_data_path("ref_output/dsm_mean_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_std.tif"),
            absolute_data_path("ref_output/dsm_std_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_n_pts.tif"),
            absolute_data_path("ref_output/dsm_n_pts_end2end_ventoux.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "dsm_pts_in_cell.tif"),
            absolute_data_path(
                "ref_output/dsm_pts_in_cell_end2end_ventoux.tif"
            ),
            atol=0.0001,
            rtol=1e-6,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_ventoux_egm96_geoid():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_ventoux/input.json")

        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "use_geoid_alt": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "write_stats": True,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Check content.json properties
        out_json = os.path.join(out_dir, "content.json")
        assert os.path.isfile(out_json)

        with open(out_json, "r", encoding="utf-8") as out_json_file:
            out_data = json.load(out_json_file)
            out_grid = out_data["applications"]["left_right"][
                "grid_generation_run"
            ]
            assert out_grid["epipolar_size_x"] == 612
            assert out_grid["epipolar_size_y"] == 612
            out_disp_compute = out_data["applications"]["left_right"][
                "disparity_range_computation_run"
            ]
            assert out_disp_compute["minimum_disparity"] > -20
            assert out_disp_compute["minimum_disparity"] < -18
            assert out_disp_compute["maximum_disparity"] > 14
            assert out_disp_compute["maximum_disparity"] < 15

            assert os.path.isfile(out_disp_compute["matches"])

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
    assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path(
            "input/phr_ventoux/input_without_color.json"
        )
        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "triangulation": {
                "method": "line_of_sight_intersection",
                "use_geoid_alt": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "write_stats": True,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_ventoux.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert os.path.exists(os.path.join(out_dir, "msk.tif")) is False


@pytest.mark.end2end_tests
def test_end2end_paca_with_mask():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ["OTB_MAX_RAM_HINT"] = "1000"

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_paca/input.json")

        # Run full resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "local_dask",
            orchestrator_parameters={"walltime": "00:10:00", "nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "write_msk": True,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_paca.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_paca.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )

    # Test we have the same results with multiprocessing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = absolute_data_path("input/phr_paca/input.json")

        # Run low resolution pipeline
        _, input_config_full_res = generate_input_json(
            input_json,
            directory,
            "sensor_to_full_resolution_dsm",
            "mp",
            orchestrator_parameters={"nb_workers": 4},
        )
        resolution = 0.5
        full_res_applications = {
            "grid_generation": {"method": "epipolar", "epi_step": 30},
            "resampling": {"method": "bicubic", "epi_tile_size": 250},
            "sparse_matching": {
                "method": "sift",
                "epipolar_error_upper_bound": 43.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "disparity_margin": 0.25,
                "save_matches": True,
            },
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "resolution": resolution,
                "sigma": 0.3,
                "dsm_no_data": -999,
                "color_no_data": 0,
                "msk_no_data": 65534,
                "write_msk": True,
            },
            "dense_matching": {"method": "census_sgm", "use_sec_disp": True},
        }
        input_config_full_res["applications"].update(full_res_applications)

        # update epsg
        final_epsg = 32631
        input_config_full_res["inputs"]["epsg"] = final_epsg

        full_res_pipeline = pipeline_full_res.SensorToFullResolutionDsmPipeline(
            input_config_full_res
        )
        full_res_pipeline.run()

        out_dir = input_config_full_res["output"]["out_dir"]

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_dir, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_paca.tif"))
        # copy2(os.path.join(out_dir, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_paca.tif"))

        assert_same_images(
            os.path.join(out_dir, "dsm.tif"),
            absolute_data_path("ref_output/dsm_end2end_paca.tif"),
            atol=0.0001,
            rtol=1e-6,
        )
        assert_same_images(
            os.path.join(out_dir, "clr.tif"),
            absolute_data_path("ref_output/clr_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        assert_same_images(
            os.path.join(out_dir, "msk.tif"),
            absolute_data_path("ref_output/msk_end2end_paca.tif"),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
