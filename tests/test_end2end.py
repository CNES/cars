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
import tempfile
import os
import json
import math
from shutil import copy2
import pyproj

import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform

from utils import temporary_dir, absolute_data_path, assert_same_images
from cars.parameters import read_input_parameters
from cars.parameters import read_preprocessing_content_file
from cars import prepare
from cars import compute_dsm
from cars import configuration_correlator as corr_cfg


@pytest.mark.unit_tests
def test_end2end_ventoux_unique():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        with open(preproc_json, 'r') as f:
            preproc_data = json.load(f)
            assert preproc_data["preprocessing"]\
                ["output"]["epipolar_size_x"] == 612
            assert preproc_data["preprocessing"]\
                ["output"]["epipolar_size_y"] == 612
            assert - \
                20 < preproc_data["preprocessing"]\
                    ["output"]["minimum_disparity"] < -18
            assert 14 < preproc_data["preprocessing"]\
                ["output"]["maximum_disparity"] < 15
            for img in [
                "matches",
                "right_epipolar_grid",
                "left_epipolar_grid"]:
                assert os.path.isfile(
                    os.path.join(
                        out_preproc,
                        preproc_data["preprocessing"]["output"][img]))

        out_stereo = os.path.join(directory, "out_preproc")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster,
            output_stats=True,
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        # Uncomment the 2 following instructions to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                                "ref_output/clr_end2end_ventoux.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = read_input_parameters(absolute_data_path(
            "input/phr_ventoux/preproc_input_without_color.json"))

        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00")

        preproc_json = os.path.join(out_preproc, "content.json")
        out_stereo = os.path.join(directory, "out_preproc")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_ventoux.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))

    # Test we have the same results with multiprocessing
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        out_stereo = os.path.join(directory, "out_preproc")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="mp",  # Multiprocessing mode
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_ventoux.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False


@pytest.mark.unit_tests
def test_prepare_ventoux_bias():
    """
    Dask prepare with bias geoms
    """
    input_json = read_input_parameters(absolute_data_path(
        "input/phr_ventoux/preproc_input_bias.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc_bias")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            epipolar_error_maximum_bias=50.,
            elevation_delta_lower_bound=-120.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00")

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        with open(preproc_json, 'r') as f:
            preproc_data = json.load(f)
            preproc_output = preproc_data["preprocessing"]["output"]
            assert preproc_output["epipolar_size_x"] == 612
            assert preproc_output["epipolar_size_y"] == 612
            assert preproc_output["minimum_disparity"] > -86
            assert preproc_output["minimum_disparity"] < -84
            assert preproc_output["maximum_disparity"] > -46
            assert preproc_output["maximum_disparity"] < -44
            for img in [
                "matches",
                "right_epipolar_grid",
                "left_epipolar_grid"]:
                assert os.path.isfile(
                    os.path.join(
                        out_preproc,
                        preproc_data["preprocessing"]["output"][img]))


@pytest.mark.unit_tests
def test_end2end_ventoux_with_color():
    """
    End to end processing with p+xs fusion
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(absolute_data_path(
        "input/phr_ventoux/preproc_input_with_pxs_fusion.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        with open(preproc_json, 'r') as f:
            preproc_data = json.load(f)
            preproc_output = preproc_data["preprocessing"]["output"]
            assert preproc_output["epipolar_size_x"] == 612
            assert preproc_output["epipolar_size_y"] == 612
            assert preproc_output["minimum_disparity"] > -20
            assert preproc_output["minimum_disparity"] < -18
            assert preproc_output["maximum_disparity"] > 14
            assert preproc_output["maximum_disparity"] < 15
            for img in [
                "matches",
                "right_epipolar_grid",
                "left_epipolar_grid"]:
                assert os.path.isfile(
                    os.path.join(
                        out_preproc,
                        preproc_data["preprocessing"]["output"][img]))

        out_stereo = os.path.join(directory, "out_preproc")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        # Uncomment the following instruction to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux_4bands.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_ventoux_4bands.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False


@pytest.mark.unit_tests
def test_compute_dsm_with_roi_ventoux():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        preproc_json = os.path.join(out_preproc, "content.json")
        out_stereo = os.path.join(directory, "out_preproc")
        final_epsg = 32631
        resolution = 0.5

        roi = [5.194, 44.2059, 5.195, 44.2064]
        roi_epsg = 4326

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=resolution,
            epsg=final_epsg,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            roi=(roi, roi_epsg))

        # Uncomment the 2 following instructions to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_end2end_ventoux_with_roi.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path(
        #      "ref_output/clr_end2end_ventoux_with_roi.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux_with_roi.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_ventoux_with_roi.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False

        # check final bounding box
        # create reference
        [roi_xmin, roi_ymin, roi_xmax, roi_ymax] = roi
        roi_poly = Polygon([(roi_xmin, roi_ymin),
                            (roi_xmax, roi_ymin),
                            (roi_xmax, roi_ymax),
                            (roi_xmin, roi_ymax),
                            (roi_xmin, roi_ymin)])

        project = pyproj.Transformer.from_proj(
            pyproj.Proj(
                init='epsg:{}'.format(roi_epsg)), pyproj.Proj(
                init='epsg:{}'.format(final_epsg)))
        ref_roi_poly = transform(project.transform, roi_poly)

        [ref_xmin, ref_ymin, ref_xmax, ref_ymax] = ref_roi_poly.bounds

        # retrieve bounding box of computed dsm
        data = rasterio.open(os.path.join(out_stereo, "dsm.tif"))
        xmin = min(data.bounds.left, data.bounds.right)
        ymin = min(data.bounds.bottom, data.bounds.top)
        xmax = max(data.bounds.left, data.bounds.right)
        ymax = max(data.bounds.bottom, data.bounds.top)

        assert math.floor(ref_xmin / resolution) * resolution == xmin
        assert math.ceil(ref_xmax / resolution) * resolution == xmax
        assert math.floor(ref_ymin / resolution) * resolution == ymin
        assert math.ceil(ref_ymax / resolution) * resolution == ymax


@pytest.mark.unit_tests
def test_compute_dsm_with_snap_to_img1():
    """
    Dask compute dsm processing with input roi (cars_stereo)
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))


    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00")

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        out_stereo = os.path.join(directory, "out_preproc")
        final_epsg = 32631
        resolution = 0.5
        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=resolution,
            epsg=final_epsg,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            snap_to_img1 = True)

        # Uncomment the 2 following instructions to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path(
        #      "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                    "ref_output/dsm_end2end_ventoux_with_snap_to_img1.tif"),
                               atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                    "ref_output/clr_end2end_ventoux_with_snap_to_img1.tif"),
                    rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False



@pytest.mark.unit_tests
def test_end2end_quality_stats():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        with open(preproc_json, 'r') as f:
            preproc_data = json.load(f)
            preproc_output = preproc_data["preprocessing"]["output"]
            assert preproc_output["epipolar_size_x"] == 612
            assert preproc_output["epipolar_size_y"] == 612
            assert preproc_output["minimum_disparity"] > -20
            assert preproc_output["minimum_disparity"] < -18
            assert preproc_output["maximum_disparity"] > 14
            assert preproc_output["maximum_disparity"] < 15
            for img in [
                "matches",
                "right_epipolar_grid",
                "left_epipolar_grid"]:
                assert os.path.isfile(
                    os.path.join(
                        out_preproc,
                        preproc_data["preprocessing"]["output"][img]))

        out_stereo = os.path.join(directory, "out_preproc")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster,
            output_stats=True,
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        # Uncomment the 2 following instructions to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'dsm_mean.tif'),
        #      absolute_data_path("ref_output/dsm_mean_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'dsm_std.tif'),
        #      absolute_data_path("ref_output/dsm_std_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'dsm_n_pts.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_n_pts_end2end_ventoux.tif"))
        #copy2(os.path.join(out_stereo, 'dsm_pts_in_cell.tif'),
        #      absolute_data_path(
        #      "ref_output/dsm_pts_in_cell_end2end_ventoux.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                            "ref_output/dsm_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                            "ref_output/clr_end2end_ventoux.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert_same_images(os.path.join(out_stereo, "dsm_mean.tif"),
                           absolute_data_path(
                            "ref_output/dsm_mean_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "dsm_std.tif"),
                           absolute_data_path(
                            "ref_output/dsm_std_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "dsm_n_pts.tif"),
                           absolute_data_path(
                            "ref_output/dsm_n_pts_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "dsm_pts_in_cell.tif"),
                           absolute_data_path(
                            "ref_output/dsm_pts_in_cell_end2end_ventoux.tif"),
                           atol=0.0001, rtol=1e-6)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False


@pytest.mark.unit_tests
def test_end2end_ventoux_egm96_geoid():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_ventoux/preproc_input.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")
        assert os.path.isfile(preproc_json)

        with open(preproc_json, 'r') as f:
            preproc_data = json.load(f)
            preproc_output = preproc_data["preprocessing"]["output"]
            assert preproc_output["epipolar_size_x"] == 612
            assert preproc_output["epipolar_size_y"] == 612
            assert preproc_output["minimum_disparity"] > -20
            assert preproc_output["minimum_disparity"] < -18
            assert preproc_output["maximum_disparity"] > 14
            assert preproc_output["maximum_disparity"] < 15
            for img in [
                "matches",
                "right_epipolar_grid",
                "left_epipolar_grid"]:
                assert os.path.isfile(
                    os.path.join(
                        out_preproc,
                        preproc_data["preprocessing"]["output"][img]))

        out_stereo = os.path.join(directory, "out_preproc")
        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster,
            nb_workers=4,
            walltime="00:10:00",
            use_geoid_alt=True,
            use_sec_disp=True
        )

        # Uncomment the 2 following instructions to update reference data
        #copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_ventoux_egm96.tif"))
        #copy2(os.path.join(out_stereo, 'clr.tif'),
        #      absolute_data_path("ref_output/clr_end2end_ventoux.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux_egm96.tif"),
                               atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_ventoux.tif"),
                               rtol=1.e-7, atol=1.e-7)
    assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False

    # Test that we have the same results without setting the color1
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        input_json = read_input_parameters(absolute_data_path(
            "input/phr_ventoux/preproc_input_without_color.json"))

        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00")

        preproc_json = os.path.join(out_preproc, "content.json")
        out_stereo = os.path.join(directory, "out_preproc")
        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            use_geoid_alt=True,
            use_sec_disp=True
        )

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_ventoux_egm96.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
            "ref_output/clr_end2end_ventoux.tif"), rtol=1.e-7, atol=1.e-7)
        assert os.path.exists(os.path.join(out_stereo, "msk.tif")) is False


@pytest.mark.unit_tests
def test_end2end_paca_with_mask():
    """
    End to end processing
    """
    # Force max RAM to 1000 to get stable tiling in tests
    os.environ['OTB_MAX_RAM_HINT'] = '1000'

    input_json = read_input_parameters(
        absolute_data_path("input/phr_paca/preproc_input.json"))

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        out_preproc = os.path.join(directory, "out_preproc")
        prepare.run(
            input_json,
            out_preproc,
            epi_step=30,
            region_size=250,
            disparity_margin=0.25,
            epipolar_error_upper_bound=43.,
            elevation_delta_lower_bound=-20.,
            elevation_delta_upper_bound=20.,
            mode="local_dask",  # Run on a local cluster
            nb_workers=4,
            walltime="00:10:00",
            check_inputs=True)

        # Check preproc properties
        preproc_json = os.path.join(out_preproc, "content.json")

        out_stereo = os.path.join(directory, "out_stereo")

        corr_config = corr_cfg.configure_correlator()

        compute_dsm.run(
            [read_preprocessing_content_file(preproc_json)],
            out_stereo,
            resolution=0.5,
            epsg=32631,
            sigma=0.3,
            dsm_radius=3,
            dsm_no_data=-999,
            color_no_data=0,
            msk_no_data=65534,
            corr_config=corr_config,
            mode="local_dask",  # Run on a local cluster,
            output_stats=True,
            nb_workers=4,
            walltime="00:10:00",
            use_sec_disp=True)

        # Uncomment the 2 following instructions to update reference data
        # copy2(os.path.join(out_stereo, 'dsm.tif'),
        #      absolute_data_path("ref_output/dsm_end2end_paca.tif"))
        # copy2(os.path.join(out_stereo, 'clr.tif'),
        #       absolute_data_path("ref_output/clr_end2end_paca.tif"))
        # copy2(os.path.join(out_stereo, 'msk.tif'),
        #      absolute_data_path("ref_output/msk_end2end_paca.tif"))

        assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                           absolute_data_path(
                               "ref_output/dsm_end2end_paca.tif"),
                           atol=0.0001, rtol=1e-6)
        assert_same_images(os.path.join(out_stereo, "clr.tif"),
                           absolute_data_path(
                               "ref_output/clr_end2end_paca.tif"),
                           rtol=1.e-7, atol=1.e-7)
        assert_same_images(os.path.join(out_stereo, "msk.tif"),
                           absolute_data_path(
                               "ref_output/msk_end2end_paca.tif"),
                           rtol=1.e-7, atol=1.e-7)

        # Test we have the same results with multiprocessing
        with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
            out_preproc = os.path.join(directory, "out_preproc")
            prepare.run(
                input_json,
                out_preproc,
                epi_step=30,
                region_size=250,
                disparity_margin=0.25,
                epipolar_error_upper_bound=43.,
                elevation_delta_lower_bound=-20.,
                elevation_delta_upper_bound=20.,
                mode="local_dask",  # Run on a local cluster
                nb_workers=4,
                walltime="00:10:00",
                check_inputs=True)

            # Check preproc properties
            preproc_json = os.path.join(out_preproc, "content.json")

            out_stereo = os.path.join(directory, "out_stereo")

            corr_config = corr_cfg.configure_correlator()

            compute_dsm.run(
                [read_preprocessing_content_file(preproc_json)],
                out_stereo,
                resolution=0.5,
                epsg=32631,
                sigma=0.3,
                dsm_radius=3,
                dsm_no_data=-999,
                color_no_data=0,
                msk_no_data=65534,
                corr_config=corr_config,
                mode="mp",
                output_stats=True,
                nb_workers=4,
                walltime="00:10:00",
                use_sec_disp=True)

            # Uncomment the 2 following instructions to update reference data
            # copy2(os.path.join(out_stereo, 'dsm.tif'),
            #      absolute_data_path("ref_output/dsm_end2end_paca.tif"))
            # copy2(os.path.join(out_stereo, 'clr.tif'),
            #       absolute_data_path("ref_output/clr_end2end_paca.tif"))
            # copy2(os.path.join(out_stereo, 'msk.tif'),
            #      absolute_data_path("ref_output/msk_end2end_paca.tif"))

            assert_same_images(os.path.join(out_stereo, "dsm.tif"),
                               absolute_data_path(
                                   "ref_output/dsm_end2end_paca.tif"),
                               atol=0.0001, rtol=1e-6)
            assert_same_images(os.path.join(out_stereo, "clr.tif"),
                               absolute_data_path(
                                   "ref_output/clr_end2end_paca.tif"),
                               rtol=1.e-7, atol=1.e-7)
            assert_same_images(os.path.join(out_stereo, "msk.tif"),
                               absolute_data_path(
                                   "ref_output/msk_end2end_paca.tif"),
                               rtol=1.e-7, atol=1.e-7)
