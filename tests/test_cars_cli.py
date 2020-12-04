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

import tempfile
import argparse
from copy import copy
import pytest


from utils import temporary_dir, absolute_data_path
from bin.cars_cli import main_cli, cars_cli_parser


@pytest.fixture(scope="module")
def prepare_default_args():
    args = argparse.Namespace()
    args.loglevel = 'INFO'
    args.command = 'prepare'
    args.disparity_margin = 0.25
    args.elevation_delta_lower_bound = -50.0
    args.elevation_delta_upper_bound = 50.0
    args.epi_step = 30
    args.epipolar_error_upper_bound = 10.0
    args.epipolar_error_maximum_bias = 0.0
    args.injson = absolute_data_path('input/phr_ventoux/preproc_input.json')
    args.mode = 'local_dask'
    args.nb_workers = 4
    args.walltime = '00:59:00'

    return args


@pytest.fixture(scope="module")
def compute_dsm_default_args():
    args = argparse.Namespace()
    args.loglevel = 'INFO'
    args.command = 'compute_dsm'
    args.sigma = None
    args.resolution = 0.5
    args.color_no_data = 0
    args.corr_config = None
    args.dsm_no_data = -32768
    args.dsm_radius = 1
    args.min_elevation_offset = None
    args.max_elevation_offset = None
    args.roi_bbox = None
    args.roi_file = None
    args.epsg = None
    args.injsons = [absolute_data_path(
        'input/cars_cli_input/content.json')]
    args.mode = 'local_dask'
    args.nb_workers = 4
    args.walltime = '00:59:00'

    return args


# ----------------------------------
# GENERAL
# ----------------------------------


@pytest.mark.unit_tests
def test_command():
    parser = cars_cli_parser()

    args = argparse.Namespace()
    args.loglevel = 'INFO'
    args.command = 'test'

    with pytest.raises(SystemExit) as e:
        main_cli(args, parser, check_inputs=True)
    assert e.type == SystemExit
    assert e.value.code == 1


# ----------------------------------
# DASK PREPARE
# ----------------------------------


@pytest.mark.unit_tests
def test_prepare_args(prepare_default_args):
    parser = cars_cli_parser()

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        prepare_default_args.outdir = directory

        # test default args
        main_cli(prepare_default_args, parser, check_inputs=True)

        prepare_default_args.loglevel = 'INFO'
        main_cli(prepare_default_args, parser, check_inputs=True)

        # degraded cases injson
        args_bad_json = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_json.injson = absolute_data_path(
                'input/cars_cli_input/test.json')
            main_cli(args_bad_json, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases disparity_margin
        args_bad_disp_margin = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_disp_margin.disparity_margin = -1.0
            main_cli(args_bad_disp_margin, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        with pytest.raises(SystemExit) as e:
            args_bad_disp_margin.disparity_margin = 1.5
            main_cli(args_bad_disp_margin, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases elevation bounds
        args_bad_elev_delta_bounds = copy(prepare_default_args)
        args_bad_elev_delta_bounds.elevation_delta_lower_bound = 50
        args_bad_elev_delta_bounds.elevation_delta_upper_bound = -50
        with pytest.raises(SystemExit) as e:
            main_cli(args_bad_elev_delta_bounds, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases epi_step
        args_bad_epi_step = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_epi_step.epi_step = 0
            main_cli(args_bad_epi_step, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases epipolar_error_upper_bound
        args_bad_epi_error_bound = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_epi_error_bound.epipolar_error_upper_bound = -10
            main_cli(args_bad_epi_error_bound, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases log level
        args_bad_loglevel = copy(prepare_default_args)
        with pytest.raises(ValueError):
            args_bad_loglevel.loglevel = 'TEST'
            main_cli(args_bad_loglevel, parser, check_inputs=True)

        # degraded cases number of workers
        args_bad_nb_workers = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_nb_workers.nb_workers = -1
            main_cli(args_bad_nb_workers, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases wall time
        args_bad_wall_time = copy(prepare_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_wall_time.walltime = '000:00:00'
            main_cli(args_bad_wall_time, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        with pytest.raises(SystemExit) as e:
            args_bad_wall_time.walltime = 'bb:bb:bb'
            main_cli(args_bad_wall_time, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1


# ----------------------------------
# DASK COMPUTE DSM
# ----------------------------------

@pytest.mark.unit_tests
def test_dsm_compute_arg(compute_dsm_default_args):
    parser = cars_cli_parser()

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        compute_dsm_default_args.outdir = directory

        # test with default args
        main_cli(compute_dsm_default_args, parser, check_inputs=True)

        # test with mp mode (multiprocessing)
        args_mode_mp = copy(compute_dsm_default_args)
        args_mode_mp.mode = "mp"
        main_cli(args_mode_mp, parser, check_inputs=True)

        # test [xmin, ymin, xmax, ymax] roi argument
        args_roi_bbox = copy(compute_dsm_default_args)
        args_roi_bbox.roi_bbox = ['1.0', '2.0', '3.0', '4.0']
        main_cli(args_roi_bbox, parser, check_inputs=True)

        # test image roi argument
        args_roi_file = copy(compute_dsm_default_args)
        args_roi_file.roi_file = absolute_data_path(
            'input/cars_cli_input/roi_image.tif')
        main_cli(args_roi_file, parser, check_inputs=True)

        # test vector roi argument
        args_roi_file.roi_file = absolute_data_path(
            'input/cars_cli_input/roi_vector.gpkg')
        main_cli(args_roi_file, parser, check_inputs=True)

        # degraded cases input jsons
        args_bad_jsons = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_jsons.injsons = [
                absolute_data_path('input/cars_cli_input/test.txt')]
            main_cli(args_bad_jsons, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        with pytest.raises(SystemExit) as e:
            args_bad_jsons.injsons = []
            main_cli(args_bad_jsons, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases sigma
        args_bad_sigma = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_sigma.sigma = -10
            main_cli(args_bad_sigma, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases dsm radius
        args_bad_dsm_radius = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_dsm_radius.dsm_radius = -10
            main_cli(args_bad_dsm_radius, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases resolution
        args_bad_resolution = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_resolution.resolution = 0
            main_cli(args_bad_resolution, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        args_bad_resolution = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_resolution.resolution = -1
            main_cli(args_bad_resolution, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases epsg
        args_bad_epsg = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_epsg.epsg = -1
            main_cli(args_bad_epsg, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases input ROI file
        args_bad_roi_file = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_roi_file.roi_file = \
                absolute_data_path('input/cars_cli_input/test.txt')
            main_cli(args_bad_roi_file, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        with pytest.raises(SystemExit) as e:
            args_bad_roi_file.roi_file = absolute_data_path(
                'input/phr_ventoux/preproc_output/content.json')
            main_cli(args_bad_roi_file, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        with pytest.raises(SystemExit) as e:
            args_bad_roi_file.roi_file = absolute_data_path(
                'input/phr_ventoux/left_image.tif')
            main_cli(args_bad_roi_file, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases correlator config file
        args_bad_correlator_conf = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_correlator_conf.corr_config = absolute_data_path(
                'input/cars_cli_input/test.txt')
            main_cli(args_bad_correlator_conf, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases elevation offsets
        args_bad_elevation_offsets = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_elevation_offsets.min_elevation_offset = 10
            args_bad_elevation_offsets.max_elevation_offset = -10
            main_cli(args_bad_elevation_offsets, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases log level
        args_bad_loglevel = copy(compute_dsm_default_args)
        with pytest.raises(ValueError):
            args_bad_loglevel.loglevel = 'TEST'
            main_cli(args_bad_loglevel, parser, check_inputs=True)

        # degraded cases number of workers
        args_bad_nb_workers = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_nb_workers.nb_workers = -1
            main_cli(args_bad_nb_workers, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1

        # degraded cases wall time
        args_bad_wall_time = copy(compute_dsm_default_args)
        with pytest.raises(SystemExit) as e:
            args_bad_wall_time.walltime = '000:00:00'
            main_cli(args_bad_wall_time, parser, check_inputs=True)
        assert e.type == SystemExit
        assert e.value.code == 1
