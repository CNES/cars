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
Test module for cars/cars.py
"""

# Standard imports
import argparse
import json
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.cars import cars_parser, main_cli

# CARS Tests imports
from .helpers import absolute_data_path, generate_input_json, temporary_dir

# ----------------------------------
# GENERAL
# ----------------------------------


@pytest.mark.unit_tests
def test_cars_parser():
    """
    Cars parser test
    """
    parser = cars_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "cars"


@pytest.mark.unit_tests
def test_main_no_argument():
    """
    Cars main_cli pytest with no argument
    """
    args = argparse.Namespace()
    args.command = None

    with pytest.raises(SystemExit) as exit_error:
        main_cli(args, dry_run=True)
    assert exit_error.type == SystemExit
    assert exit_error.value.code == 1


@pytest.mark.unit_tests
def test_main_wrong_subcommand():
    """
    Cars main_cli pytest with wrong subcommand
    """

    args = argparse.Namespace()
    args.loglevel = "INFO"
    args.command = "test"

    with pytest.raises(SystemExit) as exit_error:
        main_cli(args, dry_run=True)
    assert exit_error.type == SystemExit
    assert exit_error.value.code == 1


# ----------------------------------
# low_res_dsm
# ----------------------------------


@pytest.mark.unit_tests
def test_low_res_dsm_args():
    """
    Cars prepare arguments test with default and degraded cases
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # test default args
        args = argparse.Namespace()
        filled_absolute_path_input, conf_ = generate_input_json(
            absolute_data_path("input/phr_ventoux/input.json"),
            directory,
            "sequential",
        )
        args.conf = filled_absolute_path_input
        conf_["advanced"]["epipolar_resolutions"] = 1

        with open(filled_absolute_path_input, "w", encoding="utf8") as fstream:
            json.dump(conf_, fstream, indent=2)

        main_cli(args, dry_run=True)

        # degraded cases injson
        args_bad_conf = args
        args_bad_conf.conf = absolute_data_path("input/cars_input/test.json")
        with pytest.raises(SystemExit) as exit_error:
            main_cli(args_bad_conf, dry_run=True)
        assert exit_error.type == SystemExit
        assert exit_error.value.code == 1


# ----------------------------------
# DASK full_res_dsm
# ----------------------------------


@pytest.mark.unit_tests
def test_full_res_dsm_args():
    """
    Cars prepare arguments test with default and degraded cases
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # test default args
        args = argparse.Namespace()
        filled_absolute_path_input, conf_ = generate_input_json(
            absolute_data_path("input/phr_ventoux/input.json"),
            directory,
            "sequential",
        )
        args.conf = filled_absolute_path_input
        conf_["advanced"]["epipolar_resolutions"] = 1

        with open(filled_absolute_path_input, "w", encoding="utf8") as fstream:
            json.dump(conf_, fstream, indent=2)

        main_cli(args, dry_run=True)

        # degraded cases injson
        args_bad_conf = args
        args_bad_conf.conf = absolute_data_path("input/cars_input/test.json")
        with pytest.raises(SystemExit) as exit_error:
            main_cli(args_bad_conf, dry_run=True)
        assert exit_error.type == SystemExit
        assert exit_error.value.code == 1
