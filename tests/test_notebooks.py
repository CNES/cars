#!/usr/bin/env python
# coding: utf8
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
Test module for notebooks/*
"""

# Standard imports
from __future__ import absolute_import

import fileinput
import subprocess
import tempfile

# Third party imports
import pytest

# CARS Tests imports
from .helpers import cars_path, temporary_dir


@pytest.mark.notebook_tests
def test_sensor_to_dense_dsm_dsm_step_by_step():
    """
    Sensor to dense dsm step by step notebook test:
    notebook conversion (.ipynb->.py), copy data_samples and notebooks helper,
    modify show_data matplotlib to be executable by ipython without pause,
    run the notebook with ipython and check return code
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # Convert Jupyter notebook to script
        subprocess.run(
            [
                "jupyter nbconvert "
                "--to script "
                "{}/tutorials/sensor_to_dense_dsm_step_by_step.ipynb"
                " --output-dir {}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy notebook helpers
        subprocess.run(
            [
                "cp "
                "{}/tutorials/notebook_helpers.py "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy data samples gizeh
        subprocess.run(
            [
                "cp "
                "{}/tutorials/data_gizeh_small.tar.bz2 "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # Deactivate matplotlib show data
        for line in fileinput.input(
            "{}/sensor_to_dense_dsm_step_by_step.py".format(directory),
            inplace=True,
        ):
            if "show_data(" in line:
                line = line.replace("show_data(", "#show_data(")
            print(line)  # keep this print

        # Run notebook converted script
        out = subprocess.run(
            [
                "ipython "
                "{}/sensor_to_dense_dsm_step_by_step.py".format(directory)
            ],
            shell=True,
            check=True,
        )

        out.check_returncode()


@pytest.mark.notebook_tests
def test_sensor_to_dense_dsm_matching_methods_comparison():
    """
    sensor_to_dense_dsm_matching_methods_comparison notebook test:
    notebook conversion (.ipynb->.py), copy data_samples and notebooks helper,
    modify show_data matplotlib to be executable by ipython without pause,
    run the notebook with ipython and check return code
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        # Convert Jupyter notebook to script
        subprocess.run(
            [
                "jupyter nbconvert "
                "--to script "
                "{}/tutorials/sensor_to_dense"
                "_dsm_matching_methods_comparison.ipynb"
                " --output-dir {}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy notebook helpers
        subprocess.run(
            [
                "cp "
                "{}/tutorials/notebook_helpers.py "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy data samples gizeh
        subprocess.run(
            [
                "cp "
                "{}/tutorials/data_gizeh_small.tar.bz2 "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # Deactivate matplotlib show data
        for line in fileinput.input(
            "{}/sensor_to_dense"
            "_dsm_matching_methods_comparison.py".format(directory),
            inplace=True,
        ):
            if "show_data(" in line:
                line = line.replace("show_data(", "#show_data(")
            print(line)  # keep this print

        # Run notebook converted script
        out = subprocess.run(
            [
                "ipython "
                "{}/sensor_to_dense_dsm"
                "_matching_methods_comparison.py".format(directory)
            ],
            shell=True,
            check=True,
        )

        out.check_returncode()


@pytest.mark.notebook_tests
def test_main_tutorial():
    """
    Main tutorial test:
    notebook conversion (.ipynb->.py), copy data_samples and notebooks helper,
    modify show_data matplotlib to be executable by ipython without pause,
    run the notebook with ipython and check return code
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(
            [
                "jupyter nbconvert "
                "--to script "
                "{}/tutorials/main_tutorial.ipynb"
                " --output-dir {}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy notebook helpers
        subprocess.run(
            [
                "cp "
                "{}/tutorials/notebook_helpers.py "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy data samples gizeh
        subprocess.run(
            [
                "cp "
                "{}/tutorials/data_gizeh_small.tar.bz2 "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # Deactivate matplotlib show data
        for line in fileinput.input(
            "{}/main_tutorial.py".format(directory),
            inplace=True,
        ):
            if "show_data(" in line:
                line = line.replace("show_data(", "#show_data(")
            print(line)  # keep this print

        out = subprocess.run(
            ["ipython {}/main_tutorial.py".format(directory)],
            shell=True,
            check=True,
        )

        out.check_returncode()


@pytest.mark.notebook_tests
def test_sensor_to_dsm_from_a_priori():
    """
    sensor_to_dsm_from_a_priori notebook test:
    notebook conversion (.ipynb->.py), copy data_samples and notebooks helper,
    modify show_data matplotlib to be executable by ipython without pause,
    run the notebook with ipython and check return code
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(
            [
                "jupyter nbconvert "
                "--to script "
                "{}/tutorials/sensor_to_dsm_from_a_priori.ipynb"
                " --output-dir {}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy notebook helpers
        subprocess.run(
            [
                "cp "
                "{}/tutorials/notebook_helpers.py "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        subprocess.run(
            [
                "cp "
                "{}/tutorials/notebook_helpers_cars_free.py "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # copy data samples gizeh
        subprocess.run(
            [
                "cp "
                "{}/tutorials/data_gizeh_small.tar.bz2 "
                "{}".format(cars_path(), directory)
            ],
            shell=True,
            check=True,
        )
        # Deactivate matplotlib show data
        for line in fileinput.input(
            "{}/sensor_to_dsm_from_a_priori.py".format(directory),
            inplace=True,
        ):
            if "show_data(" in line:
                line = line.replace("show_data(", "#show_data(")
            if "orchestrator_conf" in line:
                line = line.replace("local_dask", "sequential")
                line = line.replace(', "nb_workers": 2', "")
            print(line)  # keep this print

        out = subprocess.run(
            ["ipython {}/sensor_to_dsm_from_a_priori.py".format(directory)],
            shell=True,
            check=True,
        )

        out.check_returncode()
