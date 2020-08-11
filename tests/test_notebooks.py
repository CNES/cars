#
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
import subprocess
import tempfile
import os
import fileinput
import pytest

from utils import temporary_dir, absolute_data_path
from cars import prepare
from cars.parameters import read_input_parameters


@pytest.mark.notebook_tests
def test_step_by_step_compute_dsm():
    cars_path = os.environ.get('CARSPATH')

    # uncomment the following lines to regenerate the input files
    # input_json = read_input_parameters(
    #     absolute_data_path("input/phr_ventoux/preproc_input.json"))
    # prepare.run(
    #     input_json,
    #     absolute_data_path("input/notebooks_input/content_dir/"),
    #     epi_step=30,
    #     region_size=250,
    #     disparity_margin=0.25,
    #     epipolar_error_upper_bound=43.,
    #     elevation_delta_lower_bound=-20.,
    #     elevation_delta_upper_bound=20.,
    #     mode="local",  # Run on a local cluster
    #     nb_workers=4,
    #     walltime="00:10:00",
    #     check_inputs=True)

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(['jupyter nbconvert --to script {}/notebooks/step_by_step_compute_dsm.ipynb --output-dir {}'
                       .format(cars_path, directory)], shell=True)

        for line in fileinput.input('{}/step_by_step_compute_dsm.py'.format(directory), inplace=True):
            if "cars_home = \"TODO\"" in line:
                line = line.replace("TODO", cars_path)
            elif "content_dir = \"TODO\"" in line:
                line = line.replace("TODO",
                                    absolute_data_path("input/notebooks_input/content_dir/"))
            elif "roi_file = \"TODO\"" in line:
                line = line.replace("TODO", absolute_data_path("input/notebooks_input/content_dir/"
                                                               "envelopes_intersection.gpkg"))
            elif "output_dir = \"TODO\"" in line:
                line = line.replace("TODO", directory)
            print(line) # keep this print

        out = subprocess.run(['ipython {}/step_by_step_compute_dsm.py'.format(directory)], shell=True)

        out.check_returncode()


@pytest.mark.notebook_tests
def test_epipolar_distributions():
    cars_path = os.environ.get('CARSPATH')

    # uncomment the following lines to regenerate the input files
    # input_json = read_input_parameters(
    #     absolute_data_path("input/phr_ventoux/preproc_input.json"))
    # prepare.run(
    #     input_json,
    #     absolute_data_path("input/notebooks_input/content_dir/"),
    #     epi_step=30,
    #     region_size=250,
    #     disparity_margin=0.25,
    #     epipolar_error_upper_bound=43.,
    #     elevation_delta_lower_bound=-20.,
    #     elevation_delta_upper_bound=20.,
    #     mode="local",  # Run on a local cluster
    #     nb_workers=4,
    #     walltime="00:10:00",
    #     check_inputs=True)

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(['jupyter nbconvert --to script {}/notebooks/epipolar_distributions.ipynb --output-dir {}'
                       .format(cars_path, directory)], shell=True)

        for line in fileinput.input('{}/epipolar_distributions.py'.format(directory), inplace=True):
            if "cars_home = \"TODO\"" in line:
                line = line.replace("TODO", cars_path)
            elif "content_dir = \"TODO\"" in line:
                line = line.replace("TODO",
                                    absolute_data_path("input/notebooks_input/content_dir/"))
            print(line)  # keep this print

        out = subprocess.run(['ipython {}/epipolar_distributions.py'.format(directory)], shell=True)

        out.check_returncode()


@pytest.mark.notebook_tests
def test_lowres_dem_fit():
    cars_path = os.environ.get('CARSPATH')

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(['jupyter nbconvert --to script {}/notebooks/lowres_dem_fit.ipynb --output-dir {}'
                       .format(cars_path, directory)], shell=True)

        for line in fileinput.input('{}/lowres_dem_fit.py'.format(directory), inplace=True):
            if "cars_home = \"TODO\"" in line:
                line = line.replace("TODO", cars_path)
            elif "content_dir = \"TODO\"" in line:
                line = line.replace("TODO",
                                    absolute_data_path("input/notebooks_input/lowres_dem/"))
            print(line)  # keep this print

        out = subprocess.run(['ipython {}/lowres_dem_fit.py'.format(directory)], shell=True)

        out.check_returncode()


@pytest.mark.notebook_tests
def test_compute_dsm_memory_monitoring():
    cars_path = os.environ.get('CARSPATH')

    # uncomment the following lines to regenerate the input files
    # input_json = read_input_parameters(
    #     absolute_data_path("input/phr_ventoux/preproc_input.json"))
    # prepare.run(
    #     input_json,
    #     absolute_data_path("input/notebooks_input/content_dir/"),
    #     epi_step=30,
    #     region_size=250,
    #     disparity_margin=0.25,
    #     epipolar_error_upper_bound=43.,
    #     elevation_delta_lower_bound=-20.,
    #     elevation_delta_upper_bound=20.,
    #     mode="local",  # Run on a local cluster
    #     nb_workers=4,
    #     walltime="00:10:00",
    #     check_inputs=True)

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        subprocess.run(['jupyter nbconvert --to script {}/notebooks/compute_dsm_memory_monitoring.ipynb --output-dir {}'
                       .format(cars_path, directory)], shell=True)

        for line in fileinput.input('{}/compute_dsm_memory_monitoring.py'.format(directory), inplace=True):
            if "compute_dsm_output_dir = \"TODO\"" in line:
                line = line.replace("TODO",
                                    absolute_data_path("input/notebooks_input/content_dir/"))
            print(line)  # keep this print

        out = subprocess.run(['ipython {}/compute_dsm_memory_monitoring.py'.format(directory)], shell=True)

        out.check_returncode()
