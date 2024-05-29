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
Test module for config of cars/orchestrator/cluster/
"""

import pytest

from cars.orchestrator.orchestrator import Orchestrator

# Configurations

conf_sequential = {
    "mode": "sequential",
    "max_ram_per_worker": 2000,
    "profiling": {"mode": "cars_profiling", "loop_testing": False},
}

conf_auto = {"mode": "auto"}

conf_mp = {
    "mode": "multiprocessing",
    "nb_workers": 2,
    "max_ram_per_worker": 2000,
    "max_tasks_per_worker": 10,
    "dump_to_disk": True,
    "per_job_timeout": 600,
    "factorize_tasks": True,
    "profiling": {"mode": "cars_profiling", "loop_testing": True},
}

conf_local_dask = {
    "mode": "local_dask",
    "use_memory_logger": False,
    "nb_workers": 2,
    "max_ram_per_worker": 2000,
    "walltime": "00:59:00",
    "config_name": "unknown",
    "activate_dashboard": False,
    "python": None,
    "profiling": {"mode": "cars_profiling", "loop_testing": False},
}

conf_pbs_dask = {
    "mode": "pbs_dask",
    "use_memory_logger": False,
    "nb_workers": 2,
    "max_ram_per_worker": 2000,
    "walltime": "00:59:00",
    "config_name": "unknown",
    "activate_dashboard": False,
    "python": None,
    "profiling": {"mode": "cars_profiling", "loop_testing": False},
}

conf_slurm_dask = {
    "mode": "slurm_dask",
    "account": "cars",
    "use_memory_logger": False,
    "nb_workers": 2,
    "max_ram_per_worker": 2000,
    "walltime": "00:59:00",
    "config_name": "unknown",
    "activate_dashboard": False,
    "python": None,
    "profiling": {"mode": "cars_profiling", "loop_testing": False},
    "qos": None,
}


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "conf",
    [
        conf_sequential,
        conf_auto,
        conf_mp,
        conf_local_dask,
        conf_pbs_dask,
        conf_slurm_dask,
    ],
)
def test_check_full_conf(conf):
    """
    Test configuration check for orchestrator
    """
    _ = Orchestrator(conf)
