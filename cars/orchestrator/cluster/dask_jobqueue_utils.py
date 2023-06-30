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
Contains functions for dask jobqueue cluster (PBS, SLURM)
"""

import logging
import math
import os
import sys
import warnings
from datetime import timedelta


def init_cluster_variables(
    nb_workers, walltime, out_dir, activate_dashboard, python, core_memory
):
    """
    Initialize global cluster variables
    :param nb_workers: number of workers
    :param walltime: workers walltime
    :param out_dir: ouput result directory
    :param activate_dashboard: option to activate dashboard mode
    :param python: target python used by workers (retrun system python if None)
    :param core_memory: cluster node memory (Mo)
    :return: all cluster parameters (python,
    nb_workers_per_job,
    memory,
    nb_cpus,
    stagger,
    lifetime_with_margin,
    scheduler_options,
    envs,
    log_directory,
    local_directory)
    """

    if python is None:
        python = sys.executable

    # Configure worker distribution.
    # Workers are mostly running single-threaded, GIL-holding functions, so we
    # dedicate a single thread for each worker to maximize CPU utilization.
    nb_threads_per_worker = 1

    # Network latency is not the bottleneck, so we dedicate a single worker for
    # each job in order to minimize the requested resources, which reduces our
    # scheduling delay.
    nb_workers_per_job = 1

    # Total number of CPUs is multi-threading factor times size of batch
    # (number of workers per job)
    nb_cpus = nb_threads_per_worker * nb_workers_per_job
    nb_jobs = int(math.ceil(nb_workers / nb_workers_per_job))
    # Cluster nodes have core_memory Mo per core
    memory = nb_cpus * core_memory

    # Configure worker lifetime for adaptative scaling.
    # See https://jobqueue.dask.org/en/latest/advanced-tips-and-tricks.html
    hours, minutes, seconds = map(int, walltime.split(":"))
    lifetime = timedelta(seconds=3600 * hours + 60 * minutes + seconds)
    # Use hardcoded stagger of 3 minutes. The actual lifetime will be selected
    # uniformly at random between lifetime +/- stagger.
    stagger = timedelta(minutes=3)
    # Add some margin to not get killed by scheduler during worker shutdown.
    shutdown_margin = timedelta(minutes=2)
    min_walltime = stagger + shutdown_margin
    lifetime_with_margin = lifetime - min_walltime
    if lifetime_with_margin.total_seconds() < 0:
        min_walltime_minutes = min_walltime.total_seconds() / 60
        logging.warning(
            "Could not add worker lifetime margin because specified walltime "
            "is too short. Workers might get killed by SLURM before they can "
            "cleanly exit, which might break adaptative scaling. Please "
            "specify a lifetime greater than {} minutes.".format(
                min_walltime_minutes
            )
        )
        lifetime_with_margin = lifetime

    logging.info(
        "Starting Dask SLURM cluster with {} workers "
        "({} workers with {} cores each per SLURM job)".format(
            nb_workers, nb_workers_per_job, nb_threads_per_worker
        )
    )

    logging.info(
        "Submitting {} SLURM jobs "
        "with configuration cpu={}, mem={}, walltime={}".format(
            nb_jobs, nb_cpus, memory, walltime
        )
    )

    if activate_dashboard:
        scheduler_options = None
    else:
        scheduler_options = {"dashboard": None, "dashboard_address": None}

    names = [
        "PATH",
        "PYTHONPATH",
        "CARS_STATIC_CONFIGURATION",
        "LD_LIBRARY_PATH",
        "OTB_APPLICATION_PATH",
        "OTB_MAX_RAM_HINT",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OPJ_NUM_THREADS",
        "GDAL_NUM_THREADS",
        "VIRTUAL_ENV",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "GDAL_CACHEMAX",
        "DASK_CONFIG",
        "NUMBA_THREADING_LAYER",
    ]
    names = [name for name in names if os.environ.get(name)]
    envs = ["export {}={}".format(name, os.environ[name]) for name in names]
    log_directory = os.path.join(os.path.abspath(out_dir), "dask_log")
    local_directory = "$TMPDIR"
    return (
        python,
        nb_workers_per_job,
        memory,
        nb_cpus,
        stagger,
        lifetime_with_margin,
        scheduler_options,
        envs,
        log_directory,
        local_directory,
    )


def get_dashboard_link(cluster):
    """
    This function returns the dashboard address.

    :param cluster: Dask cluster
    :type cluster: dask_jobqueue.PBSCluster
    :return: Link to the dashboard
    :rtype: string
    """
    template = "http://{host}:{port}/status"
    host = cluster.scheduler.address.split("://")[1].split(":")[0]
    port = cluster.scheduler.services["dashboard"].port
    return template.format(host=host, port=port)


def stop_cluster(cluster, client):
    """
    This function stops a dask cluster.

    :param cluster: Dask cluster
    :type cluster: dask_jobqueue.PBSCluster
    :param client: Dask client
    :type client: dask.distributed.Client
    """
    client.close()
    # Bug distributed on close cluster : Fail with AssertionError still running
    try:
        with warnings.catch_warnings():
            # Ignore internal dask_jobqueue warnings to be corrected in:
            # https://github.com/dask/dask-jobqueue/pull/506
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*ignoring was deprecated in version 2021.06.1.*",
            )
            cluster.close()
    except AssertionError as assert_error:
        logging.warning(
            "Dask cluster failed " "to stop properly: {}".format(assert_error)
        )
        # not raising to not fail tests

    logging.info("Dask cluster correctly stopped")
