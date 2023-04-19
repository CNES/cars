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
Contains abstract function for PBS dask Cluster
"""

# Standard imports
import logging
import math
import os
import sys
import warnings
from datetime import timedelta

# Third party imports
from dask.distributed import Client

with warnings.catch_warnings():
    # Ignore some internal dask_jobqueue warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*format_bytes is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*parse_bytes is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*tmpfile is deprecated.*",
    )
    from dask_jobqueue import PBSCluster

# CARS imports
from cars.orchestrator.cluster import abstract_cluster, abstract_dask_cluster


@abstract_cluster.AbstractCluster.register_subclass("pbs_dask")
class PbsDaskCluster(abstract_dask_cluster.AbstractDaskCluster):
    """
    PbsDaskCluster
    """

    def start_dask_cluster(self):
        """
        Start dask cluster
        """

        return start_cluster(
            self.nb_workers,
            self.walltime,
            self.out_dir,
            activate_dashboard=self.activate_dashboard,
            python=self.python,
        )

    def cleanup(self):
        """
        Cleanup cluster

        """
        stop_cluster(self.cluster, self.client)
        logging.info("Dask cluster closed")


def start_cluster(
    nb_workers,
    walltime,
    out_dir,
    timeout=600,
    activate_dashboard=False,
    python=None,
):
    """Create a Dask cluster.

    Each worker will be spawned in an independent job with a single CPU
    allocated to it, and will use a single process. This is done to maximize
    CPU utilization and minimize scheduling delay.

    The CARS_PBS_QUEUE environment variable, if defined, is used to specify the
    queue in which worker jobs are scheduled.

    :param nb_workers: Number of dask workers
    :type nb_workers: int
    :param walltime: Walltime for each dask worker
    :type walltime: string
    :param out_dir: Output directory
    :type out_dir: string
    :return: Dask cluster and dask client
    :rtype: (dask_jobqueue.PBSCluster, dask.distributed.Client) tuple
    """
    # retrieve current python path if None
    if python is None:
        python = sys.executable

    # Retrieve PBS queue
    pbs_queue = os.environ.get("CARS_PBS_QUEUE")

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
    # Cluster nodes have 5GB per core
    memory = nb_cpus * 5000

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
            "is too short. Workers might get killed by PBS before they can "
            "cleanly exit, which might break adaptative scaling. Please "
            "specify a lifetime greater than {} minutes.".format(
                min_walltime_minutes
            )
        )
        lifetime_with_margin = lifetime

    logging.info(
        "Starting Dask PBS cluster with {} workers "
        "({} workers with {} cores each per PBS job)".format(
            nb_workers, nb_workers_per_job, nb_threads_per_worker
        )
    )

    logging.info(
        "Submitting {} PBS jobs "
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
    cluster = PBSCluster(
        processes=nb_workers_per_job,
        cores=nb_workers_per_job,
        memory="{}MiB".format(memory),
        local_directory=local_directory,
        account="dask-test",
        walltime=walltime,
        interface="ib0",
        queue=pbs_queue,
        job_script_prologue=envs,
        log_directory=log_directory,
        python=python,
        worker_extra_args=[
            "--lifetime",
            f"{int(lifetime_with_margin.total_seconds())}s",
            "--lifetime-stagger",
            f"{int(stagger.total_seconds())}s",
        ],
        scheduler_options=scheduler_options,
    )
    logging.info("Dask cluster started")
    cluster.adapt(minimum=nb_workers, maximum=nb_workers)
    client = Client(cluster, timeout=timeout)
    logging.info("Dashboard started at {}".format(get_dashboard_link(cluster)))
    return cluster, client


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
