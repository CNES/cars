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
import warnings

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
        )

    def cleanup(self):
        """
        Cleanup cluster

        """
        stop_cluster(self.cluster, self.client)
        logging.info("Dask cluster closed")


def start_cluster(
    nb_workers, walltime, out_dir, timeout=600, activate_dashboard=False
):
    """
    This function create a dask cluster.
    Each worker has nb_cpus cpus.
    Only one python process is started on each worker.

    Threads number:
    start_cluster will use OMP_NUM_THREADS environment variable to determine
    how many threads might be used by a worker when running C/C++ code.
    (default to 1)

    Workers number:
    start_cluster will use CARS_NB_WORKERS_PER_PBS_JOB environment variable
    to determine how many workers should be started by a single PBS job.
    (default to 1)

    Queue worker:
    start_cluster will use CARS_PBS_QUEUE to determine
    in which queue worker jobs should be posted.

    :param nb_workers: Number of dask workers
    :type nb_workers: int
    :param walltime: Walltime for each dask worker
    :type walltime: string
    :param out_dir: Output directory
    :type out_dir: string
    :return: Dask cluster and dask client
    :rtype: (dask_jobqueue.PBSCluster, dask.distributed.Client) tuple
    """
    # Retrieve multi-threading factor for C/C++ code if available
    omp_num_threads = 1
    if os.environ.get("OMP_NUM_THREADS"):
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])

    # Retrieve number of workers per PBS job
    nb_workers_per_job = 1
    if os.environ.get("CARS_NB_WORKERS_PER_PBS_JOB"):
        nb_workers_per_job = int(os.environ["CARS_NB_WORKERS_PER_PBS_JOB"])

    # Retrieve PBS queue
    pbs_queue = None
    if os.environ.get("CARS_PBS_QUEUE"):
        pbs_queue = os.environ["CARS_PBS_QUEUE"]

    # Total number of cpus is multi-threading factor times size of batch
    # (number of workers per PBS job)
    nb_cpus = nb_workers_per_job * omp_num_threads
    # Cluster nodes have 5GB per core
    memory = nb_cpus * 5000
    # Resource string for PBS
    resource = "select=1:ncpus={}:mem={}mb".format(nb_cpus, memory)

    nb_jobs = int(math.ceil(nb_workers / nb_workers_per_job))

    logging.info(
        "Starting Dask PBS cluster with {} workers "
        "({} workers with {} cores each per PSB job)".format(
            nb_workers, nb_workers_per_job, omp_num_threads
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
        resource_spec=resource,
        memory="{}MB".format(memory),
        local_directory=local_directory,
        project="dask-test",
        walltime=walltime,
        interface="ib0",
        queue=pbs_queue,
        env_extra=envs,
        log_directory=log_directory,
        scheduler_options=scheduler_options,
    )
    logging.info("Dask cluster started")
    cluster.scale(nb_workers)
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
