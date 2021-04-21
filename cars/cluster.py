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
Cluster module:
provides functions to start and stop a local or PBS cluster.
"""

# Standard imports
import logging
import os
import math
import time
import psutil

# Third-party imports
import numpy as np
from dask_jobqueue import PBSCluster
from dask.distributed import Client, LocalCluster

from distributed.diagnostics.plugin import WorkerPlugin

class ComputeDSMMemoryLogger(WorkerPlugin):
    """A subclass of WorkerPlugin dedicated to monitoring workers memory

    This plugin enables two things:
    - Additional dask log traces : (for each worker internal state change)
        - amount of tasks
        - associated memory
    - A numpy data file with memory metrics and timing
    """
    def __init__(self, outdir):
        """
        Constructor
        :param outdir: output directory
        :type outdir: string
        """
        self.outdir = outdir

    def setup(self, worker):
        """
        Associate plugin with a worker
        :param worker: The worker to associate the plugin with
        """
        # Pylint Exception : Inherited attributes outside __init__
        #pylint: disable=attribute-defined-outside-init
        self.worker = worker
        self.name = worker.name
        # Measure plugin registration time
        self.start_time = time.time()
        # Data will hold the memory traces as numpy array
        self.data=([[0, 0, 0, 0, 0, 0]])


    def transition(self, key, start, finish, **kwargs):
        """
        Callback when worker changes internal state
        """
        # TODO Pylint Exception : Inherited attributes outside __init__
        #pylint: disable=attribute-defined-outside-init

        # Setup logging
        worker_logger = logging.getLogger('distributed.worker')

        # Define cumulants
        total_point_clouds_in_memory = 0
        total_point_clouds_nbytes = 0
        total_rasters_in_memory = 0
        total_rasters_nbytes = 0

        # Measure ellapsed time for the state change
        elapsed_time = time.time()-self.start_time

        # Walk the worker known memory
        for task, task_size in self.worker.nbytes.items():

            # Sort between point clouds and rasters
            if task.startswith("images_pair_to_3d_points"):
                total_point_clouds_nbytes += task_size
                total_point_clouds_in_memory += 1
            else:
                total_rasters_nbytes += task_size
                total_rasters_in_memory += 1

        # Use psutil to capture python process memory as well
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss

        # Update data records
        self.data = np.concatenate((self.data,
                                    np.array([[ elapsed_time,
                                                total_point_clouds_in_memory,
                                                total_point_clouds_nbytes,
                                                total_rasters_in_memory,
                                                total_rasters_nbytes,
                                                process_memory]])))
        # Convert nbytes size for logging
        total_point_clouds_nbytes = float(total_point_clouds_nbytes)/1000000
        total_rasters_nbytes = float(total_rasters_nbytes)/1000000
        process_memory = float(process_memory)/1000000

        # Log memory state
        worker_logger.info(
            "Memory report: point clouds = {} ({} Mb), "
                            "rasters = {} ({} Mb), "
                            "python process memory = {} Mb".format(
                        total_point_clouds_in_memory, total_point_clouds_nbytes,
                        total_rasters_in_memory, total_rasters_nbytes,
                        process_memory ))

        # Save data records in npy file
        # TODO: Save only every x seconds ?
        file = os.path.join(self.outdir,'dask_log',"memory_"+self.name+'.npy')
        np.save(file, self.data)


def start_local_cluster(nb_workers, timeout=600):
    """
    Start a local cluster

    :param nb_workers: Number of dask workers
    :type nb_workers: int
    :param timeout: Connection timeout
    :type timeout: int
    :return: Local cluster and Dask client
    :rtype: (dask.distributed.LocalCluster, dask.distributed.Client) tuple
    """
    logging.info("Local cluster with {} workers started".format(nb_workers))
    cluster = LocalCluster(n_workers=nb_workers, threads_per_worker=1)

    client = Client(cluster, timeout=timeout)
    return cluster, client


def stop_local_cluster(cluster, client):
    """
    Stop a local cluster

    :param cluster: Local cluster
    :type cluster: dask.distributed.LocalCluster
    :param client: Dask client
    :type client: dask.distributed.Client
    """
    client.close()
    cluster.close()
    logging.info("Local cluster correctly stopped")


def start_cluster(nb_workers, walltime, out_dir, timeout=600):
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
    if os.environ.get('OMP_NUM_THREADS'):
        omp_num_threads = int(os.environ['OMP_NUM_THREADS'])

    # Retrieve number of workers per PBS job
    nb_workers_per_job = 1
    if os.environ.get('CARS_NB_WORKERS_PER_PBS_JOB'):
        nb_workers_per_job = int(os.environ['CARS_NB_WORKERS_PER_PBS_JOB'])

    # Retrieve PBS queue
    pbs_queue = None
    if os.environ.get('CARS_PBS_QUEUE'):
        pbs_queue = os.environ['CARS_PBS_QUEUE']

    # Total number of cpus is multi-threading factor times size of batch
    # (number of workers per PBS job)
    nb_cpus = nb_workers_per_job * omp_num_threads
    # Cluster nodes have 5GB per core
    memory = nb_cpus * 5000
    # Ressource string for PBS
    resource = "select=1:ncpus={}:mem={}mb".format(nb_cpus, memory)

    nb_jobs = int(math.ceil(nb_workers / nb_workers_per_job))

    logging.info(
        "Starting Dask PBS cluster with {} workers "
        "({} workers with {} cores each per PSB job)".format(
        nb_workers,
        nb_workers_per_job, omp_num_threads))

    logging.info(
        "Submitting {} PBS jobs "
        "with configuration cpu={}, mem={}, walltime={}".format(
        nb_jobs,
        nb_cpus, memory, walltime))

    names = [
        'PATH',
        'PYTHONPATH',
        'CARS_STATIC_CONFIGURATION',
        'LD_LIBRARY_PATH',
        'OTB_APPLICATION_PATH',
        'OTB_GEOID_FILE',
        'OMP_NUM_THREADS',
        'NUMBA_NUM_THREADS',
        'OPJ_NUM_THREADS',
        'GDAL_NUM_THREADS',
        'OTB_MAX_RAM_HINT',
        'VIRTUAL_ENV',
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
        'GDAL_CACHEMAX',
        'DASK_CONFIG']
    names = [name for name in names if os.environ.get(name)]
    envs = ["export {}={}".format(name, os.environ[name]) for name in names]
    log_directory = os.path.join(os.path.abspath(out_dir), "dask_log")
    local_directory = '$TMPDIR'
    cluster = PBSCluster(
        processes=nb_workers_per_job,
        cores=nb_workers_per_job,
        resource_spec=resource,
        memory="{}MB".format(memory),
        local_directory=local_directory,
        project='dask-test',
        walltime=walltime,
        interface='ib0',
        queue=pbs_queue,
        env_extra=envs,
        log_directory=log_directory)
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
    This function stop a dask cluster.

    :param cluster: Dask cluster
    :type cluster: dask_jobqueue.PBSCluster
    :param client: Dask client
    :type client: dask.distributed.Client
    """
    client.close()
    cluster.close()
    logging.info("Dask cluster correctly stopped")
