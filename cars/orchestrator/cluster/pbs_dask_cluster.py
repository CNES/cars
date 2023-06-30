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
import os
import warnings

# Third party imports
from dask.distributed import Client

from cars.orchestrator.cluster.dask_jobqueue_utils import (
    get_dashboard_link,
    init_cluster_variables,
    stop_cluster,
)

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
from cars.orchestrator.cluster import (  # pylint: disable=C0412
    abstract_cluster,
    abstract_dask_cluster,
)


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

    (
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
    ) = init_cluster_variables(
        nb_workers,
        walltime,
        out_dir,
        activate_dashboard,
        python,
        core_memory=5000,
    )

    # Retrieve PBS queue
    pbs_queue = os.environ.get("CARS_PBS_QUEUE")

    with warnings.catch_warnings():
        # Ignore some internal dask_jobqueue warnings
        # TODO remove when Future warning do not exist anymore
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*extra has been renamed to worker_extra_args*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*job_extra has been renamed to job_extra_directives*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*env_extra has been renamed to job_script_prologue*",
        )
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
            resource_spec="select=1:ncpus={}:mem={}MB".format(nb_cpus, memory),
        )
        logging.info("Dask cluster started")
        cluster.adapt(minimum=nb_workers, maximum=nb_workers)
        client = Client(cluster, timeout=timeout)
        logging.info(
            "Dashboard started at {}".format(get_dashboard_link(cluster))
        )
    return cluster, client
