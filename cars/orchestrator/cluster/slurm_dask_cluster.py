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
Contains abstract function for SLURM dask Cluster
"""

import logging
import os
import warnings

from dask.distributed import Client
from json_checker import Or

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
    from dask_jobqueue import SLURMCluster

from cars.orchestrator.cluster import (  # pylint: disable=C0412
    abstract_cluster,
    abstract_dask_cluster,
)
from cars.orchestrator.cluster.dask_cluster_tools import (
    check_configuration,
    create_checker_schema,
)
from cars.orchestrator.cluster.dask_jobqueue_utils import (
    get_dashboard_link,
    init_cluster_variables,
    stop_cluster,
)


@abstract_cluster.AbstractCluster.register_subclass("slurm_dask")
class SlurmDaskCluster(abstract_dask_cluster.AbstractDaskCluster):
    """
    SlurmDaskCluster
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """
        # overload cluster schema
        overloaded_conf, cluster_schema = create_checker_schema(conf)
        if overloaded_conf["mode"] == "slurm_dask":
            overloaded_conf["account"] = conf.get("account", None)
            overloaded_conf["qos"] = conf.get("qos", None)
            cluster_schema["account"] = Or(None, str)
            cluster_schema["qos"] = Or(None, str)

            if overloaded_conf["account"] is None:
                error_msg = (
                    "'account' parameter must be set for slurm dask cluster"
                )
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        return check_configuration(overloaded_conf, cluster_schema)

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
            account=self.account,
            qos=self.qos,
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
    account=None,
    qos=None,
):
    """Create a Dask cluster.

    Each worker will be spawned in an independent job with a single CPU
    allocated to it, and will use a single process. This is done to maximize
    CPU utilization and minimize scheduling delay.

    The CARS_SLURM_QUEUE environment variable, if defined, is used
    to specify the queue in which worker jobs are scheduled.

    :param nb_workers: Number of dask workers
    :type nb_workers: int
    :param walltime: Walltime for each dask worker
    :type walltime: string
    :param out_dir: Output directory
    :type out_dir: string
    :param timeout: timeout of the cluster client
    :type timeout: int
    :param activate_dashboard: option to activate the dashborad server mode
    :type activate_dashboard: bool
    :param python: specfic python path
    :type python: string
    :param account: SLURM account
    :type account: string
    :param qos: Quality of Service parameter for TREX cluster
    :type qos: string
    :return: Dask cluster and dask client
    :rtype: (dask_jobqueue.SLURMCluster, dask.distributed.Client) tuple
    """
    # Retrieve SLURM queue
    slurm_queue = os.environ.get("CARS_SLURM_QUEUE")

    # retrieve current python path if None
    (
        python,
        nb_workers_per_job,
        memory,
        _,
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
        core_memory=7000,
        cluster_name="SLURM",
    )

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
        if qos:
            qos = ["--qos=" + qos]
            logging.info("Quality of Service option: {}".format(qos[0]))
        cluster = SLURMCluster(
            processes=nb_workers_per_job,
            cores=nb_workers_per_job,
            memory="{}MiB".format(memory),
            local_directory=local_directory,
            account=account,
            walltime=walltime,
            interface="ib0",
            queue=slurm_queue,
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
            job_extra_directives=qos,
        )
        logging.info("Dask cluster started")
        cluster.adapt(minimum=nb_workers, maximum=nb_workers)
        client = Client(cluster, timeout=timeout)
        logging.info(
            "Dashboard started at {}".format(get_dashboard_link(cluster))
        )
    return cluster, client
