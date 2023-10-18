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
Contains functions for local dask Cluster
"""

# Standard imports
import logging

# Third party imports
from dask.distributed import Client, LocalCluster

# CARS imports
from cars.orchestrator.cluster import abstract_cluster, abstract_dask_cluster
from cars.orchestrator.cluster.dask_cluster_tools import (
    check_configuration,
    create_checker_schema,
)


@abstract_cluster.AbstractCluster.register_subclass("local_dask")
class LocalDaskCluster(abstract_dask_cluster.AbstractDaskCluster):
    """
    LocalDaskCluster
    """

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        return check_configuration(*create_checker_schema(conf))

    def start_dask_cluster(self):
        """
        Start dask cluster
        """

        return start_local_cluster(
            self.nb_workers, activate_dashboard=self.activate_dashboard
        )

    def cleanup(self):
        """
        Cleanup cluster

        """
        stop_local_cluster(self.cluster, self.client)
        logging.info("Dask cluster closed")


def start_local_cluster(nb_workers, timeout=600, activate_dashboard=False):
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

    if activate_dashboard:
        dashboard_address = ":0"
    else:
        dashboard_address = None

    cluster = LocalCluster(
        n_workers=nb_workers,
        threads_per_worker=1,
        dashboard_address=dashboard_address,
    )

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
